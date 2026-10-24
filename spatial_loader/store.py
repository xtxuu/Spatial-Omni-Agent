"""Small public interface: describe -> query -> fetch."""
import hashlib
import json
import tempfile
import uuid
from pathlib import Path

import anndata as ad
import numpy as np
from PIL import Image

from .catalog import Catalog,compile_filters,cell_clause
from .contracts import FILTER_FIELDS,NUMERIC_FIELDS,FORMAT_VERSION,DataError,QueryResult
from .reader import iter_blocks
from .export import export_bundle
from .utils import dump,load,resolve,sql_string,digest


class SpatialStore:
    def __init__(self,root,*,cache_dir=None,block_rows=2048,max_read_bytes=256*1024**2):
        self.root=Path(root).resolve();self.release=load(self.root/'dataset.json')
        if self.release.get('format_version')!=FORMAT_VERSION:raise DataError('RELEASE_VERSION','Unsupported dataset format')
        self.block_rows=block_rows;self.max_read_bytes=max_read_bytes
        if block_rows<1 or max_read_bytes<1:raise DataError('INVALID_BUDGET','Positive budgets required')
        self.cache=Path(cache_dir or Path(tempfile.gettempdir())/'spatial_loader'/self.release['release_id']).resolve()
        if self.cache.is_relative_to(self.root):raise DataError('READ_ONLY_RELEASE','Query cache must be outside release')
        self.cache.mkdir(parents=True,exist_ok=True)
        self.catalog=Catalog(self.root,self.release)

    def close(self):
        self.catalog.close()

    def __enter__(self):return self

    def __exit__(self,*args):self.close()

    def _current(self):
        if load(self.root/'dataset.json')['release_id']!=self.release['release_id']:
            raise DataError('STALE_RELEASE','Reopen SpatialStore after changing the release')
        index_path=self.root/'gene_index/index.json'
        version=load(index_path)['index_id'] if index_path.exists() else None
        if version!=(self.catalog.identity.version if self.catalog.identity else None):
            raise DataError('STALE_GENE_INDEX','Reopen SpatialStore after adding/changing the gene identity index')

    def gene_systems(self,*,dataset_id=None,sample_uid=None,limit=100,offset=0):
        """Report detected namespaces/coverage, separately from source annotation release."""
        self._current()
        if self.catalog.identity is None:return dict(status='not_available',results=[])
        return self.catalog.identity.summaries(dataset_id=dataset_id,sample_uid=sample_uid,limit=limit,offset=offset)

    def resolve_genes(self,identifiers,*,species='Homo sapiens',input_namespace='ensembl_gene'):
        self._current()
        if self.catalog.identity is None:raise DataError('GENE_INDEX_REQUIRED','Build the gene identity index first')
        return dict(gene_index_id=self.catalog.identity.version,species=species,
            results=self.catalog.identity.resolver().resolve(identifiers,species,input_namespace))

    def describe(self,filters=None,fields=None,*,limit=30,offset=0,include_schema=False):
        self._current()
        fields=fields or ['source','organ','disease','cell_type','annotation_method']
        if not isinstance(fields,(list,tuple)) or not set(fields).issubset(FILTER_FIELDS):
            raise DataError('UNKNOWN_FIELD','Unknown describe field')
        if not 1<=limit<=500 or offset<0:raise DataError('INVALID_LIMIT','Use limit 1..500 and nonnegative offset')
        clauses,values=compile_filters(filters);where=' AND '.join(clauses) or 'TRUE'
        result={}
        with self.catalog.connect() as db:
            for field in fields:
                n,missing=db.execute(f'SELECT count(DISTINCT "{field}"),count(*) FILTER (WHERE "{field}" IS NULL) FROM obs WHERE {where}',values).fetchone()
                rows=db.execute(f'SELECT "{field}",count(*) AS n FROM obs WHERE {where} GROUP BY 1 ORDER BY n DESC,1 NULLS LAST LIMIT ? OFFSET ?',values+[limit,offset]).fetchall()
                result[field]=dict(n_unique=int(n),n_missing=int(missing),offset=offset,
                                  values=[dict(value=v,count=int(count)) for v,count in rows])
        response=dict(dataset_version=self.release['release_id'],fields=result,
                      cell_type_modes=['dominant','top5'],expression_representation='raw_counts')
        response['gene_identity']=dict(status='available' if self.catalog.identity else 'not_available',
            index_id=self.catalog.identity.version if self.catalog.identity else None,
            source_fields_preserved=True,canonical_keys=['canonical_gene_id','canonical_gene_symbol'] if self.catalog.identity else [])
        if include_schema:response['query_schema']={f:dict(type='number' if f in NUMERIC_FIELDS else 'string',
                operators=['eq','in','is_null']+(['gt','gte','lt','lte','between'] if f in NUMERIC_FIELDS else []))
                for f in sorted(FILTER_FIELDS)}
        return response

    def query(self,filters=None,*,cell_type=None,roi=None):
        self._current();clauses,values=compile_filters(filters)
        extra,params=cell_clause(cell_type);clauses+=extra;values+=params
        if roi is not None:
            if not isinstance(roi,dict) or set(roi)!={'sample_uid','bounds','unit'}:
                raise DataError('INVALID_ROI','ROI requires sample_uid, bounds=[xmin,ymin,xmax,ymax], unit')
            sample=self.catalog.sample(roi['sample_uid']);bounds=roi['bounds']
            if roi['unit']!='pixel' or sample['coordinate_unit']!='pixel':
                raise DataError('INVALID_ROI','Pixel ROI requires verified pixel coordinates')
            if not isinstance(bounds,(list,tuple)) or len(bounds)!=4 or not all(isinstance(v,(int,float)) and np.isfinite(v) for v in bounds):
                raise DataError('INVALID_ROI','Four finite bounds required')
            x0,y0,x1,y1=bounds
            if x0>x1 or y0>y1:raise DataError('INVALID_ROI','Bounds are reversed')
            clauses+=['sample_uid=?','x BETWEEN ? AND ?','y BETWEEN ? AND ?'];values+=[roi['sample_uid'],x0,x1,y0,y1]
        request=dict(api_version='0.1.0',release_id=self.release['release_id'],filters=filters or {},cell_type=cell_type,roi=roi)
        request['gene_index_id']=self.catalog.identity.version if self.catalog.identity else None
        try:encoded=json.dumps(request,sort_keys=True,allow_nan=False)
        except (TypeError,ValueError) as error:raise DataError('INVALID_QUERY',str(error))
        qid=hashlib.sha256(encoded.encode()).hexdigest();folder=self.cache/qid
        if (folder/'query.json').exists():return self._query_result(qid)
        folder.mkdir(exist_ok=True);selection=folder/'selection.parquet';temporary=folder/'selection.part.parquet'
        where=' AND '.join(clauses) or 'TRUE'
        with self.catalog.connect() as db:
            db.execute(f'COPY (SELECT * FROM obs WHERE {where} ORDER BY sample_uid,row_idx) TO {sql_string(temporary)} (FORMAT PARQUET, COMPRESSION ZSTD)',values)
            temporary.replace(selection)
            target=sql_string(selection)
            stats=db.execute(f'''SELECT count(*),count(DISTINCT sample_uid),count(DISTINCT donor_key),
                count(*) FILTER (WHERE donor_identity_status='missing'),
                count(*) FILTER (WHERE donor_identity_status='ambiguous') FROM read_parquet({target})''').fetchone()
            rows=db.execute(f'''SELECT source,dataset_id,disease,annotation_method,spatial_unit_type,
                count(*) AS n_obs,count(DISTINCT sample_uid) AS n_samples,count(DISTINCT donor_key) AS n_donors
                FROM read_parquet({target}) GROUP BY ALL ORDER BY 1,2,3,4 LIMIT 1001''').fetchall()
        keys=['source','dataset_id','disease','annotation_method','spatial_unit_type','n_obs','n_samples','n_donors']
        if len(rows)>1000:
            raise DataError('SUMMARY_LIMIT','Too many summary groups; narrow the query')
        result=QueryResult(qid,self.release['release_id'],*[int(v) for v in stats],groups=[dict(zip(keys,r)) for r in rows])
        if self.catalog.identity:
            with self.catalog.connect() as db:
                uids=[r[0] for r in db.execute(f'SELECT DISTINCT sample_uid FROM read_parquet({target})').fetchall()]
            result=QueryResult(qid,self.release['release_id'],*[int(v) for v in stats],
                groups=[dict(zip(keys,r)) for r in rows],
                gene_id_systems=self.catalog.identity.summaries(sample_uids=uids,limit=1000)['results'])
        dump(folder/'query.json',dict(request=request,result=result.to_dict(),selection_sha256=digest(selection)))
        return result

    def _folder(self,query_id):
        if isinstance(query_id,QueryResult):query_id=query_id.query_id
        if not isinstance(query_id,str) or len(query_id)!=64 or any(c not in '0123456789abcdef' for c in query_id):
            raise DataError('INVALID_QUERY_ID','Expected a query ID from this store')
        folder=self.cache/query_id
        if not (folder/'query.json').is_file():raise DataError('UNKNOWN_QUERY',query_id)
        record=load(folder/'query.json')
        if record['request']['release_id']!=self.release['release_id']:raise DataError('STALE_SELECTION',query_id)
        if record['request'].get('gene_index_id')!=(self.catalog.identity.version if self.catalog.identity else None):
            raise DataError('STALE_SELECTION','Rerun query after changing the gene identity index')
        if not (folder/'selection.parquet').exists() or digest(folder/'selection.parquet')!=record['selection_sha256']:
            raise DataError('STALE_SELECTION','Selection cache changed; remove this query cache directory and rerun query')
        return folder

    def _query_result(self,qid):
        result=load(self._folder(qid)/'query.json')['result']
        keys=['query_id','dataset_version','n_obs','n_samples','n_known_donors',
              'n_obs_missing_donor','n_obs_ambiguous_donor','groups']
        return QueryResult(**{k:result[k] for k in keys},gene_id_systems=result.get('gene_id_systems',[]))

    def _plans(self,folder,genes,gene_key):
        selection=sql_string(folder/'selection.parquet')
        with self.catalog.connect() as db:
            uids=[r[0] for r in db.execute(f'SELECT DISTINCT sample_uid FROM read_parquet({selection}) ORDER BY 1').fetchall()]
        # Resolve requested genes in every shard before writing any artifact files.
        if genes is not None:
            for uid in uids:self.catalog.genes(uid,genes,gene_key)
        for uid in uids:
            with self.catalog.connect() as db:
                frame=db.execute(f'SELECT * FROM read_parquet({selection}) WHERE sample_uid=? ORDER BY row_idx',[uid]).df()
            if frame.memory_usage(deep=True).sum()>self.max_read_bytes:
                raise DataError('METADATA_BUDGET','One sample metadata selection exceeds read budget; narrow query or raise budget')
            yield self.catalog.sample(uid),frame,self.catalog.genes(uid,genes,gene_key)

    def fetch(self,query_id,*,genes=None,gene_key='gene_symbol',delivery='artifact',output_dir=None,
              include_images='paths',max_memory_bytes=512*1024**2):
        self._current();folder=self._folder(query_id);qid=folder.name
        if delivery not in ['artifact','memory','iterator']:raise DataError('INVALID_DELIVERY',delivery)
        if include_images not in ['paths','copy','none']:raise DataError('INVALID_IMAGES',include_images)
        if delivery!='artifact' and include_images=='copy':raise DataError('INVALID_IMAGES','copy requires artifact delivery')
        if genes is not None and not genes:raise DataError('INVALID_GENES','Use None for all native genes, or a nonempty gene list')
        if gene_key not in ['gene_symbol','gene_id','source_var_name','canonical_gene_id','canonical_gene_symbol']:raise DataError('INVALID_GENE_KEY',gene_key)
        plans=self._plans(folder,genes,gene_key)
        if delivery=='artifact':
            target=Path(output_dir) if output_dir else self.root.parent/'exports'/uuid.uuid4().hex
            return export_bundle(self,qid,plans,target,include_images)
        def blocks():
            for sample,frame,features in plans:
                for block in iter_blocks(self.root,sample,frame,features,block_rows=self.block_rows,max_read_bytes=self.max_read_bytes):
                    block.adata.uns['data_access']={'dataset_root':str(self.root),'sample_uid':sample['sample_uid']}
                    if self.catalog.identity:
                        block.adata.uns['gene_identity']={'gene_index_id':self.catalog.identity.version,
                            'source_identifiers_preserved':True,'counts_modified':False}
                    if include_images=='none':
                        for entry in block.adata.uns.get('spatial',{}).values():entry['metadata']={}
                    yield block
        if delivery=='iterator':return blocks()
        grouped={};budget=0
        for block in blocks():
            a=block.adata
            payload=a.X.data.nbytes+a.X.indices.nbytes+a.X.indptr.nbytes
            payload+=int(a.obs.memory_usage(deep=True).sum()+a.var.memory_usage(deep=True).sum())
            payload+=sum(v.nbytes for v in a.obsm.values())
            budget+=2*payload  # Allow for final sparse stacking copies.
            if budget>max_memory_bytes:raise DataError('MEMORY_BUDGET','Use iterator/artifact delivery or narrow the query')
            grouped.setdefault(block.sample_uid,[]).append(a)
        return {uid:(parts[0] if len(parts)==1 else ad.concat(parts,join='inner',merge='same',uns_merge='same'))
                for uid,parts in grouped.items()}

    def get_image(self,sample_uid,*,resolution='lowres',load_pixels=False):
        self._current()
        if resolution not in ['hires','lowres']:raise DataError('INVALID_RESOLUTION',resolution)
        frame=self.catalog.assets(sample_uid);frame=frame[frame.resolution==resolution]
        if len(frame)!=1:raise DataError('MISSING_IMAGE',f'{sample_uid}/{resolution}')
        asset=frame.iloc[0].to_dict();path=resolve(self.root,asset['path'])
        if not path.is_file():raise DataError('MISSING_IMAGE',str(path))
        if asset['scale'] is not None and not np.isfinite(asset['scale']):asset['scale']=None
        asset['uri']=str(path);asset['coordinate_formula']='image_xy = full_resolution_xy * scale'
        if load_pixels:
            with Image.open(path) as image:asset['pixels']=np.asarray(image).copy()
        return asset

    def verify_integrity(self,*,include_matrices=True):
        self._current()
        expected=dict(self.release['catalog_sha256'])
        if include_matrices:
            with self.catalog.connect() as db:
                expected.update(dict(db.execute('SELECT matrix_path,matrix_sha256 FROM samples').fetchall()))
                expected.update(dict(db.execute('SELECT path,sha256 FROM assets').fetchall()))
        for path,checksum in expected.items():
            if digest(resolve(self.root,path))!=checksum:raise DataError('INTEGRITY_FAILED',path)
        if self.catalog.identity:
            for path,checksum in self.catalog.identity.manifest['files_sha256'].items():
                if digest(resolve(self.root/'gene_index',path))!=checksum:raise DataError('GENE_INDEX_HASH',path)
        return {'status':'verified','files':len(expected),'dataset_version':self.release['release_id']}
