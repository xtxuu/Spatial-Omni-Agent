"""Streaming counts-only exports and a block reader for analysis-agent handoff."""
import shutil
import uuid
from itertools import chain
from pathlib import Path

import anndata as ad
from anndata.io import read_elem
import h5py
import numpy as np
import pandas as pd

from .contracts import DataError
from .reader import make_obs,make_var,metadata,iter_blocks
from .utils import dump,load,resolve,digest


def write_subset(root,sample,frame,features,path,*,block_rows,max_read_bytes,uns):
    # All count data and complete composition arrays are appended in bounded blocks.
    frame=frame.sort_values('row_idx',kind='stable').reset_index(drop=True)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    a=ad.AnnData(X=None,obs=make_obs(frame),var=make_var(features),
        obsm={'spatial':frame[['x','y']].to_numpy()},uns=uns)
    a.write_h5ad(path,compression='gzip')
    with h5py.File(resolve(root,sample['matrix_path']),'r') as src:
        dtype=src[sample['counts_node']+'/data'].dtype
        n_types=src['obsm/cell_type_proportions'].shape[1]
    with h5py.File(path,'a') as out:
        layers=out.require_group('layers');layers.attrs.update({'encoding-type':'dict','encoding-version':'0.1.0'})
        group=layers.create_group('counts');group.attrs.update({'encoding-type':'csr_matrix','encoding-version':'0.1.0','shape':(len(frame),len(features))})
        data=group.create_dataset('data',shape=(0,),maxshape=(None,),dtype=dtype,chunks=True,compression='gzip')
        columns=group.create_dataset('indices',shape=(0,),maxshape=(None,),dtype=np.int64,chunks=True,compression='gzip')
        ptr=group.create_dataset('indptr',shape=(len(frame)+1,),dtype=np.int64,compression='gzip');ptr[0]=0
        proportions=out['obsm'].create_dataset('cell_type_proportions',shape=(len(frame),n_types),
            dtype=np.float32,compression='gzip');proportions.attrs.update({'encoding-type':'array','encoding-version':'0.2.0'})
        n=nnz=0
        for block in iter_blocks(root,sample,frame,features,block_rows=block_rows,max_read_bytes=max_read_bytes):
            m=block.adata.X;end=nnz+m.nnz;data.resize((end,));columns.resize((end,))
            data[nnz:end]=m.data;columns[nnz:end]=m.indices
            ptr[n+1:n+m.shape[0]+1]=m.indptr[1:]+nnz
            proportions[n:n+m.shape[0],:]=block.adata.obsm['cell_type_proportions']
            n+=m.shape[0];nnz=end
        if n!=len(frame):raise DataError('EXPORT_FAILED','Incomplete subset rows')


def export_bundle(store,query_id,plans,output_dir,include_images):
    plans=iter(plans);first=next(plans,None)
    target=Path(output_dir).resolve()
    if target.is_relative_to(store.root):raise DataError('READ_ONLY_RELEASE','Exports must be outside the dataset directory')
    if target.exists():raise DataError('OUTPUT_EXISTS',str(target))
    stage=target.with_name(target.name+'.building')
    if stage.exists():raise DataError('OUTPUT_EXISTS',str(stage))
    stage.mkdir(parents=True)
    members=[]
    for number,(sample,frame,features) in enumerate(chain([] if first is None else [first],plans)):
        _,uns,_=metadata(store.root,sample)
        if store.catalog.identity:
            uns['gene_identity']={'gene_index_id':store.catalog.identity.version,
                'source_identifiers_preserved':True,'counts_modified':False,
                'gene_systems':__import__('json').dumps(store.gene_systems(sample_uid=sample['sample_uid']))}
        assets=store.catalog.assets(sample['sample_uid']).to_dict('records')
        for asset in assets:
            if asset.get('scale') is not None and not np.isfinite(asset['scale']):asset['scale']=None
            if include_images=='copy':
                rel=Path('images')/str(number)/Path(asset['path']).name
                (stage/rel).parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(resolve(store.root,asset['path']),stage/rel)
                asset['path']=str(rel).replace('\\','/');asset['path_base']='bundle_root'
            else:
                asset['uri']=str(resolve(store.root,asset['path']));asset['path_base']='dataset_root'
        if include_images=='none':assets=[]
        for entry in uns.get('spatial',{}).values():
            meta=entry.setdefault('metadata',{})
            meta.clear();meta.update(path_base='bundle_root' if include_images=='copy' else 'dataset_root')
            for asset in assets:meta[asset['resolution']+'_image_path']=asset.get('uri',asset['path'])
        rel=Path('samples')/str(number)/'subset.h5ad'
        write_subset(store.root,sample,frame,features,stage/rel,block_rows=store.block_rows,
            max_read_bytes=store.max_read_bytes,uns=uns)
        members.append(dict(sample_uid=sample['sample_uid'],matrix_path=str(rel).replace('\\','/'),
            n_obs=len(frame),n_vars=len(features),matrix_sha256=digest(stage/rel),matrix_bytes=(stage/rel).stat().st_size,
            source_n_obs=int(sample['n_obs']),source_n_vars=int(sample['n_vars']),
            full_sample_observations=len(frame)==int(sample['n_obs']),
            full_sample_features=len(features)==int(sample['n_vars']),
            counts_node='layers/counts',assets=assets,source_release=store.release['release_id']))
    manifest=dict(format_version='spatial-bundle-1.0',bundle_id=uuid.uuid4().hex,
        query_id=query_id,dataset_version=store.release['release_id'],expression_representation='raw_counts',
        layout='per_sample',normalization_applied=False,batch_correction_applied=False,
        gene_policy='native per-sample axes; explicit missing/ambiguous genes are errors',samples=members)
    manifest['gene_index_id']=store.catalog.identity.version if store.catalog.identity else None
    manifest['selection']=load(store._folder(query_id)/'query.json')['request']
    if store.catalog.identity:
        manifest['gene_systems']=store.catalog.identity.summaries(sample_uids=[s['sample_uid'] for s in members])
    dump(stage/'manifest.json',manifest);stage.rename(target)
    return dict(status='ready',bundle_id=manifest['bundle_id'],manifest_uri=str(target/'manifest.json'),
        dataset_version=manifest['dataset_version'],n_obs=sum(x['n_obs'] for x in members),n_samples=len(members),
        expression_representation='raw_counts',layout='per_sample',images=include_images)


def load_bundle(manifest_path,*,block_rows=2048,max_read_bytes=256*1024**2,verify=True):
    """Yield SampleBlock objects with raw counts in X and layers/counts.

    Does not load a whole multi-sample bundle into memory. Image paths/transforms
    remain in each block's uns. This is suitable for the next agent's Python code.
    """
    manifest_path=Path(manifest_path).resolve();root=manifest_path.parent;manifest=load(manifest_path)
    if manifest.get('format_version')!='spatial-bundle-1.0':raise DataError('BUNDLE_VERSION','Unsupported bundle')
    for sample in manifest['samples']:
        path=resolve(root,sample['matrix_path'])
        if verify and digest(path)!=sample['matrix_sha256']:raise DataError('CORRUPT_BUNDLE',str(path))
        with h5py.File(path,'r') as f:
            obs=read_elem(f['obs']);var=read_elem(f['var']);xy=f['obsm/spatial'][:]
        frame=obs.copy()
        frame['observation_id']=obs.index.astype(str);frame['row_idx']=np.arange(len(obs));frame['x']=xy[:,0];frame['y']=xy[:,1]
        frame=frame.drop(columns=['source_row_idx'],errors='ignore').reset_index(drop=True)
        features=var.copy();features['source_var_name']=var.index.astype(str);features['col_idx']=np.arange(len(var))
        features=features.drop(columns=['source_col_idx'],errors='ignore').reset_index(drop=True)
        yield from iter_blocks(root,sample,frame,features,block_rows=block_rows,max_read_bytes=max_read_bytes)
