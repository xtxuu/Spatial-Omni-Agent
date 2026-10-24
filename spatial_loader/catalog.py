"""Parquet metadata access. No expression or image pixels are opened here."""
import duckdb
import pandas as pd

from .contracts import FILTER_FIELDS, NUMERIC_FIELDS, DataError
from .utils import resolve, sql_string


def compile_filters(filters):
    if filters is None:filters={}
    if not isinstance(filters,dict):raise DataError('INVALID_FILTER','filters must be an object')
    clauses=[];values=[]
    for field,request in filters.items():
        if field not in FILTER_FIELDS:raise DataError('UNKNOWN_FIELD',f'Unknown filter field: {field}')
        if not isinstance(request,dict):request={'in':list(request)} if isinstance(request,(list,tuple)) else {'eq':request}
        if not request:raise DataError('INVALID_FILTER',f'Empty condition for {field}')
        col='"'+field+'"'
        for op,value in request.items():
            if op=='is_null':
                if not isinstance(value,bool):raise DataError('INVALID_FILTER','is_null requires a boolean')
                clauses.append(f'{col} IS '+('NULL' if value else 'NOT NULL'));continue
            if op not in ['eq','in','gt','gte','lt','lte','between']:
                raise DataError('INVALID_OPERATOR',str(op))
            entries=value if op in ['in','between'] else [value]
            if op in ['in','between'] and not isinstance(value,(list,tuple)):
                raise DataError('INVALID_FILTER',f'{op} requires a list')
            if op=='between' and len(entries)!=2:raise DataError('INVALID_FILTER','between requires two bounds')
            if any(v is None for v in entries):raise DataError('INVALID_FILTER','Use is_null for missing values')
            if field in NUMERIC_FIELDS:
                import math
                if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in entries):
                    raise DataError('INVALID_FILTER',f'{field} requires finite numbers')
            elif any(not isinstance(v,str) for v in entries):
                raise DataError('INVALID_FILTER',f'{field} requires strings')
            if op in ['gt','gte','lt','lte','between'] and field not in NUMERIC_FIELDS:
                raise DataError('INVALID_FILTER','Range operators require numeric fields')
            if op=='in':
                clauses.append(f'{col} IN ({",".join("?" for _ in entries)})' if entries else 'FALSE')
            elif op=='between':clauses.append(f'{col} BETWEEN ? AND ?')
            else:clauses.append(col+' '+{'eq':'=','gt':'>','gte':'>=','lt':'<','lte':'<='}[op]+' ?')
            values.extend(entries)
    return clauses,values


def cell_clause(request):
    if request is None:return [],[]
    if not isinstance(request,dict) or set(request)-{'label','mode','min_weight','status'}:
        raise DataError('INVALID_CELL_TYPE','Use label, mode, min_weight and optional status')
    label=request.get('label');mode=request.get('mode','dominant');weight=request.get('min_weight',0.0)
    if not isinstance(label,str) or not label:raise DataError('INVALID_CELL_TYPE','An exact source label is required')
    if mode not in ['dominant','top5']:
        raise DataError('INVALID_CELL_TYPE','Supported modes: dominant, top5. Top5 is not any-component membership.')
    if isinstance(weight,bool) or not isinstance(weight,(int,float)) or not 0<=weight<=1:
        raise DataError('INVALID_CELL_TYPE','min_weight must be between 0 and 1')
    ranks=[1] if mode=='dominant' else range(1,6)
    parts=[];values=[]
    for i in ranks:
        parts.append(f'("cell_type_{i}" = ? AND "cell_type_weight_{i}" >= ?)');values.extend([label,weight])
    clauses=['('+' OR '.join(parts)+')']
    if 'status' in request:
        more,params=compile_filters({'annotation_status':request['status']});clauses+=more;values+=params
    return clauses,values


class Catalog:
    def __init__(self,root,release):
        self.root=root;self.release=release;self._db=None
        from .gene_identity import GeneIdentityIndex
        self.identity = GeneIdentityIndex(root,release) if (root/'gene_index/index.json').is_file() else None

    def connect(self):
        if self._db is not None:return self._db.cursor()
        db=duckdb.connect(':memory:')
        for table in ['samples','features','assets','obs']:
            spec=self.release['catalog'][table]
            paths=[str(resolve(self.root,p)) for p in spec] if isinstance(spec,list) else [str(resolve(self.root,spec))]
            if not paths or not all(__import__('pathlib').Path(p).is_file() for p in paths):
                db.close();raise DataError('MISSING_CATALOG',table)
            listing='['+','.join(sql_string(p) for p in paths)+']'
            db.execute(f'CREATE VIEW {table} AS SELECT * FROM read_parquet({listing}, union_by_name=true)')
        self._db=db
        return db.cursor()

    def close(self):
        if self._db is not None:self._db.close();self._db=None

    def sample(self,uid):
        with self.connect() as db:
            frame=db.execute('SELECT * FROM samples WHERE sample_uid=?',[uid]).df()
        if len(frame)!=1:raise DataError('UNKNOWN_SAMPLE',uid)
        return frame.iloc[0].to_dict()

    def assets(self,uid):
        with self.connect() as db:return db.execute('SELECT * FROM assets WHERE sample_uid=?',[uid]).df()

    def genes(self,uid,genes,key):
        if key not in ['gene_symbol','gene_id','source_var_name','canonical_gene_id','canonical_gene_symbol']:
            raise DataError('INVALID_GENE_KEY',key)
        sample=self.sample(uid)
        with self.connect() as db:
            if sample.get('features_path'):
                path=sql_string(resolve(self.root,sample['features_path']))
                all_vars=db.execute(f'SELECT * FROM read_parquet({path}) ORDER BY col_idx').df()
            else:all_vars=db.execute('SELECT * FROM features WHERE sample_uid=? ORDER BY col_idx',[uid]).df()
        if self.identity is not None:
            all_vars=self.identity.enrich(uid,all_vars)
        elif key.startswith('canonical_'):
            raise DataError('GENE_INDEX_REQUIRED','Build the gene identity index before canonical queries')
        if genes is None:return all_vars
        if not isinstance(genes,(list,tuple)) or any(not isinstance(g,str) for g in genes) or len(set(genes))!=len(genes):
            raise DataError('INVALID_GENES','genes must be a list of unique strings or None')
        indices=[]
        for gene in genes:
            hits=all_vars.index[all_vars[key]==gene]
            if len(hits)==0:raise DataError('MISSING_GENE',f'{gene!r} is not measured in {uid}')
            if len(hits)>1:
                candidates=all_vars.loc[hits,['gene_id','source_var_name']].to_dict('records')
                raise DataError('AMBIGUOUS_GENE',f'{gene!r} in {uid}: {candidates}')
            if key.startswith('canonical_') and not bool(all_vars.loc[hits[0],'alignment_eligible']):
                raise DataError('UNSAFE_GENE_MAPPING',f'{gene!r} has an ambiguous or duplicate target in {uid}')
            indices.append(hits[0])
        return all_vars.loc[indices].reset_index(drop=True)
