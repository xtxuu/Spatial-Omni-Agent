"""Bounded CSR row reads; never an eager read_h5ad/backed layers load."""
from dataclasses import dataclass

import anndata as ad
from anndata.io import read_elem,sparse_dataset
import h5py
import numpy as np
import pandas as pd

from .contracts import OBS_COLUMNS,DataError
from .utils import resolve


@dataclass
class SampleBlock:
    sample_uid: str
    adata: ad.AnnData


def make_obs(frame):
    frame=frame.copy().set_index('observation_id')
    frame.index.name='observation_id'
    frame=frame.rename(columns={'row_idx':'source_row_idx'})
    for key in frame:
        if pd.api.types.is_string_dtype(frame[key]) or frame[key].dtype==object:
            frame[key]=pd.Categorical(frame[key])
    return frame


def make_var(features):
    var=features.copy().set_index('source_var_name')
    var.index.name='feature_id'
    var=var.drop(columns=['sample_uid'],errors='ignore').rename(columns={'col_idx':'source_col_idx'})
    for key in var:
        if pd.api.types.is_string_dtype(var[key]) or var[key].dtype==object:var[key]=pd.Categorical(var[key])
    return var


def metadata(root,sample):
    path=resolve(root,sample['matrix_path'])
    if not path.is_file() or path.stat().st_size!=sample['matrix_bytes']:
        raise DataError('STALE_MATRIX',str(path))
    with h5py.File(path,'r') as f:
        uns=read_elem(f['uns'])
        labels=uns['cell_type_annotation']['cell_type_names']
    return path,uns,len(labels)


def iter_blocks(root,sample,frame,features,*,block_rows=2048,max_read_bytes=256*1024**2):
    if block_rows<1 or max_read_bytes<1:raise DataError('INVALID_BUDGET','Positive read budgets required')
    path,uns,n_types=metadata(root,sample)
    frame=frame.sort_values('row_idx',kind='stable').reset_index(drop=True)
    indices=frame.row_idx.to_numpy(dtype=np.int64)
    columns=features.col_idx.to_numpy(dtype=np.int64)
    if len(set(indices))!=len(indices):raise DataError('DUPLICATE_SELECTION','Duplicate rows')
    with h5py.File(path,'r') as f:
        group=f[sample['counts_node']]
        if tuple(group.attrs['shape'])!=(sample['n_obs'],sample['n_vars']):raise DataError('STALE_MATRIX','Dimensions changed')
        pointers=group['indptr'][:]
        disk=sparse_dataset(group)
        per_entry=group['data'].dtype.itemsize+group['indices'].dtype.itemsize
        # Conservative allowance for both raw-row reads and sliced output buffers.
        costs=2*(pointers[indices+1]-pointers[indices])*per_entry+8*(len(columns)+n_types)
        start=0
        while start<len(frame):
            end=start;used=0
            while end<len(frame) and end-start<block_rows and used+costs[end]<=max_read_bytes:
                used+=costs[end];end+=1
            if end==start:raise DataError('READ_BUDGET',f'One sparse row exceeds read budget in {sample["sample_uid"]}')
            selected=indices[start:end]
            counts=disk[selected,:][:,columns].tocsr()
            # Explicit selection avoids loading all layers, embeddings or source JSON.
            obs_index=f['obs'][f['obs'].attrs['_index']]
            original=obs_index.asstr()[selected]
            if not np.array_equal(original,frame.iloc[start:end].observation_id.astype(str)):
                raise DataError('STALE_SELECTION','Stored matrix row identities changed')
            a=ad.AnnData(counts,obs=make_obs(frame.iloc[start:end]),var=make_var(features),
                obsm={'spatial':f['obsm/spatial'][selected,:],
                      'cell_type_proportions':f['obsm/cell_type_proportions'][selected,:]},uns=uns.copy())
            # Same sparse object in memory; no second matrix allocation is needed.
            a.layers['counts']=a.X
            yield SampleBlock(sample['sample_uid'],a)
            start=end
