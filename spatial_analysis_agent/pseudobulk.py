"""Sum raw counts across retained spots/sections of each scoped donor."""
import numpy as np
from scipy import sparse

from .contracts import AnalysisError


def aggregate_counts(bundle, prepared, axis, metadata, request, progress):
    shape = (len(metadata), len(axis.table))
    if shape[0] * shape[1] * 8 > request.max_memory_bytes // 2:
        raise AnalysisError('MEMORY_BUDGET', 'Pseudobulk array exceeds the configured budget')
    matrix = np.zeros(shape, dtype=np.int64)
    donors = {d: i for i, d in enumerate(metadata.analysis_donor_id)}
    for item in progress(prepared, desc='Aggregate donor pseudobulk', unit='sample'):
        positions = axis.columns[item.sample.uid]
        for start, stop, counts in bundle.blocks(item.sample, request, positions):
            selected = item.selected[start:stop]
            names = item.obs.analysis_donor_id.iloc[start:stop]
            for donor in names.loc[selected].dropna().unique():
                mask = selected & names.eq(donor).fillna(False).to_numpy(bool)
                values = np.asarray(counts[mask].astype(np.int64).sum(axis=0)).ravel()
                if (values < 0).any() or (matrix[donors[donor]] > np.iinfo(np.int64).max-values).any():
                    raise AnalysisError('COUNT_OVERFLOW', 'Pseudobulk integer range exceeded')
                matrix[donors[donor]] += values
    return sparse.csr_matrix(matrix)
