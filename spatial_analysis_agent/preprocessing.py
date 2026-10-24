"""Spot QC, cell-type selection and within-sample spatial neighborhoods."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .contracts import AnalysisError


@dataclass
class PreparedSample:
    sample: object
    obs: pd.DataFrame
    qc: pd.DataFrame
    selected: np.ndarray
    library_size: np.ndarray


def cell_type_mask(obs, request):
    required = {'dominant_cell_type', 'cell_type_top_weight', 'cell_type_weight_margin'}
    if request.cell_type_mode == 'top5':
        required = {f'cell_type_{i}' for i in range(1, 6)} | {f'cell_type_weight_{i}' for i in range(1, 6)}
    if not required.issubset(obs): raise AnalysisError('CELL_TYPE_METADATA', f'Missing {sorted(required-set(obs))}')
    if request.cell_type_mode == 'top5':
        mask = np.zeros(len(obs), dtype=bool)
        for rank in range(1, 6):
            mask |= (obs[f'cell_type_{rank}'].astype('string').eq(request.cell_type).fillna(False).to_numpy(bool)
                     & pd.to_numeric(obs[f'cell_type_weight_{rank}'], errors='coerce').ge(request.min_top_weight).to_numpy(bool))
        return mask
    return (obs.dominant_cell_type.astype('string').eq(request.cell_type).fillna(False).to_numpy(bool)
            & pd.to_numeric(obs.cell_type_top_weight, errors='coerce').ge(request.min_top_weight).to_numpy(bool)
            & pd.to_numeric(obs.cell_type_weight_margin, errors='coerce').ge(request.min_margin).to_numpy(bool))


def nearest_neighbors(coordinates, k):
    """Explicitly remove self indices, including when coordinate ties occur."""
    n = len(coordinates)
    if n < 2: return np.empty((n, 0), dtype=np.int64)
    count = min(k, n-1)
    _, candidates = cKDTree(coordinates).query(coordinates, k=count+1)
    candidates = np.asarray(candidates).reshape(n, count+1)
    return np.asarray([row[row != i][:count] for i, row in enumerate(candidates)], dtype=np.int64)


def neighborhood_mask(xy, available, anchors, donors, neighbors):
    mask = np.zeros(len(xy), dtype=bool)
    for donor in pd.Series(donors).dropna().unique():
        indices = np.flatnonzero(available & np.asarray(pd.Series(donors).eq(donor).fillna(False), dtype=bool))
        if not len(indices): continue
        anchor_local = np.flatnonzero(anchors[indices])
        if not len(anchor_local): continue
        mask[indices[anchor_local]] = True
        nearest = nearest_neighbors(xy[indices], neighbors)
        if nearest.size: mask[indices[nearest[anchor_local].ravel()]] = True
    return mask


def prepare_sample(bundle, sample, obs, request):
    n = len(obs)
    library = np.zeros(n, dtype=np.float64); detected = np.zeros(n, dtype=np.int64)
    mitochondrial = np.zeros(n, dtype=np.float64)
    symbols = sample.var['gene_symbol'].astype('string').fillna('').astype(str) if 'gene_symbol' in sample.var else pd.Series(['']*len(sample.var))
    mt = symbols.str.startswith(('MT-', 'mt-')).to_numpy()
    if request.max_pct_mito is not None and not mt.any():
        raise AnalysisError('MITO_FEATURES', f'{sample.uid}: mitochondrial features cannot be identified', status='clarification_required')
    for start, stop, matrix in bundle.blocks(sample, request):
        matrix.eliminate_zeros()
        library[start:stop] = np.asarray(matrix.sum(axis=1)).ravel()
        detected[start:stop] = matrix.getnnz(axis=1)
        if mt.any(): mitochondrial[start:stop] = np.asarray(matrix[:, mt].sum(axis=1)).ravel()
    pct = np.divide(mitochondrial*100, library, out=np.zeros(n), where=library > 0) if mt.any() else np.full(n, np.nan)
    passed = (library >= request.min_counts) & (detected >= request.min_detected_genes) & (library > 0)
    if request.require_in_tissue:
        if 'in_tissue' not in obs: raise AnalysisError('IN_TISSUE', f'{sample.uid}: missing in_tissue')
        passed &= pd.to_numeric(obs.in_tissue, errors='coerce').eq(1).to_numpy()
    if request.max_pct_mito is not None: passed &= pct <= request.max_pct_mito
    cohort = obs[request.group_field].astype(str).isin([request.case, request.control]).to_numpy()
    available = passed & cohort & obs.analysis_donor_id.notna().to_numpy()
    selected = available.copy()
    anchors = np.zeros(n, dtype=bool)
    if request.method in {'cell_type_enriched_deg', 'spatial_neighborhood_deg'}:
        anchors = cell_type_mask(obs, request) & available
        selected = anchors if request.method == 'cell_type_enriched_deg' else neighborhood_mask(
            sample.xy, available, anchors, obs.analysis_donor_id, request.neighbors)
    qc = pd.DataFrame(dict(observation_id=obs.index.astype(str), sample_uid=sample.uid,
        analysis_donor_id=obs.analysis_donor_id.to_numpy(), group=obs[request.group_field].astype(str).to_numpy(),
        total_counts=library, n_genes_by_counts=detected, pct_counts_mito=pct,
        pass_qc=passed, in_requested_cohort=cohort, anchor=anchors, selected=selected))
    return PreparedSample(sample, obs, qc, selected, library)


def retain_donors(prepared, metadata, request):
    counts = {}
    for item in prepared:
        for donor, count in item.obs.loc[item.selected, 'analysis_donor_id'].value_counts().items():
            counts[donor] = counts.get(donor, 0) + int(count)
    table = metadata.copy()
    if table.empty: raise AnalysisError('NO_DONORS', 'No usable donor identities remain')
    table['n_selected_spots'] = table.analysis_donor_id.map(counts).fillna(0).astype(int)
    table['retained'] = table.n_selected_spots.ge(request.min_spots_per_donor)
    table['low_spot_count'] = table.retained & table.n_selected_spots.lt(request.warn_spots_per_donor)
    keep = set(table.loc[table.retained, 'analysis_donor_id'])
    for item in prepared:
        item.selected &= item.obs.analysis_donor_id.isin(keep).to_numpy()
        item.qc['selected'] = item.selected
    selected = table.loc[table.retained].reset_index(drop=True)
    sizes = selected.group.value_counts().to_dict() if len(selected) else {}
    if any(sizes.get(label, 0) < request.min_replicates_per_group for label in [request.case, request.control]):
        raise AnalysisError('INSUFFICIENT_DONORS', 'Too few independent donors after QC/selection',
            details={'group_sizes': {str(k): int(v) for k, v in sizes.items()}, 'donor_counts': table.to_dict('records')})
    return selected, table
