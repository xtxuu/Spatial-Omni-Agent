"""Within-sample Moran's I, donor means, Welch group comparison and BH FDR."""
import warnings

import numpy as np
import pandas as pd
from scipy import sparse, stats

from ..contracts import AnalysisError
from ..preprocessing import nearest_neighbors


def bh_adjust(pvalues):
    values = np.asarray(pvalues, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise AnalysisError('PVALUES', 'Expected finite p-values in [0,1]')
    order = np.argsort(values, kind='stable')
    adjusted = np.minimum.accumulate((values[order]*len(values)/np.arange(1, len(values)+1))[::-1])[::-1]
    output = np.empty(len(values)); output[order] = np.clip(adjusted, 0, 1)
    return output


def normalized_blocks(bundle, prepared, columns, request):
    for start, stop, counts in bundle.blocks(prepared.sample, request, columns):
        mask = prepared.selected[start:stop]
        if not mask.any(): continue
        library = prepared.library_size[start:stop][mask]
        matrix = counts[mask].astype(np.float64)
        matrix = matrix.multiply((10000/library)[:, None]).tocsr()
        np.log1p(matrix.data, out=matrix.data)
        yield matrix


def morans_i(matrix, coordinates, neighbors, max_memory_bytes):
    n = len(coordinates)
    if n < 3: return np.full(matrix.shape[1], np.nan)
    links = nearest_neighbors(coordinates, neighbors)
    graph = sparse.csr_matrix((np.ones(links.size),
        (np.repeat(np.arange(n), links.shape[1]), links.ravel())), shape=(n, n))
    graph = graph.maximum(graph.T); graph.setdiag(0); graph.eliminate_zeros()
    total = float(graph.sum())
    if not total: return np.full(matrix.shape[1], np.nan)
    chunk = min(64, max_memory_bytes // max(1, n*8*5))
    if chunk < 1: raise AnalysisError('MEMORY_BUDGET', 'One spatial gene vector exceeds the memory budget')
    scores = np.full(matrix.shape[1], np.nan)
    for start in range(0, matrix.shape[1], chunk):
        stop = min(start+chunk, matrix.shape[1])
        values = matrix[:, start:stop].toarray()
        centered = values-values.mean(axis=0)
        denominator = (centered**2).sum(axis=0)
        numerator = (centered*(graph@centered)).sum(axis=0)
        scores[start:stop] = np.divide((n/total)*numerator, denominator,
            out=np.full(stop-start, np.nan), where=denominator > 0)
    return scores


def run_spatial_pattern(bundle, prepared, axis, metadata, request, progress):
    if metadata.analysis_study_id.nunique() > 1:
        raise AnalysisError('SPATIAL_MULTISTUDY', 'Welch spatial screening is supported per study; run separate studies rather than an unadjusted pooled test', status='clarification_required')
    n, sums, squares = 0, np.zeros(len(axis.table)), np.zeros(len(axis.table))
    for item in progress(prepared, desc='Screen spatial gene variance', unit='sample'):
        for matrix in normalized_blocks(bundle, item, axis.columns[item.sample.uid], request):
            n += matrix.shape[0]; sums += np.asarray(matrix.sum(axis=0)).ravel()
            squares += np.asarray(matrix.power(2).sum(axis=0)).ravel()
    if not n: raise AnalysisError('NO_SPOTS', 'No selected observations')
    variance = squares/n-(sums/n)**2
    ordered = np.argsort(-variance, kind='stable')[:min(request.max_spatial_genes, len(variance))]
    genes = axis.table.iloc[ordered].reset_index(drop=True)
    sample_rows, skipped = [], []
    for item in progress(prepared, desc="Compute sample Moran's I", unit='sample'):
        obs = item.obs.loc[item.selected]
        if len(obs) < 3: skipped.append({'sample_uid': item.sample.uid, 'reason': 'fewer than three spots'}); continue
        if obs.analysis_donor_id.nunique() != 1 or obs[request.group_field].nunique() != 1:
            raise AnalysisError('SPATIAL_FRAME', f'{item.sample.uid}: one donor and group per sample frame required')
        blocks, size = [], 0
        for matrix in normalized_blocks(bundle, item, axis.columns[item.sample.uid][ordered], request):
            size += matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
            if size*3 > request.max_memory_bytes: raise AnalysisError('MEMORY_BUDGET', 'Spatial sample exceeds memory budget')
            blocks.append(matrix)
        matrix = sparse.vstack(blocks, format='csr')
        scores = morans_i(matrix, item.sample.xy[item.selected], request.neighbors, request.max_memory_bytes)
        for gene, score in zip(genes.to_dict('records'), scores):
            if np.isfinite(score): sample_rows.append(dict(sample_uid=item.sample.uid,
                analysis_donor_id=str(obs.analysis_donor_id.iloc[0]), group=str(obs[request.group_field].iloc[0]),
                **gene, morans_i=float(score)))
    samples = pd.DataFrame(sample_rows)
    if samples.empty: raise AnalysisError('NO_SPATIAL_SCORES', 'No finite sample-level spatial scores')
    donors = samples.groupby(['analysis_donor_id', 'group', 'gene_id', 'gene_symbol'],
        as_index=False, observed=True).morans_i.mean()
    rows = []
    for (gid, symbol), group in donors.groupby(['gene_id', 'gene_symbol'], sort=False):
        case = group.loc[group.group.eq(request.case), 'morans_i'].to_numpy(float)
        control = group.loc[group.group.eq(request.control), 'morans_i'].to_numpy(float)
        if min(len(case), len(control)) < request.min_replicates_per_group: continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            test = stats.ttest_ind(case, control, equal_var=False)
        pvalue = float(test.pvalue)
        if not np.isfinite(pvalue):
            # No within-group variability: identical constant scores contain no difference.
            if np.ptp(case) == 0 and np.ptp(control) == 0 and case[0] == control[0]: pvalue = 1.
            else: continue
        rows.append(dict(gene_id=gid, gene_symbol=symbol, effect=float(case.mean()-control.mean()),
            effect_kind='delta_morans_i_case_minus_control', statistic=float(test.statistic) if np.isfinite(test.statistic) else np.nan,
            pvalue=pvalue, case_mean_morans_i=float(case.mean()), control_mean_morans_i=float(control.mean()),
            case_donors=len(case), control_donors=len(control)))
    results = pd.DataFrame(rows)
    if results.empty: raise AnalysisError('SPATIAL_REPLICATION', 'No genes have enough finite donor scores')
    results['padj'] = bh_adjust(results.pvalue.to_numpy())
    diagnostics = dict(method="Within-sample symmetric binary kNN Moran's I; donor section means; Welch t-test; BH FDR",
        normalization='log1p(counts * 10000 / total native-feature spot counts)',
        variance_screen='case/control QC-passing spots only', n_common_genes=len(axis.table),
        n_genes_screened=len(ordered), n_tested=len(results), neighbors=request.neighbors,
        skipped_samples=skipped, graph_scope='sample_uid; never connect sections or donors')
    return results, diagnostics, {'sample_morans_i': samples, 'donor_morans_i': donors,
        'screened_genes': genes.assign(variance=variance[ordered])}
