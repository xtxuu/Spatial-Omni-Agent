"""PyDESeq2 donor-level negative-binomial Wald contrasts."""
import warnings

import numpy as np
import pandas as pd

from ..contracts import AnalysisError


def design_metadata(metadata, request):
    model = pd.DataFrame(index=metadata.analysis_donor_id.astype(str))
    used, dropped, terms = [], [], []
    numeric = list(request.numeric_covariates)
    categorical = list(request.categorical_covariates)
    if metadata.analysis_study_id.nunique() > 1:
        if not request.allow_multistudy:
            raise AnalysisError('MULTISTUDY', 'Pooling multiple studies needs explicit approval and a study-adjusted design', status='clarification_required')
        if 'analysis_study_id' not in categorical: categorical.append('analysis_study_id')
    for i, field in enumerate(numeric):
        values = pd.to_numeric(metadata[field], errors='coerce').to_numpy(float)
        if not np.isfinite(values).all(): raise AnalysisError('COVARIATE_VALUES', f'{field}: expected complete numeric values')
        if np.std(values) == 0: dropped.append({'field': field, 'reason': 'constant'}); continue
        alias = f'numeric_{i}'
        model[alias] = (values-np.mean(values))/np.std(values)
        terms.append(alias); used.append({'field': field, 'term': alias, 'type': 'numeric_zscore'})
    for i, field in enumerate(categorical):
        values = metadata[field].astype(str).to_numpy()
        if len(set(values)) < 2: dropped.append({'field': field, 'reason': 'constant'}); continue
        alias = f'categorical_{i}'; model[alias] = pd.Categorical(values)
        terms.append(alias); used.append({'field': field, 'term': alias, 'type': 'categorical'})
    model['group'] = pd.Categorical(metadata.group.astype(str).to_numpy(), categories=[request.control, request.case])
    formula = '~ ' + ' + '.join(terms + ['group'])
    from formulaic import model_matrix
    design = np.asarray(model_matrix(formula, model), dtype=float)
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise AnalysisError('CONFOUNDED_DESIGN', 'The design matrix is not full rank; group and covariates/study may be confounded')
    if design.shape[0] <= design.shape[1]: raise AnalysisError('NO_RESIDUAL_DF', 'The design leaves no residual degrees of freedom')
    return model, formula, {'covariates_used': used, 'covariates_dropped': dropped,
                           'design_rank': int(np.linalg.matrix_rank(design)), 'n_coefficients': design.shape[1]}


def run_deg(counts, metadata, genes, request):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    from threadpoolctl import threadpool_limits
    model, formula, diagnostics = design_metadata(metadata, request)
    values = counts.toarray()
    keep = (values >= request.min_gene_count).sum(axis=0) >= request.min_donors_with_gene_count
    if not keep.any(): raise AnalysisError('NO_TESTABLE_GENES', 'No genes passed the donor-level count filter')
    table = genes.loc[keep].reset_index(drop=True)
    frame = pd.DataFrame(values[:, keep], index=metadata.analysis_donor_id.astype(str), columns=table.gene_id)
    frame.index.name = 'analysis_donor_id'
    if (frame.sum(axis=1) == 0).any(): raise AnalysisError('EMPTY_DONOR', 'A donor has zero counts after gene filtering')
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=request.n_cpus):
        warnings.simplefilter('always')
        dds = DeseqDataSet(counts=frame, metadata=model, design=formula,
            n_cpus=request.n_cpus, refit_cooks=True, quiet=True)
        dds.deseq2()
        stats = DeseqStats(dds, contrast=['group', request.case, request.control],
                           n_cpus=request.n_cpus, alpha=request.fdr, quiet=True)
        stats.summary()
    results = stats.results_df.copy()
    results.index.name = 'gene_id'
    results = results.reset_index().merge(table, on='gene_id', how='left', validate='one_to_one')
    results = results.rename(columns={'log2FoldChange': 'effect', 'stat': 'statistic', 'lfcSE': 'effect_se'})
    results['effect_kind'] = 'log2_fold_change_case_vs_control'
    diagnostics.update(design=formula, method='PyDESeq2 negative-binomial Wald test',
        normalization='PyDESeq2 size-factor estimation on raw donor pseudobulk counts',
        n_input_genes=len(genes), n_count_filter_passed=int(keep.sum()),
        warnings=list(dict.fromkeys(str(w.message) for w in captured))[:30])
    return results, diagnostics
