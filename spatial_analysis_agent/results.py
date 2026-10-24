"""Stable result manifests and self-describing gene sets for downstream agents."""
from pathlib import Path
import numpy as np
import pandas as pd

from spatial_loader.utils import dump, load, digest, resolve
from .contracts import AnalysisError


GENE_SETS = ('significant_positive', 'significant_negative', 'target_positive', 'target_negative',
             'exploratory_positive', 'exploratory_negative', 'top_positive', 'top_negative')


def write_results(directory, results, request, context):
    directory = Path(directory)
    if results.gene_id.duplicated().any(): raise AnalysisError('RESULT_IDENTITY', 'Duplicate result genes')
    table = results.copy()
    table['fdr_significant'] = table.padj.lt(request.fdr)
    effect_pass = table.effect.abs().ge(request.min_abs_log2_fold_change) if request.method != 'spatial_pattern' else pd.Series(True, index=table.index)
    table['target_gene'] = table.fdr_significant & effect_pass
    table['exploratory_candidate'] = ~table.fdr_significant & table.pvalue.lt(request.exploratory_p) & effect_pass
    table = table.sort_values(['padj', 'pvalue'], na_position='last', kind='stable').reset_index(drop=True)
    table.to_csv(directory/'results_all.csv.gz', index=False)
    table.loc[table.fdr_significant].to_csv(directory/'results_fdr.csv', index=False)
    table.loc[table.target_gene].to_csv(directory/'target_genes.csv', index=False)
    table.loc[table.exploratory_candidate].to_csv(directory/'exploratory_candidates.csv', index=False)
    valid = table.loc[np.isfinite(table.pvalue) & np.isfinite(table.effect)]
    if valid.empty: raise AnalysisError('NO_VALID_RESULTS', 'No finite effect/p-value results')
    background = valid.gene_id.astype(str).tolist()
    (directory/'tested_gene_universe.txt').write_text('\n'.join(background)+'\n', encoding='utf-8')
    gene_sets = {}
    for direction, sign in [('positive', 1), ('negative', -1)]:
        directional = valid.loc[valid.effect*sign > 0].copy()
        directional['_rank_p'] = np.where(directional.fdr_significant, directional.padj, directional.pvalue)
        directional['_abs_effect'] = directional.effect.abs()
        directional = directional.sort_values(['fdr_significant', '_rank_p', '_abs_effect'], ascending=[False, True, False], kind='stable')
        masks = {'significant': directional.fdr_significant, 'target': directional.target_gene,
                 'exploratory': directional.exploratory_candidate}
        for category in ['significant', 'target', 'exploratory', 'top']:
            selected = directional.head(request.top_n) if category == 'top' else directional.loc[masks[category]]
            name = f'{category}_{direction}'
            entries = []
            for row in selected.to_dict('records'):
                entries.append(dict(gene_id=str(row['gene_id']), gene_symbol=str(row['gene_symbol']), effect=float(row['effect']),
                    pvalue=float(row['pvalue']), padj=float(row['padj']) if np.isfinite(row['padj']) else None,
                    fdr_significant=bool(row['fdr_significant'])))
            payload = dict(format_version='spatial-gene-set-1.0', name=name, method=request.method,
                warnings=context.get('warnings', []), low_spot_donors=context.get('low_spot_donors', []),
                synthetic_only=context.get('synthetic_only', False),
                case=request.case, control=request.control, gene_namespace=context['gene_namespace'],
                species=context['species'], gene_index_id=context.get('gene_index_id'),
                effect_kind=str(valid.effect_kind.iloc[0]),
                interpretation=('higher expression in case' if sign == 1 else 'higher expression in control') if request.method != 'spatial_pattern'
                    else ('stronger spatial autocorrelation in case' if sign == 1 else 'stronger spatial autocorrelation in control'),
                selection=category, fdr_threshold=request.fdr, background_path='../tested_gene_universe.txt',
                gene_ids=[r['gene_id'] for r in entries], gene_symbols=[r['gene_symbol'] for r in entries], entries=entries)
            dump(directory/'gene_sets'/f'{name}.json', payload)
            gene_sets[name] = f'gene_sets/{name}.json'
    summary = dict(result_label='SIGNIFICANT' if table.fdr_significant.any() else 'EXPLORATORY',
        n_result_genes=len(table), n_valid_tested_genes=len(valid), n_fdr_significant=int(table.fdr_significant.sum()),
        n_target_genes=int(table.target_gene.sum()), effect_kind=str(valid.effect_kind.iloc[0]),
        gene_sets=gene_sets, background_path='tested_gene_universe.txt',
        top_positive=load(directory/gene_sets['top_positive'])['entries'],
        top_negative=load(directory/gene_sets['top_negative'])['entries'])
    dump(directory/'result_summary.json', summary)
    return summary


def plot_results(directory, results, request):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table = results.loc[np.isfinite(results.effect) & np.isfinite(results.padj)].copy()
    if table.empty: return
    y = -np.log10(table.padj.clip(lower=1e-300))
    deg = request.method != 'spatial_pattern'
    fig, axes = plt.subplots(1, 2 if deg else 1, figsize=(12, 5) if deg else (6, 5), squeeze=False)
    ax = axes[0, 0]; ax.scatter(table.effect, y, s=8, alpha=.5)
    ax.axhline(-np.log10(request.fdr), linestyle='--', color='red')
    ax.set(xlabel='log2 fold change (case/control)' if deg else "Delta Moran's I (case-control)", ylabel='-log10 adjusted p-value')
    if deg:
        ax.axvline(request.min_abs_log2_fold_change, linestyle='--', color='grey')
        ax.axvline(-request.min_abs_log2_fold_change, linestyle='--', color='grey')
        axes[0, 1].scatter(table.baseMean, table.effect, s=8, alpha=.5)
        axes[0, 1].set(xscale='log', xlabel='Mean normalized pseudobulk count', ylabel='log2 fold change')
    fig.suptitle(f'{request.case} versus {request.control}'); fig.tight_layout()
    fig.savefig(Path(directory)/'result_plots.png', dpi=150); plt.close(fig)


def load_analysis_manifest(path, *, verify=True):
    path = Path(path).resolve(); manifest = load(path)
    if manifest.get('format_version') != 'spatial-analysis-result-1.0': raise AnalysisError('RESULT_FORMAT', str(path))
    if verify:
        for relative, checksum in manifest['files_sha256'].items():
            if digest(resolve(path.parent, relative)) != checksum: raise AnalysisError('RESULT_HASH', relative)
    return manifest


def load_gene_list(manifest_path, name='significant_positive', *, verify=True):
    """Return IDs plus species, effect semantics and tested-universe metadata."""
    manifest_path = Path(manifest_path).resolve()
    manifest = load_analysis_manifest(manifest_path, verify=verify)
    if manifest['status'] != 'success': raise AnalysisError('NO_RESULT', 'Analysis did not complete successfully')
    if name not in GENE_SETS: raise AnalysisError('GENE_SET', name)
    path = resolve(manifest_path.parent, manifest['summary']['gene_sets'][name])
    result = load(path)
    # Background references are resolved from the trusted result manifest, not ../ in arbitrary input.
    result['background_uri'] = str(resolve(manifest_path.parent, manifest['summary']['background_path']))
    result['analysis_manifest_uri'] = str(manifest_path)
    return result
