"""Deterministic orchestration. Statistical functions remain independently callable."""
from pathlib import Path
import json

import pandas as pd
from scipy import sparse
from tqdm import tqdm

from spatial_loader.utils import digest, dump
from .contracts import AnalysisError, AnalysisRequest
from .donors import assign_donors, donor_metadata, read_overrides
from .genes import align_features
from .inputs import BundleInput
from .methods.deg import design_metadata, run_deg
from .methods.spatial_pattern import run_spatial_pattern
from .preprocessing import prepare_sample, retain_donors
from .provenance import RunRecorder
from .results import plot_results, write_results
from .validation import validate_scope


def inspect_bundle(manifest_path): return BundleInput(manifest_path).inventory()


def run_analysis(manifest_path, request, *, output_root='analysis_runs', donor_overrides=None,
                 execute=True, show_progress=True):
    if isinstance(request, dict): request = AnalysisRequest.from_dict(request)
    if not isinstance(request, AnalysisRequest): raise TypeError('Expected AnalysisRequest or dict')
    path = Path(manifest_path).resolve()
    recorder = RunRecorder(output_root, path.parent)
    progress = lambda values, **kwargs: tqdm(values, disable=not show_progress, **kwargs)
    input_info = {'manifest_uri': str(path)}
    dump(recorder.directory/'request.json', request.to_dict())
    try:
        bundle = BundleInput(path, max_metadata_bytes=request.max_memory_bytes//2)
        input_info.update(manifest_sha256=bundle.manifest_sha256, bundle_id=bundle.manifest.get('bundle_id'),
            dataset_version=bundle.manifest.get('dataset_version'), gene_index_id=bundle.manifest.get('gene_index_id'),
            selection=bundle.manifest.get('selection'))
        if donor_overrides:
            input_info['donor_overrides_sha256'] = digest(donor_overrides)
        dump(recorder.directory/'input_inventory.json', bundle.inventory())
        samples, context = validate_scope(bundle, request)
        input_info['synthetic_only'] = context['synthetic_only']
        bundle.samples = samples
        overrides = read_overrides(donor_overrides)
        frames, audits = {}, []
        for sample in samples:
            frames[sample.uid], audit = assign_donors(sample, overrides, request); audits.append(audit)
        donor_audit = pd.concat(audits, ignore_index=True)
        donor_audit.to_csv(recorder.directory/'donor_resolution.csv', index=False)
        unresolved = donor_audit.loc[donor_audit.analysis_donor_id.isna(), ['sample_uid', 'original_donor_id']].copy()
        unresolved['analysis_donor_id'] = ''; unresolved['reviewed'] = 'false'; unresolved['reason'] = ''
        unresolved.to_csv(recorder.directory/'donor_override_template.csv', index=False)
        context['n_unresolved_sample_donor_records'] = int(donor_audit.analysis_donor_id.isna().sum())
        metadata = donor_metadata(list(frames.values()), request)
        if metadata.empty: raise AnalysisError('NO_DONORS', 'No usable donor identities for this contrast')
        group_sizes = metadata.group.value_counts().to_dict()
        if any(group_sizes.get(g, 0) < request.min_replicates_per_group for g in [request.case, request.control]):
            raise AnalysisError('INSUFFICIENT_DONORS', 'Too few independent donors before QC', details={'group_sizes': group_sizes})
        # Input samples with no eligible donor cannot contribute to the common feature axis.
        samples = [s for s in samples if frames[s.uid].analysis_donor_id.notna().any()]
        bundle.samples = samples
        axis = align_features(samples, request)
        axis.table.to_csv(recorder.directory/'common_genes.csv.gz', index=False)
        axis.audit.to_csv(recorder.directory/'gene_alignment_audit.csv', index=False)
        context.update(gene_namespace=axis.namespace, gene_index_id=bundle.manifest.get('gene_index_id'),
                       n_common_genes=len(axis.table), group_sizes_pre_qc={str(k): int(v) for k, v in group_sizes.items()})
        # Validate the proposed donor design before reading expression.
        if request.method != 'spatial_pattern':
            _, _, design = design_metadata(metadata, request)
            context['pre_qc_design'] = design
        dump(recorder.directory/'design_evidence.json', context)
        recorder.record('preflight_passed', context)
        if not execute:
            return recorder.finish('planned', request.to_dict(), input_info,
                error={'code': 'DRY_RUN', 'message': 'Metadata preflight passed; counts, QC and statistical estimability have not been executed'})
        bundle.verify(progress)
        prepared = []
        qc_path = recorder.directory/'spot_qc_metrics.csv.gz'
        # gzip append creates valid concatenated members; no giant merged QC dataframe.
        for i, sample in enumerate(progress(samples, desc='QC and region selection', unit='sample')):
            item = prepare_sample(bundle, sample, frames[sample.uid], request)
            prepared.append(item)
        try:
            metadata, donor_counts = retain_donors(prepared, metadata, request)
        finally:
            for i, item in enumerate(prepared): item.qc.to_csv(qc_path, index=False, mode='w' if i == 0 else 'a', header=i == 0)
        donor_counts.to_csv(recorder.directory/'donor_selection.csv', index=False)
        low_spots = donor_counts.loc[donor_counts.low_spot_count]
        context['low_spot_donors'] = low_spots[['analysis_donor_id', 'group', 'n_selected_spots']].to_dict('records')
        if len(low_spots):
            warning = (f'LOW_SPOT_DONORS: {len(low_spots)} retained donors have fewer than '
                       f'{request.warn_spots_per_donor} selected spots; sparse pseudobulks may be unstable. '
                       'See donor_selection.csv; significance does not remove this limitation.')
            context['warnings'].append(warning)
            if show_progress:
                print(f'WARNING: {warning}', flush=True)
                print('  group\tselected_spots\tanalysis_donor_id', flush=True)
                for donor in context['low_spot_donors']:
                    print(f"  {donor['group']}\t{donor['n_selected_spots']}\t{donor['analysis_donor_id']}", flush=True)
        dump(recorder.directory/'design_evidence.json', context)
        metadata.to_csv(recorder.directory/'donor_metadata.csv', index=False)
        context['group_sizes_post_qc'] = {str(k): int(v) for k, v in metadata.group.value_counts().items()}
        context['n_selected_spots'] = sum(int(s.selected.sum()) for s in prepared)
        recorder.record('selection_complete', {'group_sizes': context['group_sizes_post_qc'], 'n_selected_spots': context['n_selected_spots']})
        if request.method == 'spatial_pattern':
            results, diagnostics, extra = run_spatial_pattern(bundle, prepared, axis, metadata, request, progress)
            for name, table in extra.items(): table.to_csv(recorder.directory/(name+'.csv.gz'), index=False)
        else:
            counts = aggregate_and_save(bundle, prepared, axis, metadata, request, progress, recorder.directory)
            results, diagnostics = run_deg(counts, metadata, axis.table, request)
        context['statistics'] = diagnostics
        dump(recorder.directory/'design_evidence.json', context)
        dump(recorder.directory/'statistics.json', diagnostics)
        summary = write_results(recorder.directory, results, request, context)
        try: plot_results(recorder.directory, results, request)
        except Exception as error:
            recorder.record('plot_failed', {'error_type': type(error).__name__, 'message': str(error)})
        summary.update(n_selected_spots=context['n_selected_spots'], group_sizes=context['group_sizes_post_qc'],
            synthetic_only=context['synthetic_only'],
            gene_namespace=axis.namespace, species=context['species'], warnings=context['warnings'],
            low_spot_donors=context['low_spot_donors'],
            interpretation='Selected spatial regions are mixtures, not purified cell expression' if request.method in {'cell_type_enriched_deg','spatial_neighborhood_deg'} else None)
        dump(recorder.directory/'result_summary.json', summary)
        return recorder.finish('success', request.to_dict(), input_info, summary=summary)
    except AnalysisError as error:
        payload = dict(code=error.code, message=str(error), details=error.details)
        dump(recorder.directory/'failure.json', payload)
        return recorder.finish(error.status, request.to_dict(), input_info, error=payload)
    except Exception as error:
        payload = dict(code='EXECUTION_FAILED', error_type=type(error).__name__, message=str(error))
        dump(recorder.directory/'failure.json', payload)
        return recorder.finish('failed', request.to_dict(), input_info, error=payload)


def aggregate_and_save(bundle, prepared, axis, metadata, request, progress, directory):
    from .pseudobulk import aggregate_counts
    counts = aggregate_counts(bundle, prepared, axis, metadata, request, progress)
    sparse.save_npz(directory/'pseudobulk_counts_donors_by_genes.npz', counts)
    axis.table.to_csv(directory/'pseudobulk_gene_metadata.csv.gz', index=False)
    metadata.to_csv(directory/'pseudobulk_donor_metadata.csv', index=False)
    return counts
