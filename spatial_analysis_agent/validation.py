"""Design and capability checks before expensive statistical computation."""
from .contracts import AnalysisError


def validate_scope(bundle, request):
    samples, skipped = [], []
    all_groups, species, organs, studies = set(), set(), set(), set()
    warnings = []
    for sample in bundle.samples:
        if request.group_field not in sample.obs: raise AnalysisError('GROUP_FIELD', request.group_field)
        groups = set(sample.obs[request.group_field].dropna().astype(str))
        all_groups.update(groups)
        if not groups.intersection({request.case, request.control}): skipped.append(sample.uid); continue
        samples.append(sample)
        species.update(sample.obs.species.dropna().astype(str))
        organs.update(sample.obs.organ.dropna().astype(str))
        studies.update(zip(sample.obs.source.astype(str), sample.obs.dataset_id.astype(str)))
        if sample.spec.get('full_sample_features') is False and not request.allow_gene_subset:
            raise AnalysisError('GENE_SUBSET_INPUT', 'Re-export all source genes for QC/library normalization, or explicitly authorize a restricted-feature analysis', status='clarification_required')
        if 'full_sample_features' not in sample.spec:
            warnings.append('Legacy bundle lacks feature-completeness metadata; full native-feature library sizes cannot be certified')
        if request.method in {'spatial_pattern', 'spatial_neighborhood_deg'} and sample.spec.get('full_sample_observations') is not True:
            if not request.allow_partial_spatial_context:
                raise AnalysisError('SPATIAL_CONTEXT', 'Spatial analysis requires full sample observations; re-export complete sections or explicitly authorize partial-context interpretation', status='clarification_required')
            warnings.append('Spatial graph is conditional on the explicitly supplied partial sample context')
    if request.case not in all_groups or request.control not in all_groups:
        raise AnalysisError('CONTRAST_LABELS', f'Exact available {request.group_field} labels: {sorted(all_groups)}', status='clarification_required')
    if len(species) != 1: raise AnalysisError('SPECIES', 'One species per analysis is required')
    if len(organs) != 1 and not request.allow_multiple_organs:
        raise AnalysisError('MULTIPLE_ORGANS', 'Specify a single organ or explicitly authorize a combined-organ design', status='clarification_required')
    if len(studies) > 1 and not request.allow_multistudy:
        raise AnalysisError('MULTISTUDY', 'Multiple studies require an explicit study-adjusted design', status='clarification_required')
    if request.method == 'spatial_pattern' and len(studies) > 1:
        raise AnalysisError('SPATIAL_MULTISTUDY', 'Run this unadjusted spatial screen separately per study', status='clarification_required')
    if request.method in {'cell_type_enriched_deg', 'spatial_neighborhood_deg'}:
        available_types = set()
        for sample in samples:
            annotation = sample.uns.get('cell_type_annotation', {})
            available_types.update(str(x) for x in annotation.get('cell_type_names', []))
            for field in ['dominant_cell_type'] + [f'cell_type_{i}' for i in range(1,6)]:
                if field in sample.obs: available_types.update(sample.obs[field].dropna().astype(str))
        if request.cell_type not in available_types:
            raise AnalysisError('CELL_TYPE_LABEL', f'Exact available cell types: {sorted(available_types)}', status='clarification_required')
    if request.min_replicates_per_group < 3: warnings.append('Two donors per group is a minimum execution threshold; three or more is preferable')
    return samples, dict(species=next(iter(species)), organs=sorted(organs), studies=sorted([list(s) for s in studies]),
                         synthetic_only=all(source.casefold() == 'synthetic' for source, _ in studies),
                         skipped_out_of_contrast_samples=skipped, warnings=sorted(set(warnings)))
