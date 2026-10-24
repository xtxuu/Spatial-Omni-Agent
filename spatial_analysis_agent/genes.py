"""Explicit common-feature axes; no missing-feature zero filling or symbol guesses."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .contracts import AnalysisError


@dataclass
class GeneAxis:
    table: pd.DataFrame
    columns: dict
    audit: pd.DataFrame
    namespace: str


def align_features(samples, request):
    identifiers, names, positions, audit = {}, {}, {}, []
    id_key = 'canonical_gene_id' if request.gene_alignment == 'canonical' else 'gene_id'
    name_key = 'canonical_gene_symbol' if request.gene_alignment == 'canonical' else 'gene_symbol'
    for sample in samples:
        var = sample.var
        if id_key not in var:
            raise AnalysisError('GENE_INDEX_REQUIRED', f'{sample.uid}: missing {id_key}; build/re-export with the gene index or explicitly select native alignment', status='clarification_required')
        valid = var[id_key].notna() & var[id_key].astype(str).ne('')
        if request.gene_alignment == 'canonical':
            if 'alignment_eligible' not in var: raise AnalysisError('GENE_INDEX_REQUIRED', sample.uid)
            valid &= var.alignment_eligible.eq(True)
        ids = var.loc[valid, id_key].astype(str)
        duplicate = ids.duplicated(keep=False)
        if duplicate.any():
            raise AnalysisError('DUPLICATE_GENE', f'{sample.uid}: {id_key} has duplicate targets', details={'examples': ids[duplicate].head(10).tolist()})
        rows = np.flatnonzero(valid.to_numpy())
        identifiers[sample.uid] = ids.tolist()
        positions[sample.uid] = dict(zip(ids, rows))
        labels = var.loc[valid, name_key] if name_key in var else ids
        for gid, symbol in zip(ids, labels):
            if gid not in names: names[gid] = str(symbol) if pd.notna(symbol) and str(symbol) else gid
        audit.append(dict(sample_uid=sample.uid, source_features=len(var), eligible_features=len(ids), excluded_features=len(var)-len(ids)))
    common = set(identifiers[samples[0].uid])
    for ids in identifiers.values(): common.intersection_update(ids)
    ordered = [gid for gid in identifiers[samples[0].uid] if gid in common]
    if len(ordered) < request.min_common_genes:
        raise AnalysisError('COMMON_GENES', f'Only {len(ordered)} common eligible genes; at least {request.min_common_genes} required', details={'feature_audit': audit})
    for row in audit: row.update(common_features=len(ordered), eligible_not_common=row['eligible_features']-len(ordered))
    return GeneAxis(pd.DataFrame({'gene_id': ordered, 'gene_symbol': [names[g] for g in ordered]}),
        {uid: np.asarray([mapping[g] for g in ordered], dtype=np.int64) for uid, mapping in positions.items()},
        pd.DataFrame(audit), 'Ensembl_gene_versionless_with_preserved_PAR_Y' if request.gene_alignment == 'canonical' else 'source_native_exact')
