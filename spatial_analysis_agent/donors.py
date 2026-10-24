"""Auditable donor keys; missing/ambiguous identities never become sample replicates."""
import json
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from .contracts import AnalysisError


def text(value):
    if pd.isna(value): return None
    value = str(value).strip()
    return None if value.casefold() in {'', 'nan', 'none', 'null', '<na>', 'unknown', 'na', 'n/a'} else value


def key(parts): return ':'.join(quote(str(p), safe='') for p in parts)


def read_overrides(path):
    if path is None: return {}
    table = pd.read_csv(Path(path), dtype=str, keep_default_na=False)
    required = {'sample_uid', 'original_donor_id', 'analysis_donor_id', 'reviewed', 'reason'}
    if not required.issubset(table): raise AnalysisError('DONOR_OVERRIDE', f'Required columns: {sorted(required)}')
    result = {}
    for row in table.to_dict('records'):
        lookup = (row['sample_uid'], row['original_donor_id'])
        if lookup in result or row['reviewed'].casefold() != 'true' or not text(row['reason']) or not text(row['analysis_donor_id']):
            raise AnalysisError('DONOR_OVERRIDE', 'Each override must be unique, reviewed=true, and include ID and reason')
        result[lookup] = row
    return result


def assign_donors(sample, overrides, request):
    rows, assigned = [], []
    for _, obs in sample.obs.iterrows():
        raw = text(obs['donor_id']); source = text(obs['source']); dataset = text(obs['dataset_id'])
        if not source or not dataset: raise AnalysisError('DONOR_SCOPE', 'Source and dataset must be known')
        override = overrides.get((sample.uid, raw or ''))
        identity = text(obs.get('donor_identity_status'))
        if identity: identity = identity.casefold()
        existing = text(obs.get('donor_key'))
        if override:
            donor = key([source, dataset, 'reviewed', override['analysis_donor_id']])
            status, reason = 'reviewed_override', override['reason']
        elif raw is None:
            donor, status, reason = None, 'excluded_missing', 'Original donor ID is absent'
        elif identity in {'ambiguous', 'missing', 'unknown', 'unresolved'} or (dataset == 'GSE202011' and identity not in {'reviewed_scope'}):
            donor, status, reason = None, 'excluded_ambiguous', 'Donor identity needs a reviewed cohort mapping'
        elif existing:
            donor = key([source, dataset, 'existing', existing])
            status, reason = 'source_scoped', 'Preserved source-scoped donor key'
        else:
            donor, status, reason = key([source, dataset, 'native', raw]), 'source_scoped', 'Source + dataset + original donor ID'
        if donor is None and request.missing_donor == 'error':
            raise AnalysisError('DONOR_UNRESOLVED', f'{sample.uid}: {reason}', status='clarification_required')
        assigned.append(donor)
        rows.append(dict(sample_uid=sample.uid, source=source, dataset_id=dataset,
            original_donor_id=raw, source_donor_key=existing, analysis_donor_id=donor,
            donor_resolution=status, reason=reason))
    frame = sample.obs.copy(); frame['analysis_donor_id'] = pd.Series(assigned, index=frame.index, dtype='string')
    frame['analysis_study_id'] = [key([s, d]) for s, d in zip(frame.source.astype(str), frame.dataset_id.astype(str))]
    return frame, pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def donor_metadata(frames, request):
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.loc[frame.analysis_donor_id.notna() & frame[request.group_field].astype(str).isin([request.case, request.control])]
    rows = []
    fields = [request.group_field, 'source', 'dataset_id', 'analysis_study_id', 'species'] + request.numeric_covariates + request.categorical_covariates
    fields = list(dict.fromkeys(fields))
    for donor, subset in frame.groupby('analysis_donor_id', sort=True):
        row = {'analysis_donor_id': donor, 'sample_ids': json.dumps(sorted(set(subset['sample_id'].astype(str))))}
        for field in fields:
            if field not in subset or subset[field].isna().any():
                raise AnalysisError('DONOR_COVARIATE', f'{donor}: missing {field}', status='clarification_required')
            values = subset[field].astype(str).unique()
            if len(values) != 1:
                code = 'DONOR_GROUP_CONFLICT' if field == request.group_field else 'DONOR_COVARIATE_CONFLICT'
                raise AnalysisError(code, f'{donor}: {field} varies across spots/sections', status='clarification_required')
            row[field] = values[0]
        row['group'] = row[request.group_field]
        rows.append(row)
    return pd.DataFrame(rows)
