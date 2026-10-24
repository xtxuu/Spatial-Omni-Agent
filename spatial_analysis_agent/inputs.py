"""Metadata inspection and bounded CSR reads from spatial-bundle-1.0 artifacts."""
from dataclasses import dataclass
from collections import Counter
from pathlib import Path

from anndata.io import read_elem, sparse_dataset
import h5py
import numpy as np
import pandas as pd

from spatial_loader.utils import digest, load, resolve
from .contracts import AnalysisError


@dataclass
class SampleInput:
    uid: str
    spec: dict
    path: Path
    obs: pd.DataFrame
    var: pd.DataFrame
    xy: np.ndarray
    uns: dict
    snapshot: tuple


class BundleInput:
    def __init__(self, manifest_path, *, max_metadata_bytes=512*1024**2):
        self.path = Path(manifest_path).resolve(); self.root = self.path.parent
        self.manifest = load(self.path)
        if self.manifest.get('format_version') != 'spatial-bundle-1.0':
            raise AnalysisError('BUNDLE_FORMAT', 'Expected a data-agent spatial-bundle-1.0 manifest')
        if self.manifest.get('expression_representation') != 'raw_counts':
            raise AnalysisError('COUNTS_REQUIRED', 'Input must declare raw_counts')
        self.manifest_sha256 = digest(self.path)
        self.samples = []
        seen_samples, seen_obs = set(), set()
        metadata_bytes = 0
        var_cache = {}
        for spec in self.manifest.get('samples', []):
            uid = spec['sample_uid']
            if uid in seen_samples: raise AnalysisError('DUPLICATE_SAMPLE', uid)
            seen_samples.add(uid)
            path = resolve(self.root, spec['matrix_path'])
            if not path.is_file() or path.stat().st_size != spec['matrix_bytes']:
                raise AnalysisError('INPUT_CHANGED', str(path))
            with h5py.File(path, 'r') as f:
                obs, var = read_elem(f['obs']), read_elem(f['var'])
                uns = read_elem(f['uns']) if 'uns' in f else {}
                if 'obsm/spatial' not in f: raise AnalysisError('COORDINATES', uid)
                xy = f['obsm/spatial'][:]
                node = spec.get('counts_node', 'layers/counts')
                if node != 'layers/counts' or node not in f or f[node].attrs.get('encoding-type') != 'csr_matrix':
                    raise AnalysisError('COUNTS_FORMAT', 'CSR layers/counts required')
                if tuple(f[node].attrs['shape']) != (len(obs), len(var)):
                    raise AnalysisError('AXIS_SHAPE', uid)
            if (len(obs), len(var)) != (spec['n_obs'], spec['n_vars']) or not obs.index.is_unique or not var.index.is_unique:
                raise AnalysisError('AXIS_IDENTITY', uid)
            if xy.shape != (len(obs), 2) or not np.isfinite(xy).all(): raise AnalysisError('COORDINATES', uid)
            # Many source samples have exactly the same feature metadata; share it.
            signature = (len(var), int(pd.util.hash_pandas_object(var, index=True).sum()))
            if signature in var_cache and var.equals(var_cache[signature]):
                var = var_cache[signature]
            else:
                var_cache[signature] = var
                metadata_bytes += int(var.memory_usage(deep=True).sum())
            metadata_bytes += int(obs.memory_usage(deep=True).sum()) + xy.nbytes
            if metadata_bytes > max_metadata_bytes:
                raise AnalysisError('METADATA_BUDGET', 'Bundle metadata exceed the budget; select a smaller cohort or increase max_memory_bytes')
            overlap = seen_obs.intersection(obs.index.astype(str))
            if overlap: raise AnalysisError('DUPLICATE_OBSERVATION', str(next(iter(overlap))))
            seen_obs.update(obs.index.astype(str))
            for field in ['source', 'dataset_id', 'sample_id', 'species', 'organ', 'donor_id']:
                if field not in obs: raise AnalysisError('MISSING_METADATA', f'{uid}: {field}')
            if 'sample_uid' in obs and not obs['sample_uid'].astype(str).eq(uid).all():
                raise AnalysisError('SAMPLE_IDENTITY', uid)
            self.samples.append(SampleInput(uid, spec, path, obs, var, xy, uns,
                (path.stat().st_size, path.stat().st_mtime_ns)))
        if not self.samples: raise AnalysisError('EMPTY_INPUT', 'No sample artifacts in the input bundle')

    def verify(self, progress):
        if digest(self.path) != self.manifest_sha256: raise AnalysisError('INPUT_CHANGED', str(self.path))
        for sample in progress(self.samples, desc='Verify input artifacts', unit='sample'):
            if digest(sample.path) != sample.spec['matrix_sha256']:
                raise AnalysisError('INPUT_HASH', sample.uid)

    def blocks(self, sample, request, columns=None):
        stat = sample.path.stat()
        if (stat.st_size, stat.st_mtime_ns) != sample.snapshot: raise AnalysisError('INPUT_CHANGED', sample.uid)
        with h5py.File(sample.path, 'r') as f:
            group = f['layers/counts']; disk = sparse_dataset(group)
            ptr = group['indptr'][:]
            cost = group['data'].dtype.itemsize + group['indices'].dtype.itemsize
            start = 0
            while start < len(sample.obs):
                stop = min(start + request.block_rows, len(sample.obs))
                while stop > start and 3 * int(ptr[stop] - ptr[start]) * cost > request.max_memory_bytes // 3:
                    stop = start + (stop - start) // 2
                if stop == start: raise AnalysisError('MEMORY_BUDGET', 'One sparse row exceeds the read budget')
                matrix = disk[start:stop, :].tocsr()
                values = matrix.data
                if values.size and (not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.rint(values)).all()):
                    raise AnalysisError('RAW_COUNTS', f'{sample.uid}: counts must be finite nonnegative integers')
                if values.size and values.max() > 2**53:
                    raise AnalysisError('COUNT_RANGE', 'Counts exceed the safely supported numeric range')
                if columns is not None: matrix = matrix[:, columns].tocsr()
                yield start, stop, matrix
                start = stop

    def inventory(self):
        labels = {}
        for field in ['source', 'dataset_id', 'organ', 'species', 'condition', 'disease', 'cell_type', 'donor_identity_status']:
            counts = Counter()
            for sample in self.samples:
                if field in sample.obs: counts.update({str(k): int(v) for k, v in sample.obs[field].dropna().astype(str).value_counts().items()})
            if counts: labels[field] = dict(n_unique=len(counts), values=[dict(value=k, n_obs=int(v)) for k, v in counts.most_common(200)])
        return dict(bundle_id=self.manifest.get('bundle_id'), n_samples=len(self.samples), n_obs=sum(len(s.obs) for s in self.samples),
            n_features_range=[min(len(s.var) for s in self.samples), max(len(s.var) for s in self.samples)],
            obs_fields=sorted(set.intersection(*(set(s.obs.columns) for s in self.samples))), labels=labels,
            gene_index_id=self.manifest.get('gene_index_id'),
            canonical_mapping_available=all({'canonical_gene_id', 'alignment_eligible'}.issubset(s.var) for s in self.samples),
            selection=self.manifest.get('selection'),
            feature_scope='subset' if any(s.spec.get('full_sample_features') is False for s in self.samples) else 'full_or_legacy_unknown')
