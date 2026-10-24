"""Atomic run directories, software provenance and result-file hashes."""
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import json
import time
import uuid

from spatial_loader.utils import digest, dump
from .contracts import AnalysisError


def versions():
    result = {}
    for package in ['spatial-omics-loader', 'anndata', 'h5py', 'numpy', 'pandas', 'scipy', 'pydeseq2', 'formulaic', 'matplotlib']:
        try: result[package] = version(package)
        except PackageNotFoundError: result[package] = 'not_installed'
    return result


def publish_directory(source, destination):
    """Publish a completed UUID directory, retrying only transient access errors."""
    for attempt in range(8):
        try:
            source.rename(destination)
            return
        except PermissionError:
            if attempt == 7: raise
            time.sleep(min(.05 * 2**attempt, .4))


class RunRecorder:
    def __init__(self, output_root, bundle_root):
        output_root, bundle_root = Path(output_root).resolve(), Path(bundle_root).resolve()
        if output_root.is_relative_to(bundle_root): raise AnalysisError('READ_ONLY_INPUT', 'Output must be outside the input bundle')
        if any((p/'dataset.json').is_file() for p in [output_root, *output_root.parents]):
            raise AnalysisError('READ_ONLY_RELEASE', 'Analysis output cannot be inside a publication dataset')
        self.run_id = uuid.uuid4().hex
        self.final = output_root/self.run_id
        self.directory = output_root/(self.run_id+'.building')
        self.directory.mkdir(parents=True, exist_ok=False)

    def record(self, event, payload):
        with (self.directory/'evidence.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(), event=event, payload=payload),
                                    ensure_ascii=False, allow_nan=False)+'\n')

    def finish(self, status, request, input_info, summary=None, error=None):
        self.record('completed', {'status': status})
        files = {str(p.relative_to(self.directory)).replace('\\', '/'): digest(p)
                 for p in self.directory.rglob('*') if p.is_file()}
        manifest = dict(format_version='spatial-analysis-result-1.0', run_id=self.run_id,
            created_utc=datetime.now(timezone.utc).isoformat(), status=status, request=request,
            input=input_info, software_versions=versions(), summary=summary, error=error,
            files_sha256=files)
        dump(self.directory/'manifest.json', manifest)
        # Windows indexers/antivirus can briefly hold a just-written plot or CSV.
        # Retry the same atomic rename; never delete/overwrite an existing run.
        publish_directory(self.directory, self.final)
        receipt = dict(status=status, run_id=self.run_id, manifest_uri=str(self.final/'manifest.json'))
        if summary:
            receipt['warnings'] = summary.get('warnings', [])
            receipt['low_spot_donors'] = summary.get('low_spot_donors', [])
            receipt.update(result_label=summary['result_label'], n_fdr_significant=summary['n_fdr_significant'],
                gene_sets={k: str(self.final/v) for k, v in summary['gene_sets'].items()},
                top_positive=summary['top_positive'], top_negative=summary['top_negative'])
        if error: receipt['error'] = error
        return receipt
