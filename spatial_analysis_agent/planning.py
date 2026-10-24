"""Demand-driven multi-job plans, independent preflight and auditable execution."""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

from spatial_loader.utils import digest, dump, load, resolve
from .contracts import AnalysisError, AnalysisRequest, METHODS
from .provenance import RunRecorder, publish_directory


MODEL_FIELDS = {'method', 'case', 'control', 'group_field', 'cell_type',
                'numeric_covariates', 'categorical_covariates', 'top_n'}


@dataclass(frozen=True)
class PlanJob:
    request: dict
    reason: str


@dataclass(frozen=True)
class AnalysisPlan:
    scope: str
    jobs: tuple[PlanJob, ...]
    message: str = ''

    @classmethod
    def from_dict(cls, value, *, model_generated=False):
        if not isinstance(value, dict) or set(value)-{'scope','jobs','message','status'}:
            raise AnalysisError('MODEL_PLAN', 'Unexpected analysis plan fields', status='clarification_required')
        scope = value.get('scope', 'focused')
        if scope not in {'focused', 'comprehensive'}:
            raise AnalysisError('PLAN_SCOPE', 'Use focused or comprehensive', status='clarification_required')
        jobs = value.get('jobs')
        if not isinstance(jobs, list) or not 1 <= len(jobs) <= 16:
            raise AnalysisError('PLAN_SIZE', 'Use 1..16 requested analysis jobs', status='clarification_required')
        parsed = []
        for job in jobs:
            if not isinstance(job, dict) or set(job)-{'request','reason'} or not isinstance(job.get('request'), dict):
                raise AnalysisError('MODEL_PLAN', 'Each job needs request and optional reason', status='clarification_required')
            request = dict(job['request'])
            if request.get('method') not in METHODS:
                raise AnalysisError('METHOD', str(request.get('method')), status='clarification_required')
            if model_generated and set(request)-MODEL_FIELDS:
                raise AnalysisError('MODEL_PLAN', 'Model supplied unauthorized execution overrides', status='clarification_required')
            reason = job.get('reason', '')
            if not isinstance(reason, str): raise AnalysisError('MODEL_PLAN', 'Reason must be text', status='clarification_required')
            parsed.append(PlanJob(request, reason))
            if model_generated and not reason.strip():
                raise AnalysisError('MODEL_PLAN', 'Each selected job needs an objective-based reason', status='clarification_required')
        # Scope describes intent. Execute only the explicitly planned jobs; a broad
        # scope must never silently expand a partial plan into all capabilities.
        return cls(scope, tuple(parsed), str(value.get('message','')))


def execute_plan(plan, *, manifest_uri, output_root, defaults, run_job, execute, show_progress):
    if isinstance(plan, dict): plan = AnalysisPlan.from_dict(plan)
    if not isinstance(plan, AnalysisPlan): raise TypeError('Expected AnalysisPlan or dict')
    recorder = RunRecorder(Path(output_root)/'plans', Path(manifest_uri).resolve().parent)
    normalized, unique, duplicates = [], {}, []
    # Validate all proposed requests before allowing any execution.
    for i, job in enumerate(plan.jobs):
        request = {**defaults, **job.request}
        if request['method'] in {'whole_tissue_deg','spatial_pattern'}: request['cell_type'] = None
        try:
            typed = AnalysisRequest.from_dict(request)
            identity = typed.to_dict(); identity.pop('top_n')
            fingerprint = json.dumps(identity, sort_keys=True, allow_nan=False)
            if fingerprint in unique:
                existing = normalized[unique[fingerprint]]
                if typed.top_n > existing['typed'].top_n:
                    existing['typed'] = AnalysisRequest.from_dict({**existing['typed'].to_dict(), 'top_n':typed.top_n})
                duplicates.append({'requested_position':i,'duplicate_of':existing['job_id']})
                continue
            unique[fingerprint] = len(normalized)
            normalized.append({'job_id':f'job_{len(normalized)+1}', 'reason':job.reason,
                               'typed':typed,'state':'pending'})
        except AnalysisError as error:
            normalized.append({'job_id':f'job_{len(normalized)+1}', 'reason':job.reason,
                'request':request,'state':'blocked','result':{'status':error.status,
                    'error':{'code':error.code,'message':str(error),'details':error.details}}})
    def public_job(job):
        return {k:v for k,v in job.items() if k != 'typed'} | ({'request':job['typed'].to_dict()} if 'typed' in job else {})
    planned = {'format_version':'spatial-analysis-plan-1.0','scope':plan.scope,'message':plan.message,
        'execute_requested':execute,'n_original_jobs':len(plan.jobs),'duplicates_removed':duplicates,
        'jobs':[public_job(j) for j in normalized]}
    dump(recorder.directory/'plan.json',planned)
    for job in normalized:
        if job['state'] == 'blocked': continue
        try:
            check = run_job(job['typed'], execute=False, show_progress=False)
        except Exception as error:
            check = {'status':'failed','error':{'code':'PREFLIGHT_FAILED','error_type':type(error).__name__}}
        job['preflight'] = check
        if check['status'] == 'planned': job['state'] = 'ready'
        else: job['state']='blocked'; job['result']=check
    recorder.record('preflight_complete', {'jobs':[{k:v for k,v in public_job(j).items() if k in {'job_id','state','preflight','result'}} for j in normalized]})
    dump(recorder.directory/'execution.json',{'jobs':[public_job(j) for j in normalized]})
    for job in normalized:
        if job['state'] != 'ready': continue
        if execute:
            if show_progress: print(f"Analysis plan: {job['job_id']} -> {job['typed'].method}", flush=True)
            try: job['result']=run_job(job['typed'],execute=True,show_progress=show_progress)
            except Exception as error:
                job['result']={'status':'failed','error':{'code':'JOB_FAILED','error_type':type(error).__name__}}
            job['state']='completed' if job['result']['status']=='success' else 'failed'
        else: job['result']=job['preflight']; job['state']='planned'
        recorder.record('job_complete',{'job_id':job['job_id'],'state':job['state'],'result':job['result']})
        dump(recorder.directory/'execution.json',{'jobs':[public_job(j) for j in normalized]})
    jobs=[public_job(j) for j in normalized]
    successes=[j for j in jobs if j['state']=='completed']
    planned_jobs=[j for j in jobs if j['state']=='planned']
    blocked=[j for j in jobs if j['state']=='blocked']
    failed=[j for j in jobs if j['state']=='failed']
    if execute:
        status='success' if len(successes)==len(jobs) else 'partial_success' if successes else 'failed' if failed else 'clarification_required'
    else: status='planned' if len(planned_jobs)==len(jobs) else 'partially_planned' if planned_jobs else 'clarification_required'
    children=[j['result']['manifest_uri'] for j in successes if j.get('result',{}).get('manifest_uri')]
    child_records=[{'job_id':j['job_id'],'method':j['request']['method'],
        'cell_type':j['request'].get('cell_type'),'manifest_uri':j['result']['manifest_uri'],
        'manifest_sha256':digest(j['result']['manifest_uri']) if Path(j['result']['manifest_uri']).is_file() else None}
        for j in successes if j.get('result',{}).get('manifest_uri')]
    manifest={'format_version':'spatial-analysis-plan-1.0','plan_id':recorder.run_id,
        'created_utc':datetime.now(timezone.utc).isoformat(),'status':status,'scope':plan.scope,
        'input_manifest_uri':str(Path(manifest_uri).resolve()),'input_manifest_sha256':digest(manifest_uri),
        'message':plan.message,'execute_requested':execute,'jobs':jobs,'duplicates_removed':duplicates,
        'analysis_manifest_uris':children,
        'analysis_manifests':child_records,
        'n_jobs':len(jobs),'n_completed':len(successes),'n_blocked':len(blocked),'n_failed':len(failed),
        'files_sha256':{str(p.relative_to(recorder.directory)).replace('\\','/'):digest(p) for p in recorder.directory.rglob('*') if p.is_file()},
        'interpretation':'Independent analyses; each child has its own tested universe and FDR family; gene lists are not silently combined'}
    dump(recorder.directory/'manifest.json',manifest)
    publish_directory(recorder.directory,recorder.final)
    receipt={k:v for k,v in manifest.items() if k not in {'files_sha256','input_manifest_sha256'}}
    receipt['plan_manifest_uri']=str(recorder.final/'manifest.json')
    # Preserve the original single-analysis response contract for existing clients.
    if len(jobs)==1:
        child=jobs[0].get('result',{})
        receipt.update(child)
    return receipt


def load_analysis_plan(path, *, verify=True):
    path=Path(path).resolve(); manifest=load(path)
    if manifest.get('format_version')!='spatial-analysis-plan-1.0':
        raise AnalysisError('PLAN_FORMAT','Expected a spatial-analysis-plan-1.0 manifest')
    if verify:
        for relative,checksum in manifest['files_sha256'].items():
            if digest(resolve(path.parent,relative))!=checksum: raise AnalysisError('PLAN_HASH',relative)
        for child in manifest['analysis_manifests']:
            if not child.get('manifest_sha256') or digest(child['manifest_uri'])!=child['manifest_sha256']:
                raise AnalysisError('PLAN_CHILD_HASH',child['job_id'])
    return manifest


def load_plan_gene_lists(path, name='significant_positive', *, verify=True):
    """Return separate method-aware gene sets; never union incompatible effects."""
    from .results import load_gene_list
    manifest=load_analysis_plan(path,verify=verify)
    return [{'job_id':child['job_id'],'method':child['method'],'cell_type':child.get('cell_type'),
             'gene_set':load_gene_list(child['manifest_uri'],name,verify=verify)}
            for child in manifest['analysis_manifests']]
