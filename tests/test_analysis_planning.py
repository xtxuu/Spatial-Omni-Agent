"""Routing tests without LLM/network/statistical fits; verify actual call counts."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from spatial_analysis_agent import SpatialAnalysisAgent
from spatial_analysis_agent.contracts import AnalysisError, METHODS
from spatial_analysis_agent.planning import AnalysisPlan, execute_plan
from spatial_loader.utils import dump, load


def job(method, **kwargs):
    return {'request': {'method':method,'case':'case','control':'control',**kwargs},'reason':'Explicit user request'}


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory(); self.root=Path(self.temp.name)
        self.input=self.root/'bundle/manifest.json'; dump(self.input,{'fixture':True})
        self.calls=[]
        def runner(request, *, execute, show_progress):
            self.calls.append((request,execute))
            return {'status':'success' if execute else 'planned','manifest_uri':f'{request.method}.json'}
        self.runner=runner

    def tearDown(self): self.temp.cleanup()

    def run_plan(self, plan, **kwargs):
        return execute_plan(plan,manifest_uri=self.input,output_root=self.root/'runs',defaults={},
                            run_job=kwargs.pop('run_job',self.runner),execute=kwargs.pop('execute',True),show_progress=False,**kwargs)

    def test_focused_executes_one_method(self):
        result=self.run_plan({'scope':'focused','jobs':[job('whole_tissue_deg')]})
        self.assertEqual([r.method for r,execute in self.calls if execute],['whole_tissue_deg'])
        self.assertEqual(result['n_jobs'],1)
        self.assertEqual(result['manifest_uri'],'whole_tissue_deg.json')

    def test_selected_two_does_not_add_other_methods(self):
        self.run_plan({'scope':'focused','jobs':[job('whole_tissue_deg'),job('spatial_pattern')]})
        self.assertEqual([r.method for r,execute in self.calls if execute],['whole_tissue_deg','spatial_pattern'])

    def test_comprehensive_executes_four_when_four_are_planned(self):
        result=self.run_plan({'scope':'comprehensive','jobs':[job(m,cell_type='Fibroblast') for m in METHODS]})
        self.assertEqual({r.method for r,execute in self.calls if execute},set(METHODS))
        self.assertEqual(result['n_jobs'],4)
        self.assertNotIn('manifest_uri',result)
        self.assertEqual(len(result['analysis_manifest_uris']),4)

    def test_comprehensive_missing_anchor_preserves_independent_jobs(self):
        result=self.run_plan({'scope':'comprehensive','jobs':[job(m) for m in METHODS]})
        self.assertEqual(result['status'],'partial_success')
        self.assertEqual(result['n_blocked'],2)
        self.assertEqual({r.method for r,e in self.calls if e},{'whole_tissue_deg','spatial_pattern'})
        self.assertTrue(all(r.cell_type is None for r,e in self.calls))

    def test_duplicate_up_down_requests_fit_once(self):
        result=self.run_plan({'jobs':[job('whole_tissue_deg',top_n=10),job('whole_tissue_deg',top_n=20)]})
        executed=[r for r,e in self.calls if e]
        self.assertEqual(len(executed),1); self.assertEqual(executed[0].top_n,20)
        self.assertEqual(len(result['duplicates_removed']),1)

    def test_blocked_preflight_does_not_block_valid_job(self):
        def runner(request, *, execute, show_progress):
            self.calls.append((request,execute))
            return {'status':'not_estimable','error':{'code':'SPATIAL_CONTEXT'}} if request.method=='spatial_pattern' else {'status':'success' if execute else 'planned'}
        result=self.run_plan({'jobs':[job('whole_tissue_deg'),job('spatial_pattern')]},run_job=runner)
        self.assertEqual(result['status'],'partial_success')
        self.assertEqual([r.method for r,e in self.calls if e],['whole_tissue_deg'])

    def test_failed_execution_does_not_expand_or_hide_other_results(self):
        def runner(request, *, execute, show_progress):
            if execute and request.method=='whole_tissue_deg': raise RuntimeError('fit failed')
            return {'status':'success' if execute else 'planned'}
        result=self.run_plan({'jobs':[job('whole_tissue_deg'),job('spatial_pattern')]},run_job=runner)
        self.assertEqual(result['status'],'partial_success')
        self.assertEqual(result['n_failed'],1)

    def test_preflight_exception_preserves_other_jobs(self):
        def runner(request, *, execute, show_progress):
            if request.method=='spatial_pattern': raise RuntimeError('unavailable metadata')
            return {'status':'success' if execute else 'planned'}
        result=self.run_plan({'jobs':[job('whole_tissue_deg'),job('spatial_pattern')]},run_job=runner)
        self.assertEqual(result['status'],'partial_success')
        self.assertEqual(result['n_blocked'],1)

    def test_dry_run_never_executes_statistics(self):
        result=self.run_plan({'scope':'comprehensive','jobs':[job(m,cell_type='Fibroblast') for m in METHODS]},execute=False)
        self.assertEqual(result['status'],'planned')
        self.assertEqual(len(self.calls),4)
        self.assertFalse(any(execute for _,execute in self.calls))
        stored=load(result['plan_manifest_uri'])
        self.assertEqual(stored['format_version'],'spatial-analysis-plan-1.0')

    def test_model_overrides_and_unjustified_jobs_rejected(self):
        with self.assertRaises(AnalysisError):
            AnalysisPlan.from_dict({'jobs':[job('whole_tissue_deg',allow_multistudy=True)]},model_generated=True)
        with self.assertRaises(AnalysisError):
            AnalysisPlan.from_dict({'jobs':[{'request':job('whole_tissue_deg')['request'],'reason':''}]},model_generated=True)

    def test_comprehensive_scope_never_adds_unplanned_methods(self):
        result=self.run_plan({'scope':'comprehensive','jobs':[job('cell_type_enriched_deg',cell_type='Fibroblast')]})
        self.assertEqual(result['n_jobs'],1)
        self.assertEqual([r.method for r,e in self.calls if e],['cell_type_enriched_deg'])

    def test_gemini_multi_plan_uses_dispatcher(self):
        reply={'status':'ready','scope':'focused','jobs':[job('whole_tissue_deg'),job('spatial_pattern')]}
        backend=SimpleNamespace(client=SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw:SimpleNamespace(text=json.dumps(reply)))),
                                types=SimpleNamespace(GenerateContentConfig=lambda **kw:kw))
        agent=SpatialAnalysisAgent(self.input,output_root=self.root/'runs',backend=backend)
        agent.inspect=Mock(return_value={'conditions':['case','control']})
        agent.run_request=self.runner
        result=agent.run('Compare DEG and spatial patterns',execute=True,show_progress=False)
        self.assertEqual(result['n_jobs'],2)
        self.assertEqual(result['status'],'success')


if __name__=='__main__': unittest.main()
