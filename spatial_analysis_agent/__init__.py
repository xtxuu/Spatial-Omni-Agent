"""Manifest-based spatial analysis with deterministic, independently callable methods."""
from .contracts import AnalysisRequest, AnalysisError
from .agent import SpatialAnalysisAgent
from .workflow import run_analysis, inspect_bundle
from .results import load_analysis_manifest, load_gene_list
from .planning import AnalysisPlan, PlanJob, load_analysis_plan, load_plan_gene_lists

__all__ = ['AnalysisRequest', 'AnalysisError', 'SpatialAnalysisAgent', 'run_analysis',
           'inspect_bundle', 'load_analysis_manifest', 'load_gene_list', 'AnalysisPlan', 'PlanJob',
           'load_analysis_plan', 'load_plan_gene_lists']
