"""Typed, validated requests. No free-form formulas, code or model-supplied paths."""
from dataclasses import asdict, dataclass, field
import math


METHODS = ('whole_tissue_deg', 'cell_type_enriched_deg', 'spatial_neighborhood_deg', 'spatial_pattern')


class AnalysisError(ValueError):
    def __init__(self, code, message, *, status='not_estimable', details=None):
        super().__init__(message)
        self.code, self.status, self.details = code, status, details or {}


@dataclass(frozen=True)
class AnalysisRequest:
    method: str
    case: str
    control: str
    group_field: str = 'condition'
    cell_type: str | None = None
    gene_alignment: str = 'canonical'
    numeric_covariates: list[str] = field(default_factory=list)
    categorical_covariates: list[str] = field(default_factory=list)
    allow_multistudy: bool = False
    allow_multiple_organs: bool = False
    allow_gene_subset: bool = False
    allow_partial_spatial_context: bool = False
    missing_donor: str = 'exclude'
    min_replicates_per_group: int = 2
    min_spots_per_donor: int = 1
    warn_spots_per_donor: int = 20
    min_counts: int = 100
    min_detected_genes: int = 200
    max_pct_mito: float | None = None
    require_in_tissue: bool = True
    cell_type_mode: str = 'dominant_confident'
    min_top_weight: float = 0.60
    min_margin: float = 0.15
    neighbors: int = 6
    min_common_genes: int = 20
    min_gene_count: int = 10
    min_donors_with_gene_count: int = 2
    max_spatial_genes: int = 1000
    fdr: float = 0.05
    min_abs_log2_fold_change: float = 0.5
    exploratory_p: float = 0.001
    top_n: int = 10
    n_cpus: int = 1
    block_rows: int = 1024
    max_memory_bytes: int = 1024 * 1024**2

    def __post_init__(self):
        if self.method not in METHODS: raise AnalysisError('METHOD', str(self.method), status='clarification_required')
        if not isinstance(self.case, str) or not isinstance(self.control, str) or not self.case or not self.control or self.case == self.control:
            raise AnalysisError('CONTRAST', 'Specify different, exact case and control labels', status='clarification_required')
        if self.gene_alignment not in {'canonical', 'native'}: raise AnalysisError('GENE_ALIGNMENT', self.gene_alignment)
        if self.missing_donor not in {'exclude', 'error'}: raise AnalysisError('DONOR_POLICY', self.missing_donor)
        if self.cell_type_mode not in {'dominant_confident', 'top5'}: raise AnalysisError('CELL_SELECTION', self.cell_type_mode)
        if self.method in {'cell_type_enriched_deg', 'spatial_neighborhood_deg'} and not self.cell_type:
            raise AnalysisError('CELL_TYPE', 'Specify an exact target cell type', status='clarification_required')
        for key in ['allow_multistudy', 'allow_multiple_organs', 'allow_gene_subset', 'allow_partial_spatial_context', 'require_in_tissue']:
            if not isinstance(getattr(self, key), bool): raise AnalysisError('PARAMETER', f'{key} must be boolean')
        for key in ['min_replicates_per_group', 'min_spots_per_donor', 'warn_spots_per_donor', 'min_detected_genes', 'neighbors',
                    'min_common_genes', 'min_donors_with_gene_count', 'max_spatial_genes', 'top_n', 'n_cpus', 'block_rows', 'max_memory_bytes']:
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1: raise AnalysisError('PARAMETER', f'{key} must be positive integer')
        if self.min_replicates_per_group < 2: raise AnalysisError('REPLICATION', 'At least two independent donors per group are required')
        for key in ['min_counts', 'min_gene_count']:
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0: raise AnalysisError('PARAMETER', key)
        for key in ['fdr', 'exploratory_p', 'min_top_weight', 'min_margin', 'min_abs_log2_fold_change']:
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise AnalysisError('PARAMETER', key)
        if not 0 < self.fdr < 1 or not 0 < self.exploratory_p < 1 or self.min_top_weight > 1 or self.min_margin > 1:
            raise AnalysisError('PARAMETER', 'Invalid probability/composition threshold')
        if self.max_pct_mito is not None and (isinstance(self.max_pct_mito, bool) or not isinstance(self.max_pct_mito, (float, int)) or not math.isfinite(self.max_pct_mito) or not 0 <= self.max_pct_mito <= 100):
            raise AnalysisError('PARAMETER', 'max_pct_mito must be 0..100 or None')
        if not isinstance(self.numeric_covariates, list) or not isinstance(self.categorical_covariates, list):
            raise AnalysisError('COVARIATES', 'Use field-name lists')
        covariates = self.numeric_covariates + self.categorical_covariates
        if (any(not isinstance(v, str) or not v for v in covariates) or len(covariates) != len(set(covariates))):
            raise AnalysisError('COVARIATES', 'Use distinct field-name lists')
        if self.group_field in covariates: raise AnalysisError('COVARIATES', 'Group field cannot also be a covariate')
        if self.method == 'spatial_pattern' and covariates:
            raise AnalysisError('SPATIAL_COVARIATES', 'The Welch spatial screen does not implement covariate adjustment', status='clarification_required')

    def to_dict(self): return asdict(self)

    @classmethod
    def from_dict(cls, value):
        try: return cls(**value)
        except TypeError as error: raise AnalysisError('REQUEST', str(error), status='clarification_required') from error
