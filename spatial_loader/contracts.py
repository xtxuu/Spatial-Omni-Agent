"""Public query/result types and the frozen source observation contract."""
from dataclasses import dataclass, asdict, field

TEXT_COLUMNS = ['source','dataset_id','sample_id','donor_id','organ','tissue_general','tissue',
    'species','assay','disease','condition','sex','batch','treatment','lesion_status',
    'source_url','metadata_status','deg_eligible','donor_id_source','barcode','age_unit',
    'sex_normalized','spatial_unit_type','spatial_unit_id','cell_type','dominant_cell_type',
    'cell_type_conservative','annotation_status','annotation_method','annotation_reference','annotation_version']
TEXT_COLUMNS += [f'cell_type_{i}' for i in range(1,6)]
INT_COLUMNS = ['in_tissue','array_row','array_col']
FLOAT_COLUMNS = ['age','pxl_row_in_fullres','pxl_col_in_fullres','cell_type_top_weight',
    'cell_type_weight_margin'] + [f'cell_type_weight_{i}' for i in range(1,6)]
OBS_COLUMNS = TEXT_COLUMNS + INT_COLUMNS + FLOAT_COLUMNS
EXTRA_TEXT = ['sample_uid','observation_id','donor_key','donor_identity_status']
EXTRA_NUMERIC = ['row_idx','x','y']
FILTER_FIELDS = set(OBS_COLUMNS + EXTRA_TEXT + EXTRA_NUMERIC)
NUMERIC_FIELDS = set(INT_COLUMNS + FLOAT_COLUMNS + EXTRA_NUMERIC)
FORMAT_VERSION = 'spatial-counts-1.0'


class DataError(ValueError):
    """A stable machine-readable error code plus a human-readable message."""
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)

    def to_dict(self):
        return {'status':'error', 'code':self.code, 'message':str(self)}


@dataclass(frozen=True)
class QueryResult:
    query_id: str
    dataset_version: str
    n_obs: int
    n_samples: int
    n_known_donors: int
    n_obs_missing_donor: int
    n_obs_ambiguous_donor: int
    groups: list
    gene_id_systems: list = field(default_factory=list)

    def to_dict(self):
        return dict(status='matched' if self.n_obs else 'empty',
                    spatial_unit_types=sorted({g['spatial_unit_type'] for g in self.groups if g.get('spatial_unit_type')}),
                    **asdict(self))
