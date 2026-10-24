"""Framework-neutral JSON tool adapter; expression payloads never enter JSON."""
from pathlib import Path
import uuid

from .contracts import FILTER_FIELDS,NUMERIC_FIELDS,DataError


def tool_specs():
    """Return JSON Schema specifications; map these to your agent framework."""
    properties={}
    for f in sorted(FILTER_FIELDS):
        scalar={'type':'number' if f in NUMERIC_FIELDS else 'string'}
        array={'type':'array','items':scalar}
        operators={'eq':scalar,'in':array,'is_null':{'type':'boolean'}}
        if f in NUMERIC_FIELDS:
            operators.update({k:scalar for k in ['gt','gte','lt','lte']})
            operators['between']={'type':'array','items':scalar,'minItems':2,'maxItems':2}
        properties[f]={'oneOf':[scalar,array,{'type':'object','properties':operators,
                         'additionalProperties':False,'minProperties':1}]}
    filters={'type':'object','properties':properties,'additionalProperties':False}
    cell={'type':'object','properties':{'label':{'type':'string'},
        'mode':{'type':'string','enum':['dominant','top5']},
        'min_weight':{'type':'number','minimum':0,'maximum':1},
        'status':{'oneOf':[{'type':'string'},{'type':'array','items':{'type':'string'}}]}},
        'required':['label'],'additionalProperties':False}
    roi={'type':'object','properties':{'sample_uid':{'type':'string'},'unit':{'const':'pixel'},
        'bounds':{'type':'array','items':{'type':'number'},'minItems':4,'maxItems':4}},
        'required':['sample_uid','unit','bounds'],'additionalProperties':False}
    definitions=[
        ('describe','Discover exact source labels, filter fields and counts; does not read expression.',
         {'filters':filters,'fields':{'type':'array','items':{'type':'string','enum':sorted(FILTER_FIELDS)}},
          'limit':{'type':'integer','minimum':1,'maximum':500},'offset':{'type':'integer','minimum':0},
          'include_schema':{'type':'boolean'}},[]),
        ('query','Select spatial observations by metadata and dominant/top5 compositions. Returns a query ID and cohort summary.',
         {'filters':filters,'cell_type':cell,'roi':roi},[]),
        ('fetch','Export selected raw sparse counts and metadata as per-sample artifacts. Returns a manifest URI, not matrix JSON.',
         {'query_id':{'type':'string'},'genes':{'type':['array','null'],'items':{'type':'string'}},
          'gene_key':{'type':'string','enum':['gene_symbol','gene_id','source_var_name','canonical_gene_id','canonical_gene_symbol']},
          'include_images':{'type':'string','enum':['paths','copy','none']}},['query_id']),
        ('gene_systems','Report detected gene ID namespaces, mapping coverage and source-release uncertainty.',
         {'dataset_id':{'type':'string'},'sample_uid':{'type':'string'},'limit':{'type':'integer'},'offset':{'type':'integer'}},[]),
        ('resolve_genes','Resolve gene identifiers using frozen references; returns candidates, never guesses.',
         {'identifiers':{'type':'array','items':{'type':'string'}},'species':{'type':'string'},
          'input_namespace':{'type':'string','enum':['ensembl_gene','gene_symbol','hgnc_id','ncbi_gene']}},['identifiers'])]
    return [dict(name=name,description=description,input_schema=dict(type='object',properties=props,
            required=required,additionalProperties=False)) for name,description,props,required in definitions]


class DataAgentTools:
    def __init__(self,store,exports_dir):
        self.store=store;self.exports_dir=Path(exports_dir).resolve()
        if self.exports_dir.is_relative_to(store.root):raise DataError('READ_ONLY_RELEASE','Exports must be outside release')

    def call(self,name,arguments):
        """Dispatch allowlisted tools, returning JSON-serializable data."""
        allowed={'describe':{'filters','fields','limit','offset','include_schema'},
                 'query':{'filters','cell_type','roi'},
                 'fetch':{'query_id','genes','gene_key','include_images'},
                 'gene_systems':{'dataset_id','sample_uid','limit','offset'},
                 'resolve_genes':{'identifiers','species','input_namespace'}}
        try:
            if name not in allowed:raise DataError('UNKNOWN_TOOL',name)
            if not isinstance(arguments,dict) or set(arguments)-allowed[name]:
                raise DataError('INVALID_ARGUMENTS','Unexpected tool parameters')
            if name=='fetch':
                return self.store.fetch(**arguments,delivery='artifact',
                    output_dir=self.exports_dir/uuid.uuid4().hex)
            result=getattr(self.store,name)(**arguments)
            return result.to_dict() if name=='query' else result
        except DataError as error:return error.to_dict()
        except TypeError as error:return DataError('INVALID_ARGUMENTS',str(error)).to_dict()
        except FileNotFoundError as error:return DataError('MISSING_FILE',str(error)).to_dict()
        except OSError as error:return DataError('IO_ERROR',str(error)).to_dict()
