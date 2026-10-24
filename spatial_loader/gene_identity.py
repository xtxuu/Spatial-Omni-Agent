"""Read-only access to frozen gene identities and reference resolution."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import duckdb
from .contracts import DataError
from .utils import digest, load, resolve, sql_string

HUMAN = 'Homo sapiens'
INDEX_FORMAT = 'spatial-gene-identity-1.0'
ENSEMBL = re.compile(r'^(ENSG\d+)(?:\.(\d+))?(_PAR_Y)?$')


def normalize_id(value):
    """Remove only a recognized gene version; preserve PAR_Y as a distinct feature."""
    match = ENSEMBL.fullmatch(str(value))
    return match.group(1) + (match.group(3) or '') if match else str(value)


def namespace(value):
    value = str(value)
    if re.fullmatch(r'DEPRECATED_ENSG\d+(?:\.\d+)?', value):
        return 'source_tagged_deprecated_ensembl'
    if ENSEMBL.fullmatch(value):
        return 'ensembl_gene_PAR_Y' if value.endswith('_PAR_Y') else 'ensembl_gene'
    if re.fullmatch(r'ENSMUSG\d+(?:\.\d+)?', value): return 'ensembl_mouse_gene'
    if re.fullmatch(r'ENST\d+(?:\.\d+)?', value): return 'ensembl_transcript'
    if re.fullmatch(r'HGNC:\d+', value): return 'hgnc_id'
    if re.fullmatch(r'\d+', value): return 'numeric_candidate'
    return 'symbol_or_custom'


def read_parquet(path):
    with duckdb.connect() as db:
        return db.execute(f'SELECT * FROM read_parquet({sql_string(path)})').df()


class ReferenceResolver:
    """Candidate sets are explicit; no LLM inference or silent one-to-many choice."""
    def __init__(self, genes, hgnc):
        self.ids = defaultdict(list)
        self.symbols = defaultdict(lambda: defaultdict(set))
        self.names = defaultdict(set)
        self.external = defaultdict(set)
        self.display = {}
        self.current = set()
        for row in genes.to_dict('records'):
            gid, symbol = row['gene_id'], row['gene_symbol']
            self.ids[gid].append(row['reference']); self.names[gid].add(symbol)
            if row['reference'] == 'GENCODE_v49':
                self.current.add(gid); self.display[gid] = symbol
            else: self.display.setdefault(gid, symbol)
            # Historical PAR copies have separate identities, not symbol-based aliases.
            if not gid.endswith('_PAR_Y'):
                self.symbols[symbol][1 if row['reference'] == 'GENCODE_v49' else 2].add(gid)
        for row in hgnc.fillna('').to_dict('records'):
            if row.get('status', 'Approved') != 'Approved': continue
            gid = str(row.get('ensembl_gene_id', ''))
            if not re.fullmatch(r'ENSG\d+', gid): continue
            symbol = str(row['symbol'])
            self.ids[gid].append('HGNC_snapshot'); self.names[gid].add(symbol)
            self.display[gid] = symbol
            self.symbols[symbol][0].add(gid)
            for column, priority in [('prev_symbol', 3), ('alias_symbol', 4)]:
                for token in str(row.get(column, '')).split('|'):
                    if token:
                        self.symbols[token][priority].add(gid); self.names[gid].add(token)
            for column, ns in [('hgnc_id', 'hgnc_id'), ('entrez_id', 'ncbi_gene')]:
                value = str(row.get(column, ''))
                if value: self.external[(ns, value)].add(gid)

    def symbol_candidates(self, symbol):
        levels = self.symbols.get(str(symbol), {})
        if not levels: return set(), None
        best = min(levels)
        return set(levels[best]), best

    def map_feature(self, gid, symbol, species, feature_type='Gene Expression'):
        gid, symbol = str(gid), str(symbol)
        ns = namespace(gid)
        result = dict(detected_namespace=ns, canonical_gene_id=None, canonical_gene_symbol=None,
            mapping_status='unmapped', mapping_method='none', canonical_candidates='[]',
            mapping_reference=None, symbol_relation='not_checked', alignment_eligible=False)
        if species != HUMAN:
            result['mapping_status'] = 'unsupported_species'; return result
        if ns == 'source_tagged_deprecated_ensembl':
            result['mapping_status'] = 'source_marked_deprecated'; return result
        if str(feature_type) not in {'Gene Expression', 'gene', '', 'None', 'nan', '<NA>'}:
            result['mapping_status'] = 'non_gene_feature'; return result
        candidates, method, priority = set(), '', None
        normalized = normalize_id(gid)
        if ns.startswith('ensembl_gene'):
            if normalized in self.ids: candidates = {normalized}; method = 'ensembl_exact_versionless'
            else: result['mapping_status'] = 'unverified_ensembl'; return result
        elif ns == 'hgnc_id': candidates = self.external.get(('hgnc_id', gid), set()); method = 'hgnc_exact'
        elif ns == 'numeric_candidate': candidates = self.external.get(('ncbi_gene', gid), set()); method = 'ncbi_gene_exact'
        elif ns == 'ensembl_transcript' or ns == 'ensembl_mouse_gene':
            result['mapping_status'] = 'wrong_feature_or_species'; return result
        else:
            candidates, priority = self.symbol_candidates(gid); method = 'symbol_exact' if priority in {0, 1, 2} else 'symbol_alias'
            if candidates: result['detected_namespace'] = 'gene_symbol'
        result['canonical_candidates'] = json.dumps(sorted(candidates))
        result['mapping_method'] = method or 'none'
        if not candidates: return result
        if len(candidates) != 1: result['mapping_status'] = 'ambiguous'; return result
        target = next(iter(candidates))
        symbol_targets, _ = self.symbol_candidates(symbol)
        if symbol in self.names[target] or symbol == gid or normalize_id(symbol) == target:
            relation = 'compatible'
        elif symbol_targets and target not in symbol_targets:
            result['mapping_status'] = 'id_symbol_conflict'; result['symbol_relation'] = 'conflict'
            return result
        else: relation = 'unverified_source_symbol'
        status = 'mapped' if target in self.current else 'mapped_historical_or_hgnc_only'
        result.update(canonical_gene_id=target, canonical_gene_symbol=self.display[target],
            mapping_status=status, mapping_reference='|'.join(sorted(set(self.ids[target]))),
            symbol_relation=relation, alignment_eligible=True)
        return result

    def resolve(self, identifiers, species, input_namespace):
        if input_namespace not in {'ensembl_gene', 'gene_symbol', 'hgnc_id', 'ncbi_gene'}:
            raise DataError('INVALID_GENE_NAMESPACE', input_namespace)
        if species != HUMAN: raise DataError('UNSUPPORTED_SPECIES', species)
        if not isinstance(identifiers, list) or not 1 <= len(identifiers) <= 1000 or any(not isinstance(x, str) for x in identifiers):
            raise DataError('INVALID_GENES', 'Use 1..1000 identifier strings')
        rows = []
        for token in identifiers:
            if input_namespace == 'ensembl_gene':
                target = normalize_id(token); hits = {target} if target in self.ids else set()
            elif input_namespace == 'gene_symbol': hits, _ = self.symbol_candidates(token)
            else: hits = self.external.get((input_namespace, token), set())
            unique = next(iter(hits)) if len(hits) == 1 else None
            rows.append(dict(input_id=token, status='mapped' if unique else 'ambiguous' if hits else 'unmapped',
                canonical_gene_id=unique, canonical_gene_symbol=self.display.get(unique), candidates=sorted(hits)))
        return rows


def summarize_systems(frame):
    rows = []
    for (source, dataset, species), group in frame.groupby(['source', 'dataset_id', 'species'], sort=True):
        ns, statuses = Counter(), Counter()
        for _, sample in group.iterrows():
            ns.update(json.loads(sample.namespaces)); statuses.update(json.loads(sample.mapping_status_counts))
        total = int(group.n_features.sum())
        rows.append(dict(source=source, dataset_id=dataset, species=species, n_samples=len(group),
            n_feature_profiles=int(group.profile_id.nunique()), min_n_genes=int(group.n_features.min()),
            max_n_genes=int(group.n_features.max()), feature_occurrences=total,
            namespaces=dict(ns), mapping_status_counts=dict(statuses),
            mapped_fraction=float(group.n_mapped.sum() / total),
            alignment_eligible_fraction=float(group.n_alignment_eligible.sum() / total),
            n_duplicate_target_features=int(group.n_duplicate_target_features.sum()),
            source_annotation_release='unknown_not_inferred_from_IDs'))
    return rows


class GeneIdentityIndex:
    """Read-only loader overlay, independently versioned and integrity checked."""
    def __init__(self, root, release):
        self.root = Path(root); self.directory = self.root / 'gene_index'
        self.manifest = load(self.directory / 'index.json')
        if self.manifest.get('format_version') != INDEX_FORMAT or self.manifest.get('source_release_id') != release['release_id']:
            raise DataError('GENE_INDEX_VERSION', 'Index belongs to another release')
        if self.manifest['source_catalog_sha256'] != {k: release['catalog_sha256'][k]
                for k in self.manifest['source_catalog_sha256'] if k in release['catalog_sha256']}:
            raise DataError('GENE_INDEX_SOURCE', 'Source catalog fingerprints differ')
        self.verified, self.profiles, self.reference = set(), {}, None
        self.systems = read_parquet(self.checked('sample_systems.parquet'))
        if self.systems.sample_uid.duplicated().any(): raise DataError('GENE_INDEX_SAMPLE', 'Duplicate sample mappings')
        self.sample_lookup = self.systems.set_index('sample_uid').to_dict('index')
        self.version = self.manifest['index_id']

    def checked(self, relative):
        path = resolve(self.directory, relative)
        expected = self.manifest['files_sha256'].get(relative)
        if not expected: raise DataError('GENE_INDEX_FILE', relative)
        if relative not in self.verified:
            if not path.is_file() or digest(path) != expected: raise DataError('GENE_INDEX_HASH', relative)
            self.verified.add(relative)
        return path

    def enrich(self, uid, features):
        info = self.sample_lookup.get(uid)
        if info is None: raise DataError('GENE_INDEX_SAMPLE', uid)
        profile = info['profile_id']
        if profile not in self.profiles:
            table = read_parquet(self.checked(info['mapping_path'])).sort_values('col_idx').reset_index(drop=True)
            self.profiles[profile] = table
        table = self.profiles[profile]
        if (len(features) != len(table) or features.col_idx.tolist() != table.col_idx.tolist() or
                features.gene_id.astype(str).tolist() != table.source_gene_id.astype(str).tolist() or
                features.gene_symbol.astype(str).tolist() != table.source_gene_symbol.astype(str).tolist()):
            raise DataError('GENE_INDEX_AXIS', uid)
        result = features.copy()
        for column in table:
            if column not in {'col_idx', 'source_gene_id', 'source_gene_symbol'}:
                result[column] = table[column].to_numpy()
        result['gene_index_id'] = self.version
        return result

    def summaries(self, *, dataset_id=None, sample_uid=None, sample_uids=None, limit=100, offset=0):
        if not isinstance(limit, int) or not 1 <= limit <= 1000 or not isinstance(offset, int) or offset < 0:
            raise DataError('INVALID_LIMIT', 'Use limit 1..1000 and nonnegative offset')
        frame = self.systems
        if dataset_id is not None: frame = frame.loc[frame.dataset_id == dataset_id]
        if sample_uid is not None: frame = frame.loc[frame.sample_uid == sample_uid]
        if sample_uids is not None: frame = frame.loc[frame.sample_uid.isin(sample_uids)]
        rows = summarize_systems(frame)
        if sample_uid is not None:
            rows = frame.to_dict('records')
            for row in rows:
                row['namespaces'] = json.loads(row['namespaces'])
                row['mapping_status_counts'] = json.loads(row['mapping_status_counts'])
        return dict(status='available', gene_index_id=self.version, n_results=len(rows),
                    offset=offset, results=rows[offset:offset+limit])

    def resolver(self):
        if self.reference is None:
            self.reference = ReferenceResolver(read_parquet(self.checked('reference_genes.parquet')),
                                               read_parquet(self.checked('reference_hgnc.parquet')))
        return self.reference
