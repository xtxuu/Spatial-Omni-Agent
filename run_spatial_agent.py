"""Explore and analyze spatial transcriptomics through cooperating Gemini agents.

No synthetic fixture is prepared. Without --execute, extraction still writes a
local counts bundle, but the analysis stage only plans and validates metadata.
"""
import argparse
import json
from pathlib import Path
import uuid

from spatial_loader import SpatialStore
from spatial_loader.config import resolve_dataset_path
from spatial_loader.utils import dump
from spatial_data_agent import GeminiDataAgent
from spatial_analysis_agent import SpatialAnalysisAgent, load_plan_gene_lists


ROOT = Path(__file__).resolve().parent


def extraction_request(question):
    return (
        'This is an explicitly requested data-extraction-to-analysis workflow. Your role '
        'is data discovery and export only; a separate analysis agent will handle '
        'the analysis. Discover exact catalog labels for the user request below. '
        'Export one cohort bundle containing the requested comparison groups, all '
        'native genes and complete observations of the relevant tissue sections. '
        'Keep full spatial context: do not prefilter spots by target cell type, '
        'confidence or QC; the analysis agent will select regions/cell types later. '
        'Use image paths rather than copying images, unless the user requests copies. '
        'Do not silently combine independent studies or choose an arbitrary study '
        'when several match; ask for clarification instead. Do not invent controls '
        'or narrow the requested cohort to make it fit. If the request explicitly '
        'requires a gene/spot subset incompatible with full-context export, explain '
        'and ask rather than silently changing it. Never perform analysis yourself. '
        'The following JSON contains the original user request:\n'
        + json.dumps({'user_request': question}, ensure_ascii=False)
    )


def clarification_text(result):
    """Surface blocked-job reasons instead of only a generic plan message."""
    messages = [result.get('message', '')]
    if result.get('error'):
        messages.append(result['error'].get('message', ''))
    for job in result.get('jobs', []):
        if job.get('state') == 'blocked':
            error = job.get('result', {}).get('error', {})
            messages.append(f"{job['request']['method']}: {error.get('message', 'Needs clarification')}")
    return '\n'.join(dict.fromkeys(message for message in messages if message))


def run_pipeline(question, *, dataset, output_root, key_file, model, execute=False,
                 donor_overrides=None, reply=None):
    """Return an auditable receipt; ambiguous extraction never starts analysis."""
    run_root = Path(output_root).resolve() / uuid.uuid4().hex
    run_root.mkdir(parents=True)
    dump(run_root / 'request.json', {
        'question': question, 'dataset': str(Path(dataset).resolve()),
        'model': model, 'execute': execute,
    })
    print(f'Run directory: {run_root}', flush=True)
    conversation = [{'role': 'user', 'stage': 'initial', 'message': question}]
    dump(run_root / 'conversation.json', conversation)
    end_reason = None

    def ask(stage, result):
        nonlocal end_reason
        message = clarification_text(result)
        conversation.append({'role': 'agent', 'stage': stage, 'message': message})
        dump(run_root / 'conversation.json', conversation)
        if reply is None:
            return None
        print(f'\n[{stage}] {message}', flush=True)
        while True:
            try:
                answer = reply('Your reply (or /quit): ').strip()
            except (EOFError, KeyboardInterrupt):
                end_reason = 'input_closed'
                return None
            if answer.casefold() in {'/quit', '/exit'}:
                end_reason = 'user_exit'
                return None
            if answer:
                conversation.append({'role': 'user', 'stage': stage, 'message': answer})
                dump(run_root / 'conversation.json', conversation)
                return answer

    with SpatialStore(dataset, cache_dir=run_root / 'query_cache') as store:
        agent = GeminiDataAgent(store, key_file=key_file, model=model,
                                exports_dir=run_root / 'data_exports', progress=print)
        try:
            message = extraction_request(question)
            turn = 0
            while True:
                data_result = agent.run(message)
                turn += 1
                dump(run_root / f'turns/data_{turn:03d}.json', data_result)
                dump(run_root / 'data_agent_result.json', data_result)
                artifacts = data_result.get('artifacts', [])
                if data_result['status'] == 'ready' and len(artifacts) == 1:
                    break
                if data_result['status'] not in {'ready', 'answer', 'selected', 'clarification_required'}:
                    break
                prompt = dict(data_result)
                if len(artifacts) > 1:
                    prompt['message'] = ('Multiple bundles were returned. Please specify one cohort to '
                                         're-export as one bundle.\n' + prompt.get('message', ''))
                answer = ask('data', prompt)
                if answer is None:
                    break
                message = ('Clarification for the existing extraction request: ' + answer +
                           '\nContinue discovery/export with the same constraints. Return one complete '
                           'cohort bundle in this turn when resolved; do not analyze the data.')
        finally:
            agent.close()
    dump(run_root / 'data_agent_result.json', data_result)
    artifacts = data_result.get('artifacts', [])
    receipt = {'run_root': str(run_root), 'data_status': data_result['status'],
               'data_message': data_result.get('message', ''), 'analysis': None}
    if data_result['status'] != 'ready' or len(artifacts) != 1:
        receipt['status'] = ('clarification_required' if data_result['status'] in
                             {'ready', 'answer', 'selected', 'clarification_required'} else 'failed')
        receipt['message'] = ('Analysis requires one completed cohort export. '
                              'Review data_agent_result.json and clarify the extraction request.')
    else:
        print('Data export complete. Planning requested analyses.', flush=True)
        analysis = SpatialAnalysisAgent(
            artifacts[0]['manifest_uri'], output_root=run_root / 'analysis',
            key_file=key_file, model=model, defaults={'gene_alignment': 'canonical'},
            donor_overrides=donor_overrides,
        )
        try:
            # Carry data-stage choices (study, disease, etc.) into the analysis planner.
            answers = [entry for entry in conversation if entry['role'] == 'user' and entry['stage'] == 'data']
            message = question if not answers else json.dumps({
                'original_request': question, 'extraction_clarifications': answers,
                'instruction': 'Use the clarified choices with this exported bundle.'}, ensure_ascii=False)
            turn = 0
            while True:
                result = analysis.run(message, execute=execute, show_progress=True)
                turn += 1
                dump(run_root / f'turns/analysis_{turn:03d}.json', result)
                dump(run_root / 'analysis_agent_result.json', result)
                if result['status'] not in {'clarification_required', 'partially_planned'}:
                    break
                answer = ask('analysis', result)
                if answer is None:
                    break
                message = answer
        finally:
            analysis.close()
        dump(run_root / 'analysis_agent_result.json', result)
        receipt.update(status=result['status'], analysis=result)
        if result.get('analysis_manifest_uris'):
            # Preserve each job's method, direction, background and gene namespace.
            gene_sets = {direction: load_plan_gene_lists(result['plan_manifest_uri'], direction)
                         for direction in ['significant_positive', 'significant_negative']}
            downstream_path = run_root / 'downstream_gene_lists.json'
            dump(downstream_path, gene_sets)
            receipt['downstream_gene_lists_uri'] = str(downstream_path)
    if end_reason:
        receipt['conversation_end'] = end_reason
    receipt['conversation_uri'] = str(run_root / 'conversation.json')
    dump(run_root / 'pipeline_result.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query', help='One natural-language extraction and analysis request')
    parser.add_argument('--config', type=Path, default=ROOT / 'agent_config.json')
    parser.add_argument('--dataset', type=Path, help='Override the configured dataset root')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'runtime/runs')
    parser.add_argument('--key-file', type=Path, default=ROOT / 'secrets/gemini_api_key.txt')
    parser.add_argument('--model', default='gemini-flash-latest')
    parser.add_argument('--donor-overrides', type=Path)
    parser.add_argument('--execute', action='store_true', help='Run selected statistical analyses after export')
    parser.add_argument('--non-interactive', action='store_true',
                        help='Return clarification immediately instead of waiting for terminal replies')
    args = parser.parse_args()
    try:
        args.dataset = resolve_dataset_path(args.config, override=args.dataset)
    except ValueError as error:
        parser.error(str(error))
    print(f'Dataset root: {args.dataset}', flush=True)
    question = args.query if args.query is not None else input('Enter your data/analysis request: ')
    if not question.strip():
        parser.error('A nonempty natural-language request is required')
    receipt = run_pipeline(question, dataset=args.dataset, output_root=args.output_root,
                           key_file=args.key_file, model=args.model, execute=args.execute,
                           donor_overrides=args.donor_overrides,
                           reply=None if args.non_interactive else input)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    if receipt['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
