"""Command-line entry point for inspected bundles and explicit analysis requests."""
import argparse
import json
from pathlib import Path

from spatial_loader.utils import load
from .agent import SpatialAnalysisAgent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--inspect', action='store_true')
    actions.add_argument('--request-json', type=Path)
    actions.add_argument('--plan-json', type=Path)
    actions.add_argument('--query')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output-root', type=Path, default=Path('analysis_runs'))
    parser.add_argument('--donor-overrides', type=Path)
    parser.add_argument('--key-file', type=Path, default=Path('secrets/gemini_api_key.txt'))
    parser.add_argument('--model', default='gemini-flash-latest')
    agent = None
    args = parser.parse_args()
    try:
        agent = SpatialAnalysisAgent(args.manifest, output_root=args.output_root,
            donor_overrides=args.donor_overrides, key_file=args.key_file, model=args.model)
        if args.inspect: result = agent.inspect()
        elif args.request_json: result = agent.run_request(load(args.request_json), execute=args.execute)
        elif args.plan_json: result = agent.run_plan(load(args.plan_json), execute=args.execute)
        else: result = agent.run(args.query, execute=args.execute)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        if agent is not None: agent.close()


if __name__ == '__main__': main()
