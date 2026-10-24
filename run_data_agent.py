"""Gemini data discovery and export CLI. Never runs on import."""
import argparse
import json
from pathlib import Path

from tqdm import tqdm
from spatial_loader import SpatialStore
from spatial_loader.config import resolve_dataset_path
from spatial_data_agent import GeminiDataAgent

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "agent_config.json")
    parser.add_argument("--dataset", type=Path, help="Override the configured dataset root")
    parser.add_argument("--key-file", type=Path, default=ROOT / "secrets/gemini_api_key.txt")
    parser.add_argument("--model", default="gemini-flash-latest")
    parser.add_argument("--exports", type=Path, default=ROOT / "runtime/data_exports")
    parser.add_argument("--query", help="One request; omit to enter an interactive conversation")
    parser.add_argument("--max-export-obs", type=int, default=250000)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--output", type=Path, help="Optional JSON result file outside the release/secrets")
    args = parser.parse_args()
    try:
        args.dataset = resolve_dataset_path(args.config, override=args.dataset)
    except ValueError as error:
        parser.error(str(error))
    print(f'Dataset root: {args.dataset}', flush=True)
    if args.output:
        target = args.output.resolve()
        if target.is_relative_to(args.dataset.resolve()) or target.is_relative_to(args.key_file.resolve().parent):
            parser.error("--output must be outside the dataset and API key directory")
    with SpatialStore(args.dataset, cache_dir=ROOT / "runtime/query_cache") as store:
        agent = GeminiDataAgent(store, key_file=args.key_file, model=args.model,
            exports_dir=args.exports, max_steps=args.max_steps, max_export_obs=args.max_export_obs,
            progress=lambda text: tqdm.write(text))
        try:
            while True:
                if args.query is not None:
                    question = args.query
                else:
                    try:
                        question = input("You (exit to quit): ").strip()
                    except EOFError:
                        break
                    if question.lower() in {"exit", "quit"}:
                        break
                    if not question:
                        continue
                result = agent.run(question)
                print(json.dumps(result, ensure_ascii=False, indent=2))
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                if args.query is not None:
                    break
        finally:
            agent.close()


if __name__ == "__main__":
    main()
