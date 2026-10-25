"""Prepare Serbian Wikipedia + FineWeb2 without downloading entire datasets."""
import argparse
import json
from pathlib import Path

from serbian_llm.config import load_config
from serbian_llm.preparation import prepare_project_data

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fineweb-mb", type=float, default=350)
    parser.add_argument("--wiki-mb", type=float, default=150)
    parser.add_argument("--config", default=str(ROOT / "configs/data.json"))
    parser.add_argument("--sources-config", default=str(ROOT / "configs/data_mix.json"))
    parser.add_argument("--candidates-only", action="store_true", help="Collect source candidates for expand-corpus; do not create new evaluation splits.")
    args = parser.parse_args()
    result = prepare_project_data(load_config(args.config), load_config(args.sources_config),
                                  args.output, fineweb_mb=args.fineweb_mb, wiki_mb=args.wiki_mb,
                                  candidates_only=args.candidates_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
