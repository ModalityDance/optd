#!/usr/bin/env python3
"""Resolve and validate the pinned LCB-v5 test dataset on the current host."""

import argparse
import json

from deepcoder_lcb_common import discover_lcb_glob, load_lcb_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    pattern = discover_lcb_glob()
    paths = __import__("glob").glob(pattern)
    result = {"data_glob": pattern, "shards": len(paths), "problems": len(load_lcb_rows(pattern))}
    if args.as_json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(pattern)
        print(f"validated shards={result['shards']} problems={result['problems']}")


if __name__ == "__main__":
    main()
