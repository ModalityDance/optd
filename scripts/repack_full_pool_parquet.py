#!/usr/bin/env python3
"""Repack the full Math+Code pool into small Parquet row groups for HF Datasets."""

import hashlib
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main(source: Path, dest: Path, rows_per_group: int = 128) -> None:
    if dest.exists():
        raise SystemExit(f"Refusing to overwrite {dest}")
    parquet = pq.ParquetFile(source)
    rows = 0
    with pq.ParquetWriter(dest, parquet.schema_arrow, compression="snappy") as writer:
        for batch in parquet.iter_batches(batch_size=rows_per_group):
            writer.write_table(pa.Table.from_batches([batch]), row_group_size=rows_per_group)
            rows += batch.num_rows
            if rows % 4096 < rows_per_group:
                print(f"repacked {rows}/{parquet.metadata.num_rows}", flush=True)
    assert rows == parquet.metadata.num_rows
    digest = hashlib.sha256()
    with dest.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    manifest = json.loads(Path(str(source) + ".manifest.json").read_text())
    manifest.update(
        output=str(dest),
        output_sha256=digest.hexdigest(),
        repacked_from_sha256=manifest["output_sha256"],
        parquet_row_group_size=rows_per_group,
    )
    Path(str(dest) + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"repacked {rows} rows to {dest}; sha256={digest.hexdigest()}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
