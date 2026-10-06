"""Job: fetch post-2020 OHLCV (extend_fetch) then rebuild dataset_1962_2026.parquet from the RAW union with the ONE uniform rule set (build_dataset).
params: start, limit (0 = all tickers), workers."""
import sys

from config import BASE, DATASET
from src import build_dataset, extend_fetch
from src.jobs import proto


def run(p, run_dir):
    proto.progress("fetch", 0)
    sys.argv = ["extend_fetch", "--start", str(p["start"]), "--limit", str(int(p["limit"])),
                "--workers", str(int(p["workers"]))]
    extend_fetch.main()
    proto.progress("clean", 50)
    sys.argv = ["build_dataset", "--force"]
    build_dataset.main()
    return {"parquet": str(DATASET.relative_to(BASE))}


if __name__ == "__main__":
    proto.main("data_extend", run)
