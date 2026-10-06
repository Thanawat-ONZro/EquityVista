"""Job: data/processed/dataset_1962_2026.parquet -> data/processed/features.parquet (the 25 features; no target:
each train job builds its own H-day target from adj_close, so one features file serves every horizon).
params: source (default dataset_1962_2026; the old name us_market_ext is accepted too, both mean DATASET), leakage_test (1 = run the slow no-lookahead check first),
start_year (rows before Jan 1 of that year are left out of the features file)."""
import polars as pl

from config import BASE, DATASET, FEATURES
from src import features as F
from src.jobs import proto


def run(p, run_dir):
    if p["source"] not in (DATASET.stem, "us_market_ext"):
        raise ValueError(f"unknown source {p['source']!r}: only {DATASET.stem} exists")
    proto.progress("load", 0)
    raw = pl.read_parquet(DATASET)
    raw = raw.filter(pl.col("date").dt.year() >= int(p.get("start_year", 1962)))
    if int(p["leakage_test"]):
        proto.progress("leakage", 5)
        F.test_no_lookahead(raw)
    proto.progress("features", 40)
    df = F.add_features(raw)
    dest = FEATURES
    tmp = dest.with_name(dest.name + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(dest)  # atomic: a kill / app close mid-write must not leave a truncated "must end with PAR1" file
    proto.metric("rows", df.height)
    proto.metric("tickers", df["ticker"].n_unique())
    return {"features": str(FEATURES.relative_to(BASE))}


if __name__ == "__main__":
    proto.main("build_features", run)
