"""Small CNN-LSTM (deep-learning arm) on 60-day windows of scale-free features.

Features (F=11, all scale-free; 'r_*' cross-sectional ranks and ret60/vol60 skipped to stay compact):
  ret1 ret5 ret20 vol20 close_ma20 close_ma50 rsi14 macd_n vol_z hl_range mkt_ret
Scaling: ONE global standardisation (mean/std) from TRAIN rows only, clipped to +-8 (no per-ticker scaler).
Windows are never materialised: a batch gathers X[idx[:,None]+arange(-59,1)] from the flat
float32 feature matrix. A window ending at row r is valid iff row r-59 belongs to the same ticker.
Targets: direction = (y>=0) [sigmoid/BCE], return = y*100 [Huber]; predictions are divided back by 100.
"""
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import polars as pl
import keras
from keras import layers

from src.split_us import FEATS, make_splits

W = 60
FEATURES = ["ret1", "ret5", "ret20", "vol20", "close_ma20", "close_ma50",
            "rsi14", "macd_n", "vol_z", "hl_range", "mkt_ret"]
RET_SCALE = 100.0
OFFS = np.arange(-W + 1, 1)
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "cnn_lstm_us.keras"


class Windows(keras.utils.PyDataset):
    """Lazy batches of windows ending at `rows` (indices into X)."""
    def __init__(self, X, rows, y=None, bs=512, shuffle=False, seed=0, **kw):
        super().__init__(**kw)
        self.X, self.rows, self.y, self.bs, self.shuffle = X, rows, y, bs, shuffle
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(rows))
        if shuffle:
            self.rng.shuffle(self.order)

    def __len__(self):
        return -(-len(self.rows) // self.bs)

    def __getitem__(self, i):
        o = self.order[i * self.bs:(i + 1) * self.bs]
        xb = self.X[self.rows[o][:, None] + OFFS]
        if self.y is None:
            return xb
        yb = self.y[o]
        return xb, {"direction": (yb >= 0).astype("float32"), "return": (yb * RET_SCALE).astype("float32")}

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)


def build_model():
    inp = keras.Input((W, len(FEATURES)))
    x = layers.Conv1D(32, 5, padding="causal", activation="relu")(inp)
    x = layers.Conv1D(32, 3, padding="causal", activation="relu")(x)
    x = layers.LSTM(64)(x)
    x = layers.Dense(32, activation="relu")(x)
    m = keras.Model(inp, [layers.Dense(1, activation="sigmoid", name="direction")(x),
                          layers.Dense(1, name="return")(x)])
    m.compile("adam", loss={"direction": "binary_crossentropy", "return": keras.losses.Huber()})
    return m


def _valid(ids, tick, y):
    """Rows whose 60-step window stays inside one ticker; returns (rows, next-day y)."""
    r = ids["row"].to_numpy().astype(np.int64)   # `row` is UInt32: r - 59 would wrap for r < 59
    ok = (r >= W - 1) & (tick[np.maximum(r - (W - 1), 0)] == tick[r])
    return r[ok], y[ok, 0]


def predict(m, X, rows, bs=2048):
    """Predicted next-day log return (model output / RET_SCALE)."""
    return m.predict(Windows(X, rows, bs=bs), verbose=0)[1].ravel() / RET_SCALE


def train(X, tick, sp, n_tickers=300, frac=1.0, epochs=3, bs=512, seed=0, verbose=2):
    """Train on a ticker subsample (n_tickers) and optionally a row fraction `frac`."""
    rng = np.random.default_rng(seed)
    tr_rows, tr_y = _valid(sp["train"][0], tick, sp["train"][1])
    tr_tick = tick[tr_rows]
    uniq = np.unique(tr_tick)
    sel = np.isin(tr_tick, rng.choice(uniq, size=min(n_tickers, len(uniq)), replace=False))
    if frac < 1:
        sel &= rng.random(len(sel)) < frac
    tr_rows, tr_y = tr_rows[sel], tr_y[sel]
    va_rows, va_y = _valid(sp["val"][0], tick, sp["val"][1])
    vs = rng.choice(len(va_rows), size=min(50_000, len(va_rows)), replace=False)  # val subsample for speed
    m = build_model()
    t = time.time()
    h = m.fit(Windows(X, tr_rows, tr_y, bs, True, seed), epochs=epochs, verbose=verbose,
              validation_data=Windows(X, va_rows[vs], va_y[vs], 2048))
    return {"model": m, "history": h.history, "sec_per_epoch": (time.time() - t) / epochs,
            "subsample_n": len(tr_rows)}


def load():
    feat = pl.read_parquet(FEATS)
    sp = make_splits(feat)
    X = feat.select(FEATURES).to_numpy().astype("float32")
    tick = feat["ticker"].cast(pl.Categorical).to_physical().to_numpy()
    tr = sp["train"][0]["row"].to_numpy()
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9      # TRAIN-only stats
    return np.clip((X - mu) / sd, -8, 8).astype("float32"), tick, sp


def score(name, p, y):
    corr, dacc = np.corrcoef(p, y)[0, 1], np.mean((p >= 0) == (y >= 0))
    print(f"{name}: n={len(y):,} corr={corr:.4f} diracc={dacc:.4f} base(up)={np.mean(y >= 0):.4f}")


if __name__ == "__main__":
    keras.utils.set_random_seed(0)
    t0 = time.time()
    X, tick, sp = load()
    print(f"loaded X{X.shape} in {time.time()-t0:.0f}s")
    n_tr = len(_valid(sp["train"][0], tick, sp["train"][1])[0])
    print(f"valid train windows: {n_tr:,}")
    # 1) time on ~1% of train windows (all tickers, 1% of rows)
    r = train(X, tick, sp, n_tickers=10**6, frac=0.01, epochs=2)
    print(f"[1% timing] n={r['subsample_n']:,} windows -> {r['sec_per_epoch']:.1f}s/epoch "
          f"=> full {n_tr:,} windows ~{r['sec_per_epoch']*n_tr/r['subsample_n']/3600:.1f} h/epoch")
    # 2) real run on a ticker subsample
    NT, EP = int(os.environ.get("NT", 300)), int(os.environ.get("EP", 3))
    r = train(X, tick, sp, n_tickers=NT, epochs=EP)
    print(f"[run] {NT} tickers, n={r['subsample_n']:,} windows, {EP} epochs, {r['sec_per_epoch']:.0f}s/epoch")
    MODEL_PATH.parent.mkdir(exist_ok=True)
    r["model"].save(MODEL_PATH)
    for k in ("test_seen", "test_unseen"):
        rows, y = _valid(sp[k][0], tick, sp[k][1])
        t = time.time()
        r[f"{k}_pred"] = predict(r["model"], X, rows)
        print(f"{k} predict {len(rows):,} in {time.time()-t:.0f}s")
        score(k, r[f"{k}_pred"], y)
