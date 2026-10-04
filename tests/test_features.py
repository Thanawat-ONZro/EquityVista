import numpy as np
import pandas as pd
import pytest
from src.features import add_indicators, make_sequences

@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    n = 150
    dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
    base_price = 100.0 + np.cumsum(np.random.randn(n))
    df = pd.DataFrame(
        {
            "Open": base_price + np.random.uniform(-0.5, 0.5, n),
            "High": base_price + np.random.uniform(0.5, 1.5, n),
            "Low": base_price - np.random.uniform(0.5, 1.5, n),
            "Close": base_price,
            "Volume": np.random.randint(1000, 5000, n),
        },
        index=dates,
    )
    return df

def test_add_indicators_columns(synthetic_data):
    df_feat = add_indicators(synthetic_data)
    expected_cols = {"Open", "High", "Low", "Close", "Volume", "Return", "MA20", "MA50", "RSI", "MACD", "Signal"}
    assert expected_cols.issubset(df_feat.columns)
    assert not df_feat.isnull().values.any()

def test_make_sequences_shape(synthetic_data):
    df_feat = add_indicators(synthetic_data)
    window = 10
    X, y = make_sequences(df_feat, window)
    n_rows = len(df_feat)
    n_features = 6  # Close, Return, MA20, MA50, RSI, MACD
    assert X.shape == (n_rows - window, window, n_features)
    assert y.shape == (n_rows - window,)
