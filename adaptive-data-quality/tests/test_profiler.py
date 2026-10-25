"""Profiler tests (Section 5)."""
import numpy as np
import pandas as pd
import pytest

from profiling.profiler import profile_batch


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 500
    d = pd.DataFrame({"record_id": [f"r{i}" for i in range(n)],
                      "amount": rng.normal(75, 15, n),
                      "status": rng.choice(["Active", "Inactive", "Pending"], n, p=[.6, .3, .1])})
    d.loc[:4, "amount"] = np.nan
    return d


def test_numerical_fields(df):
    p = profile_batch(df, ["amount"], ["status"], batch="T01")["columns"]["amount"]
    s = df["amount"].dropna()
    for k in ("count", "min", "max", "mean", "median", "std", "quantiles", "missing_pct", "unique_count"):
        assert k in p
    assert p["count"] == len(s)
    assert p["missing_pct"] == pytest.approx(1.0)
    assert p["mean"] == pytest.approx(s.mean()) and p["std"] == pytest.approx(s.std())
    assert p["median"] == pytest.approx(s.median())
    assert p["quantiles"]["p25"] == pytest.approx(s.quantile(0.25))


def test_categorical_fields(df):
    p = profile_batch(df, ["amount"], ["status"])["columns"]["status"]
    assert p["cardinality"] == 3
    assert p["most_common"][0] == "Active"
    assert sum(p["frequencies"].values()) == pytest.approx(1.0)
    assert p["missing_pct"] == 0


def test_dataset_level(df):
    df = df.dropna().reset_index(drop=True)    # null rows with the same status would also count as duplicates
    d2 = pd.concat([df, df.iloc[:10].assign(record_id=[f"x{i}" for i in range(10)])], ignore_index=True)
    p = profile_batch(d2, ["amount"], ["status"])["dataset"]
    n = len(df) + 10
    assert p["record_count"] == n and p["column_count"] == 3
    assert p["duplicate_pct"] == pytest.approx(100 * 10 / n)   # record_id is ignored for duplicates
    assert set(p["schema"]) == {"record_id", "amount", "status"}
