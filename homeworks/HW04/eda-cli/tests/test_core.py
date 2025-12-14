from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)

def test_quality_flags_detect_constant_and_id_duplicates():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "const": [7, 7, 7],
            "value": [10, 20, 30],
        }
    )

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    assert flags["has_constant_columns"] is True
    assert "const" in flags["constant_columns"]

    assert flags["has_suspicious_id_duplicates"] is True
    assert "user_id" in flags["id_columns_with_duplicates"]


def test_quality_flags_detect_id_duplicates() -> None:
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "value": [10, 20, 30],
        }
    )

    summary = summarize_dataset(df)
    miss = missing_table(df)
    flags = compute_quality_flags(summary, miss)

    assert flags["has_suspicious_id_duplicates"] is True
    assert "user_id" in flags["id_columns_with_duplicates"]


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2
