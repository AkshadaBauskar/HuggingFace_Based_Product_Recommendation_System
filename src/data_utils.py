from __future__ import annotations
import pandas as pd


RECENT_COL = "product_status"


def load_catalog(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"product_id", "title", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def mark_recently_viewed(df: pd.DataFrame, recent_ids: list[str]) -> pd.DataFrame:
    df = df.copy()
    df[RECENT_COL] = "not_viewed"
    if recent_ids:
        df.loc[df["product_id"].isin(recent_ids), RECENT_COL] = "recently_viewed"
    return df


def build_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Simple concat; customize if you want (category, brand, etc.)
    df["combined"] = (df["title"].astype(str).str.strip() +
    ". " + df["description"].astype(str).str.strip())
    return df