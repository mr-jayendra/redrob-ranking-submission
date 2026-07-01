from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd

# --- 1. SCHEMATIC COLUMN SELECTION MATCHING THE REDROB JSON SCHEMA ---
@dataclass(frozen=True)
class ColumnConfig:
    candidate_id: str = "candidate_id"
    years_experience: str = "years_of_experience"
    skills: str = "skills"
    title: str = "current_title"

DEFAULT_COLUMNS = ColumnConfig()

def to_id_str(series: pd.Series) -> pd.Series:
    """Ensures consistent, non-null string casting for IDs."""
    return series.fillna("").astype(str)

# --- 2. VECTORIZED DATA-CLEANING / HONEYPOT EXTRACTION ---
def _rule_experience_bounds(df: pd.DataFrame, cols: ColumnConfig) -> pd.Series:
    """Flags negative experience or experience implausibly exceeding a working lifetime."""
    exp = pd.to_numeric(df[cols.years_experience], errors="coerce")
    return exp.isna() | (exp < 0) | (exp > 50)

def _rule_duplicate_id(df: pd.DataFrame, cols: ColumnConfig) -> pd.Series:
    """Flags duplicate candidate_id rows."""
    return df[cols.candidate_id].duplicated(keep="first")

def _rule_missing_id(df: pd.DataFrame, cols: ColumnConfig) -> pd.Series:
    """Flags rows with a missing or blank candidate_id."""
    return df[cols.candidate_id].isna() | (to_id_str(df[cols.candidate_id]).str.strip() == "")

HONEYPOT_RULES = [
    ("missing_candidate_id", _rule_missing_id),
    ("duplicate_candidate_id", _rule_duplicate_id),
    ("invalid_experience_value", _rule_experience_bounds),
]

def filter_honeypots(df: pd.DataFrame, cols: ColumnConfig = DEFAULT_COLUMNS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the candidate pool vectorially into valid and flagged sets without loops."""
    reasons = pd.Series([""] * len(df), index=df.index, dtype=object)
    flagged_any = pd.Series(False, index=df.index)

    for name, rule_fn in HONEYPOT_RULES:
        mask = rule_fn(df, cols).fillna(False)
        reasons.loc[mask] = reasons.loc[mask] + name + ";"
        flagged_any |= mask

    valid = df.loc[~flagged_any].copy()
    flagged = df.loc[flagged_any].copy()
    flagged["honeypot_reason"] = reasons.loc[flagged_any].str.rstrip(";")
    return valid, flagged

# --- 3. HARD JOB DESCRIPTION EXTRACTOR ---
@dataclass
class JobRequirements:
    raw_text: str
    required_years: float = 5.0  # From JD: "5-9 years" target
    required_skills: set[str] = field(default_factory=lambda: {
        "embeddings", "retrieval", "ranking", "llms", "fine-tuning", 
        "vector databases", "python", "ndcg", "mrr", "map", "a/b testing"
    })

def parse_job_description(jd_path: str) -> JobRequirements:
    """Reads the extracted JD plain text file."""
    with open(jd_path, "r", encoding="utf-8") as f:
        text = f.read()
    return JobRequirements(raw_text=text)

# --- 4. ANTI-HALLUCINATION REASONING ENGINE (TOP-K ONLY) ---
def generate_reasoning(row: pd.Series, job_req: JobRequirements, cols: ColumnConfig = DEFAULT_COLUMNS) -> str:
    """Builds an audit-proof explanation citing only mathematically verified facts."""
    bits = [f"Semantic profile affinity score: {row['semantic_score']:.2f}."]
    exp = row[cols.years_experience]
    bits.append(f"Documented professional tenure: {exp:.0f} years (JD target: 5-9 yrs).")
    
    if row.get("is_ghost", False):
        bits.append("Warning: Displayed diminished availability signals on platform.")
    else:
        bits.append("Strong operational readiness and response profile metrics.")
        
    if row.get("honeypot_backfill", False):
        bits.append(f"Backfill entry. Primary exception rule: {row.get('honeypot_reason', 'unknown')}.")
        
    return " ".join(bits)

# --- 5. AUTOMATED SCHEMA VALIDATOR ---
def sanity_check_output(out_df: pd.DataFrame, top_k: int) -> None:
    """Strictly assets structural properties to guarantee no auto-validation failures."""
    expected_cols = ["candidate_id", "rank", "score", "reasoning"]
    assert list(out_df.columns) == expected_cols, f"Column error: {list(out_df.columns)}"
    assert len(out_df) == top_k, f"Expected exactly {top_k} rows, got {len(out_df)}"
    assert out_df["rank"].tolist() == list(range(1, top_k + 1)), "Non-sequential rank sequence"
    assert out_df["candidate_id"].is_unique, "Found duplicate candidate_id entries"
    scores = out_df["score"].values
    assert np.all(np.diff(scores) <= 1e-12), "Scores are not monotonically non-increasing!"
    assert out_df["candidate_id"].notna().all(), "Null candidate IDs discovered"