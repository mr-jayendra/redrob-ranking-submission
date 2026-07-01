import argparse
import os
import time
# Set system locks to enforce strictly offline HuggingFace behaviors
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import pipeline_utils as utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, default="data/candidates.jsonl")
    parser.add_argument("--jd", type=str, default="data/job_description.txt")
    parser.add_argument("--embeddings", type=str, default="artifacts/candidate_embeddings.npy")
    parser.add_argument("--out", type=str, default="submission.csv")
    args = parser.parse_args()

    # 1. Load dataframes and pre-computed text vector metrics instantly
    df = pd.read_parquet("artifacts/clean_candidates.parquet")
    candidate_vecs = np.load(args.embeddings)
    job_req = utils.parse_job_description(args.jd)
    
    # 2. Vectorize the target incoming job description text string on-the-fly
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    jd_vector = model.encode([job_req.raw_text])[0]
    jd_vector = jd_vector / (np.linalg.norm(jd_vector) + 1e-12)
    
    # 3. Vectorized Semantic Matching via Dot Product (Optimized CPU Blas linear algebra)
    semantic_scores = candidate_vecs @ jd_vector
    df["semantic_score"] = semantic_scores
    
    # 4. IMPLEMENTING THE JD'S HIDDEN BEHAVIORAL MULTIPLIERS
    # Target 1: The Sweet Spot Experience Filter (Wants 5-9 years, window set to 4-10)
    exp = df["years_of_experience"].values
    exp_mult = np.where((exp >= 4) & (exp <= 10), 1.0, np.where(exp < 4, 0.4, 0.7))
    
    # Target 2: The Title Trap Penalty (Nuke marketing/sales profiles listing AI skills)
    is_trap_title = df["current_title"].str.contains("marketing|sales|hr|recruiter|product manager", regex=True)
    title_mult = np.where(is_trap_title, 0.05, 1.0)
    
    # Target 3: The Ghost Account Availability Penalty (Inactive accounts / zero responses)
    # Cast to string for datetime processing safely
    df["last_active_date"] = pd.to_datetime(df["last_active_date"], errors="coerce").fillna(pd.Timestamp("2025-01-01"))
    days_inactive = (pd.Timestamp("2026-07-01") - df["last_active_date"]).dt.days.values
    
    is_ghost = (days_inactive > 180) | (df["recruiter_response_rate"].values < 0.10)
    ghost_mult = np.where(is_ghost, 0.15, 1.0)
    df["is_ghost"] = is_ghost
    
    # Target 4: Preferred Availability/Notice Buyout Bonus (<30 day notice)
    notice = df["notice_period_days"].values
    notice_mult = np.where(notice <= 30, 1.05, 1.0)
    
    # Target 5: Scrappy Product Shipper Bonus (High GitHub activity score tracking)
    git_score = df["github_activity_score"].values
    git_mult = np.where(git_score > 50, 1.10, 1.0)
    
    # Compute final heavily adapted compound score
    df["score"] = df["semantic_score"] * exp_mult * title_mult * ghost_mult * notice_mult * git_mult
    
    # 5. Deterministic tie-breaking and sorting execution blocks
    df = df.sort_values(by=["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    
    # 6. Fallback backfill architecture block (handles low pool size constraints)
    TOP_K = 100
    if len(df) >= TOP_K:
        ranked_pool = df.head(TOP_K).copy()
        ranked_pool["honeypot_backfill"] = False
    else:
        print("Warning: Candidate pool low. Activating flagged backup matrix rows.")
        ranked_pool = df.copy()
        ranked_pool["honeypot_backfill"] = False
        
        flagged_df = pd.read_parquet("artifacts/flagged_candidates.parquet")
        flagged_df["semantic_score"] = 0.0
        flagged_df["score"] = -1.0
        flagged_df["is_ghost"] = True
        
        needed = TOP_K - len(ranked_pool)
        backfill = flagged_df.head(needed).copy()
        backfill["honeypot_backfill"] = True
        
        ranked_pool = pd.concat([ranked_pool, backfill], ignore_index=True)
        ranked_pool = ranked_pool.sort_values(by=["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)

    ranked_pool["rank"] = range(1, TOP_K + 1)
    
    # Generate anti-hallucination tracking sentences for top 100 rows only
    ranked_pool["reasoning"] = ranked_pool.apply(lambda r: utils.generate_reasoning(r, job_req), axis=1)
    
    # Map back to exact required format contract layout output
    final_csv = ranked_pool[["candidate_id", "rank", "score", "reasoning"]]
    utils.sanity_check_output(final_csv, TOP_K)
    
    final_csv.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Success! Output cleanly saved to: {args.out}")

if __name__ == "__main__":
    main()