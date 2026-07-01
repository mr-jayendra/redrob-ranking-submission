#  Candidate Ranking Pipeline

This repository contains a high-efficiency, multi-stage Retrieve-and-Rank pipeline designed to evaluate 100K+ candidates against the "Senior AI Engineer — Founding Team" Job Description. 

It is engineered to run entirely on a **CPU-only environment** within the **≤ 5-minute wall-clock limit** with **zero network calls** during the ranking execution phase.

## 🧠 Pipeline Architecture & Methodology

To comply with the strict compute constraints and properly penalize "keyword stuffers" as explicitly warned in the Job Description, the pipeline is split into two isolated stages:

1. **Stage 1: Offline Pre-computation (`prepare_data.py`)**
   * **Vectorized Schema Flattening:** Parses the deeply nested Redrob JSON schema into a flat, highly optimized Pandas DataFrame.
   * **Honeypot Pruning:** Applies vectorized boolean masks to instantly drop structurally impossible profiles (e.g., negative experience, time-traveling employment dates) before embeddings are generated, ensuring a 0% honeypot rate.
   * **Dense Embedding:** Converts flattened career histories and skills into 384-dimensional vectors using `all-MiniLM-L6-v2`. Artifacts are cached locally to disk.

2. **Stage 2: Online Timed Ranking (`rank.py`)**
   * **Pure Linear Algebra Matching:** Bypasses heavy libraries like FAISS in favor of a mathematically equivalent, ultra-fast Numpy dot-product (`candidate_vecs @ jd_vec`) against the normalized vectors.
   * **Behavioral Multipliers:** The semantic baseline score is heavily modified using JD-specific logic derived from `redrob_signals`:
     * *Ghost Penalty:* Severe penalty for profiles inactive > 180 days or with < 10% response rates.
     * *Title Trap Penalty:* Down-weights non-engineering titles (e.g., Marketing, Sales) to avoid keyword matches.
     * *Experience Sweet-Spot:* Normalizes the score for the target 5-9 year band.
     * *Shipper Bonus:* Positive multiplier for high GitHub activity scores.
   * **Anti-Hallucination Reasoning:** The final 100 explanations are deterministically generated from verifiable dataframe columns, ensuring 100% factual accuracy for manual review.

---

## 📂 Repository Structure

```text
redrob-ranking-submission/
├── data/
│   ├── job_description.txt        # The manual text extraction of the JD
│   └── candidates.jsonl           # [IGNORED IN GIT] Raw candidate pool
├── artifacts/                     # [IGNORED IN GIT] Generated binary vectors 
├── pipeline_utils.py              # Shared schema config, filtering, and scoring math
├── prepare_data.py                # Offline data flattener and embedder
├── rank.py                        # The 5-minute timed execution script
├── requirements.txt               # Pinned dependencies
├── submission_metadata.yaml       # Hackathon portal metadata match
└── README.md                      # Setup and reproduction instructions

```

---

## ⚙️ Setup & Reproduction Instructions

### 1. Environment Setup

Install the strictly pinned dependencies.

```bash
pip install -r requirements.txt

```

*Note on HuggingFace caching:* Ensure your environment has downloaded the `all-MiniLM-L6-v2` weights locally. `rank.py` automatically sets `HF_HUB_OFFLINE=1` to enforce the "no-network" constraint during the timed run.

### 2. Provide the Data

Ensure the following files are present in the `data/` directory:

* `job_description.txt` (Included in the repo)
* `candidates.jsonl` (You must place this file here; it is `.gitignore`d to prevent repo bloat).

---

## 🚀 Execution

### Step A: Pre-Compute Embeddings (Offline / Untimed)

Run this command once per candidate pool. It flattens the JSON, filters honeypots, downloads the embedding model, and caches the `.parquet` and `.npy` artifacts to the `artifacts/` folder.

```bash
python prepare_data.py --candidates data/candidates.jsonl

```

### Step B: The Timed Ranking (Stage 3 Reproduction)

This is the single execution command that produces the final submission CSV. It runs entirely offline, utilizes CPU-only linear algebra, and finishes well within the 5-minute limit (typically ~3-10 seconds for 100K profiles).

```bash
python rank.py --candidates data/candidates.jsonl --jd data/job_description.txt --embeddings artifacts/candidate_embeddings.npy --out submission.csv

```

---

## 📊 Output Contract Assurance

The pipeline utilizes an automated `sanity_check_output` function before writing the CSV to guarantee zero formatting rejections at Stage 1. It strictly enforces:

* Exactly 100 rows (plus header).
* `rank` column is strictly sequential from 1 to 100.
* `score` column is mathematically validated to be monotonically non-increasing.
* `candidate_id` values are unique, utilizing a deterministic ID-ascending tie-breaker for identical scores.

## 🧪 Interactive Sandbox Sandbox

A fully functional, end-to-end Google Colab environment containing a mock sample dataset is available to verify the environment constraints and execution limits:
**https://colab.research.google.com/drive/1rguNINvKGiHLOzvXjoQVOCjvlvcEcSAA?hl=en#scrollTo=MGo7uyAG1f-h**


