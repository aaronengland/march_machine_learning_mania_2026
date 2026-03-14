# March Machine Learning Mania 2026

## Competition Goal

Predict the probability of every possible team matchup in the 2026 NCAA Division I Men's and Women's basketball tournaments. Scored by **Brier score** (mean squared error of predicted probabilities vs actual 0/1 outcomes). Lower is better.

**Deadline: March 19, 2026**

## Submission Format

CSV with two columns: `ID` and `Pred`.

- `ID` format: `2026_XXXX_YYYY` where XXXX is the lower TeamID and YYYY is the higher TeamID
- `Pred`: probability that the lower-ID team (XXXX) beats the higher-ID team (YYYY)
- Men's TeamIDs: 1000–1999. Women's TeamIDs: 3000–3999. They do not overlap.
- You must predict EVERY possible matchup, not just tournament teams
- Stage 1 sample submission covers 2022–2025 (for validation). Stage 2 covers 2026 (the real submission).
- Stage 1: ~519K rows. Stage 2: ~132K rows.
- The final submission combines both men's and women's predictions into a single CSV.

## Infrastructure

- **Execution environment**: AWS SageMaker notebook instances
- **Data storage**: S3 bucket `s3://march-machine-learning-mania-2026/`
- **Local data**: `00_data_collection/` exists locally only for Claude's access. It is gitignored and not pushed.
- **S3 layout**: Each pipeline stage stores its outputs in a matching S3 prefix:
  - `s3://march-machine-learning-mania-2026/00_data_collection/` — raw Kaggle data
  - `s3://march-machine-learning-mania-2026/01_data_joining/` — joined datasets
  - `s3://march-machine-learning-mania-2026/02_eda/` — EDA outputs (if any large artifacts)
  - `s3://march-machine-learning-mania-2026/03_data_split/` — train/val/test splits
  - `s3://march-machine-learning-mania-2026/04_preprocessing/` — preprocessed features
  - `s3://march-machine-learning-mania-2026/05_models/` — trained model artifacts
  - `s3://march-machine-learning-mania-2026/06_model_eval/` — evaluation results
  - `s3://march-machine-learning-mania-2026/07_submission/` — final submission CSVs
- **Local output folders**: Each pipeline stage directory contains an `output/` subfolder for small tables, graphs, and artifacts that are useful to review locally (e.g., plots, summary stats, small CSVs).
- **S3 I/O in notebooks**: Use `boto3` or `s3fs` + `pandas` for reading/writing data. Every notebook should define the bucket name and relevant S3 prefixes at the top.

## Key Data Notes

- Raw data lives in `s3://march-machine-learning-mania-2026/00_data_collection/` (and locally in `00_data_collection/` for Claude)
- Men's files prefixed with `M`, women's with `W`. Cities.csv and Conferences.csv span both.
- Seasons are referenced by the year the tournament is played (e.g., 2026 = the 2025–26 season)
- `DayNum` is an offset from `DayZero` in the Seasons file. DayNum 154 = men's championship game. DayNum 132 = Selection Sunday.
- Compact Results: scores only (men from 1985, women from 1998)
- Detailed Results: full box scores — FGM, FGA, FGM3, FGA3, FTM, FTA, OR, DR, Ast, TO, Stl, Blk, PF (men from 2003, women from 2010)
- `WTeamID`/`WScore` = winning team (not women's). `LTeamID`/`LScore` = losing team.
- Massey Ordinals (`MMasseyOrdinals.csv`): weekly rankings from dozens of systems (Pomeroy, Sagarin, RPI, etc.) — men's only, from 2003
- Tournament seeds use format like `W01`, `X16a` — first char is region (W/X/Y/Z), next two digits are seed number, optional a/b for play-in
- ~1.5% of women's games in 2010–2012 may be missing detailed results

## Men's vs Women's: Separate Pipelines

Men's and women's data are processed in **separate notebooks** within each pipeline stage. This is necessary because:

- Men's detailed stats go back to 2003; women's only to 2010
- Massey Ordinals (public rankings) exist for men only
- Tournament structures and scheduling differ
- Feature availability and history length are different, so models may need different approaches

Each pipeline stage folder contains exactly two notebooks:

- `mens_notebook.ipynb`
- `womens_notebook.ipynb`

The folder name describes the step. The notebook name only distinguishes gender.

S3 paths also separate by gender within each stage:
- `s3://march-machine-learning-mania-2026/01_data_joining/mens/`
- `s3://march-machine-learning-mania-2026/01_data_joining/womens/`

The final submission notebook in `07_submission/` combines both men's and women's predictions into the single required CSV.

## Project Structure

```
march_machine_learning_mania_2026/
├── CLAUDE.md                       # This file — project context for Claude
├── 00_data_collection/             # Raw Kaggle data (local + S3). Do not modify.
├── 01_data_joining/
│   ├── mens_notebook.ipynb
│   ├── womens_notebook.ipynb
│   └── output/
├── 02_eda/
│   ├── mens_notebook.ipynb
│   ├── womens_notebook.ipynb
│   └── output/
├── 03_data_split/
│   ├── mens_notebook.ipynb
│   ├── womens_notebook.ipynb
│   └── output/
├── 04_preprocessing/
│   ├── mens_notebook.ipynb
│   ├── womens_notebook.ipynb
│   └── output/
├── 05_models/
│   ├── xgboost/
│   │   ├── mens_notebook.ipynb
│   │   ├── womens_notebook.ipynb
│   │   └── output/
│   ├── lightgbm/
│   │   ├── mens_notebook.ipynb
│   │   ├── womens_notebook.ipynb
│   │   └── output/
│   ├── catboost/
│   │   ├── mens_notebook.ipynb
│   │   ├── womens_notebook.ipynb
│   │   └── output/
│   ├── pytorch/
│   │   ├── mens_notebook.ipynb
│   │   ├── womens_notebook.ipynb
│   │   └── output/
│   └── tensorflow/
│       ├── mens_notebook.ipynb
│       ├── womens_notebook.ipynb
│       └── output/
├── 06_model_eval/
│   ├── mens_notebook.ipynb
│   ├── womens_notebook.ipynb
│   └── output/
├── 07_submission/
│   ├── generate_submission.ipynb    # Combines men's + women's into final CSV
│   └── output/
└── data/                            # Gitignored. Not used in SageMaker — use S3 instead.
```

## Code Conventions

- **Format**: Jupyter notebooks (`.ipynb`) for all pipeline stages
- **Language**: Python 3
- **Libraries**: XGBoost, LightGBM, CatBoost, PyTorch, TensorFlow, plus pandas, numpy, sklearn, matplotlib, seaborn, boto3, s3fs as needed
- **Style**: Clean and well-commented. Explain the reasoning behind modeling decisions, not just what the code does. Use markdown cells in notebooks to narrate the analysis.
- **Data flow**: Each stage reads from the previous stage's S3 outputs. Raw data in `00_data_collection/` (S3) is never modified.
- **Naming**: Numbered folder prefixes to show pipeline order (00_, 01_, 02_, etc.). Every stage folder contains `mens_notebook.ipynb` and `womens_notebook.ipynb` — the folder name describes the step, the notebook name only distinguishes gender. Exception: `05_models/` has subfolders per model (xgboost, lightgbm, catboost, pytorch, tensorflow), each with `mens_notebook.ipynb` and `womens_notebook.ipynb`.
- **Notebook boilerplate**: Every notebook should start with a cell defining:
  ```python
  BUCKET = "march-machine-learning-mania-2026"
  GENDER = "mens"  # or "womens"
  STAGE = "01_data_joining"  # matches the folder name
  INPUT_PREFIX = f"s3://{BUCKET}/00_data_collection/"
  OUTPUT_PREFIX = f"s3://{BUCKET}/{STAGE}/{GENDER}/"
  ```
- **Local outputs**: Small tables, plots, and summary artifacts are saved to the `output/` subfolder within each stage directory.
- **Documentation**: After completing any pipeline stage, update `README.md` in the repo root to document what was done, why, and what the outputs are. The README should serve as a complete narrative of the project — someone reading it should understand the entire process from data collection through final submission without opening a single notebook. Include sections for each pipeline stage, key decisions made, and results/metrics where applicable.

## Research

Before building models, run the `researcher` agent (defined in `.claude/agents/researcher.md`) to investigate past winning solutions, common features, and top approaches. The agent writes its findings to `RESEARCH.md` in the repo root. This file should be read before starting feature engineering or model building, as it informs decisions across the entire pipeline.

## Modeling Strategy Notes

- Build separate models for men's and women's using XGBoost, LightGBM, CatBoost, PyTorch, and TensorFlow
- Predictions must be well-calibrated probabilities (not just rankings) since Brier score penalizes overconfident wrong predictions heavily
- Validation should use Stage 1 data (2022–2025 tournament results) to simulate leaderboard scoring
- `06_model_eval/` has two responsibilities:
  1. **Individual model evaluation** — compare each model's Brier score, calibration curves, and prediction distributions
  2. **Ensemble construction** — build an ensemble combining all 5 models (e.g., weighted averaging, stacking, or blending). The ensemble predictions are what get passed to `07_submission/`.
- The final submission in `07_submission/` merges men's and women's ensemble predictions into one CSV
