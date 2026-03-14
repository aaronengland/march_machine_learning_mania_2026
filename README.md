# March Machine Learning Mania 2026

Predict the probability of every possible team matchup in the 2026 NCAA Division I Men's and Women's basketball tournaments. Scored by **Brier score** (mean squared error of predicted probabilities vs actual 0/1 outcomes). Lower is better.

**Deadline**: March 19, 2026

## Table of Contents

1. [Competition Overview](#competition-overview)
2. [Infrastructure](#infrastructure)
3. [Project Structure](#project-structure)
4. [Pipeline Stages](#pipeline-stages)
   - [00 Data Collection](#00-data-collection)
   - [01 Data Joining](#01-data-joining)
   - [02 EDA](#02-exploratory-data-analysis)
   - [03 Data Split](#03-data-split)
   - [04 Preprocessing](#04-preprocessing)
   - [05 Models](#05-models)
   - [06 Model Evaluation & Ensemble](#06-model-evaluation--ensemble)
   - [07 Submission](#07-submission)
5. [Men's vs Women's Pipeline Differences](#mens-vs-womens-pipeline-differences)
6. [Results](#results)
7. [Research Foundation](#research-foundation)
8. [How to Reproduce](#how-to-reproduce)

---

## Competition Overview

### Submission Format
- CSV with two columns: `ID` and `Pred`
- `ID` format: `YYYY_XXXX_YYYY` where XXXX < YYYY are TeamIDs
- `Pred`: probability that the lower-ID team (XXXX) beats the higher-ID team (YYYY)
- Men's TeamIDs: 1000–1999. Women's TeamIDs: 3000–3999
- Must predict every possible matchup, not just tournament games
- Stage 1 covers 2022–2025 (validation). Stage 2 covers 2026 (real submission)

### Scoring
Brier score = mean of (predicted probability - actual outcome)^2 across all games that actually occurred in the tournament. Since 2024, the Brier score is averaged over 6 tournament rounds rather than all games equally, heavily weighting late-round accuracy.

---

## Infrastructure

- **Execution environment**: AWS SageMaker notebook instances (Python 3.12)
- **Data storage**: S3 bucket `s3://march-machine-learning-mania-2026/`
- **S3 layout**: Each pipeline stage stores outputs in a matching prefix, separated by gender:
  - `s3://.../{stage}/{gender}/filename.parquet`
- **Local data**: `00_data_collection/` exists locally for development. It is gitignored.

---

## Project Structure

```
march_machine_learning_mania_2026/
├── CLAUDE.md                       # Project instructions for Claude
├── RESEARCH.md                     # Research findings on past winning strategies
├── README.md                       # This file
├── 00_data_collection/             # Raw Kaggle data (local + S3)
├── 01_data_joining/
│   ├── mens_notebook.ipynb
│   ├── womens_notebook.ipynb
│   └── output/
├── 02_eda/
│   ├── mens_notebook.ipynb
│   ├── womens_notebook.ipynb
│   └── output/                     # Plots saved here
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
│   │   └── womens_notebook.ipynb
│   ├── lightgbm/
│   │   ├── mens_notebook.ipynb
│   │   └── womens_notebook.ipynb
│   ├── catboost/
│   │   ├── mens_notebook.ipynb
│   │   └── womens_notebook.ipynb
│   └── pytorch/
│       ├── mens_notebook.ipynb
│       └── womens_notebook.ipynb
├── 06_model_eval/
│   ├── mens_notebook.ipynb
│   ├── womens_notebook.ipynb
│   └── output/                     # Plots saved here
└── 07_submission/
    ├── generate_submission.ipynb    # Combines men's + women's
    └── output/
```

---

## Pipeline Stages

### 00 Data Collection

Raw Kaggle competition data. Not modified by any pipeline stage.

**Men's files** (prefixed `M`):
- `MRegularSeasonCompactResults.csv` — game scores, 1985–2026 (198,079 games)
- `MRegularSeasonDetailedResults.csv` — box scores, 2003–2026 (124,031 games)
- `MNCAATourneyCompactResults.csv` — tournament scores, 1985–2025 (2,585 games)
- `MNCAATourneyDetailedResults.csv` — tournament box scores, 2003–2025 (1,449 games)
- `MNCAATourneySeeds.csv` — tournament seeds (2,626 entries)
- `MMasseyOrdinals.csv` — weekly rankings from ~197 systems, 2003–2026 (5.8M rows)
- `MTeams.csv`, `MTeamConferences.csv`, `MTeamCoaches.csv`, `MSeasons.csv`

**Women's files** (prefixed `W`):
- Same structure but compact results start 1998, detailed start 2010
- No Massey Ordinals, no coaches data

**Shared**: `Conferences.csv`, `Cities.csv`, `SampleSubmissionStage1.csv`, `SampleSubmissionStage2.csv`

---

### 01 Data Joining

**Purpose**: Join raw CSV files into clean, unified parquet datasets for downstream analysis.

**Men's notebook** (`01_data_joining/mens_notebook.ipynb`):

1. **Load** all raw men's CSV files from S3 (with local fallback)
2. **Merge compact + detailed results** for regular season and tournament games. Compact results (1985–2026) are left-joined with detailed box scores (2003–2026), preserving all rows with NaN for pre-2003 seasons
3. **Build team-centric game rows**: Convert winner/loser format into two rows per game (one per team), with team's own stats and opponent stats. This is the "unpivot" step
4. **Aggregate per team per season**: From the team-centric rows, compute season averages — wins, losses, WinPct, scoring, all box score stats, shooting percentages, estimated possessions (FGA - OR + TO + 0.475*FTA), offensive/defensive efficiency (points per possession), and NetEff
5. **Clean tournament seeds**: Extract numeric seed from strings like `W01`, `X16a` → SeedNum, Region, PlayIn
6. **Filter Massey Ordinals**: Remove post-tournament rankings (DayNum >= 132) to prevent data leakage. Keep only the latest pre-tournament ranking per system/team/season. Pivot to wide format (one column per ranking system). Compute average across all systems and average of top 4 systems (POM, SAG, MOR, WLK)
7. **Build team metadata**: Join team names, conference affiliations (with power conference flag for ACC, Big Ten, Big 12, SEC, Pac-12, Big East), and coaches (filtered to those active through Selection Sunday)

**Outputs** (S3 `01_data_joining/mens/`):
| File | Shape | Description |
|------|-------|-------------|
| `regular_season_games.parquet` | (198079, 34) | All regular season games with box scores where available |
| `tourney_games.parquet` | (2585, 34) | All tournament games with box scores where available |
| `team_season_stats.parquet` | (13753, 45) | Per-team per-season aggregates |
| `tourney_seeds.parquet` | (2626, 6) | Cleaned seeds with numeric values |
| `massey_ordinals_pre_tourney.parquet` | (8356, 195) | Wide-format pre-tournament rankings |
| `team_metadata.parquet` | (13753, 7) | Team names, conferences, coaches |

**Women's notebook**: Same structure, outputs same datasets minus `massey_ordinals_pre_tourney` and minus coach info. Compact results start 1998, detailed start 2010. ~1.5% of 2010–2012 games may have missing detailed results.

---

### 02 Exploratory Data Analysis

**Purpose**: Understand data patterns to inform feature engineering and modeling decisions.

**Men's notebook** (`02_eda/mens_notebook.ipynb`) — 10 sections:

1. **Seed analysis**: Tournament win rate by seed (1-seeds win ~80%+ of games), first-round matchup win probabilities (1v16: 98.8%, 5v12: 64.4%, 8v9: 48.1%), seed matchup heatmap
2. **Upset trends**: Overall upset rate is 27.3%. Plotted over time with 5-year rolling average — no strong trend
3. **Massey Ordinal system ranking**: Tested all ~191 systems for predictive accuracy on tournament games. Top systems include BKM, DP, LYD, plus the research-recommended POM/SAG/MOR/WLK. The composite `TopSystemsAvgRank` is competitive with individual systems
4. **Tournament vs non-tournament teams**: Compared 14 stats. Tournament teams have 63% higher WinPct, 7.8% higher OffEff, 6.2% lower DefEff. Biggest separators: WinPct, AvgPointDiff, NetEff
5. **Feature correlation with tournament wins**: SeedNum (-0.558), AvgPointDiff (+0.442), NetEff (+0.435) are top correlates
6. **Scoring and pace trends**: Scoring has declined over the decades. Pace (possessions) has been relatively stable 2003+
7. **Seed vs Massey rank**: Strong correlation (r ≈ 0.9+) between tournament seed and Massey top systems rank, but variance within each seed creates opportunity
8. **Power conference analysis**: Power conference teams average 1.46 tournament wins vs 0.54 for mid-majors, but this is largely captured by seed
9. **Feature correlation matrix**: Identified highly correlated pairs (WinPct/AvgPointDiff/NetEff at r=0.94–0.996, OffEff/FGPct at r=0.81) — important for feature selection
10. **Key takeaways**: Recommended feature priority list

**Women's notebook**: Same structure minus Massey analysis and seed-vs-Massey comparison (no Massey data available for women's).

**Plots saved to**: `02_eda/output/` (PNG files)

---

### 03 Data Split

**Purpose**: Define the matchup-level datasets and cross-validation strategy.

**Key decisions**:
- **Target format**: Matches the submission format — TeamA (lower ID) vs TeamB (higher ID), Label = P(TeamA wins)
- **CV strategy**: Leave-One-Season-Out (LOGO). Each season is one fold. Train on all other seasons, validate on the held-out season's tournament games
- **Exclude 2020**: No tournament was played
- **No explicit train/test split**: The "test set" is Stage 2 (2026). Stage 1 (2022–2025) serves as the held-out evaluation set for model comparison

**Men's notebook** (`03_data_split/mens_notebook.ipynb`):

1. **Build tournament matchups**: Convert tournament games from winner/loser format to (Season, TeamA, TeamB, Label) where TeamA < TeamB and Label = 1 if TeamA won. Creates the submission-format ID string
2. **Assign LOGO folds**: Fold column = Season. 40 folds (1985–2025, excluding 2020). ~63–67 games per fold. Label balance: 0.512
3. **Mark Stage 1 validation**: IsStage1Val flag for 2022–2025 seasons (268 games)
4. **Parse prediction grids**: Extract (Season, TeamA, TeamB) from sample submission CSVs, filter to men's (TeamIDs 1000–1999). Stage 1: 261,013 rows. Stage 2: 66,430 rows
5. **Merge labels into Stage 1 grid**: Attach actual tournament outcomes where games occurred (268 of 261,013 rows)

**Outputs** (S3 `03_data_split/mens/`):
| File | Shape | Description |
|------|-------|-------------|
| `tourney_matchups.parquet` | (2585, 7) | Historical tournament matchups with labels and fold assignments |
| `prediction_grid_stage1.parquet` | (261013, 5) | All possible men's matchups 2022–2025, with labels where available |
| `prediction_grid_stage2.parquet` | (66430, 4) | All possible men's matchups 2026 |

**Women's notebook**: Same structure, TeamIDs 3000+, tournament data starts 1998.

---

### 04 Preprocessing

**Purpose**: Construct matchup-level features from team-season data. The dominant pattern is **difference features**: `TeamA_stat - TeamB_stat`.

**Men's notebook** (`04_preprocessing/mens_notebook.ipynb`):

1. **Load data** from 01_data_joining (team stats, seeds, Massey, metadata) and 03_data_split (matchups, grids)
2. **Build matchup features** for each (Season, TeamA, TeamB):
   - **Seed features**: SeedA, SeedB, SeedDiff
   - **Massey rank differences**: TopSystemsAvgRankDiff, AvgOrdinalRankDiff, POMDiff, SAGDiff, MORDiff, WLKDiff
   - **Team stat differences**: WinPctDiff, AvgPointDiffDiff, OffEffDiff, DefEffDiff, NetEffDiff, FGPctDiff, FG3PctDiff, FTPctDiff, AvgTODiff, AvgStlDiff, AvgBlkDiff, AvgORDiff, AvgDRDiff, AvgAstDiff, OppFGPctDiff, OppFG3PctDiff, AvgPossDiff
   - **Conference**: IsPowerConfDiff
3. **Flip and double** training data: Each matchup produces two rows — original and mirror (features negated, label flipped). This prevents the model from learning artifacts from the arbitrary lower-ID-first ordering. Label balance becomes exactly 0.500
4. **Apply to prediction grids**: Same feature construction for Stage 1 and Stage 2 (no flip-and-double)
5. **Sanity checks**: Verify diff feature means are ~0 after augmentation. Verify feature-label correlations go the expected direction (SeedDiff: -0.492, WinPctDiff: +0.324, NetEffDiff: +0.402)

**Total features**: 27 (men's), ~21 (women's, no Massey features)

**Missing data**: ~43.9% of training features are NaN for pre-2003 games (no detailed stats or Massey). Gradient boosting models handle NaN natively. Neural nets impute with 0 (meaning "no difference"). Stage 2 (2026) has 100% missing for seeds (tournament hasn't started) and SAG.

**Outputs** (S3 `04_preprocessing/mens/`):
| File | Shape | Description |
|------|-------|-------------|
| `train_features.parquet` | (5170, 82) | Flip-doubled training matchups with features |
| `stage1_features.parquet` | (261013, 80) | Stage 1 grid with features |
| `stage2_features.parquet` | (66430, 79) | Stage 2 grid with features |
| `feature_columns.parquet` | (27, 1) | List of feature column names |

**Women's notebook**: Same structure, 21 features (no Massey-related features).

---

### 05 Models

**Purpose**: Train 4 models using LOGO-CV, generate calibrated out-of-fold predictions, and produce Stage 1/2 predictions.

All models follow the same framework:
1. Load preprocessed features from 04_preprocessing
2. Run 40-fold LOGO CV (train on flip-doubled data, validate on original rows only)
3. Collect OOF predictions and per-fold Brier scores
4. Train final model on all data (using median best round from CV)
5. Calibrate with isotonic regression fit on OOF predictions
6. Generate Stage 1/2 predictions, clip to [0.02, 0.98]
7. Save OOF predictions, Stage 1/2 predictions, CV results

#### XGBoost (`05_models/xgboost/`)

- **Version**: 2.1.4 (pinned to `>=2.0,<3.0` for SageMaker compatibility — 3.x requires CMake 3.18+ which isn't available)
- **Hyperparameters**: max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_alpha=1.0, reg_lambda=1.0
- **Rounds**: 500 max with early stopping at 50. Final model uses median best round from CV (138)
- **Key detail**: Uses `iteration_range=(0, model.best_iteration + 1)` in CV predict calls to ensure best model is used regardless of XGBoost version

**Men's results**: OOF Brier: 0.1832 raw → 0.1804 calibrated. Top features by gain: SeedDiff (78.2), SeedB (15.0), SeedA (14.6), AvgPointDiffDiff (14.6), MORDiff (11.4)

**Women's results**: OOF Brier: 0.1427 raw → 0.1395 calibrated

#### LightGBM (`05_models/lightgbm/`)

- **Version**: 4.6.0
- **Hyperparameters**: max_depth=3, num_leaves=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_samples=10, reg_alpha=1.0, reg_lambda=1.0
- **Rounds**: 500 max, early stopping 50. Final model: 115 rounds (men's)

**Men's results**: OOF Brier: 0.1834 raw → 0.1799 calibrated

**Women's results**: OOF Brier: 0.1432 raw → 0.1401 calibrated

#### CatBoost (`05_models/catboost/`)

- **Version**: 1.2.10
- **Hyperparameters**: depth=4, learning_rate=0.05, l2_leaf_reg=3.0
- **Rounds**: 500 max, early stopping 50. Final model: 156 rounds (men's)
- **Note**: Handles NaN natively via ordered boosting

**Men's results**: OOF Brier: 0.1837 raw → 0.1802 calibrated

**Women's results**: OOF Brier: 0.1422 raw → 0.1384 calibrated

#### PyTorch (`05_models/pytorch/`)

- **Architecture**: BrierNet — feedforward network with 2 hidden layers (64 → 32 → 1), ReLU activations, Dropout(0.3), Sigmoid output
- **Loss function**: **Brier loss (MSE)** directly, not binary cross-entropy. Research shows this produces better-calibrated probabilities for Brier-scored competitions
- **Preprocessing**: NaN imputed with 0 (difference features — 0 means no difference), then StandardScaler fit per fold
- **Training**: Adam optimizer (lr=0.001, weight_decay=1e-4), batch_size=128, max 200 epochs, patience=20
- **Note**: TensorFlow was dropped due to incompatible GCC version (7.3) on SageMaker

**Men's results**: OOF Brier: 0.1827 raw → 0.1797 calibrated (best single model)

**Women's results**: OOF Brier: 0.1387 raw → 0.1347 calibrated (best single model)

---

### 06 Model Evaluation & Ensemble

**Purpose**: Compare all models, build an optimized ensemble, and generate final predictions.

**Men's notebook** (`06_model_eval/mens_notebook.ipynb`):

1. **Individual model comparison**: Side-by-side Brier scores (raw, calibrated, Stage 1, CV mean/std)
2. **Calibration curves**: Raw vs calibrated predictions plotted against perfect calibration diagonal
3. **Prediction distributions**: Histograms showing how spread out each model's probabilities are
4. **Per-fold Brier scores**: Line plot across all 40 seasons — models track each other closely, with 2007 easiest and 2022 hardest
5. **Model correlation**: Heatmap of prediction correlations between models — high correlation between gradient boosting models, PyTorch provides some diversity
6. **Ensemble weight optimization**: Scipy SLSQP minimization of Brier score on OOF predictions, constrained to non-negative weights summing to 1
7. **Ensemble evaluation**: Compare optimized vs equal-weight ensemble vs best single model on both full OOF and Stage 1 subset

**Men's ensemble results**:

| Metric | Score |
|--------|-------|
| Best single model (PyTorch, calibrated) | 0.1797 |
| Equal-weight ensemble | 0.1783 |
| Optimized ensemble | **0.1781** |
| Ensemble improvement over best single | +0.0016 |

**Optimized weights** (men's):
| Model | Weight |
|-------|--------|
| PyTorch | 0.4678 |
| LightGBM | 0.2216 |
| CatBoost | 0.1736 |
| XGBoost | 0.1369 |

PyTorch earned nearly half the ensemble weight, confirming that training with Brier loss directly produces predictions that complement the gradient boosting models.

**Outputs** (S3 `06_model_eval/mens/`):
| File | Description |
|------|-------------|
| `ensemble_stage1_predictions.parquet` | Final ensemble predictions for 2022–2025 matchups |
| `ensemble_stage2_predictions.parquet` | Final ensemble predictions for 2026 matchups |
| `ensemble_weights.parquet` | Optimized model weights |
| `model_comparison.parquet` | Side-by-side model metrics |

**Women's notebook** (`06_model_eval/womens_notebook.ipynb`): Same structure with 4 models.

**Women's ensemble results**:

| Metric | Score |
|--------|-------|
| Best single model (PyTorch, calibrated) | 0.1347 |
| Equal-weight ensemble | 0.1363 |
| Optimized ensemble | **0.1343** |
| Ensemble improvement over best single | +0.0004 |

**Optimized weights** (women's):
| Model | Weight |
|-------|--------|
| PyTorch | 0.7709 |
| CatBoost | 0.1700 |
| XGBoost | 0.0460 |
| LightGBM | 0.0131 |

PyTorch dominates even more in the women's ensemble (77%), reflecting the stronger benefit of Brier-loss training when the tournament is more predictable (fewer upsets).

---

### 07 Submission

**Purpose**: Combine men's and women's ensemble predictions into the final Kaggle submission CSV.

**Notebook** (`07_submission/generate_submission.ipynb`):

1. **Load sample submissions** to get required row IDs
2. **Load men's ensemble predictions** from 06_model_eval
3. **Load women's ensemble predictions** from 06_model_eval
4. **Combine** men's and women's predictions by concatenating
5. **Merge with sample submission** to ensure every required row has a prediction. Missing predictions filled with 0.5
6. **Validate**: Row count, column names, no nulls, prediction range [0,1], ID matching
7. **Final clip** to [0.02, 0.98]

**Outputs** (S3 `07_submission/`):
- `submission_stage1.csv` — 519,144 rows (2022–2025 validation)
- `submission_stage2.csv` — 132,133 rows (2026 real submission)

---

## Men's vs Women's Pipeline Differences

| Aspect | Men's | Women's |
|--------|-------|---------|
| Compact results | 1985–2026 | 1998–2026 |
| Detailed box scores | 2003–2026 | 2010–2026 |
| Massey Ordinals | Yes (~191 systems) | Not available |
| Coaches data | Yes | Not available |
| TeamID range | 1000–1999 | 3000–3999 |
| Number of features | 27 | ~21 (no Massey) |
| Tournament upset rate | ~27% | Lower (top seeds dominant) |
| Models | XGBoost, LightGBM, CatBoost, PyTorch | Same 4 models |
| Separate pipelines | Yes | Yes |

The pipelines are structurally identical but run on separate data. The women's pipeline has fewer features due to the absence of Massey Ordinals, which means team quality must be derived entirely from game results and box scores.

---

## Results

### Men's Model Comparison (OOF Brier Score, Calibrated)

| Rank | Model | OOF Brier | Stage 1 Brier |
|------|-------|-----------|---------------|
| 1 | PyTorch (Brier loss) | 0.1797 | 0.1865 |
| 2 | LightGBM | 0.1799 | 0.1893 |
| 3 | CatBoost | 0.1802 | 0.1909 |
| 4 | XGBoost | 0.1804 | 0.1866 |
| **E** | **Ensemble (optimized)** | **0.1781** | **0.1853** |

### Women's Model Comparison (OOF Brier Score, Calibrated)

| Rank | Model | OOF Brier | Stage 1 Brier |
|------|-------|-----------|---------------|
| 1 | PyTorch (Brier loss) | 0.1347 | 0.1290 |
| 2 | CatBoost | 0.1384 | 0.1350 |
| 3 | XGBoost | 0.1395 | 0.1334 |
| 4 | LightGBM | 0.1401 | 0.1358 |
| **E** | **Ensemble (optimized)** | **0.1343** | **0.1286** |

Women's Brier scores are significantly lower than men's (~0.134 vs ~0.178), reflecting the women's tournament being more predictable (fewer upsets, top seeds more dominant).

### Final Submission

| Component | Stage 1 Rows | Stage 2 Rows | Pred Range |
|-----------|-------------|-------------|------------|
| Men's ensemble | 261,013 | 66,430 | [0.020, 0.980] |
| Women's ensemble | 258,131 | 65,703 | [0.020, 0.980] |
| **Combined** | **519,144** | **132,133** | **[0.020, 0.980]** |

### Brier Score Benchmarks (from research)

| Benchmark | Approx. Score |
|-----------|--------------|
| Naive (predict 0.5) | ~0.250 |
| Seed-only model | ~0.210–0.230 |
| Mid-tier solution | ~0.126 |
| Top 10% | ~0.115–0.125 |

Note: Our OOF scores are computed on all historical tournament games including pre-2003/pre-2010 seasons with sparser features. Scores on recent seasons with full features are better.

---

## Research Foundation

Before building models, the `researcher` agent investigated past winning strategies. Key findings documented in `RESEARCH.md`:

1. **XGBoost is the consensus top performer** — confirmed by 2025 winner (Mohammad Odeh) and multiple top solutions
2. **Massey Ordinals are the strongest men's feature** — averaging top systems (POM, SAG, MOR, WLK) is competitive with more complex approaches
3. **Seed difference is universally used** — even naive seed-only models are competitive
4. **Train neural nets with Brier loss** (not BCE) for Brier-scored competitions
5. **Leave-One-Season-Out CV** is the standard validation strategy
6. **Flip and double training data** to eliminate winner/loser ordering bias
7. **Isotonic regression or Platt scaling** for post-hoc calibration
8. **Clip predictions to [0.02, 0.98]** to avoid extreme overconfidence penalties
9. **Filter Massey Ordinals to DayNum < 132** to prevent data leakage from post-tournament rankings
10. **Exclude 2020** from CV (tournament was canceled)

---

## How to Reproduce

### Prerequisites
- AWS SageMaker notebook instance with S3 access to `s3://march-machine-learning-mania-2026/`
- Python 3.12 with: pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost (>=2.0,<3.0), lightgbm, catboost, pytorch, boto3, s3fs

### Step-by-step

**Stage 0**: Raw data should already be in S3 at `s3://march-machine-learning-mania-2026/00_data_collection/`

**Stage 1 — Data Joining**: Run `01_data_joining/mens_notebook.ipynb`, then `womens_notebook.ipynb`. Each reads raw CSVs and writes parquet datasets to S3.

**Stage 2 — EDA**: Run `02_eda/mens_notebook.ipynb`, then `womens_notebook.ipynb`. Generates plots in `02_eda/output/`. No data written to S3.

**Stage 3 — Data Split**: Run `03_data_split/mens_notebook.ipynb`, then `womens_notebook.ipynb`. Creates matchup datasets and prediction grids.

**Stage 4 — Preprocessing**: Run `04_preprocessing/mens_notebook.ipynb`, then `womens_notebook.ipynb`. Builds difference features and flip-doubles training data.

**Stage 5 — Models**: Run all 4 model notebooks for men's, then all 4 for women's:
- `05_models/xgboost/mens_notebook.ipynb` (then womens)
- `05_models/lightgbm/mens_notebook.ipynb` (then womens)
- `05_models/catboost/mens_notebook.ipynb` (then womens)
- `05_models/pytorch/mens_notebook.ipynb` (then womens)

Each model can be run independently. Order within gender doesn't matter.

**Stage 6 — Model Eval**: Run `06_model_eval/mens_notebook.ipynb`, then `womens_notebook.ipynb`. Compares models and builds the ensemble.

**Stage 7 — Submission**: Run `07_submission/generate_submission.ipynb`. Combines men's and women's ensemble predictions into the final CSV.

### Notes
- Each notebook reads from the previous stage's S3 outputs. Local fallback paths are provided for development
- XGBoost must be pinned to `>=2.0,<3.0` on SageMaker (3.x requires CMake 3.18+ which isn't available)
- TensorFlow was dropped due to GCC 7.3 incompatibility on SageMaker
- All notebooks use the same structure: imports → functions → constants → make output dir → pipeline steps → save → summary
