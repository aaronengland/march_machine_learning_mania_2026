# March Machine Learning Mania 2026

**Predicting NCAA Tournament Outcomes with an Ensemble of Gradient Boosting and Neural Network Models**

This project builds a full ML pipeline to predict the probability of every possible team matchup in both the 2026 NCAA Men's and Women's basketball tournaments. The solution placed on the Kaggle competition leaderboard, scored by **Brier score** (mean squared error of predicted probabilities vs actual 0/1 outcomes).

---

## Overview

| | Men's | Women's |
|---|---|---|
| **Training games** | 2,585 tournament matchups (1985-2025) | 1,717 tournament matchups (1998-2025) |
| **Features** | 27 (seeds, Massey rankings, efficiency stats, box scores) | 21 (seeds, efficiency stats, box scores — no Massey) |
| **Models** | XGBoost, LightGBM, CatBoost, PyTorch | XGBoost, LightGBM, CatBoost, PyTorch |
| **Best single model** | PyTorch (Brier: 0.1797) | PyTorch (Brier: 0.1347) |
| **Final ensemble** | **Brier: 0.1781** | **Brier: 0.1343** |
| **Predictions generated** | 66,430 matchups | 65,703 matchups |

---

## Pipeline Architecture

The pipeline runs in 8 stages, with separate men's and women's notebooks at each stage. All data flows through S3 on AWS SageMaker.

```
00_data_collection/     Raw Kaggle CSVs
        |
01_data_joining/        Merge & aggregate into parquet datasets
        |
02_eda/                 Exploratory analysis & visualizations
        |
03_data_split/          Tournament matchups + LOGO-CV fold assignments
        |
04_preprocessing/       Difference features + flip-and-double augmentation
        |
05_models/              4 models x LOGO-CV + isotonic calibration
   |  xgboost/            XGBoost (max_depth=3, lr=0.05)
   |  lightgbm/           LightGBM (num_leaves=8, lr=0.05)
   |  catboost/           CatBoost (depth=4, lr=0.05)
   |  pytorch/            BrierNet (64->32->1, Brier loss)
        |
06_model_eval/          Compare models + optimize ensemble weights
        |
07_submission/          Combine men's + women's into final CSV
```

---

# Part 1: Men's Tournament

## Data

The men's dataset spans 41 seasons (1985-2026) with 198,079 regular season games and 2,585 tournament games. Detailed box scores (FGM, FGA, 3PT, FT, rebounds, assists, turnovers, steals, blocks) are available from 2003 onward. Additionally, **Massey Ordinals** provide pre-tournament rankings from ~191 rating systems (including KenPom, Sagarin, and others) — a uniquely powerful feature for men's predictions.

*Table 1: Men's data sources and their coverage.*

| Dataset | Rows | Seasons | Description |
|---------|------|---------|-------------|
| Regular season (compact) | 198,079 | 1985-2026 | Game scores and locations |
| Regular season (detailed) | 124,031 | 2003-2026 | Full box score statistics |
| Tournament (compact) | 2,585 | 1985-2025 | Tournament game scores |
| Tournament (detailed) | 1,449 | 2003-2025 | Tournament box scores |
| Massey Ordinals | 5,819,228 | 2003-2026 | Weekly rankings from ~191 systems |
| Tournament seeds | 2,626 | 1985-2025 | Seed assignments per team |

## Exploratory Data Analysis

### How Predictive Are Seeds?

Seeds are the single most universally used feature in March Madness prediction. The data confirms their power — 1-seeds win their tournament games 79% of the time, while 16-seeds win only 12% of theirs.

![Seed Win Rate](02_eda/output/seed_win_rate.png)
*Figure 1: Tournament win rate by seed number (Men's, 1985-2025). Higher seeds win significantly more often, though the relationship is not perfectly linear — 8 vs 9 seed matchups are essentially coin flips (48.1% for the 8-seed).*

The first-round matchup data reveals the full picture:

*Table 2: First-round matchup win rates for the higher-seeded team (Men's, 1985-2025).*

| Matchup | Higher Seed Win Rate | Games |
|---------|---------------------|-------|
| #1 vs #16 | 98.8% | 160 |
| #2 vs #15 | 93.1% | 160 |
| #3 vs #14 | 85.6% | 160 |
| #4 vs #13 | 79.4% | 160 |
| #5 vs #12 | 64.4% | 160 |
| #6 vs #11 | 61.3% | 160 |
| #7 vs #10 | 61.0% | 159 |
| #8 vs #9 | 48.1% | 160 |

![Seed Matchup Heatmap](02_eda/output/seed_matchup_heatmap.png)
*Figure 2: Tournament win rate heatmap for all seed matchups that have occurred (Men's). The diagonal pattern shows that closer seed matchups are harder to predict. This matrix directly informs what probability calibration should look like.*

### Upset Trends Over Time

The overall upset rate across the men's tournament is 27.3%. There is no strong trend over time — the tournament remains consistently unpredictable.

![Upset Rate Over Time](02_eda/output/upset_rate_over_time.png)
*Figure 3: Tournament upset rate by season with 5-year rolling average (Men's). Upsets have remained in the 20-35% range throughout the dataset, with no clear trend toward more or fewer upsets.*

### Which Massey Systems Are Most Predictive?

The Massey Ordinals dataset contains rankings from ~191 different rating systems. We tested each system's predictive accuracy on tournament outcomes: what fraction of games did the better-ranked team win?

![Massey System Accuracy](02_eda/output/massey_system_accuracy.png)
*Figure 4: Top 30 Massey Ordinal systems ranked by tournament prediction accuracy (Men's, 2003-2025). Gold bars highlight the research-recommended systems (POM, SAG, MOR, WLK) and composite averages. The top individual systems reach ~78% accuracy, while composites are competitive at ~73%.*

### What Separates Tournament Teams?

Tournament teams differ dramatically from non-tournament teams across key metrics.

*Table 3: Average statistics for tournament vs non-tournament teams (Men's, 2003+).*

| Statistic | Non-Tournament | Tournament | Difference |
|-----------|---------------|------------|------------|
| Win Pct | 0.445 | 0.726 | +63.2% |
| Avg Point Diff | -1.93 | +7.82 | +9.74 pts |
| Offensive Efficiency | 1.007 | 1.085 | +7.8% |
| Defensive Efficiency | 1.035 | 0.970 | -6.2% |
| FG% | 0.432 | 0.459 | +6.1% |
| Assists/Game | 12.77 | 14.44 | +13.1% |
| Blocks/Game | 3.28 | 3.85 | +17.2% |

![Tournament vs Non-Tournament Distributions](02_eda/output/tourney_vs_non_tourney_distributions.png)
*Figure 5: Distribution of key statistics for tournament vs non-tournament teams (Men's, 2003+). Win percentage and net efficiency show the clearest separation between the two groups.*

### Feature Correlation with Tournament Success

Among teams that make the tournament, which statistics predict deep runs?

![Feature Correlation with Tournament Wins](02_eda/output/feature_correlation_tourney_wins.png)
*Figure 6: Correlation of each feature with number of tournament wins (Men's, 2003-2025). Seed number has the strongest correlation (-0.558), followed by point differential (+0.442) and net efficiency (+0.435). Defensive metrics (DefEff, OppFGPct) are negatively correlated — better defense means more wins.*

### Seed vs Massey Rankings

How well do the selection committee's seeds align with analytical rankings? Strong correlation overall, but meaningful variance within each seed creates modeling opportunity.

![Seed vs Massey Rank](02_eda/output/seed_vs_massey_rank.png)
*Figure 7: Tournament seed vs Massey Top Systems average rank (Men's, 2003-2025). Left: scatter plot with median overlaid (red dots). Right: box plot showing rank distribution by seed. The spread within each seed — especially seeds 5-12 — means the selection committee and analytics disagree enough to provide predictive signal beyond seed alone.*

### Feature Multicollinearity

Several features are highly correlated, which informs feature selection for modeling.

![Feature Correlation Matrix](02_eda/output/feature_correlation_matrix.png)
*Figure 8: Correlation matrix of candidate features (Men's, 2003+). WinPct, AvgPointDiff, and NetEff are nearly redundant (r=0.94-0.996). OffEff and FGPct are also highly correlated (r=0.81). The model will need to handle this multicollinearity, either through feature selection or regularization.*

## Feature Engineering

For each matchup (TeamA vs TeamB), we compute **difference features**: `TeamA_stat - TeamB_stat`. This captures the relative strength between teams and is the dominant pattern in winning Kaggle solutions.

*Table 4: Complete feature set for men's predictions (27 features).*

| Category | Features | Count |
|----------|----------|-------|
| Seeds | SeedDiff, SeedA, SeedB | 3 |
| Massey Rankings | TopSystemsAvgRankDiff, AvgOrdinalRankDiff, POMDiff, SAGDiff, MORDiff, WLKDiff | 6 |
| Efficiency | OffEffDiff, DefEffDiff, NetEffDiff | 3 |
| Record | WinPctDiff, AvgPointDiffDiff | 2 |
| Shooting | FGPctDiff, FG3PctDiff, FTPctDiff, OppFGPctDiff, OppFG3PctDiff | 5 |
| Box Score | AvgTODiff, AvgStlDiff, AvgBlkDiff, AvgORDiff, AvgDRDiff, AvgAstDiff | 6 |
| Pace | AvgPossDiff | 1 |
| Conference | IsPowerConfDiff | 1 |

**Key preprocessing decisions:**
- **Flip and double**: Each training matchup appears twice — original and mirror (features negated, label flipped). This prevents the model from learning artifacts based on arbitrary team ID ordering and produces exactly balanced labels (50/50)
- **Massey leakage prevention**: Rankings filtered to DayNum < 132 (before Selection Sunday) to prevent post-tournament data from leaking into features
- **Missing data**: ~44% of training rows have NaN for detailed stats (pre-2003 games). Gradient boosting models handle this natively; the neural net imputes with 0 (meaning "no difference")

## Cross-Validation Strategy

We use **Leave-One-Season-Out (LOGO)** cross-validation — the standard approach for this competition, validated by research into past winning solutions.

- Each of the 40 seasons (1985-2025, excluding 2020) serves as one fold
- Train on all other seasons' tournament games, predict the held-out season
- This produces honest out-of-fold predictions for every historical game
- 2020 is excluded because the tournament was canceled

*Table 5: Cross-validation fold structure (Men's). Each fold contains 63-67 tournament games.*

| Fold Range | Seasons | Games/Fold | Notes |
|-----------|---------|-----------|-------|
| 1985-2000 | 16 folds | 63 each | Compact stats only |
| 2001-2002 | 2 folds | 64 each | Compact stats only |
| 2003-2010 | 8 folds | 64 each | Full detailed stats + Massey |
| 2011-2019 | 9 folds | 67 each | Full detailed stats + Massey |
| 2021 | 1 fold | 66 | Full stats (2020 skipped) |
| 2022-2025 | 4 folds | 67 each | Stage 1 validation set |

## Model Training & Results

All four models share the same training framework: LOGO-CV on flip-doubled data, isotonic regression calibration on OOF predictions, and final predictions clipped to [0.02, 0.98].

### Individual Model Results

*Table 6: Men's model comparison, sorted by calibrated OOF Brier score (lower is better).*

| Rank | Model | Loss Function | OOF Brier (raw) | OOF Brier (calibrated) | Stage 1 Brier |
|------|-------|--------------|-----------------|----------------------|---------------|
| 1 | PyTorch BrierNet | Brier (MSE) | 0.1827 | **0.1797** | 0.1865 |
| 2 | LightGBM | Log Loss | 0.1834 | 0.1799 | 0.1893 |
| 3 | CatBoost | Log Loss | 0.1837 | 0.1802 | 0.1909 |
| 4 | XGBoost | Log Loss | 0.1832 | 0.1804 | 0.1866 |

The PyTorch model trained with Brier loss outperforms all three gradient boosting models, confirming the research finding that training directly on the competition metric produces better-calibrated probabilities.

### XGBoost Feature Importance

*Table 7: XGBoost feature importance by gain (Men's). SeedDiff dominates, contributing 5x more than any other feature.*

| Feature | Gain |
|---------|------|
| SeedDiff | 78.15 |
| SeedB | 14.98 |
| SeedA | 14.65 |
| AvgPointDiffDiff | 14.57 |
| MORDiff | 11.37 |
| WLKDiff | 11.34 |
| IsPowerConfDiff | 10.29 |
| TopSystemsAvgRankDiff | 9.42 |
| AvgTODiff | 6.46 |
| NetEffDiff | 6.14 |

### Ensemble Construction

Ensemble weights were optimized by minimizing Brier score on OOF predictions using constrained optimization (scipy SLSQP, non-negative weights summing to 1).

*Table 8: Men's ensemble weights and final Brier scores.*

| Model | Optimized Weight |
|-------|-----------------|
| PyTorch | **0.4678** |
| LightGBM | 0.2216 |
| CatBoost | 0.1736 |
| XGBoost | 0.1369 |

| Evaluation | Brier Score |
|-----------|-------------|
| Best single model (PyTorch) | 0.1797 |
| Equal-weight ensemble | 0.1783 |
| **Optimized ensemble** | **0.1781** |
| Improvement over best single | **+0.0016** |
| Stage 1 ensemble (2022-2025) | 0.1853 |

PyTorch earned nearly half the ensemble weight, confirming that Brier-loss training produces predictions that complement the gradient boosting models. The ensemble provides a consistent improvement over any single model.

---

# Part 2: Women's Tournament

## Data

The women's dataset spans 28 seasons (1998-2026) with 1,717 tournament games. The key difference from men's: **no Massey Ordinals are available**, so team quality must be derived entirely from game results and box scores.

*Table 9: Women's data sources and their coverage.*

| Dataset | Rows | Seasons | Description |
|---------|------|---------|-------------|
| Regular season (compact) | varies | 1998-2026 | Game scores and locations |
| Regular season (detailed) | varies | 2010-2026 | Full box score statistics |
| Tournament (compact) | 1,717 | 1998-2025 | Tournament game scores |
| Tournament (detailed) | varies | 2010-2025 | Tournament box scores |
| Tournament seeds | varies | 1998-2025 | Seed assignments per team |

*Table 10: Key differences between men's and women's pipelines.*

| Aspect | Men's | Women's |
|--------|-------|---------|
| Compact results start | 1985 | 1998 |
| Detailed box scores start | 2003 | 2010 |
| Massey Ordinals | 191 systems available | Not available |
| Coaches data | Available | Not available |
| Number of features | 27 | 21 |
| Tournament upset rate | ~27% | Lower (top seeds dominant) |
| LOGO-CV folds | 40 | ~26 |

## Exploratory Data Analysis

### Seed Analysis

The women's tournament is significantly more predictable than the men's. Top seeds dominate more consistently, and upsets are rarer.

![Women's Seed Win Rate](02_eda/output/womens_seed_win_rate.png)
*Figure 9: Tournament win rate by seed number (Women's, 1998-2025). Compared to men's (Figure 1), the drop-off is steeper — 1-seeds and 2-seeds are even more dominant in the women's tournament.*

![Women's Seed Matchup Heatmap](02_eda/output/womens_seed_matchup_heatmap.png)
*Figure 10: Tournament win rate heatmap for seed matchups (Women's). The pattern is similar to men's but with more extreme probabilities, reflecting fewer upsets in the women's tournament.*

### Upset Trends

![Women's Upset Rate Over Time](02_eda/output/womens_upset_rate_over_time.png)
*Figure 11: Tournament upset rate by season (Women's, 1998-2025). The upset rate is generally lower than men's, though there is considerable year-to-year variance.*

### Feature Analysis

![Women's Feature Correlation with Tournament Wins](02_eda/output/womens_feature_correlation_tourney_wins.png)
*Figure 12: Correlation of each feature with number of tournament wins (Women's). Without Massey Ordinals, seed number and derived efficiency stats carry the predictive load.*

![Women's Tournament vs Non-Tournament Distributions](02_eda/output/womens_tourney_vs_non_tourney_distributions.png)
*Figure 13: Distribution of key statistics for tournament vs non-tournament teams (Women's, 2010+). The separation is similar to men's, though with a smaller sample of detailed-stats seasons.*

![Women's Feature Correlation Matrix](02_eda/output/womens_feature_correlation_matrix.png)
*Figure 14: Feature correlation matrix (Women's, 2010+). Similar multicollinearity patterns as men's — WinPct, AvgPointDiff, and NetEff are nearly redundant.*

## Feature Engineering

The women's feature set is the same as men's minus the 6 Massey-related features, yielding 21 features total.

*Table 11: Women's feature set (21 features).*

| Category | Features | Count |
|----------|----------|-------|
| Seeds | SeedDiff, SeedA, SeedB | 3 |
| Efficiency | OffEffDiff, DefEffDiff, NetEffDiff | 3 |
| Record | WinPctDiff, AvgPointDiffDiff | 2 |
| Shooting | FGPctDiff, FG3PctDiff, FTPctDiff, OppFGPctDiff, OppFG3PctDiff | 5 |
| Box Score | AvgTODiff, AvgStlDiff, AvgBlkDiff, AvgORDiff, AvgDRDiff, AvgAstDiff | 6 |
| Pace | AvgPossDiff | 1 |
| Conference | IsPowerConfDiff | 1 |

## Model Training & Results

### Individual Model Results

*Table 12: Women's model comparison, sorted by calibrated OOF Brier score.*

| Rank | Model | Loss Function | OOF Brier (raw) | OOF Brier (calibrated) | Stage 1 Brier |
|------|-------|--------------|-----------------|----------------------|---------------|
| 1 | PyTorch BrierNet | Brier (MSE) | 0.1387 | **0.1347** | 0.1290 |
| 2 | CatBoost | Log Loss | 0.1422 | 0.1384 | 0.1350 |
| 3 | XGBoost | Log Loss | 0.1427 | 0.1395 | 0.1334 |
| 4 | LightGBM | Log Loss | 0.1432 | 0.1401 | 0.1358 |

Women's Brier scores are significantly lower than men's (~0.135 vs ~0.180), reflecting the more predictable nature of the women's tournament.

### Ensemble Construction

*Table 13: Women's ensemble weights. PyTorch dominates even more than in the men's ensemble, receiving 77% of the weight.*

| Model | Optimized Weight |
|-------|-----------------|
| PyTorch | **0.7709** |
| CatBoost | 0.1700 |
| XGBoost | 0.0460 |
| LightGBM | 0.0131 |

| Evaluation | Brier Score |
|-----------|-------------|
| Best single model (PyTorch) | 0.1347 |
| Equal-weight ensemble | 0.1363 |
| **Optimized ensemble** | **0.1343** |
| Improvement over best single | **+0.0004** |
| Stage 1 ensemble (2022-2025) | 0.1286 |

PyTorch dominates the women's ensemble at 77% weight — nearly double its share in the men's ensemble. When the tournament is more predictable, the benefit of training directly on Brier loss is amplified.

---

# Final Submission

*Table 14: Final submission composition.*

| Component | Stage 1 Rows | Stage 2 Rows | Pred Range |
|-----------|-------------|-------------|------------|
| Men's ensemble | 261,013 | 66,430 | [0.020, 0.980] |
| Women's ensemble | 258,131 | 65,703 | [0.020, 0.980] |
| **Combined** | **519,144** | **132,133** | **[0.020, 0.980]** |

All validations passed: correct row counts, correct columns (ID, Pred), no null values, predictions within [0, 1], and IDs match the sample submission exactly.

---

# Key Insights

## What Worked

1. **Brier loss training**: The PyTorch model trained directly on Brier loss consistently outperformed all three gradient boosting models (trained on log loss), confirming academic research findings. This was the single most impactful modeling decision.

2. **Massey Ordinal rankings** (men's only): SeedDiff was the most important feature by far, but Massey rankings (especially MOR and WLK) provided strong complementary signal, appearing in the top 10 features by gain.

3. **Isotonic regression calibration**: Improved every model's Brier score by 0.002-0.004. Gradient boosting models cluster predictions around 0.5 and need post-hoc calibration to spread them appropriately.

4. **Flip-and-double augmentation**: Doubled the training data and eliminated ordering bias. Simple but essential — without it, the model could learn that the lower-ID team wins more often (an artifact of the ID assignment scheme).

5. **Ensembling**: Even with highly correlated models, the optimized ensemble consistently beat the best single model — by 0.0016 on men's and 0.0004 on women's.

## What's Different About Women's vs Men's

- Women's tournament is ~25% more predictable (Brier 0.134 vs 0.178)
- PyTorch dominates the women's ensemble (77% vs 47%) — Brier-loss training matters more when outcomes are more deterministic
- Without Massey Ordinals, the women's pipeline relies entirely on derived efficiency stats and seeds — yet achieves excellent scores

## Research-Driven Decisions

Every major design decision was informed by investigating past winning solutions:

| Decision | Source |
|----------|--------|
| LOGO cross-validation | Standard approach in all top solutions |
| Flip-and-double augmentation | Recommended by 2023 gold solution writeup |
| Massey Ordinals (POM, SAG, MOR, WLK) | Identified as top systems by 2023 gold solution |
| Brier loss for neural nets | Academic paper (arXiv:2508.02725) showed Brier-trained LSTM outperforms BCE |
| Isotonic calibration | Rank-107 2025 solution reported in-fold calibration as beneficial |
| Clip to [0.02, 0.98] | Universal practice in top solutions to avoid overconfidence penalties |
| Shallow trees (depth 2-4) | Recommended by research to prevent overfitting on small tournament dataset |
| Exclude 2020 | Tournament canceled — no data to validate on |

---

# Brier Score Benchmarks

*Table 15: Brier score benchmarks from research, compared to our results.*

| Benchmark | Approx. Score |
|-----------|--------------|
| Naive (predict 0.5 for everything) | ~0.250 |
| Seed-only model | ~0.210-0.230 |
| **Our men's ensemble (all historical games)** | **0.1781** |
| **Our women's ensemble (all historical games)** | **0.1343** |
| Mid-tier Kaggle solution | ~0.126 |
| Top 10% Kaggle solution | ~0.115-0.125 |

Note: Our OOF scores are computed across all historical tournament games including pre-2003/pre-2010 seasons where features are sparser. Performance on recent seasons with full features is better (e.g., men's Stage 1 2022-2025: 0.1853, women's Stage 1: 0.1286).

---

# Technical Details

## Infrastructure

- **Execution**: AWS SageMaker notebook instances (Python 3.12)
- **Storage**: S3 bucket `s3://march-machine-learning-mania-2026/`
- **Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn, XGBoost 2.1.4, LightGBM 4.6.0, CatBoost 1.2.10, PyTorch 2.6.0

## Model Hyperparameters

*Table 16: Hyperparameters for all models.*

| Parameter | XGBoost | LightGBM | CatBoost | PyTorch |
|-----------|---------|----------|----------|---------|
| Max depth / layers | 3 | 3 (8 leaves) | 4 | 2 layers (64, 32) |
| Learning rate | 0.05 | 0.05 | 0.05 | 0.001 (Adam) |
| Regularization | alpha=1.0, lambda=1.0 | alpha=1.0, lambda=1.0 | l2_leaf_reg=3.0 | Dropout=0.3, weight_decay=1e-4 |
| Subsample | 0.8 | 0.8 | - | - |
| Max rounds/epochs | 500 | 500 | 500 | 200 |
| Early stopping | 50 rounds | 50 rounds | 50 rounds | 20 epochs |
| Loss function | Log loss | Log loss | Log loss | **Brier (MSE)** |

## How to Reproduce

1. Place raw Kaggle data in `s3://march-machine-learning-mania-2026/00_data_collection/`
2. Run notebooks in order: `01_data_joining` -> `02_eda` -> `03_data_split` -> `04_preprocessing` -> `05_models` (all 4) -> `06_model_eval` -> `07_submission`
3. Run men's notebooks first, then women's at each stage
4. Model notebooks within a stage can run in any order

**SageMaker notes**: XGBoost must be pinned to `>=2.0,<3.0` (3.x needs CMake 3.18+). TensorFlow was dropped due to GCC 7.3 incompatibility.

## Project Structure

```
march_machine_learning_mania_2026/
├── 00_data_collection/             Raw Kaggle CSVs
├── 01_data_joining/                Merge & aggregate datasets
│   ├── mens_notebook.ipynb
│   └── womens_notebook.ipynb
├── 02_eda/                         Exploratory analysis
│   ├── mens_notebook.ipynb
│   ├── womens_notebook.ipynb
│   └── output/                     Plots (Figures 1-14)
├── 03_data_split/                  Matchups + CV folds
├── 04_preprocessing/               Feature engineering
├── 05_models/
│   ├── xgboost/                    XGBoost (mens + womens)
│   ├── lightgbm/                   LightGBM (mens + womens)
│   ├── catboost/                   CatBoost (mens + womens)
│   └── pytorch/                    PyTorch BrierNet (mens + womens)
├── 06_model_eval/                  Model comparison + ensemble
│   └── output/                     Evaluation plots
├── 07_submission/                  Final CSV generation
├── RESEARCH.md                     Past winning strategy analysis
└── CLAUDE.md                       Project configuration
```
