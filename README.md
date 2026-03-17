# March Machine Learning Mania 2026

**Predicting NCAA Tournament Outcomes with an Ensemble of Gradient Boosting and Neural Network Models**

This project builds a full ML pipeline to predict the probability of every possible team matchup in both the 2026 NCAA Men's and Women's basketball tournaments. The solution is scored by **Brier score** (mean squared error of predicted probabilities vs actual 0/1 outcomes).

---

## At a Glance

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MARCH MACHINE LEARNING MANIA 2026                    │
│                                                                             │
│  Goal: Predict P(TeamA beats TeamB) for every possible NCAA tournament      │
│        matchup. Scored by Brier score (lower = better).                     │
│                                                                             │
│  ┌───────────── DATA ──────────────┐  ┌────────── FEATURES ──────────────┐  │
│  │ Men's:  41 seasons (1985-2026)  │  │ Men's:  74 difference features   │  │
│  │         199K regular season games│  │   Seeds, Massey ranks, Elo,     │  │
│  │         2,585 tournament games  │  │   shooting, momentum, SOS...    │  │
│  │         192 Massey ranking systems│ │                                 │  │
│  │                                 │  │ Women's: 63 difference features  │  │
│  │ Women's: 28 seasons (1998-2026) │  │   Seeds, synthetic ranks, Elo,  │  │
│  │          143K regular season games│ │   shooting, momentum, SOS...    │  │
│  │          1,717 tournament games │  │   (no Massey Ordinals available) │  │
│  └─────────────────────────────────┘  └──────────────────────────────────┘  │
│                                                                             │
│  ┌──────────── PIPELINE ──────────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │  Raw CSVs ──► Join & ──► EDA ──► Split ──► Feature ──► 3 Models ──►  │  │
│  │              Aggregate          (LOGO-CV)  Engineering   per gender   │  │
│  │                                                              │       │  │
│  │                                              ┌───────────────┘       │  │
│  │                                              ▼                       │  │
│  │                                      Ensemble (SLSQP            │  │
│  │                                      optimized weights)             │  │
│  │                                              │                       │  │
│  │                                              ▼                       │  │
│  │                                      Final Submission                │  │
│  │                                      (132,133 predictions)           │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────── MODEL TRAINING (3 phases) ────────────────────────────────┐  │
│  │                                                                       │  │
│  │  Phase 1: TUNE           Phase 2: EVALUATE         Phase 3: PREDICT   │  │
│  │  ┌─────────────────┐    ┌─────────────────────┐   ┌────────────────┐  │  │
│  │  │ Optuna Bayesian │    │ Full LOGO-CV        │   │ Train on ALL   │  │  │
│  │  │ optimization    │    │ (40 folds men's,    │   │ historical     │  │  │
│  │  │                 │    │  27 folds women's)  │   │ data           │  │  │
│  │  │ 4 recent folds  │──►│                     │──►│                │  │  │
│  │  │ (2022-2025)     │    │ Produces OOF preds  │   │ Generate 2026  │  │  │
│  │  │                 │    │ for ensemble weights │   │ predictions    │  │  │
│  │  │ Finds best      │    │ & model comparison  │   │                │  │  │
│  │  │ hyperparams     │    │                     │   │ Apply ensemble │  │  │
│  │  └─────────────────┘    └─────────────────────┘   │ weights        │  │  │
│  │                                                    └────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────── RESULTS ──────────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │              Men's Ensemble              Women's Ensemble             │  │
│  │          ┌──────────────────┐        ┌──────────────────┐            │  │
│  │          │ XGBoost    44.9% │        │ CatBoost   33.2% │            │  │
│  │          │ PyTorch    55.1% │        │ PyTorch    66.8% │            │  │
│  │          │ LogReg      0.0% │        │ LogReg      0.0% │            │  │
│  │          └────────┬─────────┘        └────────┬─────────┘            │  │
│  │                   ▼                           ▼                      │  │
│  │          Brier: 0.1747               Brier: 0.1299                  │  │
│  │                                                                       │  │
│  │    ──────────────────────────────────────────────── Benchmarks ──     │  │
│  │    Naive (all 0.5)           ████████████████████████████  0.250     │  │
│  │    Seed-only model           █████████████████████         0.220     │  │
│  │    Our men's ensemble        █████████████████             0.175     │  │
│  │    Our women's ensemble      █████████████                 0.130     │  │
│  │    Top Kaggle solutions      ████████████                  0.120     │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Overview

| | Men's | Women's |
|---|---|---|
| **Training games** | 2,585 tournament matchups (1985-2025) | 1,717 tournament matchups (1998-2025) |
| **Features** | 74 difference features | 63 difference features |
| **Models** | XGBoost, PyTorch, Logistic Regression | CatBoost, PyTorch, Logistic Regression |
| **Best single model** | PyTorch (Brier: 0.1760) | PyTorch (Brier: 0.1313) |
| **Final ensemble** | **Brier: 0.1747** | **Brier: 0.1299** |
| **Stage 1 ensemble** | **Brier: 0.1800** | **Brier: 0.1244** |
| **Predictions generated** | 66,430 matchups | 65,703 matchups |

---

## How the Models Are Fit

Understanding the model training process is critical, because there are three distinct phases that serve different purposes:

### Phase 1: Hyperparameter Tuning (Optuna)

**Goal:** Find the best model configuration (learning rate, depth, dropout, etc.).

- Uses **4 recent seasons (2022-2025)** as validation folds
- For each Optuna trial, the model trains on 3 of those 4 seasons and validates on the held-out season, cycling through all 4
- The mean Brier score across 4 folds is the trial's objective
- 30-50 trials per model using Bayesian optimization (TPE sampler)
- **Why only 4 folds?** Speed (4 folds vs 40 is 10x faster per trial) and recency (recent seasons are most relevant to 2026). The hyperparameters found here are locked in for subsequent phases

| Model | Trials | Tuning Folds | Best Trial Brier |
|-------|--------|-------------|-----------------|
| XGBoost (men's) | 50 | 2022-2025 | 0.1836 |
| PyTorch (men's) | 30 | 2022-2025 | 0.1861 |
| Logistic Regression (men's) | 50 | 2022-2025 | 0.1918 |
| CatBoost (women's) | 50 | 2022-2025 | 0.0874 |
| PyTorch (women's) | 30 | 2022-2025 | tuned |
| Logistic Regression (women's) | 50 | 2022-2025 | tuned |

### Phase 2: Full Cross-Validation (LOGO-CV)

**Goal:** Generate honest out-of-fold (OOF) predictions for every historical tournament game, used for model evaluation and ensemble weight optimization.

- Uses the **tuned hyperparameters from Phase 1** (locked, no further tuning)
- **Leave-One-Season-Out (LOGO):** 40 folds for men's (1985-2025), 27 folds for women's (1998-2025), excluding 2020 (canceled)
- For each fold: train on all other seasons' tournament games, predict the held-out season
- This produces one OOF prediction per historical game — no data leakage
- **Isotonic regression calibration** is applied to OOF predictions to improve probability calibration
- These OOF predictions are what determine ensemble weights in `06_model_eval`

### Phase 3: Final Model for 2026 Predictions

**Goal:** Generate predictions for the actual 2026 tournament submission.

- Uses the **same tuned hyperparameters from Phase 1**
- Trains on **ALL available historical data** (every season from 1985-2025 for men's, 1998-2025 for women's)
- No held-out fold — the final model sees every historical tournament game
- This model generates Stage 1 (2022-2025 validation) and Stage 2 (2026) predictions
- Predictions are calibrated using the isotonic regression model from Phase 2, then clipped to [0.02, 0.98]

### Ensemble (from `06_model_eval`)

- Ensemble weights are optimized by minimizing **recency-weighted Brier score** on Phase 2 OOF predictions using constrained optimization (scipy SLSQP, non-negative weights summing to 1)
- **Recency weighting**: each game's squared error is weighted linearly by season — the most recent season (2025) gets 40x the weight of the oldest (1985) for men's, and 27x for women's. This tells the optimizer "care more about getting recent tournaments right" since modern basketball is most relevant to 2026
- The same weights are applied to Phase 3 final model predictions to produce the submission

```
Phase 1: Tune hyperparams     →  locked config per model
              ↓
Phase 2: LOGO-CV (all folds)  →  OOF predictions → ensemble weights
              ↓
Phase 3: Train on ALL data    →  2026 predictions × ensemble weights → submission
```

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
05_models/              3 models x LOGO-CV + Optuna tuning + isotonic calibration
   |  xgboost/            XGBoost (Optuna-tuned, custom Brier objective) — men's
   |  catboost/           CatBoost (Optuna-tuned, RMSE loss) — women's
   |  pytorch/            BrierNet (Optuna-tuned architecture, Brier loss) — both
   |  logistic_regression/ LogisticRegression (Optuna-tuned C/penalty) — both
        |
06_model_eval/          Compare models + optimize ensemble weights
        |
07_submission/          Combine men's + women's into final CSV
```

---

# Part 1: Men's Tournament

## Data

The men's dataset spans 41 seasons (1985-2026) with 198,577 regular season games and 2,585 tournament games. Detailed box scores (FGM, FGA, 3PT, FT, rebounds, assists, turnovers, steals, blocks) are available from 2003 onward. Additionally, **Massey Ordinals** provide pre-tournament rankings from ~192 rating systems (including KenPom, Sagarin, and others) — a uniquely powerful feature for men's predictions.

| Dataset | Rows | Seasons | Description |
|---------|------|---------|-------------|
| Regular season (compact) | 198,577 | 1985-2026 | Game scores and locations |
| Regular season (detailed) | 124,529 | 2003-2026 | Full box score statistics |
| Tournament (compact) | 2,585 | 1985-2025 | Tournament game scores |
| Tournament (detailed) | 1,449 | 2003-2025 | Tournament box scores |
| Massey Ordinals | 5,865,001 | 2003-2026 | Weekly rankings from ~192 systems |
| Tournament seeds | 2,694 | 1985-2026 | Seed assignments per team |

*Table 1: Men's data sources and their coverage.*

## Exploratory Data Analysis

### How Predictive Are Seeds?

Seeds are the single most universally used feature in March Madness prediction. The data confirms their power — 1-seeds win their tournament games 79% of the time, while 16-seeds win only 12% of theirs.

![Seed Win Rate](02_eda/output/seed_win_rate.png)
*Figure 1: Tournament win rate by seed number (Men's, 1985-2025). Higher seeds win significantly more often, though the relationship is not perfectly linear — 8 vs 9 seed matchups are essentially coin flips (48.1% for the 8-seed).*

The first-round matchup data reveals the full picture:

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

*Table 2: First-round matchup win rates for the higher-seeded team (Men's, 1985-2025).*

![Seed Matchup Heatmap](02_eda/output/seed_matchup_heatmap.png)
*Figure 2: Tournament win rate heatmap for all seed matchups that have occurred (Men's). The diagonal pattern shows that closer seed matchups are harder to predict. This matrix directly informs what probability calibration should look like.*

### Upset Trends Over Time

The overall upset rate across the men's tournament is 27.3%. There is no strong trend over time — the tournament remains consistently unpredictable.

![Upset Rate Over Time](02_eda/output/upset_rate_over_time.png)
*Figure 3: Tournament upset rate by season with 5-year rolling average (Men's). Upsets have remained in the 20-35% range throughout the dataset, with no clear trend toward more or fewer upsets.*

### Which Massey Systems Are Most Predictive?

The Massey Ordinals dataset contains rankings from ~192 different rating systems. We tested each system's predictive accuracy on tournament outcomes: what fraction of games did the better-ranked team win?

![Massey System Accuracy](02_eda/output/massey_system_accuracy.png)
*Figure 4: Top 30 Massey Ordinal systems ranked by tournament prediction accuracy (Men's, 2003-2025). Gold bars highlight the research-recommended systems (POM, SAG, MOR, WLK) and composite averages. The top individual systems reach ~78% accuracy, while composites are competitive at ~73%.*

### What Separates Tournament Teams?

Tournament teams differ dramatically from non-tournament teams across key metrics.

| Statistic | Non-Tournament | Tournament | Difference |
|-----------|---------------|------------|------------|
| Win Pct | 0.442 | 0.726 | +64.3% |
| Avg Point Diff | -2.04 | +7.86 | +9.89 pts |
| Offensive Efficiency | 1.005 | 1.087 | +8.3% |
| Defensive Efficiency | 1.034 | 0.973 | -5.9% |
| FG% | 0.432 | 0.459 | +6.3% |
| Assists/Game | 12.74 | 14.48 | +13.7% |
| Blocks/Game | 3.18 | 3.84 | +20.7% |

*Table 3: Average statistics for tournament vs non-tournament teams (Men's, 2003+).*

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

For each matchup (TeamA vs TeamB), we compute **difference features**: `TeamA_stat - TeamB_stat`. This captures the relative strength between teams and is the dominant pattern in winning Kaggle solutions. The feature set includes 74 features across 10 categories.

| Category | Features | Count | Missing Rate |
|----------|----------|-------|-------------|
| Seeds | SeedDiff, SeedA, SeedB | 3 | 0% |
| Massey Rankings | TopSystemsAvgRankDiff, AvgOrdinalRankDiff, POMDiff, SAGDiff, MORDiff, WLKDiff | 6 | 30% |
| Efficiency | OffEffDiff, DefEffDiff, NetEffDiff | 3 | 44% |
| Record & Momentum | WinPctDiff, WeightedWinPctDiff, WinStreakDiff, LossStreakDiff, AvgPointDiffDiff | 5 | 0-44% |
| Shooting | FGPctDiff, FG3PctDiff, FTPctDiff, OppFGPctDiff, OppFG3PctDiff | 5 | 44% |
| Advanced Shooting | eFGPctDiff, TSPctDiff, FTrDiff, ThreePArDiff, OppeFGPctDiff, OppTSPctDiff | 6 | 44% |
| Box Score Margins | FGM/FGA/FGM3/FTM/FTA/OR/DR/DRB_Pct/TotalReb/TO/Stl/Blk/Ast MarginDiff | 13 | 44% |
| Pace | AvgPossDiff | 1 | 44% |
| Conference & Elo | IsPowerConfDiff, ConfTourneyChampDiff, ConfRegSeasonChampDiff, EloDiff | 4 | 0% |
| Schedule Strength | SOSDiff, PowerConfWinPctDiff, QualityWinPctDiff, ScoreStdDiff, Top25/10/5WinPctDiff | 7 | 0% |
| Home/Road Splits | HomeWinPctDiff, RoadWinPctDiff, Home/RoadTop25/10/5WinPctDiff | 8 | 0% |
| Recent Form | RecentWinPctDiff, RecentAvgPointDiffDiff, RecentOff/Def/NetEffDiff, RecentFGPctDiff | 6 | 29% |
| Interaction | Seed_x_Rank (SeedDiff x TopSystemsAvgRankDiff) | 1 | 0% |
| Box Score Stats | AvgTO/Stl/Blk/OR/DR/Ast Diff, DRB_PctDiff | 7 | 44% |

*Table 4: Complete feature set for men's predictions (74 features). Missing rates reflect pre-2003 seasons lacking detailed box scores.*

**Key feature engineering decisions:**
- **Difference features**: `TeamA_stat - TeamB_stat` for each matchup — the dominant pattern in winning Kaggle solutions
- **Margin features**: `AvgFGM - AvgOppFGM` captures both offense and defense in a single feature
- **Advanced shooting**: eFG%, TS%, FTr, 3PAr — better than raw FG% because they weight 3-pointers and free throws appropriately
- **Momentum features**: WeightedWinPct (linearly increasing weights on recent games), WinStreak, LossStreak
- **DRB%**: Defensive rebound percentage — `DR / (DR + OppOR)` — measures how well a team prevents offensive rebounds, normalized for pace
- **Strength of opponent**: Win% against Top 5/10/25 ranked opponents and power conference teams
- **Flip and double**: Each training matchup appears twice — original and mirror (features negated, label flipped) — preventing ordering bias and producing exactly balanced labels (50/50)
- **Massey leakage prevention**: Rankings filtered to DayNum < 132 (before Selection Sunday)
- **Elo ratings**: Computed iteratively from all historical games with K=32, margin-of-victory adjustment, home-court advantage, and season-start mean reversion

## Cross-Validation Strategy

We use **Leave-One-Season-Out (LOGO)** cross-validation — the standard approach for this competition, validated by research into past winning solutions.

- Each of the 40 seasons (1985-2025, excluding 2020) serves as one fold
- Train on all other seasons' tournament games, predict the held-out season
- This produces honest out-of-fold predictions for every historical game
- 2020 is excluded because the tournament was canceled

| Fold Range | Seasons | Games/Fold | Notes |
|-----------|---------|-----------|-------|
| 1985-2002 | 18 folds | 63 each | Compact stats only |
| 2003-2019 | 17 folds | 64-67 each | Full detailed stats + Massey |
| 2021 | 1 fold | 66 | Full stats (2020 skipped) |
| 2022-2025 | 4 folds | 67 each | Stage 1 validation set |

*Table 5: Cross-validation fold structure (Men's). Each fold contains 63-67 tournament games.*

## Model Training & Results

### Tuned Hyperparameters

| Parameter | XGBoost | PyTorch | Logistic Regression |
|-----------|---------|---------|---------------------|
| Architecture | max_depth=2 | BrierNet(64 -> 32 -> 1) | Linear |
| Learning rate | 0.136 | 0.00989 (Adam) | - |
| Regularization | alpha=0.017, lambda=0.071 | dropout=0.120, weight_decay=0.00392 | C=0.00839 (elasticnet, l1_ratio=0.909) |
| Subsample | 0.500, colsample=0.772 | - | - |
| Other | min_child_weight=4 | batch_size=256 | solver=saga |
| Loss function | Custom Brier | Brier (MSE) | Log loss |
| Final rounds/epochs | 72 | early stopping | - |

*Table 6: Tuned hyperparameters for men's models.*

### Individual Model Results

| Rank | Model | OOF Brier (raw) | OOF Brier (calibrated) | Stage 1 Brier | CV Weighted Mean |
|------|-------|-----------------|----------------------|---------------|-----------------|
| 1 | PyTorch BrierNet | 0.1794 | **0.1760** | 0.1833 | 0.1802 |
| 2 | XGBoost | 0.1812 | 0.1777 | 0.1813 | 0.1819 |
| 3 | Logistic Regression | 0.1846 | 0.1817 | 0.1900 | 0.1865 |

*Table 7: Men's model comparison, sorted by calibrated OOF Brier score (lower is better).*

### XGBoost Feature Importance

| Feature | Gain |
|---------|------|
| SeedDiff | 67.94 |
| TopSystemsAvgRankDiff | 39.16 |
| EloDiff | 23.80 |
| Seed_x_Rank | 16.06 |
| FGM_MarginDiff | 9.12 |
| IsPowerConfDiff | 8.25 |

*Table 8: Top XGBoost features by gain (Men's). SeedDiff dominates, followed by Massey rankings and Elo.*

### Ensemble Construction

Ensemble weights were optimized by minimizing **recency-weighted Brier score** on OOF predictions using constrained optimization (scipy SLSQP, non-negative weights summing to 1). Each game's error is weighted linearly by season — 2025 games count 40x more than 1985 games — so the ensemble favors accuracy on modern basketball.

| Model | Optimized Weight |
|-------|-----------------|
| PyTorch | **0.5508** |
| XGBoost | 0.4492 |
| Logistic Regression | 0.0000 |

| Evaluation | Brier Score |
|-----------|-------------|
| Best single model (PyTorch) | 0.1760 |
| Equal-weight ensemble | 0.1762 |
| **Optimized ensemble** | **0.1747** |
| Improvement over best single | **-0.0013** |
| **Stage 1 ensemble (2022-2025)** | **0.1800** |

*Table 9: Men's ensemble weights and final Brier scores.*

The ensemble splits weight between PyTorch (55.1%) and XGBoost (44.9%). Compared to unweighted optimization (57.5/42.5), recency weighting shifted weight toward XGBoost — indicating it has been relatively stronger on recent seasons. Logistic regression receives 0% weight — its predictions are too correlated with the other models to provide ensemble diversity.

---

# Part 2: Women's Tournament

## Data

The women's dataset spans 28 seasons (1998-2026) with 1,717 tournament games. The key difference from men's: **no Massey Ordinals are available**, so team quality must be derived entirely from game results, box scores, and a synthetic ranking built via Ridge regression.

| Dataset | Rows | Seasons | Description |
|---------|------|---------|-------------|
| Regular season (compact) | 142,507 | 1998-2026 | Game scores and locations |
| Regular season (detailed) | 87,187 | 2010-2026 | Full box score statistics |
| Tournament (compact) | 1,717 | 1998-2025 | Tournament game scores |
| Tournament (detailed) | 961 | 2010-2025 | Tournament box scores |
| Tournament seeds | 1,812 | 1998-2026 | Seed assignments per team |

*Table 10: Women's data sources and their coverage.*

| Aspect | Men's | Women's |
|--------|-------|---------|
| Compact results start | 1985 | 1998 |
| Detailed box scores start | 2003 | 2010 |
| Massey Ordinals | 192 systems available | Not available (synthetic rankings via Ridge regression, R²=0.423) |
| Number of features | 74 | 63 |
| Ensemble models | XGBoost + PyTorch + LogReg | CatBoost + PyTorch + LogReg |
| LOGO-CV folds | 40 | 27 |

*Table 11: Key differences between men's and women's pipelines.*

## Exploratory Data Analysis

### Seed Analysis

The women's tournament is significantly more predictable than the men's. Top seeds dominate more consistently, and the overall upset rate is 21.1% (vs 27.3% for men's).

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

The women's feature set uses 63 features. It replaces the 6 Massey-related features with a single synthetic ranking feature (built via Ridge regression on team stats, R²=0.423), and shares all other feature engineering with the men's pipeline. The Seed_x_Rank interaction uses SyntheticRankDiff instead of TopSystemsAvgRankDiff. Top-N opponent features use SyntheticRank thresholds. The women's pipeline does not include the men's Home/Road Top-N split features, but does include HomeWinPct and RoadWinPct.

| Category | Features | Count |
|----------|----------|-------|
| Seeds | SeedDiff, SeedA, SeedB | 3 |
| Synthetic Rankings | SyntheticRankDiff | 1 |
| Efficiency | OffEffDiff, DefEffDiff, NetEffDiff | 3 |
| Record & Momentum | WinPctDiff, WeightedWinPctDiff, WinStreakDiff, LossStreakDiff, AvgPointDiffDiff | 5 |
| Shooting | FGPctDiff, FG3PctDiff, FTPctDiff, OppFGPctDiff, OppFG3PctDiff | 5 |
| Advanced Shooting | eFGPctDiff, TSPctDiff, FTrDiff, ThreePArDiff, OppeFGPctDiff, OppTSPctDiff | 6 |
| Box Score Margins | FGM/FGA/FGM3/FTM/FTA/OR/DR/DRB_Pct/TotalReb/TO/Stl/Blk/Ast MarginDiff | 13 |
| Pace | AvgPossDiff | 1 |
| Conference & Elo | IsPowerConfDiff, ConfTourneyChampDiff, ConfRegSeasonChampDiff, EloDiff | 4 |
| Schedule Strength | SOSDiff, PowerConfWinPctDiff, QualityWinPctDiff, ScoreStdDiff, Top25/10/5WinPctDiff | 7 |
| Home/Road Splits | HomeWinPctDiff, RoadWinPctDiff | 2 |
| Recent Form | RecentWinPctDiff, RecentAvgPointDiffDiff, RecentOff/Def/NetEffDiff, RecentFGPctDiff | 6 |
| Box Score Stats | AvgTO/Stl/Blk/OR/DR/Ast Diff, DRB_PctDiff | 7 |
| Interaction | Seed_x_Rank (SeedDiff x SyntheticRankDiff) | 1 |

*Table 12: Women's feature set (63 features).*

## Model Training & Results

The women's ensemble uses CatBoost, PyTorch, and Logistic Regression. CatBoost was chosen over XGBoost for women's because it handles NaN natively — important since ~44% of women's training data is missing detailed stats (pre-2010). All models were tuned with Optuna.

### Tuned Hyperparameters

| Parameter | CatBoost | PyTorch | Logistic Regression |
|-----------|----------|---------|---------------------|
| Architecture | depth=7 | BrierNet(256 -> 128 -> 1) | Linear |
| Learning rate | 0.291 | 0.00639 (Adam) | - |
| Regularization | l2_leaf_reg=0.366, random_strength=9.390 | dropout=0.358, weight_decay=2.97e-6 | C=0.01491 (elasticnet, l1_ratio=0.624) |
| Subsample | 0.706 (Bernoulli) | - | - |
| Other | iterations=500 | batch_size=128 | solver=saga |
| Loss function | RMSE | Brier (MSE) | Log loss |
| Final rounds/epochs | 30 | early stopping | - |

*Table 13: Tuned hyperparameters for women's models.*

### Individual Model Results

| Rank | Model | OOF Brier (raw) | OOF Brier (calibrated) | Stage 1 Brier | CV Weighted Mean |
|------|-------|-----------------|----------------------|---------------|-----------------|
| 1 | PyTorch BrierNet | 0.1353 | **0.1313** | 0.1303 | 0.1332 |
| 2 | CatBoost | 0.1421 | 0.1378 | 0.1287 | 0.1404 |
| 3 | Logistic Regression | 0.1415 | 0.1379 | 0.1358 | 0.1399 |

*Table 14: Women's model comparison, sorted by calibrated OOF Brier score.*

Women's Brier scores are significantly lower than men's (~0.13 vs ~0.18), reflecting the more predictable nature of the women's tournament.

### Ensemble Construction

Ensemble weights were optimized using recency-weighted Brier score — 2025 games count 27x more than 1998 games.

| Model | Optimized Weight |
|-------|-----------------|
| PyTorch | **0.6676** |
| CatBoost | 0.3324 |
| Logistic Regression | 0.0000 |

| Evaluation | Brier Score |
|-----------|-------------|
| Best single model (PyTorch) | 0.1313 |
| Equal-weight ensemble | 0.1315 |
| **Optimized ensemble** | **0.1299** |
| Improvement over best single | **-0.0014** |
| **Stage 1 ensemble (2022-2025)** | **0.1244** |

*Table 15: Women's ensemble weights and final Brier scores.*

---

# Final Submission

| Component | Stage 1 Rows | Stage 2 Rows | Pred Range | Pred Mean |
|-----------|-------------|-------------|------------|-----------|
| Men's ensemble | 261,013 | 66,430 | [0.020, 0.980] | — |
| Women's ensemble | 258,131 | 65,703 | [0.020, 0.980] | — |
| **Combined Stage 1** | **519,144** | — | **[0.020, 0.980]** | **0.499** |
| **Combined Stage 2** | — | **132,133** | **[0.020, 0.980]** | **0.498** |

*Table 16: Final submission composition.*

All validations passed: correct row counts, correct columns (ID, Pred), no null values, predictions within [0, 1], and IDs match the sample submission exactly.

---

# Key Insights

## What Worked

1. **Brier loss training**: The PyTorch model trained directly on Brier loss outperformed all gradient boosting models for both men's and women's. This was the single most impactful modeling decision.

2. **Feature engineering expansion** (74 features for men's, 63 for women's): Margin features, advanced shooting metrics (eFG%, TS%), momentum features (WeightedWinPct, WinStreak, LossStreak), defensive rebound rate (DRB%), and strength-of-opponent features all contributed signal.

3. **Bayesian hyperparameter tuning**: Optuna tuning on Stage 1 validation folds improved every model. Key findings: XGBoost prefers shallow trees (depth=2) with moderate learning rate, PyTorch benefits from moderate weight decay, and elastic net with high L1 ratio works best for logistic regression.

4. **Massey Ordinal rankings** (men's only): SeedDiff was the most important feature by far, but Massey rankings and Elo provided strong complementary signal. For women's, a synthetic ranking via Ridge regression partially compensated.

5. **Isotonic regression calibration**: Improved every model's Brier score by 0.003-0.004. Gradient boosting models cluster predictions around 0.5 and need post-hoc calibration to spread them appropriately.

6. **Flip-and-double augmentation**: Doubled the training data and eliminated ordering bias. Simple but essential — without it, the model could learn that the lower-ID team wins more often (an artifact of the ID assignment scheme).

7. **Different ensembles for men's vs women's**: Men's uses XGBoost + PyTorch; women's uses CatBoost + PyTorch. CatBoost's native NaN handling benefits the women's data (more missing stats). Both ensembles give majority weight to PyTorch.

## What's Different About Women's vs Men's

- Women's tournament is ~26% more predictable (Brier 0.130 vs 0.175)
- Both ensembles give PyTorch majority weight, but men's is a more balanced split (55.1/44.9) vs women's (66.8/33.2)
- Without Massey Ordinals, the women's pipeline uses a synthetic ranking feature (Ridge regression, R²=0.423) — competitive but weaker than real Massey systems
- Women's PyTorch prefers a larger network (256->128) vs men's (64->32), possibly compensating for the lack of Massey features by learning more complex interactions

## Research-Driven Decisions

Every major design decision was informed by investigating past winning solutions:

| Decision | Source |
|----------|--------|
| LOGO cross-validation | Standard approach in all top solutions |
| Flip-and-double augmentation | Recommended by 2023 gold solution writeup |
| Massey Ordinals (POM, SAG, MOR, WLK) | Identified as top systems by 2023 gold solution |
| Brier loss for neural nets | Academic paper showed Brier-trained networks outperform BCE |
| Margin features (team - opponent) | Common pattern in top Kaggle March Madness solutions |
| Isotonic calibration | Rank-107 2025 solution reported in-fold calibration as beneficial |
| Clip to [0.02, 0.98] | Universal practice in top solutions to avoid overconfidence penalties |
| Bayesian HP tuning | More efficient than grid search for continuous hyperparameter spaces |
| Recency-weighted ensemble optimization | Modern seasons are most relevant to 2026 — weighting recent games more heavily improved Stage 1 scores |
| Exclude 2020 | Tournament canceled — no data to validate on |

---

# Brier Score Benchmarks

| Benchmark | Approx. Score |
|-----------|--------------|
| Naive (predict 0.5 for everything) | ~0.250 |
| Seed-only model | ~0.210-0.230 |
| **Our men's ensemble (all historical games)** | **0.1747** |
| **Our men's ensemble (Stage 1 2022-2025)** | **0.1800** |
| **Our women's ensemble (all historical games)** | **0.1299** |
| **Our women's ensemble (Stage 1 2022-2025)** | **0.1244** |
| Mid-tier Kaggle solution | ~0.126 |
| Top 10% Kaggle solution | ~0.115-0.125 |

*Table 17: Brier score benchmarks from research, compared to our results.*

Note: Our OOF scores are computed across all historical tournament games including pre-2003/pre-2010 seasons where features are sparser. Performance on recent seasons with full features is better.

---

# Technical Details

## Infrastructure

- **Execution**: AWS SageMaker notebook instances (Python 3.12)
- **Storage**: S3 bucket `s3://march-machine-learning-mania-2026/`
- **Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn, XGBoost, CatBoost, PyTorch, Optuna

## How to Reproduce

1. Place raw Kaggle data in `s3://march-machine-learning-mania-2026/00_data_collection/`
2. Run notebooks in order: `01_data_joining` -> `02_eda` -> `03_data_split` -> `04_preprocessing` -> `05_models` (all 3) -> `06_model_eval` -> `07_submission`
3. Run men's notebooks first, then women's at each stage
4. Model notebooks within a stage can run in any order (or in parallel)
5. To update for a new season: re-run the full pipeline once Kaggle updates the data

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
│   ├── xgboost/                    XGBoost (men's only)
│   ├── catboost/                   CatBoost (women's only)
│   ├── pytorch/                    PyTorch BrierNet (both ensembles)
│   └── logistic_regression/        Logistic Regression (both ensembles)
├── 06_model_eval/                  Model comparison + ensemble
│   └── output/                     Evaluation plots
├── 07_submission/                  Final CSV generation
├── RESEARCH.md                     Past winning strategy analysis
└── CLAUDE.md                       Project configuration
```
