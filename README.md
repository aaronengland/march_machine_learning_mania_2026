# March Machine Learning Mania 2026

**Predicting NCAA Tournament Outcomes with an Ensemble of Gradient Boosting and Neural Network Models**

This project builds a full ML pipeline to predict the probability of every possible team matchup in both the 2026 NCAA Men's and Women's basketball tournaments. The solution is scored by **Brier score** (mean squared error of predicted probabilities vs actual 0/1 outcomes).

---

## Overview

| | Men's | Women's |
|---|---|---|
| **Training games** | 2,585 tournament matchups (1985-2025) | 1,717 tournament matchups (1998-2025) |
| **Features** | 62 (seeds, Massey rankings, Elo, margins, advanced shooting, Top-N win%, SOS, recent form) | 57 (seeds, synthetic rankings, Elo, margins, advanced shooting, Top-N win%, SOS, recent form) |
| **Models** | XGBoost, PyTorch, Logistic Regression | CatBoost, PyTorch, Logistic Regression |
| **Hyperparameter tuning** | Optuna Bayesian optimization (50 trials, 4-fold Stage 1 CV) | Optuna Bayesian optimization (30-50 trials, 4-fold Stage 1 CV) |
| **Best single model** | PyTorch (Brier: 0.1731) | PyTorch (Brier: 0.1340) |
| **Final ensemble** | **Brier: 0.1723** | **Brier: 0.1314** |
| **Stage 1 ensemble** | **Brier: 0.1759** | **Brier: 0.1211** |
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

The men's dataset spans 41 seasons (1985-2026) with 198,079 regular season games and 2,585 tournament games. Detailed box scores (FGM, FGA, 3PT, FT, rebounds, assists, turnovers, steals, blocks) are available from 2003 onward. Additionally, **Massey Ordinals** provide pre-tournament rankings from ~191 rating systems (including KenPom, Sagarin, and others) — a uniquely powerful feature for men's predictions.

| Dataset | Rows | Seasons | Description |
|---------|------|---------|-------------|
| Regular season (compact) | 198,079 | 1985-2026 | Game scores and locations |
| Regular season (detailed) | 124,031 | 2003-2026 | Full box score statistics |
| Tournament (compact) | 2,585 | 1985-2025 | Tournament game scores |
| Tournament (detailed) | 1,449 | 2003-2025 | Tournament box scores |
| Massey Ordinals | 5,819,228 | 2003-2026 | Weekly rankings from ~191 systems |
| Tournament seeds | 2,626 | 1985-2025 | Seed assignments per team |

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

The Massey Ordinals dataset contains rankings from ~191 different rating systems. We tested each system's predictive accuracy on tournament outcomes: what fraction of games did the better-ranked team win?

![Massey System Accuracy](02_eda/output/massey_system_accuracy.png)
*Figure 4: Top 30 Massey Ordinal systems ranked by tournament prediction accuracy (Men's, 2003-2025). Gold bars highlight the research-recommended systems (POM, SAG, MOR, WLK) and composite averages. The top individual systems reach ~78% accuracy, while composites are competitive at ~73%.*

### What Separates Tournament Teams?

Tournament teams differ dramatically from non-tournament teams across key metrics.

| Statistic | Non-Tournament | Tournament | Difference |
|-----------|---------------|------------|------------|
| Win Pct | 0.445 | 0.726 | +63.2% |
| Avg Point Diff | -1.93 | +7.82 | +9.74 pts |
| Offensive Efficiency | 1.007 | 1.085 | +7.8% |
| Defensive Efficiency | 1.035 | 0.970 | -6.2% |
| FG% | 0.432 | 0.459 | +6.1% |
| Assists/Game | 12.77 | 14.44 | +13.1% |
| Blocks/Game | 3.28 | 3.85 | +17.2% |

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

For each matchup (TeamA vs TeamB), we compute **difference features**: `TeamA_stat - TeamB_stat`. This captures the relative strength between teams and is the dominant pattern in winning Kaggle solutions. The feature set expanded from 38 to 62 features through the addition of margin features, advanced shooting metrics, and strength-of-opponent features.

| Category | Features | Count |
|----------|----------|-------|
| Seeds | SeedDiff, SeedA, SeedB | 3 |
| Massey Rankings | TopSystemsAvgRankDiff, AvgOrdinalRankDiff, POMDiff, SAGDiff, MORDiff, WLKDiff | 6 |
| Efficiency | OffEffDiff, DefEffDiff, NetEffDiff | 3 |
| Record | WinPctDiff, AvgPointDiffDiff | 2 |
| Shooting | FGPctDiff, FG3PctDiff, FTPctDiff, OppFGPctDiff, OppFG3PctDiff | 5 |
| Advanced Shooting | eFGPctDiff, TSPctDiff, FTrDiff, ThreePArDiff, OppeFGPctDiff, OppTSPctDiff | 6 |
| Box Score Margins | FGM_MarginDiff, FGA_MarginDiff, FGM3_MarginDiff, FTM_MarginDiff, FTA_MarginDiff, OR_MarginDiff, DR_MarginDiff, TotalReb_MarginDiff, TO_MarginDiff, Stl_MarginDiff, Blk_MarginDiff, Ast_MarginDiff | 12 |
| Pace | AvgPossDiff | 1 |
| Conference | IsPowerConfDiff, ConfTourneyChampDiff, ConfRegSeasonChampDiff | 3 |
| Elo | EloDiff | 1 |
| Schedule Strength | SOSDiff, PowerConfWinPctDiff, QualityWinPctDiff, Top25WinPctDiff, Top10WinPctDiff, Top5WinPctDiff | 6 |
| Consistency | ScoreStdDiff | 1 |
| Recent Form | RecentWinPctDiff, RecentAvgPointDiffDiff, RecentOffEffDiff, RecentDefEffDiff, RecentNetEffDiff, RecentFGPctDiff | 6 |
| Interaction | Seed_x_Rank (SeedDiff x TopSystemsAvgRankDiff) | 1 |

*Table 4: Complete feature set for men's predictions (62 features).*

**Key feature engineering innovations:**
- **Margin features**: Instead of raw averages (AvgFGM), we compute margins (AvgFGM - AvgOppFGM) which capture both offensive and defensive quality in a single feature. This follows the pattern from top Kaggle solutions.
- **Advanced shooting**: Effective FG% (eFG%), True Shooting % (TS%), Free Throw Rate (FTr), and 3-Point Attempt Rate (3PAr) — better shooting metrics than raw FG% because they weight 3-pointers and free throws appropriately.
- **Strength of opponent features**: Win% against Top 5, Top 10, and Top 25 ranked opponents (by Massey rankings), plus win% specifically against power conference teams. These capture how a team performs against elite competition.
- **Conference champions**: Both conference tournament champion and regular season conference champion flags.
- **Seed x Rank interaction**: Captures non-linear relationships between seeding and analytical rankings.

**Key preprocessing decisions:**
- **Flip and double**: Each training matchup appears twice — original and mirror (features negated, label flipped). This prevents the model from learning artifacts based on arbitrary team ID ordering and produces exactly balanced labels (50/50)
- **Massey leakage prevention**: Rankings filtered to DayNum < 132 (before Selection Sunday) to prevent post-tournament data from leaking into features
- **Missing data**: ~44% of training rows have NaN for detailed stats (pre-2003 games). Gradient boosting models handle this natively; the neural net and logistic regression impute with 0 (representing no difference between teams)
- **Elo ratings**: Computed iteratively from all historical games with K=32, margin-of-victory adjustment, home-court advantage, and season-start mean reversion

## Cross-Validation Strategy

We use **Leave-One-Season-Out (LOGO)** cross-validation — the standard approach for this competition, validated by research into past winning solutions.

- Each of the 40 seasons (1985-2025, excluding 2020) serves as one fold
- Train on all other seasons' tournament games, predict the held-out season
- This produces honest out-of-fold predictions for every historical game
- 2020 is excluded because the tournament was canceled

| Fold Range | Seasons | Games/Fold | Notes |
|-----------|---------|-----------|-------|
| 1985-2000 | 16 folds | 63 each | Compact stats only |
| 2001-2002 | 2 folds | 64 each | Compact stats only |
| 2003-2010 | 8 folds | 64 each | Full detailed stats + Massey |
| 2011-2019 | 9 folds | 67 each | Full detailed stats + Massey |
| 2021 | 1 fold | 66 | Full stats (2020 skipped) |
| 2022-2025 | 4 folds | 67 each | Stage 1 validation set |

*Table 5: Cross-validation fold structure (Men's). Each fold contains 63-67 tournament games.*

## Hyperparameter Tuning

All models were tuned using **Optuna** (Bayesian optimization with TPE sampler). Tuning used the 4 Stage 1 validation folds (2022-2025) as the objective — fast (4 folds vs 40) and directly relevant since these match the competition's scoring window.

| Model | Trials | Folds | Parameters Tuned |
|-------|--------|-------|-----------------|
| XGBoost | 50 | 4 | max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, reg_alpha, reg_lambda |
| PyTorch | 30 | 4 | hidden1, hidden2, dropout, lr, weight_decay, batch_size |
| Logistic Regression | 50 | 4 | C, penalty (l1/l2/elasticnet), l1_ratio |

*Table 6: Optuna tuning configuration (Men's).*

## Model Training & Results

All models share the same training framework: LOGO-CV on flip-doubled data, isotonic regression calibration on OOF predictions, and final predictions clipped to [0.02, 0.98].

### Tuned Hyperparameters

| Parameter | XGBoost | PyTorch | Logistic Regression |
|-----------|---------|---------|---------------------|
| Architecture | max_depth=2 | BrierNet(64 -> 128 -> 1) | Linear |
| Learning rate | 0.2204 | 0.00993 (Adam) | - |
| Regularization | alpha=0.363, lambda=0.437 | dropout=0.336, weight_decay=0.00892 | C=0.00863 (L1) |
| Subsample | 0.629, colsample=0.831 | - | - |
| Other | min_child_weight=7 | batch_size=64 | solver=saga |
| Loss function | Custom Brier | Brier (MSE) | Log loss |
| Final rounds/epochs | 34 | 23 | - |

*Table 7: Tuned hyperparameters for men's models.*

### Individual Model Results

| Rank | Model | OOF Brier (raw) | OOF Brier (calibrated) | Stage 1 Brier | CV Mean +/- Std |
|------|-------|-----------------|----------------------|---------------|-----------------|
| 1 | PyTorch BrierNet | 0.1774 | **0.1731** | 0.1822 | 0.1773 +/- 0.0189 |
| 2 | XGBoost | 0.1810 | 0.1781 | 0.1804 | 0.1809 +/- 0.0183 |
| 3 | Logistic Regression | 0.1847 | 0.1816 | 0.1917 | 0.1845 +/- 0.0172 |

*Table 8: Men's model comparison, sorted by calibrated OOF Brier score (lower is better).*

PyTorch trained with Brier loss outperforms XGBoost, confirming the research finding that training directly on the competition metric produces better-calibrated probabilities.

### XGBoost Feature Importance

| Feature | Gain |
|---------|------|
| SeedDiff | 185.89 |
| TopSystemsAvgRankDiff | 29.55 |
| EloDiff | 16.83 |
| **Seed_x_Rank** | **14.05** |
| MORDiff | 13.70 |
| AvgOrdinalRankDiff | 8.33 |
| SeedB | 7.65 |
| SOSDiff | 6.01 |
| AvgPointDiffDiff | 5.99 |
| WLKDiff | 5.82 |

*Table 9: Top 10 XGBoost features by gain (Men's). SeedDiff dominates. The new Seed_x_Rank interaction feature is the 4th most important.*

### Ensemble Construction

Ensemble weights were optimized by minimizing Brier score on OOF predictions using constrained optimization (scipy SLSQP, non-negative weights summing to 1).

| Model | Optimized Weight |
|-------|-----------------|
| PyTorch | **0.6978** |
| XGBoost | 0.3022 |
| Logistic Regression | 0.0000 |

| Evaluation | Brier Score |
|-----------|-------------|
| Best single model (PyTorch) | 0.1731 |
| Equal-weight ensemble | 0.1746 |
| **Optimized ensemble** | **0.1723** |
| Improvement over best single | **-0.0008** |
| **Stage 1 ensemble (2022-2025)** | **0.1759** |

*Table 10: Men's ensemble weights and final Brier scores.*

The ensemble heavily favors PyTorch (70%) with XGBoost (30%). Logistic regression receives 0% weight — its predictions are too correlated with the other models to provide ensemble diversity.

---

# Part 2: Women's Tournament

## Data

The women's dataset spans 28 seasons (1998-2026) with 1,717 tournament games. The key difference from men's: **no Massey Ordinals are available**, so team quality must be derived entirely from game results, box scores, and a synthetic ranking built via Ridge regression.

| Dataset | Rows | Seasons | Description |
|---------|------|---------|-------------|
| Regular season (compact) | 142,093 | 1998-2026 | Game scores and locations |
| Regular season (detailed) | 86,773 | 2010-2026 | Full box score statistics |
| Tournament (compact) | 1,717 | 1998-2025 | Tournament game scores |
| Tournament (detailed) | 961 | 2010-2025 | Tournament box scores |
| Tournament seeds | 1,744 | 1998-2025 | Seed assignments per team |

*Table 11: Women's data sources and their coverage.*

| Aspect | Men's | Women's |
|--------|-------|---------|
| Compact results start | 1985 | 1998 |
| Detailed box scores start | 2003 | 2010 |
| Massey Ordinals | 191 systems available | Not available (synthetic rankings via Ridge regression) |
| Coaches data | Available | Not available |
| Number of features | 62 | 57 |
| Ensemble models | XGBoost + PyTorch + LogReg | CatBoost + PyTorch + LogReg |
| LOGO-CV folds | 40 | 27 |

*Table 12: Key differences between men's and women's pipelines.*

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

The women's feature set replaces the 6 Massey-related features with a single synthetic ranking feature (built via Ridge regression on team stats, R^2=0.416), and shares all other feature engineering with the men's pipeline. The Seed_x_Rank interaction uses SyntheticRankDiff instead of TopSystemsAvgRankDiff. Top-N opponent features use SyntheticRank thresholds.

| Category | Features | Count |
|----------|----------|-------|
| Seeds | SeedDiff, SeedA, SeedB | 3 |
| Synthetic Rankings | SyntheticRankDiff | 1 |
| Efficiency | OffEffDiff, DefEffDiff, NetEffDiff | 3 |
| Record | WinPctDiff, AvgPointDiffDiff | 2 |
| Shooting | FGPctDiff, FG3PctDiff, FTPctDiff, OppFGPctDiff, OppFG3PctDiff | 5 |
| Advanced Shooting | eFGPctDiff, TSPctDiff, FTrDiff, ThreePArDiff, OppeFGPctDiff, OppTSPctDiff | 6 |
| Box Score Margins | FGM_MarginDiff, FGA_MarginDiff, FGM3_MarginDiff, FTM_MarginDiff, FTA_MarginDiff, OR_MarginDiff, DR_MarginDiff, TotalReb_MarginDiff, TO_MarginDiff, Stl_MarginDiff, Blk_MarginDiff, Ast_MarginDiff | 12 |
| Pace | AvgPossDiff | 1 |
| Conference | IsPowerConfDiff, ConfTourneyChampDiff, ConfRegSeasonChampDiff | 3 |
| Elo | EloDiff | 1 |
| Schedule Strength | SOSDiff, PowerConfWinPctDiff, QualityWinPctDiff, Top25WinPctDiff, Top10WinPctDiff, Top5WinPctDiff | 6 |
| Consistency | ScoreStdDiff | 1 |
| Recent Form | RecentWinPctDiff, RecentAvgPointDiffDiff, RecentOffEffDiff, RecentDefEffDiff, RecentNetEffDiff, RecentFGPctDiff | 6 |
| Interaction | Seed_x_Rank (SeedDiff x SyntheticRankDiff) | 1 |

*Table 13: Women's feature set (57 features).*

## Model Training & Results

The women's ensemble uses CatBoost, PyTorch, and Logistic Regression. CatBoost was chosen over XGBoost for women's because it handles NaN natively — important since ~44% of women's training data is missing detailed stats (pre-2010). All models were tuned with Optuna.

### Tuned Hyperparameters

| Parameter | CatBoost | PyTorch | Logistic Regression |
|-----------|----------|---------|---------------------|
| Architecture | depth=8 | BrierNet(32 -> 32 -> 1) | Linear |
| Learning rate | 0.1663 | 0.001189 (Adam) | - |
| Regularization | l2_leaf_reg=0.630, random_strength=2.022 | dropout=0.103, weight_decay=1.15e-6 | C=0.02401 (L1) |
| Subsample | 0.680 (Bernoulli) | - | - |
| Other | - | batch_size=256 | solver=saga |
| Loss function | RMSE | Brier (MSE) | Log loss |
| Final rounds/epochs | 30 | early stopping | - |

*Table 14: Tuned hyperparameters for women's models.*

### Individual Model Results

| Rank | Model | OOF Brier (raw) | OOF Brier (calibrated) | Stage 1 Brier | CV Mean +/- Std |
|------|-------|-----------------|----------------------|---------------|-----------------|
| 1 | PyTorch BrierNet | 0.1380 | **0.1340** | 0.1286 | 0.1381 +/- 0.0240 |
| 2 | CatBoost | 0.1403 | 0.1353 | 0.1308 | 0.1404 +/- 0.0224 |
| 3 | Logistic Regression | 0.1417 | 0.1378 | 0.1403 | 0.1417 +/- 0.0195 |

*Table 15: Women's model comparison, sorted by calibrated OOF Brier score.*

Women's Brier scores are significantly lower than men's (~0.134 vs ~0.173), reflecting the more predictable nature of the women's tournament.

### Ensemble Construction

| Model | Optimized Weight |
|-------|-----------------|
| PyTorch | **0.5358** |
| CatBoost | 0.4642 |
| Logistic Regression | 0.0000 |

| Evaluation | Brier Score |
|-----------|-------------|
| Best single model (PyTorch) | 0.1340 |
| Equal-weight ensemble | 0.1325 |
| **Optimized ensemble** | **0.1314** |
| Improvement over best single | **-0.0026** |
| **Stage 1 ensemble (2022-2025)** | **0.1211** |

*Table 16: Women's ensemble weights and final Brier scores.*

---

# Final Submission

| Component | Stage 1 Rows | Stage 2 Rows | Pred Range | Pred Mean |
|-----------|-------------|-------------|------------|-----------|
| Men's ensemble | 261,013 | 66,430 | [0.020, 0.980] | 0.431 |
| Women's ensemble | 258,131 | 65,703 | [0.020, 0.980] | 0.518 |
| **Combined** | **519,144** | **132,133** | **[0.020, 0.980]** | |

*Table 17: Final submission composition.*

All validations passed: correct row counts, correct columns (ID, Pred), no null values, predictions within [0, 1], and IDs match the sample submission exactly.

---

# Key Insights

## What Worked

1. **Brier loss training**: The PyTorch model trained directly on Brier loss outperformed all gradient boosting models for both men's and women's, confirming academic research findings. This was the single most impactful modeling decision.

2. **Feature engineering expansion** (38 -> 62 features for men's, 33 -> 57 for women's): Margin features, advanced shooting metrics (eFG%, TS%), and strength-of-opponent features (Top 5/10/25 win%, PowerConfWinPct) all contributed meaningful signal. The Seed_x_Rank interaction became the 4th most important XGBoost feature.

3. **Bayesian hyperparameter tuning**: Optuna tuning on Stage 1 validation folds improved every model. Key findings: XGBoost prefers very shallow trees (depth=2) with high learning rate (0.22), PyTorch benefits from higher weight decay (0.009), and L1 regularization (LASSO) works best for logistic regression.

4. **Massey Ordinal rankings** (men's only): SeedDiff was the most important feature by far, but Massey rankings provided strong complementary signal. For women's, a synthetic ranking via Ridge regression partially compensated.

5. **Isotonic regression calibration**: Improved every model's Brier score by 0.003-0.005. Gradient boosting models cluster predictions around 0.5 and need post-hoc calibration to spread them appropriately.

6. **Flip-and-double augmentation**: Doubled the training data and eliminated ordering bias. Simple but essential — without it, the model could learn that the lower-ID team wins more often (an artifact of the ID assignment scheme).

7. **Different ensembles for men's vs women's**: Men's uses XGBoost + PyTorch; women's uses CatBoost + PyTorch. CatBoost's native NaN handling benefits the women's data (more missing stats).

## What's Different About Women's vs Men's

- Women's tournament is ~24% more predictable (Brier 0.131 vs 0.172)
- Both ensembles converge to ~50/50 splits between tree and neural net models
- Without Massey Ordinals, the women's pipeline uses a synthetic ranking feature (Ridge regression, R^2=0.416) — competitive but weaker than real Massey systems
- Women's PyTorch prefers a smaller network (32->32) vs men's (64->128), reflecting simpler patterns

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
| Shallow trees (depth 2-4) | Recommended by research to prevent overfitting on small tournament dataset |
| Bayesian HP tuning | More efficient than grid search for continuous hyperparameter spaces |
| Exclude 2020 | Tournament canceled — no data to validate on |

---

# Brier Score Benchmarks

| Benchmark | Approx. Score |
|-----------|--------------|
| Naive (predict 0.5 for everything) | ~0.250 |
| Seed-only model | ~0.210-0.230 |
| **Our men's ensemble (all historical games)** | **0.1723** |
| **Our men's ensemble (Stage 1 2022-2025)** | **0.1759** |
| **Our women's ensemble (all historical games)** | **0.1314** |
| **Our women's ensemble (Stage 1 2022-2025)** | **0.1211** |
| Mid-tier Kaggle solution | ~0.126 |
| Top 10% Kaggle solution | ~0.115-0.125 |

*Table 18: Brier score benchmarks from research, compared to our results.*

Note: Our OOF scores are computed across all historical tournament games including pre-2003/pre-2010 seasons where features are sparser. Performance on recent seasons with full features is better.

---

# Technical Details

## Infrastructure

- **Execution**: AWS SageMaker notebook instances (Python 3.12)
- **Storage**: S3 bucket `s3://march-machine-learning-mania-2026/`
- **Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn, XGBoost 2.1.4, CatBoost, PyTorch 2.6.0, Optuna

## How to Reproduce

1. Place raw Kaggle data in `s3://march-machine-learning-mania-2026/00_data_collection/`
2. Run notebooks in order: `01_data_joining` -> `02_eda` -> `03_data_split` -> `04_preprocessing` -> `05_models` (all 3) -> `06_model_eval` -> `07_submission`
3. Run men's notebooks first, then women's at each stage
4. Model notebooks within a stage can run in any order (or in parallel)

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
