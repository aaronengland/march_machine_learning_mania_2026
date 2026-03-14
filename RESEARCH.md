# March Machine Learning Mania — Research Findings

Research compiled March 2026, focusing on past Kaggle competition years 2021–2025.

---

## 1. Competition Format Context

The competition has evolved significantly over the years:

- **Pre-2023**: Separate men's and women's competitions, scored by log loss.
- **2023 onwards**: Men's and women's combined into one submission, scoring switched to **Brier score** (MSE of predicted probabilities vs. 0/1 outcomes). Lower is better.
- **2024 scoring change**: Brier score averaged over 6 rounds (not per-game), which heavily weights late-round games. A championship game loss is worth far more than a first-round loss.
- The task is to predict **every possible matchup** (not just actual ones), so you must produce ~132K rows for the Stage 2 (2026) submission.

---

## 2. Past Winning Solutions

### 2025 — 1st Place: Mohammad Odeh

- **Winner**: Mohammad Odeh (Kaggle handle: modeh7)
- Writeup and notebook available at: https://www.kaggle.com/competitions/march-machine-learning-mania-2025/writeups/mohammad-odeh-first-place-solution
- Final solution notebook: https://www.kaggle.com/code/modeh7/final-solution-ncaa-2025
- Specific details of features and model are not publicly summarized in any blog post yet, but **XGBoost was reported to outperform CatBoost and LightGBM** for this competition winner.

### 2025 — Rank ~107 (LinkedIn writeup, A. G.):

- Used a **single XGBoost model** with 23 features selected after permutation importance filtering.
- Features: attacking/defensive metrics, seed differences, GLM-derived team quality metrics, historical point-difference statistics.
- **Cross-validation strategy**: Leave-1-Group-Out (LOGO), leaving out 1 season per fold across 2003–2024 (20-fold, excluding 2020).
- **Calibration**: Cubic-spline calibration applied in-fold (not post-hoc), which improved CV score slightly.
- CV Score: 9.258 (MAE). Lesson: "Simple single models like XGBoost are sufficient."

### 2024 — 1st Place

- Winner was a high school science and stats teacher using **R to implement a Monte Carlo simulation** based on a combination of third-party team ratings and personal intuition.
- Second place also used R.
- This year's winner essentially did **team rating simulation** rather than pure ML, illustrating that well-calibrated domain knowledge can beat complex ensembles.

### 2023 — Top 1% Gold (maze508, Medium writeup)

This is the most detailed publicly available solution writeup for the modern competition format.

**Features used:**
- External rating systems: Selected the **top 10 historically accurate Massey Ordinal systems** (Pomeroy/POM, Sagarin/SAG, Moore/MOR, Whitlock/WLK were consistently among the best).
- Win rates and point differentials from regular season.
- Home/away split statistics.
- Team box score aggregates.
- A derived "Risk Appetite" feature that improved model performance.
- Seed number and seed differences.

**Feature selection:** Recursive Feature Elimination (RFE) on an expanding-window cross-validation across 2015–2019.

**Model:** XGBoost (men's). A simpler Logistic Regression was used for the women's bracket (see Section 6).

**Ensembling:** Women's: LR ensembled with XGB. Men's: primarily XGB.

**Calibration / Post-processing:**
- Acknowledged that XGB predictions initially clustered around 0.5.
- Explored scaling predictions during post-processing.
- For the women's bracket: aggressive overrides — set Seeds 1, 2, 3 to beat Seeds 14, 15, 16 with very high probability (women's tournament has fewer upsets historically).
- For the men's bracket: typically set Seed 1 to beat Seed 16 with high confidence, but more conservative elsewhere.

**Key insight on Brier vs. Log Loss:** The switch to Brier score encourages slightly more "gambling" (extreme overrides) since Brier penalizes extreme wrong predictions less severely than log loss. However, large overrides on moderate-confidence games still hurt.

### 2021 (John Edwards' writeup, ncaam-march-mania-2021)

- 9 team-strength metrics from historical schedule data: Elo, LRMC, SRS, RPI, TrueSkill, Mixed Model (LME4), Colley Matrix, Win%, Margin of Victory.
- Per-possession box-score adjusted features added for 2002+ data.
- **Feature selection:** Boruta algorithm identified 42 confirmed predictive features from ~500 candidates.
- **Model:** XGBoost (tree depth=2, learning rate=0.012, subsample=44.3%), tuned with Bayesian optimization over 1000 iterations.
- **Result:** Log loss ~0.501 on 2019 validation — about middle of the pack.
- **Calibration tricks:**
  - Probability capping at 0.95/0.05 to limit penalty exposure from overconfident wrong picks.
  - Dual submissions: coinflip games (45–55% win probability) assigned artificially high confidence in one submission.
  - Injury adjustment using Vegas spreads.

### Andrew Landgraf — Earlier 1st Place (pre-2023 era)

- Used simple combination of rating systems (Sagarin, Pomeroy, Moore, Whitlock) averaged together.
- Fitted logistic regression on the composite score.
- **Log loss on combined ratings: ~0.543** — competitive with more complex approaches at the time.
- Key insight: thoughtfully combining a few proven ratings systems is extremely competitive.

---

## 3. Feature Engineering Patterns

These are the features that consistently appear in top solutions:

### Tier 1 — Highest Predictive Value

1. **Massey Ordinal rankings** (men's only): Pre-tournament ratings from systems like Pomeroy (POM), Sagarin (SAG), Moore (MOR), Whitlock (WLK). Top solutions select the 5–10 historically best systems and average or stack them. Use the **latest Massey ranking before the tournament** (DayNum < 133).

2. **Seed number / seed difference**: The most universally used feature. Directly encodes selection committee expectations. Even naive seed-only models are competitive.

3. **Elo ratings**: Computed iteratively from historical game results. FiveThirtyEight-style Elo (incorporating score margin and home court) is commonly used. Elo and Massey ratings have similar solo predictive power (log loss ~0.543 each).

4. **Adjusted Offensive and Defensive Efficiency** (AdjOE, AdjDE): KenPom-style tempo-free stats. AdjEM (efficiency margin = AdjOE - AdjDE) is one of the best single predictors of tournament success. Feature is computed as the **difference** between the two teams' values.

### Tier 2 — Strong Secondary Features

5. **Win percentage** (regular season): Simple but useful.
6. **Point differential / Margin of victory**: Raw or adjusted for pace.
7. **Strength of schedule**: Adjusts win percentage for opponent quality.
8. **Recent form**: Average stats from last 10–15 games highlight tournament momentum.
9. **GLM-derived team quality**: Logistic/linear mixed-effect models fit to game results give probabilistic strength estimates.
10. **Defensive metrics**: Steals, blocks, defensive rebounds are important features in gradient boosting models.
11. **Shooting efficiency**: FG%, 3P%, eFG%, adjusted for opponents.
12. **Turnovers and free throws**: Turnover margin and FTA rate are predictive.
13. **Home/away splits**: Some models use separate statistics for neutral-site performance.

### Tier 3 — Supplementary

14. **Conference indicators / Power conference membership**: Power 6 conferences correlate with tournament depth.
15. **Coach/program prestige metrics**: History of tournament appearances.
16. **Injury adjustments**: Vegas spreads can proxy injury information (not in the Kaggle dataset, must be external).

### Feature Construction Pattern

The dominant pattern is to compute **per-team season averages**, then create **difference features** for each matchup: `feature_diff = team_A_stat - team_B_stat`. This reduces dimensionality and captures relative strength, which is what actually drives outcomes.

**Data augmentation:** "Flip and double" each game — add the mirror record with features negated and label flipped. This prevents the model from learning artifacts based on which team is listed first and doubles training data.

---

## 4. Model Choices

### Gradient Boosting (Dominant)

XGBoost is the most commonly used model in winning and top solutions. Key observations:
- **XGBoost outperforms LightGBM and CatBoost** in several reported experiments for this specific competition.
- Shallow trees (max depth 2–4) with high regularization help prevent overfitting on limited data (~20–40 tournaments of training data).
- LightGBM achieves Brier scores around **0.126** in mid-tier solutions.
- CatBoost is less favored here.

### Logistic Regression (Strong Baseline)

- Fitting LR on a composite of top Massey ratings achieved log loss ~0.543 on men's data.
- Simple, well-calibrated, and competitive.
- Frequently used as the women's bracket model or as an ensemble component.

### Bradley-Terry Model

- Directly models pairwise win probabilities from game results.
- Used in several top solutions (including the 2024 winner's R implementation).
- Naturally outputs well-calibrated probabilities.
- Limitation: Teams with perfect records (e.g., South Carolina women's 2024) can cause numerical instability.

### Neural Networks (Deep Learning)

Academic research (arXiv:2508.02725) comparing LSTM and Transformers on NCAA prediction found:
- **LSTM trained with Brier loss**: Best Brier score (**0.1589**), best calibration.
- **Transformer trained with BCE**: Best AUC (0.8473), better discriminative ranking.
- Recommendation: For Brier-scored competitions, train neural nets directly on Brier loss.
- ECE (Expected Calibration Error): LSTM/Brier ~2.3–3.2%; Transformer/BCE ~4.1–6.2%.
- Neural nets require more training data and careful regularization given the limited annual tournament samples.

### Model Recommendation Priority

1. XGBoost (primary) — well-tested, strong performance, feature importance available
2. LightGBM — fast alternative, competitive
3. Bradley-Terry / Logistic Regression — excellent calibration baselines
4. PyTorch / TensorFlow — use Brier loss directly; require careful regularization

---

## 5. Calibration Techniques

Calibration is critical because Brier score penalizes miscalibrated probabilities. A well-ordered model that predicts 0.9 when it should predict 0.7 will be severely penalized.

### Core Methods

- **Platt Scaling** (sigmoid calibration): Apply logistic regression on model outputs. Works well when calibration curve is sigmoid-shaped. Better with less calibration data.
- **Isotonic Regression**: Non-parametric, monotone mapping. Can cut ECE by ~50% and reduce Brier scores by ~44% in general applications. Preferred when calibration curve is non-sigmoid and when there is more data.
- **Temperature Scaling**: Apply softmax(logit / T) where T is tuned on held-out data. Simple and effective.
- **In-fold calibration**: Apply calibration within each CV fold rather than as a separate post-hoc step. The rank-107 2025 solution reported cubic-spline in-fold calibration as slightly better than post-hoc.

### Competition-Specific Calibration Notes

- **Clip or clip-and-scale probabilities**: Many practitioners clip predictions to [0.05, 0.95] to avoid extreme overconfidence. Brier score penalizes a confident wrong prediction (e.g., predicting 0.99 for a team that loses) heavily.
- **Post-hoc probability shifting**: Multiplying all probabilities by a factor < 1 and redistributing can improve calibration when the model is overconfident.
- **Women's bracket overrides**: Set top seeds (1, 2, 3) to near-certain wins over low seeds (14, 15, 16). Women's tournaments have historically had far fewer upsets, so extreme probabilities are correct here.
- **Men's bracket**: 1 vs. 16 matchups can have aggressive overrides, but 2 vs. 15, 3 vs. 14 should retain some upset probability.
- **Gradient boosting calibration**: XGBoost raw probabilities often cluster around 0.5 and need post-processing. Platt scaling or isotonic regression on a held-out season is recommended.

---

## 6. Ensemble Methods

### What Works

- **Simple averaging** of multiple model outputs is robust and often competitive with more complex stacking.
- **Weighted averaging** (optimize weights on validation seasons) can outperform equal weighting.
- **Stacking with a meta-learner**: Use fold-held-out predictions from each base model as features for a meta-learner (e.g., ridge regression, logistic regression). Reduces overfitting versus arbitrary weighting.
- The 2023 gold solution ensembled XGBoost with logistic regression for the women's bracket.

### Ensemble Architecture for This Project

Given the 5 models (XGBoost, LightGBM, CatBoost, PyTorch, TensorFlow):
1. Train each model with leave-one-season-out CV, generating out-of-fold (OOF) predictions.
2. Optimize ensemble weights by minimizing Brier score on OOF predictions across all validation years (2022–2025).
3. Alternatively, train a simple logistic regression meta-learner on OOF predictions.
4. Expected gain from ensembling: 2–5% Brier score reduction versus best single model.

### Diversity Matters

- Gradient boosting models (XGB, LGBM, CatBoost) are correlated. Include at least one structurally different model (Bradley-Terry, neural network, logistic regression on ratings) for diversity.
- Bradley-Terry or LR models on Massey ratings provide different "view" than feature-engineered gradient boosting.

---

## 7. Men's vs. Women's Differences

| Aspect | Men's | Women's |
|--------|-------|---------|
| Massey Ordinals | Available (dozens of systems) | Not available in competition data |
| Historical depth | Compact: 1985+; Detailed: 2003+ | Compact: 1998+; Detailed: 2010+ |
| Tournament upset frequency | Higher (avg. 8+ upsets/tournament) | Lower (top seeds dominant) |
| South Carolina 2024 problem | N/A | Undefeated teams break Bradley-Terry |
| Feature richness | Rich (KenPom-equivalents in Massey) | Must derive from box scores |
| Prediction difficulty | Harder (more upsets) | Easier (chalk-heavy) |

### Practical Implications

**Men's model:** Use Massey Ordinals as primary features. Average POM, SAG, MOR, WLK at minimum. Supplement with derived efficiency stats and Elo. Model upsets explicitly.

**Women's model:** No Massey Ordinals — must derive quality estimates entirely from box score stats (AdjOE, AdjDE via rolling regression), game results, and seeds. Simpler models may generalize better due to less noise. Consider using a simpler LR or Bradley-Terry baseline alongside gradient boosting. Apply aggressive seed-based overrides in low-seed vs. high-seed matchups.

**Women's external data opportunity:** The competition does not prohibit external data. BartTorvik.com and HerHoopStats provide KenPom-style women's analytics. Incorporating these could significantly improve women's model quality.

**Separate models are necessary.** Using the same model/features for both will underperform because feature availability and tournament dynamics differ substantially.

---

## 8. Cross-Validation Strategy

Temporal validation is critical to avoid data leakage and get honest performance estimates.

### Recommended Approach: Leave-One-Season-Out (LOGO-CV)

- Train on all seasons except one, validate on the held-out season's tournament games.
- Repeat for each season from 2003 (men's) or 2010 (women's) onwards.
- Exclude 2020 (tournament canceled; no games to validate on).
- Use 2022–2025 specifically for final model comparison (these are the Stage 1 evaluation years).
- Advantage: Each season acts as a test of true generalization. Given ~20 tournament seasons, this gives 20-fold-equivalent CV.

### Alternative: Expanding Window CV

- Train on seasons 1 through N-1, test on season N.
- More realistic than LOGO because it respects temporal ordering.
- The 2023 gold solution used expanding-window CV from 2015–2019 for feature selection.

### Key Rule: Never Use Test-Season Data in Training

- Do not use features derived from the tournament you are predicting.
- Pre-tournament Massey ratings (DayNum < 132) are safe.
- Post-tournament stats (results, final rankings) must be excluded from features for that season.

---

## 9. Data Leakage Concerns

### Known Issue: End-of-Season Massey Ratings

Massey Ordinals historically included **post-tournament data** in their end-of-season rankings. Using these end-of-season ratings as features for predicting that same tournament is leakage. **Solution:** Filter Massey Ordinals to only use rankings from before Selection Sunday (DayNum < 132) for each season.

### Season-in-Training leakage

If you build a feature like "did team X win their conference tournament?" using post-regular-season data, and that data includes conference tournament results that happened after the NCAA tournament started in some edge cases, you have leakage. Be precise about cutoff dates.

### Rating System Leakage

Some external rating sites publish end-of-season ratings that include tournament performance. If incorporating external data beyond the provided Massey Ordinals, verify the ratings are published before the tournament start.

### Winner/Loser Bias in Training Data

The competition data labels games with WTeamID/WScore (winner) and LTeamID/LScore (loser), meaning naive feature construction always has the winning team first. This introduces a systematic bias. **Solution:** Flip and double each training game (swap team order, negate difference features, flip label) to ensure balanced representation.

---

## 10. Common Pitfalls

1. **Overfitting to regular season patterns**: Tournament games are different (neutral site, higher stakes, knockout format). Models that overfit regular season noise perform poorly on tournament games. Use only a modest number of features.

2. **Ignoring the women's bracket**: Most public solutions focus on men's. The women's bracket is half of the score. A strong women's model with simple calibrated predictions can provide significant gains.

3. **Poor calibration on low seeds vs. high seeds**: A model that assigns 70% to a 1-seed over a 16-seed (instead of 99%+) will be crushed on easy matchups. Use seed-based priors or manual overrides for extreme mismatches.

4. **Using raw win-loss records without opponent adjustment**: Raw win% rewards teams in weak conferences. Must adjust for strength of schedule or use pace-adjusted efficiency metrics.

5. **Leaking post-tournament Massey ratings**: See Section 9.

6. **Over-tuning to a specific year's tournament**: March Madness has high variance. A model can have excellent cross-validated Brier score but still get hit by a Cinderella run. Don't chase a single year's results.

7. **Gradient boosting raw probabilities without calibration**: XGBoost and LightGBM outputs cluster around 0.5. Apply Platt scaling or isotonic regression on held-out seasons before submitting.

8. **Single submission risk**: The competition allows multiple submissions. Consider submitting a "safe" calibrated ensemble as primary and an aggressive override version as secondary.

9. **Neglecting recent form**: Season-long averages can mask teams that peaked or declined heading into the tournament. Weight recent games more heavily or use separate late-season features.

10. **Ignoring the scoring change (averaged over rounds)**: Since 2024, Brier is averaged over 6 rounds. This means Round of 64 games count less than Elite Eight+ games. It is more important to get Final Four and championship predictions right than first-round predictions.

---

## 11. Brier Score Benchmarks

The Brier score for this competition is MSE of predicted probabilities vs. binary outcomes. Due to the inherent randomness of the tournament, there is a floor even for a perfectly calibrated model.

| Benchmark | Approx. Brier Score |
|-----------|---------------------|
| Predict 0.5 for every game (no information) | ~0.250 |
| Seed-only simple model | ~0.210–0.230 |
| Composite Massey ratings (log-reg on POM+SAG+MOR+WLK) | ~0.185–0.195 (log-loss era equivalent) |
| Mid-tier Kaggle solution (LightGBM, basic features) | ~0.126 |
| Competitive top 10% solution | ~0.115–0.125 |
| Top 1% / medal-worthy solution | ~0.105–0.115 |
| LSTM with Brier loss (academic benchmark) | ~0.159 (single tournament validation) |

Note: Brier scores vary significantly year-to-year due to tournament randomness. A strong model in a chaotic year (many upsets) will still post higher Brier than a weak model in a chalky year. Validate across multiple seasons rather than relying on a single year's score.

The competition scoring (since 2024) averages over 6 rounds rather than all games equally, so late-round accuracy dominates the leaderboard metric. This may make the effective range tighter since there are fewer games at higher weight.

---

## 12. Actionable Takeaways for This Pipeline

### Feature Engineering (Priority Order)

1. Extract **Massey Ordinal** pre-tournament rankings for men's (DayNum < 132). Average POM, SAG, MOR, WLK, and at least 6 others. Use difference between teams' ratings.
2. Build **Elo ratings** from all historical game results, resetting each season with decay, incorporating score margin.
3. Compute **season-average efficiency stats** from detailed results: OffEff, DefEff, AdjEM per possession. Use rolling window for recent form.
4. Include **seed number** and **seed difference** as explicit features.
5. Build **win percentage**, **point differential**, and **SOS** metrics.
6. For women's: build all of the above except Massey. Consider external women's analytics (BarTorvik).

### Data Preparation

- Flip and double training games to eliminate winner/loser bias.
- Use DayNum filters to ensure features are pre-tournament only.
- Exclude 2020 from cross-validation.

### Modeling

- Primary: **XGBoost** with shallow trees (max depth 2–4), regularization tuned via LOGO-CV.
- Secondary: **LightGBM**, **CatBoost**, **Bradley-Terry**, **LR on Massey ratings**.
- Neural nets (PyTorch/TensorFlow): Train with **Brier loss directly**, not BCE.
- Apply **in-fold Platt scaling or isotonic regression** calibration.
- Clip all final predictions to [0.02, 0.98] or [0.05, 0.95] to avoid extreme overconfidence.

### Ensembling

- Generate OOF predictions for all models using LOGO-CV.
- Optimize ensemble weights by minimizing Brier score on OOF predictions.
- Or train a simple ridge regression meta-learner on OOF predictions.

### Separate Men's and Women's Pipelines

- Men's: Full pipeline with Massey Ordinals as primary features.
- Women's: Derived efficiency stats + seeds + game results. Simpler model may generalize better.
- Apply manual override probabilities for extreme seed matchups in women's bracket (1 vs. 16, 2 vs. 15 etc.).

---

## Sources

- [First Place Solution 2025 — Mohammad Odeh | Kaggle](https://www.kaggle.com/competitions/march-machine-learning-mania-2025/writeups/mohammad-odeh-first-place-solution)
- [Final Solution Notebook 2025 — modeh7 | Kaggle](https://www.kaggle.com/code/modeh7/final-solution-ncaa-2025)
- [Top 1% Gold — March Machine Learning Mania 2023 Solution Writeup | maze508 | Medium](https://medium.com/@maze508/top-1-gold-kaggle-march-machine-learning-mania-2023-solution-writeup-2c0273a62a78)
- [March Machine Learning Mania 2025 — Rank 107 Approach | LinkedIn](https://www.linkedin.com/pulse/march-machine-learning-mania-2025-rank-107-approach-g13jf)
- [March Machine Learning Mania 2024 | Kaggle](https://www.kaggle.com/competitions/march-machine-learning-mania-2024)
- [March Machine Learning Mania, 1st Place Winner's Interview: Andrew Landgraf | Kaggle Blog | Medium](https://medium.com/kaggle-blog/march-machine-learning-mania-1st-place-winners-interview-andrew-landgraf-f18214efc659)
- [2021 March Madness Kaggle Solution | John Edwards](https://johnbedwards.io/blog/march_madness_2021/)
- [Forecasting NCAA Basketball Outcomes with Deep Learning: LSTM and Transformer | arXiv:2508.02725](https://arxiv.org/html/2508.02725v1)
- [NCAA Bracket Prediction Using Machine Learning and Combinatorial Fusion Analysis | arXiv:2603.10916](https://arxiv.org/html/2603.10916v1)
- [March Madness Tournament Predictions Model: A Mathematical Modeling Approach | arXiv:2503.21790](https://arxiv.org/html/2503.21790v1)
- [My completely uninformed guide to March Madness — Stats in the Wild (Bradley-Terry, 2024)](https://statsinthewild.com/2024/03/21/my-completely-uninformed-guide-to-march-madness-and-some-thoughts-on-my-kaggle-entry/)
- [Machine Learning Madness: Predicting Every Tournament Matchup | Conor Dewey](https://www.conordewey.com/blog/machine-learning-madness-predicting-every-ncaa-tournament-matchup)
- [March ML Mania 2026 GitHub Baseline | ngusadeep](https://github.com/ngusadeep/March-ML-Mania)
- [2025 March Machine Learning Mania Leaderboard | Kaggle Dataset](https://www.kaggle.com/datasets/brisamarina/2025-march-machine-learning-mania-leaderboard)
- [KenPom Explained: March Madness Betting Guide | betstamp](https://betstamp.com/education/kenpom-march-madness-betting-guide)
- [How Our March Madness Predictions Work | FiveThirtyEight](https://fivethirtyeight.com/features/how-our-march-madness-predictions-work/)
