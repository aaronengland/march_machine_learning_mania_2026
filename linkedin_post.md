# Training Neural Nets on the Right Loss Function Beat Gradient Boosting for March Madness Prediction

Every March, Kaggle hosts the March Machine Learning Mania competition, challenging data scientists to predict the outcome of every possible NCAA tournament matchup -- both men's and women's -- scored by Brier score (the mean squared error between your predicted probabilities and actual 0/1 outcomes). This year I built a full end-to-end pipeline for the 2026 edition, and the results surprised me in a few ways worth sharing.

## The Setup

The task sounds deceptively simple: for every possible pair of tournament teams, output a probability that one beats the other. In practice, it means building models on decades of game-level data -- box scores, rankings, seedings, conference performance -- and producing well-calibrated probabilities that don't get punished by overconfident wrong picks.

I built separate 3-model ensembles for men's and women's tournaments:

- **Men's**: XGBoost + PyTorch + Logistic Regression (62 engineered features)
- **Women's**: CatBoost + PyTorch + Logistic Regression (57 features)

The entire pipeline ran on AWS SageMaker with data stored in S3, spanning eight stages from raw data collection through final submission. Men's and women's data were processed in separate notebooks at every stage, since the datasets differ substantially in history length, available rankings data, and statistical patterns.

## The Results

- **Men's ensemble Brier score**: 0.172 (all historical tournaments), 0.176 (Stage 1 validation on 2022-2025)
- **Women's ensemble Brier score**: 0.131 (all historical), 0.121 (Stage 1 validation)

For context, a naive model predicting 50/50 for every game scores 0.250, and a simple seed-based model lands around 0.210-0.230. Competitive Kaggle submissions in this competition typically fall in the 0.115-0.126 range.

## What Actually Worked (and What Surprised Me)

**1. Training directly on Brier loss outperformed gradient boosting.**

This was the biggest surprise. My PyTorch "BrierNet" -- a straightforward feedforward network trained with Brier score as its loss function -- was the best single model for both men's and women's predictions. XGBoost and CatBoost, despite extensive Optuna hyperparameter tuning, couldn't match a neural network that was simply optimizing the exact metric the competition uses. The lesson is almost too obvious in retrospect: if you can train directly on the evaluation metric, do it.

**2. Women's tournament basketball is roughly 24% more predictable than men's.**

The women's ensemble achieved a Brier score of 0.131 compared to 0.172 for men's. This isn't a modeling artifact -- it reflects a genuine structural difference. The women's tournament has historically had fewer upsets, with top seeds advancing more reliably. Higher seeds in the women's bracket convert at noticeably higher rates through the later rounds.

**3. Isotonic calibration improved every single model.**

Post-hoc isotonic calibration reduced Brier scores by 0.003 to 0.005 across the board. That may sound small, but in a competition where the difference between placements can be thousandths of a point, it's significant. Every model benefited, no exceptions.

**4. Feature engineering mattered more than model complexity.**

Expanding from 38 to 62 features for men's made a measurable difference. The most impactful additions were margin-based features (average scoring margin, close-game win percentage), advanced shooting metrics (effective FG%, true shooting %), and strength-of-opponent features (win percentage against top-5, top-10, and top-25 teams). For men's, Massey Ordinal rankings -- aggregated weekly rankings from systems like Pomeroy and Sagarin -- provided strong signal. Since these don't exist for women's, I built synthetic rankings using Ridge regression, which partially compensated.

**5. Data augmentation eliminated a subtle but real bias.**

A technique called "flip-and-double" augmentation -- where you duplicate every training row with the teams swapped and the target inverted -- was essential for ensuring the model didn't learn an ordering bias from the ID format (lower-ID team is always "Team A"). Without it, predictions were systematically skewed.

## Key Takeaways

- **Match your loss function to the evaluation metric.** This sounds like textbook advice, but it genuinely beat more sophisticated models that were optimizing surrogate losses.
- **Calibration is not optional.** If your competition metric cares about probability quality (Brier, log loss), invest in calibration.
- **Research past solutions before building.** Every major design decision -- from flip-and-double augmentation to ensemble weighting with SLSQP optimization -- was informed by studying past winning Kaggle solutions. I ran a dedicated research phase before writing a single model, and it saved significant time.
- **Separate pipelines for separate populations.** Men's and women's college basketball have different data availability, different competitive dynamics, and different levels of predictability. Treating them as one problem would have left performance on the table.

The competition deadline is March 19, 2026. Brackets tip off shortly after. We'll see how the models hold up against the chaos of the real tournament.

---

#MarchMadness #MachineLearning #Kaggle #DataScience #NCAA #PyTorch #XGBoost #DeepLearning #SportsAnalytics #PredictiveModeling #BrierScore #MLEngineering
