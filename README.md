# March Machine Learning Mania 2026

Predict the probability of every possible team matchup in the 2026 NCAA Division I Men's and Women's basketball tournaments. Scored by Brier score (lower is better). Deadline: March 19, 2026.

## Pipeline

### 00 Data Collection

Raw Kaggle competition data stored in S3 (`s3://march-machine-learning-mania-2026/00_data_collection/`). Men's files prefixed with `M`, women's with `W`. Includes game results (compact: 1985+, detailed box scores: 2003+), tournament seeds, Massey Ordinal rankings (~197 systems), team/conference/coach metadata, and sample submissions.

### 01 Data Joining

**Men's** (`01_data_joining/mens_notebook.ipynb`): Joins all raw men's data sources into clean, unified datasets. Key steps:
- Merges compact (1985–2026) and detailed (2003–2026) regular season results into one game-level dataset
- Same merge for tournament results
- Builds per-team per-season aggregated statistics: win/loss, scoring, shooting percentages, possessions estimate, offensive/defensive efficiency
- Cleans tournament seeds — extracts numeric seed, region, and play-in indicator
- Filters Massey Ordinals to pre-tournament only (DayNum < 132) to avoid leakage, pivots to wide format with one column per ranking system, computes average rank and top-systems average (POM, SAG, MOR, WLK)
- Joins team metadata: names, conference affiliations (with power conference flag), and coaches

**Outputs** (parquet, saved to S3 and `01_data_joining/output/`):
- `regular_season_games` — all regular season games with box scores where available
- `tourney_games` — all tournament games with box scores where available
- `team_season_stats` — per-team per-season aggregates (wins, losses, WinPct, shooting %, OffEff, DefEff, NetEff, etc.)
- `tourney_seeds` — cleaned seeds with numeric values
- `massey_ordinals_pre_tourney` — wide-format pre-tournament rankings from ~197 systems
- `team_metadata` — team names, conferences, power conference flag, coaches per season

**Women's** (`01_data_joining/womens_notebook.ipynb`): Same structure as men's, adapted for women's data differences:
- Compact results start 1998 (vs. 1985 for men's), detailed box scores start 2010 (vs. 2003)
- No Massey Ordinals available — team quality must be derived entirely from game results and box scores in later stages
- No coaches data available
- ~1.5% of 2010–2012 games may be missing detailed results (tracked in notebook output)
- Outputs same datasets as men's minus `massey_ordinals_pre_tourney` and minus coach info in `team_metadata`
