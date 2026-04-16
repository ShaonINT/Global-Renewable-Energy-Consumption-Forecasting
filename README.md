# Global Renewable Energy Consumption Forecasting
## A Comparative Benchmarking Study of Statistical, Machine Learning, Deep Learning, and Hybrid Models

**Authors:** Shaon Biswas · Paramita Roy  
 

---

## Overview

This repository contains the full reproducible pipeline for a comparative benchmarking study of **13 forecasting model families** applied to global renewable energy consumption data. The study introduces a rigorous **5-window expanding Walk-Forward Cross-Validation (WF-CV)** protocol that eliminates the information asymmetry present in prior benchmarking work (rolling one-step-ahead ARIMA vs. frozen DL evaluation).

**Champion model:** ETS (Holt Linear Trend) — RMSE = 0.5430, Skill Score = +14.8% vs. Naïve  
**Key finding:** Only 3 of 9 non-baseline models beat the Naïve baseline. All standalone deep learning models fail to outperform it on this data regime.

---

## Dataset

| Field | Value |
|-------|-------|
| Source | [World Bank Open Data](https://data.worldbank.org/indicator/EG.FEC.RNEW.ZS) |
| Indicator | EG.FEC.RNEW.ZS — Renewable energy consumption (% of total final energy consumption) |
| Time period | 1990–2020 (31 annual observations) |
| Regions | 11 aggregate World Bank regional groupings |
| Access | Public, free download; CSV or XLS via World Bank API |

> **Biomass note:** High renewable shares in developing regions reflect traditional solid biomass (energy poverty), not modern renewable deployment — a critical distinction for SDG-7 policy interpretation.

---

## Evaluation Protocol

| Parameter | Value |
|-----------|-------|
| Protocol | Expanding Walk-Forward Cross-Validation (WF-CV) |
| Windows | 5 (train sizes: 20, 22, 24, 26, 28 observations) |
| Forecast horizon | H = 3 steps ahead (multi-step iterative) |
| Test observations | 15 total (3 per window) |
| Structural break | 2014 (Chow F = 31.97, p < 0.001; Bai-Perron confirmed) |
| Break alignment | W3–W5 straddle the break, measuring regime-transition robustness |

```
W1: train 1990–2009 (20 obs) → forecast 2010–2012
W2: train 1990–2011 (22 obs) → forecast 2012–2014
W3: train 1990–2013 (24 obs) → forecast 2014–2016  ← straddles structural break
W4: train 1990–2015 (26 obs) → forecast 2016–2018
W5: train 1990–2017 (28 obs) → forecast 2018–2020  ← steepest post-break segment
```

**Consistency with M4 Competition:** The protocol mirrors the M4 Competition design (Makridakis et al., 2018) — a single multi-step forecast per window, no rolling refitting, identical rules for all model classes.

---

## Models Benchmarked

| Category | Models |
|----------|--------|
| Baselines | Naïve, RW-Drift, Linear Trend |
| Statistical | ETS (Holt Linear), Damped ETS, Theta, ARIMA (auto-order, AIC) |
| Machine Learning | XGBoost (lag features, lags 1–5 + rolling stats) |
| Deep Learning | GRU, LSTM, N-BEATS |
| Additive | Prophet (custom changepoints 2001/2007) |
| Hybrid | ETS-GRU (equal-weight ensemble) |

---

## Key Findings

1. **ETS is the champion model** (SS = +14.8%, RMSE = 0.543). This is a data-regime-specific finding explained by four quantified conditions: severe underdetermination (~150 parameters per training observation for DL), near-linear piecewise trend (Chow R² = 0.844 post-break), no seasonality on annual data, and iterative multi-step error compounding. Consistent with M4 Competition results.

2. **Only 3/9 non-baseline models beat Naïve:** ETS (+14.8%), Prophet (+4.8%), and ETS-GRU hybrid (+7.0%). All standalone DL models fail to beat Naïve.

3. **ARIMA ranks 7th (SS = −0.217)** without rolling refit — exposing the information asymmetry in prior work that uses rolling one-step-ahead evaluation to inflate ARIMA accuracy.

4. **N-BEATS has the worst average RMSE (0.867)** with catastrophic post-break degradation (W5/W1 ratio = 3.7×). XGBoost has the highest W5/W1 ratio (10.6×), indicating lag-feature models overfit the stable pre-break phase.

5. **DM tests:** No pair reaches p < 0.05 at n = 15 test observations. The ARIMA–GRU gap approaches significance (p = 0.066). The n = 15 test sample is a fundamental constraint of annual frequency data.

6. **Multi-region (§25):** ETS achieves the best average rank (2.73) across 11 regions. Naïve has the overall best rank (2.00), reinforcing that simple baselines are hard to beat on diverse aggregate regional series.

7. **SDG-7 Gap Analysis (§26):** High income and North America forecast < 20% renewable by 2030 — well below the IEA NZE 30% milestone.

---

## Novel Contributions

1. **Walk-Forward CV protocol** — first application to renewable energy benchmarking; eliminates the rolling-ARIMA information asymmetry.
2. **Structural Break analysis** (Chow F = 31.97, 2014) — Paris Agreement regime shift confirmed; WF windows aligned to measure regime-transition robustness.
3. **Data Regime Analysis (§28)** — first explicit quantification of the four conditions under which statistical models outperform DL on this data, bounding the finding and motivating DL-appropriate extensions.
4. **Transition Velocity Index (TVI)** — a novel scale-invariant regional transition metric.
5. **Beta-Convergence** (β = −0.014, p = 0.011) — convergence in renewable shares strengthens post-break.
6. **SDG-7 Gap Analysis** with traditional biomass disambiguation.

---

## Notebook Structure

| Section | Content |
|---------|---------|
| §0.1–0.2 | Reproducibility settings + experiment metadata |
| §1–4 | Data loading, EDA, ADF stationarity, WF-CV protocol |
| §5 | PyTorch helpers (dataset, training loop, WF prediction wrapper) |
| §6–14 | Nine core models (Naïve, ETS, Baselines, ARIMA, XGBoost, GRU, LSTM, N-BEATS, Prophet, ETS-GRU) |
| §15b–15d | Multi-seed robustness · Direct multi-output · Ranking stability |
| §15 | Benchmark results (Table A) + DL stability (Table B) |
| §16 | Visualisations |
| §17–17c | Diebold-Mariano tests · Bootstrap · Model Confidence Set |
| §18–18c | Error decomposition · Residual diagnostics · Prediction intervals |
| §19 | Nested CV |
| §20–20b | Complexity table (Table C) · Publication figures |
| §21 | 20-year ETS forecast across all regions |
| §22 | Structural breaks (Chow · Bai-Perron · CUSUM · Ruptures) |
| §23–26 | TVI · Beta-convergence · Multi-region · SDG-7 gap |
| §28 | Data regime analysis |
| §29 | Final summary |
| §30 | Auto-generated methodology text |
| §31 | CSV exports (4 publication-ready tables) |

---

## Reproducibility

All randomness is controlled through a single `set_global_seed(42)` call at the top of the notebook:

```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
os.environ['PYTHONHASHSEED'] = '42'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Re-running all cells top-to-bottom with `SEED = 42` reproduces every table, figure, and test statistic exactly.

**Multi-seed experiments (§15b):** 10 seeds = `[42, 7, 13, 21, 37, 55, 73, 89, 101, 117]`

---

## Hyperparameters

| Parameter | Value | Selection method |
|-----------|-------|-----------------|
| SEQ_LENGTH | 3 | Nested WF-CV inner loop |
| HIDDEN | 64 | Nested WF-CV inner loop |
| EPOCHS | 300 | Fixed |
| LR | 0.001 | Fixed |
| BATCH_SIZE | 8 | Fixed |
| XGBoost | n_estimators=100, max_depth=3, lr=0.05 | Nested CV |
| ARIMA | Auto-order (AIC), non-seasonal | Auto |
| Prophet | changepoints=[2001, 2007], prior_scale=0.3 | Manual |

---

## Statistical Tests

| Test | Purpose | Implementation |
|------|---------|----------------|
| ADF | Unit root / stationarity | `statsmodels.tsa.stattools.adfuller` |
| Diebold-Mariano | Pairwise forecast accuracy | Custom (Harvey et al. 1997 small-sample correction) |
| Paired Bootstrap | RMSE-diff CIs vs ETS and Naïve | `numpy.random.default_rng`, n_boot = 1000 |
| Model Confidence Set | Formal model selection | Bootstrap MCS (Hansen, Lunde & Nason 2011), α = 0.10 |
| Chow F-test | Single structural break | `scipy.stats` |
| Bai-Perron | Multiple breakpoint candidates | Exhaustive Chow grid |
| CUSUM | Gradual parameter instability | `statsmodels.stats.diagnostic.breaks_cusumolsresid` |
| Ljung-Box Q(3) | Residual autocorrelation | `statsmodels.stats.diagnostic.acorr_ljungbox` |
| Shapiro-Wilk | Residual normality | `scipy.stats.shapiro` |

---

## Exported Files

Running §31 generates four publication-ready CSVs:

| File | Contents |
|------|---------|
| `benchmark_results.csv` | Full 13-model WF-CV results (RMSE, MAE, MAPE, Skill Score, Rank) |
| `deep_learning_seed_stability.csv` | GRU/LSTM/N-BEATS/XGBoost across 10 seeds (mean, std, min, max, CoV%) |
| `bootstrap_results.csv` | Paired bootstrap RMSE-diff vs ETS and vs Naïve (CI, p-value, significance) |
| `rank_stability.csv` | Per-window ranks and stability coefficients for all 13 models |

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
torch
scikit-learn
statsmodels
scipy
xgboost
prophet
```

Install via:

```bash
pip install numpy pandas matplotlib seaborn torch scikit-learn statsmodels scipy xgboost prophet
```

**Device support:** The notebook auto-detects MPS (Apple Silicon), CUDA, or CPU.

---

## Data Access

Download the dataset from the World Bank:

1. Visit: https://data.worldbank.org/indicator/EG.FEC.RNEW.ZS
2. Download as Excel (`.xls`)
3. Update `FILEPATH` in §1 to point to the downloaded file

---

*Generated from `forecasting_benchmark_v10.ipynb` — set SEED = 42 and run all cells top-to-bottom to reproduce all results.*
