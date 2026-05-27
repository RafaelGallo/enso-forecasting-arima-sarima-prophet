# 🌊 ENSO Early Phase Prediction
### Time Series Forecasting with ARIMA · SARIMA · Prophet

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14-orange)
![Prophet](https://img.shields.io/badge/Prophet-1.1-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?logo=kaggle)

## 📋 Overview

This project builds and evaluates classical time series models for
**early prediction of ENSO phase transitions** at lead times of
**1, 3, and 6 months**, using monthly climate indices from 1980 to 2026.

The El Niño–Southern Oscillation (ENSO) is the dominant driver of
interannual climate variability on Earth. Accurate early forecasting
of its three phases — **El Niño**, **La Niña**, and **Neutral** —
is critical for agricultural planning, water resource management,
humanitarian preparedness, and climate risk assessment across
Latin America, Southeast Asia, Africa, and the Pacific Basin.

## 🎯 Business Problem

> Can classical time series models accurately forecast the Niño 3.4
> SST anomaly and ENSO phase transitions 1, 3, and 6 months ahead,
> using only observed historical climate indices, without relying on
> complex numerical weather prediction systems?

## 📁 Project Structure

```
enso-early-phase-prediction/
│
├── data/
│   ├── enso_train.csv              # Train set (1980–2018, 468 months)
│   ├── enso_test.csv               # Test set  (2019–2026, 91 months)
│   └── data_dictionary.csv         # Column descriptions and dtypes
│
├── models/
│   ├── arima_200_fitted.pkl        # ARIMA(2,0,0) fitted model
│   ├── sarima_200x200_12_fitted.pkl # SARIMA fitted model
│   ├── prophet_base_fitted.pkl     # Prophet base model
│   ├── prophet_lags_fitted.pkl     # Prophet + lagged regressors
│   ├── training_series.pkl         # Full training series 1980–2026
│   ├── walk_forward_forecasts.pkl  # Walk-forward forecast arrays
│   ├── prophet_walk_forward_forecasts.pkl
│   ├── models_metadata.json        # ARIMA/SARIMA orders and metrics
│   ├── prophet_metadata.json       # Prophet hyperparameters and metrics
│   ├── final_metrics.csv           # ARIMA vs SARIMA metrics table
│   └── prophet_metrics.csv         # Prophet metrics table
│
├── outputs/
│   ├── eda_*.png                   # EDA plots (6 charts)
│   ├── anomaly_*.png               # Anomaly detection plots
│   ├── outlier_*.png               # Outlier analysis plots
│   ├── stationarity_*.png          # ADF/KPSS diagnostic plots
│   ├── autoarima_*.png             # Auto ARIMA grid search plots
│   ├── arima_*.png                 # ARIMA model plots
│   ├── sarima_*.png                # SARIMA model plots
│   ├── prophet_*.png               # Prophet model plots
│   ├── wf_forecast_*.png           # Walk-forward forecast plots
│   ├── forecast_2025_*.png         # 2025 retrospective forecast
│   └── final_*.png                 # Final comparison plots
│
└── notebook.ipynb                  # Full analysis notebook
```

## 📊 Dataset

| Split | Period | Rows | Description |
|---|---|---|---|
| **Train** | 1980–2018 | 468 months | Full ENSO cycle history |
| **Test** | 2019–2026 | 91 months | Includes 2020–2023 triple-dip La Niña |

**Source:** [ENSO Early Phase Prediction — Kaggle](https://www.kaggle.com/datasets/ferariz/enso-forecast)

### Key Features

| Feature | Description |
|---|---|
| `sst_anom_nino34` | Primary ENSO index — Niño 3.4 SST anomaly (°C) |
| `southern_oscillation_index` | Normalized Tahiti − Darwin pressure difference |
| `zonal_wind_850_anom` | 850 hPa equatorial zonal wind anomaly |
| `wwv_anom_std` | Warm Water Volume — subsurface heat content |
| `sst_anom_nino34_lag1/3/6` | Lagged Niño 3.4 at 1, 3, 6 months |
| `crosses_spring_tL` | Does forecast window cross boreal spring (MAM)? |
| `init_in_growth_phase` | Is initialization in JJA–SON high-skill window? |

### Targets

| Variable | Type | Description |
|---|---|---|
| `enso_t1/t3/t6` | Classification | ENSO phase 1, 3, 6 months ahead |
| `nino34_t1/t3/t6` | Regression | Niño 3.4 anomaly (°C) ahead |

## 🤖 Models

### ARIMA(2, 0, 0)
Found by `auto_arima` minimizing AIC. The AR(2) process captures
strong month-to-month autocorrelation (ar.L1 = +1.475) and
mean-reverting oscillation (ar.L2 = −0.544) characteristic of ENSO.

### SARIMA(2, 0, 0)x(2, 0, 0, 12)
Extends ARIMA with seasonal AR terms at lags 12 and 24 months.
The ar.S.L24 term (p = 0.046) captures the quasi-biennial ENSO
oscillation of approximately 28–30 months.

### Prophet + Lagged Regressors
Base Prophet produces a flat forecast near zero because ENSO has
no monotonic trend. Adding lagged regressors (lags 1, 2, 3, 6, 12
months) provides the autoregressive information needed to track
ENSO dynamics.

## 📈 Results — Walk-Forward Test Set (2019–2026)

| Model | Horizon | RMSE (°C) | MAE (°C) | F1 Macro | Skill Score |
|---|---|---|---|---|---|
| ARIMA(2,0,0) | t+1 | 0.3365 | 0.2827 | 0.7539 | 0.7375 |
| SARIMA(2,0,0)x(2,0,0,12) | t+1 | **0.3355** | 0.2836 | **0.7622** | **0.7383** |
| ARIMA(2,0,0) | t+3 | 0.5176 | 0.4245 | 0.5772 | 0.6007 |
| SARIMA(2,0,0)x(2,0,0,12) | t+3 | **0.5111** | **0.4185** | **0.5974** | **0.6057** |
| ARIMA(2,0,0) | t+6 | 0.7348 | 0.5956 | **0.3996** | 0.4423 |
| SARIMA(2,0,0)x(2,0,0,12) | t+6 | **0.7226** | **0.5905** | 0.3127 | **0.4516** |

### Best Model per Horizon

| Horizon | Best Model | Reason |
|---|---|---|
| **t+1** | SARIMA | Better F1 (+0.008) and RMSE |
| **t+3** | SARIMA | Better on all metrics |
| **t+6** | ARIMA (F1) / SARIMA (RMSE) | Trade-off between metrics |

### Benchmark Comparison (LightGBM reference)

| Horizon | Best Classical F1 | LightGBM F1 | Gap |
|---|---|---|---|
| t+1 | 0.7622 | 0.955 | −0.193 |
| t+3 | 0.5974 | 0.798 | −0.201 |
| t+6 | 0.3996 | 0.617 | −0.217 |

## 🔑 Key Findings

**1. Walk-Forward is essential**
Direct multi-step forecasting causes ARIMA/SARIMA to converge to
zero within months. Walk-forward refitting at each step preserves
the signal throughout the entire test period.

**2. Spring Predictability Barrier confirmed**
F1 drops from 0.76 (t+1) to 0.40 (t+6) — a 47% degradation —
consistent with the known collapse in ENSO forecast skill through
boreal spring (MAM).

**3. SARIMA seasonal component adds value**
The ar.S.L24 term captures the quasi-biennial ENSO cycle and
improves performance at t+1 and t+3. At t+6 it adds noise.

**4. Both models beat persistence by 44–74%**
Even simple AR(2) provides substantial forecast skill across
all three horizons.

**5. Prophet requires lagged regressors**
Base Prophet produces near-zero forecasts. Adding lags 1–12 months
as regressors enables competitive performance.

## 🗓️ 2025 Retrospective Forecast

Trained on 1980–December 2024, evaluated against known 2025 data:

| Model | Phase Accuracy | Over-estimated La Niña |
|---|---|---|
| ARIMA(2,0,0) | 50% (6/12 months) | +2 months |
| SARIMA(2,0,0)x(2,0,0,12) | 33% (4/12 months) | +4 months |

The actual 2025 series was predominantly **Neutral** (8 months)
with La Niña only in January, October, November, and December.
Both models over-predicted La Niña persistence because the AR(2)
propagated the December 2024 negative signal too strongly.

## 🔮 12-Month Forecast — 2027

Both models forecast **Neutral phase** for all 12 months of 2027,
with central forecast near −0.05°C. The 95% CI widens to ±1.73°C
by December 2027.

| Month | P(El Niño) | P(Neutral) | P(La Niña) |
|---|---|---|---|
| Jan 2027 | 0.0% | 100.0% | 0.0% |
| Jun 2027 | 31.4% | 34.0% | 34.6% |
| Dec 2027 | 34.0% | 29.6% | 36.4% |

## ⚙️ Installation

```bash
git clone https://github.com/RafaelGallo/enso-early-phase-prediction.git
cd enso-early-phase-prediction
pip install -r requirements.txt
```

### Requirements

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
statsmodels>=0.14
pmdarima>=2.0
prophet>=1.1
scikit-learn>=1.3
tqdm>=4.65
pyarrow>=12.0
```

## 🚀 Quick Start

```python
import pickle
import pandas as pd

# load SARIMA model / carrega modelo SARIMA
with open("models/sarima_200x200_12_fitted.pkl", "rb") as f:
    sarima = pickle.load(f)

# forecast next 3 months / prevê próximos 3 meses
fc = sarima.forecast(steps=3)
print(fc)

# load Prophet + lags / carrega Prophet + lags
with open("models/prophet_lags_fitted.pkl", "rb") as f:
    prophet = pickle.load(f)
```

## 📋 Methodology

```
1. Data Loading & Feature Engineering
   └── Forward-shifted regression targets (nino34_t1/t3/t6)

2. Exploratory Data Analysis
   └── Series visualization, phase distribution, SOI scatter,
       moving averages, anomaly detection, outlier analysis

3. Stationarity Testing
   └── ADF + KPSS on 6 series → d=0 confirmed for all

4. Auto ARIMA / Auto SARIMA
   └── pmdarima grid search minimizing AIC
   └── Best: ARIMA(2,0,0) | SARIMA(2,0,0)x(2,0,0,12)

5. Walk-Forward Forecasting
   └── Expanding window, refit at each observation
   └── Horizons: t+1, t+3, t+6

6. Prophet + Lagged Regressors
   └── Hyperparameter grid search by validation RMSE
   └── Lags: 1, 2, 3, 6, 12 months as regressors

7. Evaluation
   └── RMSE, MAE, Bias, F1 Macro, Skill Score, Phase Accuracy

8. Retrospective 2025 Forecast
   └── Train cutoff: Dec 2024 → evaluate vs actual 2025

9. Future Forecast 2027
   └── Rolling walk-forward → CI-derived phase probabilities
```

## 🔮 Next Steps

- **LightGBM / XGBoost** with the full 68-feature dataset
- **LSTM / Transformer** for long-range temporal dependencies
- **Ensemble** combining SARIMA (linear) + gradient boosting residual corrector
- **Probabilistic forecasting** with quantile regression or conformal prediction
- **Real-time pipeline** with automated monthly data ingestion from NOAA CPC

## 📚 Data Sources

| Variable | Source |
|---|---|
| Niño 3.4, 1+2, 3, 4 anomalies | NOAA CPC (ERSSTv5, base 1991–2020) |
| Southern Oscillation Index | NOAA CPC |
| 850 hPa equatorial zonal wind | NOAA CPC |
| Warm Water Volume (WWV) | NOAA PMEL |

## 👤 Author

**Rafael Gallo**
- Kaggle: [Rafael Gallo](https://www.kaggle.com/rafaelgallo)
- GitHub: [@RafaelGallo](https://github.com/RafaelGallo)
- Instagram: [@Rafael Gallo](https://instagram.com/Rafael Gallo)

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgements

Dataset by [ferariz](https://www.kaggle.com/ferariz) —
ENSO Early Phase Prediction (ferariz/enso-forecast).
All climate data sourced from NOAA CPC and NOAA PMEL,
freely available without API key.

*Train: 1980–2018 · Test: 2019–2026 · Method: Walk-Forward*
*Primary metric: Macro F1 · Models: ARIMA · SARIMA · Prophet*
