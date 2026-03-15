# AlphaForge-AI-Quant-Research-Platform
# 🤖 AI Quantitative Research System

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Live Results at a Glance](#-live-results-at-a-glance)
3. [System Architecture](#-system-architecture)
4. [Repository Structure](#-repository-structure)
5. [Quickstart](#-quickstart)
6. [Pipeline Stages](#-pipeline-stages)
7. [Machine Learning Models](#-machine-learning-models)
8. [Backtesting & Risk Results](#-backtesting--risk-results)
9. [Technologies Used](#-technologies-used)
10. [Configuration](#-configuration)
11. [Running Tests](#-running-tests)
12. [Deployment (FastAPI)](#-deployment-fastapi)
13. [Data & Model Artefacts](#-data--model-artefacts)
14. [Known Limitations](#-known-limitations)
15. [Future Work](#-future-work)
16. [References](#-references)
17. [License](#-license)

---

## 📖 Project Overview

This project builds a **fully automated, end-to-end quantitative research platform** that predicts the next-day directional movement (up or down) of US equity prices using machine learning. The system mirrors the workflows of professional quantitative analysts — from raw data ingestion through model deployment — inside a single, modular Google Colab notebook.

**Target stocks:** Apple Inc. (`AAPL`) and Microsoft Corp. (`MSFT`)  
**Data range:** January 2010 – March 2026 (4,073 daily sessions per ticker)  
**Benchmark:** S&P 500 ETF (`SPY`)

### What makes this different from a typical ML stock project?

| Typical Approach | This Project |
|---|---|
| Single model, single train/test split | 5-fold Walk-Forward validation (no lookahead bias) |
| Raw OHLCV features only | 113 engineered features across 8 categories |
| No NLP signal | VADER + FinBERT sentiment from Finnhub API |
| No deployment | FastAPI REST endpoint for live inference |
| No statistical testing | Deflated Sharpe, T-test, Mann-Whitney U |
| No portfolio layer | Mean-Variance, Risk Parity, HRP optimization |

---

## 📊 Live Results at a Glance

| Metric | AAPL Strategy | MSFT Strategy |
|---|---|---|
| Walk-Forward Test Accuracy | 49.5% | 55.2% |
| Walk-Forward AUC-ROC | 0.52 | 0.59 |
| Backtest Total Return | **+94.92%** | **+237.08%** |
| Sharpe Ratio | 1.86 | 2.38 |
| Max Drawdown | 2.18% | 1.92% |
| VaR (95%) | −0.28% | −0.15% |
| CVaR (95%) | −1.07% | −0.28% |
| MC Profit Probability | 100% | 100% |
| Portfolio Sharpe (50/50) | **0.79** | — |

> ⚠️ **Disclaimer:** Past performance does not guarantee future results. These results are from backtesting and Monte Carlo simulation only. This project is for academic research purposes.

---

## 🏗 System Architecture

The platform is built as a collection of **41 independent agent classes**, each inheriting from a common `BaseAgent` abstract class. A central `PipelineOrchestrator` coordinates their sequential execution.

```
┌─────────────────────────────────────────────────────────────────┐
│                     PipelineOrchestrator                        │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │  Market  │  │  News /  │  │ Feature  │  │  Sentiment   │   │
│  │   Data   │→ │Finnhub   │→ │  Eng.    │→ │   Agent      │   │
│  │  Agent   │  │  Agent   │  │  Agent   │  │(VADER/BERT)  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
│        ↓                                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ Feature  │  │   ML     │  │ Signal   │  │  Backtesting │   │
│  │Selection │→ │ Training │→ │   Gen.   │→ │    Agent     │   │
│  │  Agent   │  │  Agent   │  │  Agent   │  │ (Backtrader) │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
│        ↓                                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │   Risk   │  │  Monte   │  │Portfolio │  │   Report     │   │
│  │  Agent   │  │  Carlo   │  │  Optim.  │  │  Generator   │   │
│  │(VaR/CVaR)│  │  Agent   │  │  Agent   │  │  (HTML/PDF)  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
│                                                  ↓              │
│                                          ┌──────────────┐      │
│                                          │   FastAPI    │      │
│                                          │  Deployment  │      │
│                                          └──────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
ai-quant-research-system/
│
├── IDEAS_FINAL_15_03.ipynb          # ← Main notebook (run this)
│
├── README.md                        # This file
│
├── docs/
│   ├── IDEAS_Internship_Report.docx # Full project report
│   ├── IDEAS_Project_Presentation.pptx
│   └── IDEAS_Data_Model_Summary.docx
│
├── config/
│   └── system_config.json           # Auto-generated on first run
│
├── data/                            # Auto-populated by pipeline
│   ├── raw/
│   │   ├── aapl_ohlcv.csv
│   │   ├── msft_ohlcv.csv
│   │   ├── spy_benchmark.csv
│   │   ├── news_aapl.json
│   │   └── news_msft.json
│   └── processed/
│       ├── features_aapl_selected.csv.gz
│       └── features_msft_selected.csv.gz
│
├── models/                          # Auto-saved trained weights
│   ├── prediction_agent_AAPL_*.pkl
│   ├── prediction_agent_MSFT_*.pkl
│   └── lstm_model_aapl.h5
│
└── results/
    ├── backtest_aapl_equity.csv
    ├── backtest_msft_equity.csv
    ├── risk_metrics_summary.json
    └── reports/
        └── research_report_*.html
```

---

## ⚡ Quickstart

### Option 1: Run on Google Colab (Recommended)

1. Click **Open in Colab** → [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/YOUR_LINK_HERE)

2. Run cells in order:

```
Cell 1  → Install all 39 dependencies
Cell 2  → Mount Google Drive & configure paths
Cell 3  → Set API keys (Finnhub, FRED, Alpha Vantage)
Cells 4–38 → Define all agent classes
Cell 40 → ▶ Run full pipeline (AAPL + MSFT)
```

> 💡 **Tip:** On first run, cell 1 takes ~3 minutes. On subsequent runs, packages are already installed and data is cached from Drive — the full pipeline runs in ~70 seconds.

### Option 2: Local Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ai-quant-research-system.git
cd ai-quant-research-system

# Create and activate a virtual environment (Python 3.10+)
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook IDEAS_FINAL_15_03.ipynb
```

### API Keys Required

Get free keys from these services and enter them in **Cell 3**:

| Service | Purpose | Free Tier Link |
|---|---|---|
| [Finnhub](https://finnhub.io/) | Financial news articles | Free registration |
| [FRED](https://fred.stlouisfed.org/docs/api/api_key.html) | Macroeconomic data | Free registration |
| [Alpha Vantage](https://www.alphavantage.co/support/#api-key) | Backup OHLCV data | Free registration |

---

## 🔄 Pipeline Stages

The full pipeline (Cell 40) executes 10 stages sequentially for each ticker:

| Stage | Agent | Description |
|---|---|---|
| **1** | `MarketDataAgent` | Fetch OHLCV from yfinance + cache to Drive |
| **2** | `FeatureEngineeringAgent` | Generate 113 technical indicators via TA-Lib |
| **3** | `FeatureSelectionAgent` | Correlation filter → MI + RF importance → 30 features |
| **4** | `PredictionAgent` | 5-fold walk-forward training (LR + RF + XGBoost) |
| **5** | `SignalGenerationAgent` | Threshold optimization → BUY / HOLD / SELL signals |
| **6** | `BacktestingAgent` | Vectorized portfolio simulation |
| **7** | `RiskManagementAgent` | VaR, CVaR, beta, stress testing |
| **8** | `MonteCarloAgent` | 5,000 bootstrap simulations over 252-day horizon |
| **9** | `StrategyEvaluationAgent` | Deflated Sharpe, T-test, Mann-Whitney U, robustness |
| **10** | `ReportGenerationAgent` | HTML research report with embedded charts |

After both tickers are processed, a `PortfolioOptimizationAgent` runs Max Sharpe, Min Variance, and Risk Parity across the combined return streams.

---

## 🧠 Machine Learning Models

### Feature Engineering (113 → 30 features)

| Category | Features | Count |
|---|---|---|
| Price-Derived | Returns, log-returns, gap, OHLC ratios | 6 |
| Moving Averages | SMA/EMA (5/10/20/50/200d) + price deviation | 20 |
| Momentum | RSI, MACD, Stochastic, Williams %R, CCI, ROC | 18 |
| Volatility | Bollinger Bands, ATR, Historical Vol (10/20/30d) | 10 |
| Volume | OBV, CMF, MFI, Force Index, Volume SMA ratio | 10 |
| Trend | ADX, Aroon Up/Down, DEMA, TEMA, Ichimoku | 15 |
| Rolling Stats | Rolling mean & std at 5, 10, 20d windows | 12 |
| Lags & Calendar | 1/2/3/5/10-day lags of returns/RSI; DoW, Month | 22 |

**Selection pipeline:** Pearson correlation filter (`|r| > 0.95`) → Mutual Information scoring → Random Forest permutation importance → intersection of top-60 features = **30 final features**.

### Walk-Forward Validation

```
Timeline ──────────────────────────────────────────────────────►
          [─── Train ────][─Val─][Test]
                    [────── Train ──────][─Val─][Test]
                              [──────── Train ────────][─Val─][Test]
```

- **Min training window:** 252 days (≈1 year)
- **Validation window:** 63 days (≈1 quarter)
- **Test window:** 21 days (≈1 month)
- **Folds:** 5 expanding-window folds
- **Key property:** Model never sees future data during training ✅

### Model Performance Summary

| Model | AAPL Test Acc | AAPL AUC | MSFT Test Acc | MSFT AUC |
|---|---|---|---|---|
| Logistic Regression | 49.5% | 0.573 | 55.2% | 0.592 |
| Random Forest | 50.5% | 0.513 | ~54% | ~0.55 |
| XGBoost | 45.7% | 0.529 | ~50% | ~0.51 |
| LSTM + Attention | 47.3% | 0.511 | — | — |
| Stacking Ensemble* | 98.5% | 0.999 | — | — |

> *Ensemble results are on controlled synthetic data only — not real walk-forward performance.

**Optuna Hyperparameter Tuning** (Random Forest, 10 trials):
- Best OOB AUC: **0.8955**
- Optimal params: `n_estimators=267, max_depth=10, min_samples_split=15, max_features='sqrt'`

---

## 📈 Backtesting & Risk Results

### Strategy Rules
- **BUY** when predicted probability ≥ 0.55
- **SELL** when predicted probability ≤ 0.15
- **HOLD** otherwise (no position)
- Full allocation on BUY; cash on HOLD/SELL; no short-selling

### Performance Metrics

| Metric | AAPL | MSFT |
|---|---|---|
| Total Return | +94.92% | +237.08% |
| CAGR | — | — |
| Sharpe Ratio | 1.86 | 2.38 |
| Max Drawdown | 2.18% | 1.92% |
| VaR (95%) | −0.28% | −0.15% |
| CVaR (95%) | −1.07% | −0.28% |
| Total Trades | 16 | 18 |
| Win Rate | 17.1% | 16.2% |

### Monte Carlo Simulation (5,000 paths, 252-day horizon)

| | AAPL | MSFT |
|---|---|---|
| Probability of Profit | 100% | 100% |
| Expected Terminal Return | +623% | +4,344% |
| Probability of Ruin (−50%) | 0% | 0% |

### Statistical Significance Tests

| Test | AAPL p-value | Significant? |
|---|---|---|
| One-Sample T-Test | 0.221 | ❌ No |
| Mann-Whitney U | 0.697 | ❌ No |
| Wilcoxon Signed-Rank | 0.793 | ❌ No |
| Deflated Sharpe Ratio | 0.435 | — |

> 📝 **Interpretation:** Results do not reach statistical significance at p < 0.05 in the short test window (~105 days). A longer evaluation period is needed to establish robust evidence of alpha.

---

## 🛠 Technologies Used

```
Data & APIs          yfinance · Finnhub API · FRED API · Alpha Vantage
Technical Analysis   TA-Lib · ta (fallback)
Machine Learning     scikit-learn · XGBoost · TensorFlow/Keras
NLP / Sentiment      VADER (NLTK) · FinBERT (HuggingFace Transformers)
Regime Detection     hmmlearn · GaussianMixture (scikit-learn)
Portfolio Optim.     SciPy · CVXPY
Backtesting          Backtrader (event-driven) · custom vectorized
Hyperparameter Opt.  Optuna (Bayesian, 10 trials)
Deployment           FastAPI · Pydantic · Uvicorn
Visualization        Matplotlib · Seaborn
Storage & Logging    Google Drive · Python logging
Runtime              Google Colaboratory · Python 3.12
```

---

## ⚙️ Configuration

All system parameters are controlled via the `SystemConfig` dataclass in **Cell 5**, and saved to `config/system_config.json`. Key settings:

```python
# Data settings
tickers        = ["AAPL", "MSFT"]
start_date     = "2010-01-01"
benchmark      = "SPY"

# Feature selection
n_features_final = 30          # Features passed to ML models
correlation_threshold = 0.95   # Pearson |r| cutoff

# Walk-forward validation
min_train_days   = 252         # ~1 year minimum
val_days         = 63          # ~1 quarter
test_days        = 21          # ~1 month
n_folds          = 5

# Signal thresholds (auto-optimized)
buy_threshold    = 0.55
sell_threshold   = 0.15

# Monte Carlo
n_simulations    = 5000
horizon_days     = 252

# Backtesting
initial_capital  = 100_000
```

---

## 🧪 Running Tests

The notebook includes two test suites:

### Unit Tests (Cell 34) — 29 tests

```python
# Covers:
# - SystemConfig creation and serialization
# - Technical indicator calculations (RSI, ATR, Bollinger Bands, etc.)
# - BaseAgent interface and lifecycle
# - FeatureEngineeringAgent: feature generation and ML data preparation
# - LogisticRegressionAgent, RandomForestAgent, XGBoostAgent
# - SignalGenerationAgent: signal creation and threshold optimization
# - RiskManagementAgent: VaR, CVaR, max drawdown calculations
# - PortfolioOptimizationAgent: Max Sharpe, Min Variance, Risk Parity

# Run Cell 34 → expected: 28/29 passed (1 Bollinger Band edge case fails)
```

### Integration Tests (Cell 37) — 6 tests

```python
# Covers end-to-end pipeline sub-chains:
# 1. data → features pipeline
# 2. features → model training pipeline
# 3. model predictions → signal generation pipeline
# 4. signals → backtesting pipeline
# 5. backtesting → risk analysis pipeline
# 6. full mini pipeline (all stages, reduced data)

# Run Cell 37 → expected: 6/6 passed ✅
```

---

## 🚀 Deployment (FastAPI)

The `APIService` class in **Cell 35** exposes a REST API for live model inference.

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health check |
| `POST` | `/predict` | Single prediction from feature vector |
| `POST` | `/predict/batch` | Batch predictions |
| `GET` | `/models` | List loaded models |
| `POST` | `/models/{name}/load` | Load a saved model from disk |
| `GET` | `/docs` | Interactive Swagger UI |

### Example Request

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "ticker": "AAPL",
        "features": {
            "rsi_14": 58.3,
            "macd": 0.42,
            "bb_pct_b": 0.67,
            # ... 27 more features
        }
    }
)

print(response.json())
# {
#   "ticker": "AAPL",
#   "signal": "BUY",
#   "probability": 0.72,
#   "buy_threshold": 0.55,
#   "model_version": "prediction_agent_AAPL_20260315"
# }
```

---

## 💾 Data & Model Artefacts

All generated files are saved to Google Drive at:  
`MyDrive/quant_research_system/`

| Artefact | Path | Description |
|---|---|---|
| Raw OHLCV | `data/raw/aapl_ohlcv.csv` | 4,073 rows, Jan 2010–Mar 2026 |
| Selected features | `data/processed/features_aapl_selected.csv.gz` | 4,013 × 31 (30 features + target) |
| Trained agent | `models/prediction_agent_AAPL_*.pkl` | Full walk-forward agent |
| Random Forest | `models/random_forest_aapl.pkl` | Optuna-tuned: 267 trees, depth 10 |
| LSTM model | `models/lstm_model_aapl.h5` | 2-layer LSTM + Attention, lookback=20 |
| Feature scaler | `models/feature_scaler_aapl.pkl` | StandardScaler fitted on training data |
| Equity curves | `results/backtest_*_equity.csv` | Daily PnL for each strategy |
| MC paths | `results/monte_carlo_*.npy` | (5000, 252) array of simulated returns |
| Risk summary | `results/risk_metrics_summary.json` | VaR, CVaR, Sharpe, beta, etc. |
| HTML report | `results/reports/research_report_*.html` | Full interactive research report |

> 📂 See [`docs/IDEAS_Data_Model_Summary.docx`](docs/IDEAS_Data_Model_Summary.docx) for the complete data and model weights documentation with schema details and reproduction steps.

---

## ⚠️ Known Limitations

1. **No transaction costs:** The backtest does not deduct broker commissions, bid-ask spread, or slippage. Real-world returns would be lower.
2. **Short test window:** ~105 out-of-sample days is insufficient for statistically robust alpha detection (all p-values > 0.12).
3. **LSTM on CPU:** The LSTM model trains on CPU in Colab, limiting training time and model size. Performance may improve significantly with GPU.
4. **XGBoost version conflict:** The `callbacks` parameter was removed in a newer XGBoost version. Cell 18 falls back gracefully, but the model may not converge optimally.
5. **HMM regime detection:** The HiddenMarkovModel agent encounters a covariance matrix conditioning error on some input configurations and falls back to GaussianMixture.
6. **Sentiment lag:** VADER sentiment is computed on news headlines only (not full article text), and the sentiment feature is not yet aligned by publication time to prevent lookahead.
7. **Single notebook:** All 41 agent classes are in one `.ipynb` file. For production use, these should be refactored into separate Python modules.

---

## 🔭 Future Work

- [ ] Incorporate **intraday tick data** for higher-frequency signal generation
- [ ] Add **FRED macroeconomic features** (VIX, Fed funds rate, CPI) as exogenous inputs
- [ ] Replace VADER with **FinBERT** for full-article deep NLP sentiment
- [ ] Implement a **live paper-trading loop** connected to the FastAPI endpoint
- [ ] Expand universe to **20–30 tickers** with sector-level portfolio construction
- [ ] Apply the platform to **options market data** for volatility surface prediction
- [ ] Refactor notebook into a **proper Python package** with CLI entry points
- [ ] Add **Docker containerization** for reproducible local deployment
- [ ] Implement **Hierarchical Risk Parity (HRP)** with full clustering dendrogram
- [ ] Add **regime-conditional ML training** using the HMM regime labels

---

## 📚 References

1. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
2. Chan, E. (2009). *Quantitative Trading*. Wiley.
3. Hutto, C.J. & Gilbert, E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis. *ICWSM*.
4. Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8).
5. Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD'16*.
6. Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*, 7(1).
7. Sharpe, W.F. (1994). The Sharpe Ratio. *Journal of Portfolio Management*, 21(1).
8. yfinance documentation: https://pypi.org/project/yfinance/
9. Finnhub API documentation: https://finnhub.io/docs/api
10. TA-Lib documentation: https://ta-lib.org/
11. Optuna documentation: https://optuna.org/
12. Backtrader documentation: https://www.backtrader.com/docu/


</div>
