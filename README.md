# 🤖 AI Quantitative Research System: A Multi-Agent Framework for Stock Price Direction Prediction

A production-grade, multi-agent machine learning platform that predicts the next-day directional movement of US equity prices. The system covers the full quantitative research lifecycle — from raw data ingestion through model deployment — and is structured as a reusable Python package refactored from a 41-cell Google Colab notebook.
---
## 📊 Key Results at a Glance

| Ticker | WF Test Accuracy | AUC-ROC | Total Return | Sharpe Ratio | Max Drawdown |
|--------|-----------------|---------|--------------|--------------|--------------|
| AAPL   | 49.5%           | 0.573   | +94.92%      | 1.86         | 2.18%        |
| MSFT   | 55.2%           | 0.592   | +237.08%     | 2.38         | 1.92%        |
| Portfolio (50/50) | — | — | —         | 0.79         | < 2.2%       |

> Results are from a 5-fold walk-forward validation on daily data (Jan 2010 – Mar 2026). All backtesting is out-of-sample. Past performance does not guarantee future results.

---
## 📁 Repository Structure
```
ai-quant-research-system/
│
├── 📓 IDEAS_FINAL_15_03.ipynb       Original 41-cell Google Colab notebook
│
├── 📦 quant_research/               Modular Python package
│   ├── config.py                    PathManager · APIConfig · SystemConfig
│   ├── utils.py                     Shared helpers · caching · metrics
│   ├── base.py                      BaseAgent · MLAgent · DataAgent
│   ├── pipeline.py                  PipelineOrchestrator · run_complete_pipeline
│   ├── __main__.py                  CLI entry point
│   ├── setup.py                     pip-installable package
│   ├── requirements.txt             All 39 dependencies
│   ├── agents/                      Data · Features · Sentiment · Prediction · Evaluation
│   ├── models/                      LR · RF · XGBoost · LSTM · Ensemble · Optuna
│   ├── trading/                     Signals · Backtest · Risk · Monte Carlo · Portfolio
│   ├── reporting/                   HTML Reports · Charts · Dashboard
│   ├── api/                         FastAPI inference server
│   └── tests/                       29 unit tests + 6 integration tests
│
├── 🖼️  assets/                      Charts and visualizations
├── README.md                        This file
├── .env.example                     API key template
├── .gitignore
└── CHANGELOG.md
```

---

## 🏗️ System Architecture

The platform is built as **41 modular agent classes**, each inheriting from a common `BaseAgent` abstract base. A central `PipelineOrchestrator` co-ordinates their sequential execution.

```
                     ┌──────────────────────────┐
                     │    PipelineOrchestrator   │
                     └────────────┬─────────────┘
                                  │
  ┌──────────────┐  ┌────────────┐│┌─────────────────┐  ┌─────────────────┐
  │MarketData    │  │News Agent  │││FeatureEngineering│  │FeatureSelection │
  │Agent         │─▶│(Finnhub)   │─┤│Agent (113 indic.)│─▶│Agent (→30 feat) │
  └──────────────┘  └────────────┘││└─────────────────┘  └─────────────────┘
                                  ││
  ┌──────────────┐  ┌────────────┐││┌─────────────────┐  ┌─────────────────┐
  │Sentiment     │  │Prediction  │││ Signal           │  │  Backtesting    │
  │Agent         │─▶│Agent (WF)  │─┤│ Generation      │─▶│  Agent          │
  │(VADER/BERT)  │  │            │││ Agent            │  │  (Backtrader)   │
  └──────────────┘  └────────────┘││└─────────────────┘  └─────────────────┘
                                  ││
  ┌──────────────┐  ┌────────────┐││┌─────────────────┐  ┌─────────────────┐
  │Risk          │  │MonteCarlo  │││ Portfolio        │  │  Report         │
  │Management    │─▶│Agent       │─┤│ Optimization     │─▶│  Generation     │
  │(VaR/CVaR)    │  │(5000 paths)│││ (MaxSharpe/RP)   │  │  (HTML+FastAPI) │
  └──────────────┘  └────────────┘│└─────────────────┘  └─────────────────┘
```

---

## ⚡ Quickstart

### Option A — Google Colab

1. Click the **Open in Colab** badge at the top
2. Open **Cell 3** and add your API keys
3. Click **Runtime → Run all**
4. Wait ~5 minutes — results appear in your Google Drive under `quant_research_system/`

### Option B — Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/ai-quant-research-system.git
cd ai-quant-research-system/quant_research

# 2. Install the package
pip install -e ".[full]"

# 3. Install TA-Lib (requires C library)
#    Ubuntu/Colab:  sudo apt-get install libta-lib-dev && pip install TA-Lib
#    macOS:         brew install ta-lib && pip install TA-Lib

# 4. Configure API keys
cp .env.example .env
# Open .env and add your keys

# 5. Run the full pipeline
python -m quant_research

# Custom options
python -m quant_research --tickers AAPL MSFT GOOGL --no-report
```

### Option C — Start the Inference API Only

```bash
uvicorn quant_research.api.server:app --host 0.0.0.0 --port 8000
# Open: http://localhost:8000/docs
```

---

## 🔑 API Keys Required

All three are completely free:

| Service | Purpose | Sign Up |
|---------|---------|---------|
| **Finnhub** | Financial news articles | [finnhub.io](https://finnhub.io) |
| **FRED** | Macroeconomic data | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) |
| **Alpha Vantage** | Backup OHLCV data | [alphavantage.co](https://www.alphavantage.co) |

Add to your `.env` file (never commit this):

```
FINNHUB_API_KEY=your_key_here
FRED_API_KEY=your_key_here
ALPHA_VANTAGE_API_KEY=your_key_here
```

---

## 🧠 Machine Learning Models

| Model | AAPL WF Accuracy | AAPL AUC | Notes |
|-------|-----------------|---------|-------|
| Logistic Regression | 49.5% | 0.573 | L2 regularised, interpretable baseline |
| Random Forest | 50.5% | 0.513 | 267 trees, depth 10 (Optuna-tuned) |
| XGBoost | 45.7% | 0.529 | 100 estimators, learning rate 0.1 |
| LSTM + Attention | 47.3% | 0.511 | 128→64 units, lookback=20 (CPU) |
| Stacking Ensemble | 98.5%* | 0.999* | *Synthetic data only |

> All models use **5-fold walk-forward time-series cross-validation** — no lookahead bias.

**Feature pipeline:** 113 raw TA-Lib indicators → correlation filter (|r|>0.95) → mutual information + RF importance → **30 final features**

---

## 📈 Full Pipeline Stages

| Stage | Agent | What it does |
|-------|-------|-------------|
| 1 | `MarketDataAgent` | Fetch OHLCV from yfinance; Drive caching |
| 2 | `FeatureEngineeringAgent` | 113 indicators across 8 categories |
| 3 | `FeatureSelectionAgent` | Correlation filter → MI + RF → 30 features |
| 4 | `PredictionAgent` | 5-fold walk-forward LR + RF + XGBoost |
| 5 | `SignalGenerationAgent` | Grid-search threshold → BUY / HOLD / SELL |
| 6 | `BacktestingAgent` | Vectorized + Backtrader simulation |
| 7 | `RiskManagementAgent` | VaR, CVaR, Beta, stress testing |
| 8 | `MonteCarloAgent` | 5,000 bootstrap simulations |
| 9 | `StrategyEvaluationAgent` | Deflated Sharpe, T-test, Mann-Whitney |
| 10 | `ReportGenerationAgent` | Full HTML research report |
| + | `PortfolioOptimizationAgent` | Max Sharpe · Min Variance · Risk Parity |

**Total execution time: ~70 seconds** (AAPL + MSFT, Google Colab CPU)

---

## 🛠️ Technology Stack

```
Data & APIs          yfinance · Finnhub API · FRED API · Alpha Vantage
Feature Engineering  TA-Lib · ta (fallback) · numpy · pandas
Machine Learning     scikit-learn · XGBoost · TensorFlow/Keras
NLP / Sentiment      VADER (NLTK) · FinBERT (HuggingFace Transformers)
Regime Detection     hmmlearn · GaussianMixture (sklearn)
Portfolio Optim.     SciPy · CVXPY
Backtesting          Backtrader (event-driven) · vectorized simulation
Hyperparameter Opt.  Optuna (Bayesian TPE, 10 trials)
Deployment           FastAPI · Pydantic · Uvicorn
Visualisation        Matplotlib · Seaborn
Runtime              Google Colaboratory · Python 3.12
```

---

## 💾 Files Saved to Google Drive

All outputs land under `MyDrive/quant_research_system/`:

| Folder | File(s) | Description |
|--------|---------|-------------|
| `config/` | `system_config.json` | Configuration snapshot |
| `logs/` | `QUANT_SYSTEM_YYYYMMDD.log` | Date-stamped run log |
| `cache/` | `{cache_key}.pkl` | Cached market data |
| `models/` | `prediction_agent_{TICKER}_{timestamp}.pkl` | Trained agent |
| `models/` | `{model}_scaler.pkl` · `lstm_model.h5` | Scaler + Keras weights |
| `results/reports/` | `research_report_{timestamp}.html` | Full HTML report |

---

## 🧪 Tests

```bash
# All 35 tests
python -m pytest quant_research/tests/ -v

# Unit tests only
python -m pytest quant_research/tests/unit_tests.py -v

# Integration tests only
python -m pytest quant_research/tests/integration_tests.py -v
```

| Suite | Tests | Status |
|-------|-------|--------|
| Unit Tests | 29 | 28/29 ✅ |
| Integration Tests | 6 | 6/6 ✅ |

---

## 🌐 API Reference

```
GET  /health              System health and model load status
POST /predict             Single prediction from 30-feature vector
POST /predict/batch       Batch predictions
GET  /models              List all loaded models
POST /models/{name}/load  Load a saved model from disk
GET  /docs                Interactive Swagger UI
```

**Example:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL","features":{"rsi_14":58.3,"macd_histogram":0.42}}'

# Response:
# {"signal":"BUY","probability":0.72,"buy_threshold":0.55,"execution_ms":12}
```

---

## ⚠️ Known Limitations

1. **No transaction costs** — real returns would be lower after commissions and slippage
2. **Short test window** — ~105 out-of-sample days; all p-values > 0.12 (not statistically significant)
3. **Bull-market bias** — test period (2024–2026) is a strong bull market; bear results would differ
4. **LSTM on CPU** — limited epochs and training time; GPU would improve convergence significantly
5. **Sentiment snapshot** — VADER applied to a 7-day news snapshot, not a live rolling feed
6. **XGBoost version conflict** — `callbacks` param removed in newer versions; graceful fallback applied

---

## 🔭 Future Work

- [ ] Intraday tick data (Polygon.io / Interactive Brokers)
- [ ] FRED macroeconomic features (VIX, Fed rate, CPI)
- [ ] FinBERT on full article text with rolling 30-day feed
- [ ] Live paper-trading loop via FastAPI endpoint
- [ ] 20–30 ticker universe with sector portfolio
- [ ] Options volatility surface prediction
- [ ] Regime-conditional model selection (HMM labels → separate models)
- [ ] Docker + GitHub Actions CI/CD pipeline
- [ ] Temporal Fusion Transformer to replace LSTM

---

## 📚 References

1. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
2. Chan, E. (2009). *Quantitative Trading*. Wiley.
3. Hutto, C.J. & Gilbert, E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment. *ICWSM*.
4. Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8).
5. Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD'16*.
6. Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*, 7(1).
7. Sharpe, W.F. (1994). The Sharpe Ratio. *Journal of Portfolio Management*, 21(1).
8. yfinance: https://pypi.org/project/yfinance/
9. Finnhub API: https://finnhub.io/docs/api
10. TA-Lib: https://ta-lib.org/ · Optuna: https://optuna.org/ · Backtrader: https://www.backtrader.com
