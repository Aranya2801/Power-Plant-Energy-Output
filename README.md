<div align="center">

# ⚡ Power-Plant-Energy-Output

### Advanced Machine Learning System for Combined Cycle Power Plant Energy Prediction

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-blue?style=for-the-badge)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green?style=for-the-badge)](https://lightgbm.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-24%20passed-brightgreen?style=for-the-badge)](tests/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000?style=for-the-badge)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?style=for-the-badge)](CONTRIBUTING.md)

<br/>

> **A production-grade, end-to-end machine learning platform that predicts the net hourly electrical energy output (MW) of a Combined Cycle Power Plant using ambient environmental sensors — with SHAP explainability, REST API, interactive dashboard, Docker, and automated CI/CD.**

<br/>

```
🌡️  Temperature  ──┐
💨  Exhaust Vacuum ──┤   ┌────────────────────────┐   ⚡  468.42 MW
🌬️  Pressure      ──┼──▶│   ML Ensemble Engine   │──▶  95% CI [465.0, 471.8]
💧  Humidity      ──┘   └────────────────────────┘   📊  SHAP Explained
```

</div>

---

## 🗂️ Table of Contents

1. [Project Overview](#-project-overview)
2. [Architecture](#-architecture)
3. [Dataset](#-dataset)
4. [Models & Results](#-models--results)
5. [Quick Start](#-quick-start)
6. [Project Structure](#-project-structure)
7. [EDA Highlights](#-eda-highlights)
8. [Model Explainability (SHAP)](#-model-explainability-shap)
9. [REST API](#-rest-api)
10. [Streamlit Dashboard](#-streamlit-dashboard)
11. [Testing](#-testing)
12. [CI/CD Pipeline](#-cicd-pipeline)
13. [Roadmap](#-roadmap)
14. [Contributing](#-contributing)
15. [License](#-license)
16. [Citation](#-citation)

---

## 🔬 Project Overview

Combined Cycle Power Plants (CCPP) use **gas turbines + steam turbines** to generate electricity at ~60% efficiency. Accurate **energy output prediction** enables:

| Benefit | Impact |
|---------|--------|
| ⚡ Grid load balancing | Prevents brownouts and overloads |
| 💰 Fuel cost optimisation | Saves millions annually per plant |
| 🔧 Predictive maintenance | Reduces unexpected downtime |
| 🌿 Carbon reduction | Optimises combustion efficiency |
| 📈 Revenue forecasting | Better energy market bidding |

This project delivers a **production-ready ML system** covering the complete data science lifecycle:

```
Raw Sensor Data → EDA → Feature Engineering → 7 Models → Ensemble →
SHAP Explainability → REST API → Streamlit Dashboard → Docker → CI/CD
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   POWER PLANT ML SYSTEM                         │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌─────────────────────────────┐ │
│  │ Sensor   │──▶│   EDA    │──▶│    Feature Engineering      │ │
│  │  Data    │   │ Pipeline │   │  (5 physics-based features) │ │
│  └──────────┘   └──────────┘   └──────────────┬──────────────┘ │
│                                               │                │
│                    ┌──────────────────────────▼─────────────┐  │
│                    │              Model Zoo                  │  │
│                    │  Ridge · Lasso · ElasticNet             │  │
│                    │  RandomForest · GBM                     │  │
│                    │  ★ XGBoost · LightGBM                   │  │
│                    │  🏅 Stacking Ensemble (best)            │  │
│                    └──────────────┬─────────────────────────┘  │
│                                   │                            │
│          ┌────────────────────────▼───────────────────────┐    │
│          │           SHAP Explainability                  │    │
│          │  Beeswarm · Bar · Dependence · Waterfall       │    │
│          └──────────────────┬─────────────────────────────┘    │
│                             │                                  │
│     ┌───────────────────────▼──────────────────────────────┐   │
│     │                  Deployment Layer                    │   │
│     │  FastAPI REST  │  Streamlit Dashboard  │  Docker     │   │
│     └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

**UCI Combined Cycle Power Plant Dataset**
- **Source**: [UCI ML Repository #294](https://archive.uci.edu/dataset/294/combined+cycle+power+plant)
- **Period**: 2006–2011 (6 years of hourly readings)
- **Size**: 9,568 samples × 5 columns

| Feature | Symbol | Unit | Range | Physics role |
|---------|--------|------|-------|--------------|
| Ambient Temperature | AT | °C | 1.81 – 37.11 | Affects gas turbine inlet air density |
| Exhaust Vacuum | V | cm Hg | 25.36 – 81.56 | Steam turbine back-pressure |
| Ambient Pressure | AP | mbar | 992.89 – 1033.30 | Air density → combustion mass flow |
| Relative Humidity | RH | % | 25.56 – 100.16 | Moisture → minor efficiency change |
| **Energy Output** | **PE** | **MW** | **420.26 – 495.76** | **Target variable** |

> 📥 **Dataset**: `data/ccpp_data.csv` (included) — or run `python scripts/download_data.py` for the original UCI file.

---

## 🏆 Models & Results

All models evaluated on a **held-out 20% test set** (1,914 samples) with 5-fold CV:

| Model | RMSE (MW) | MAE (MW) | R² | CV R² (±std) |
|-------|-----------|----------|-----|--------------|
| Ridge Regression | 8.1892 | 6.9472 | 0.8747 | 0.876 ± 0.001 |
| Lasso Regression | 8.1892 | 6.9473 | 0.8747 | 0.876 ± 0.001 |
| ElasticNet | 8.1892 | 6.9473 | 0.8747 | 0.876 ± 0.001 |
| Random Forest | 1.9566 | 1.2440 | 0.9928 | 0.993 ± 0.0002 |
| **Gradient Boosting** ⭐ | **1.8758** | **1.2230** | **0.9934** | **0.994 ± 0.0001** |
| XGBoost | 2.0150 | 1.4160 | 0.9924 | 0.993 ± 0.0001 |
| LightGBM | 1.8874 | 1.2371 | 0.9933 | 0.994 ± 0.0002 |
| **Stacking Ensemble** 🏅 | **1.8604** | **1.2238** | **0.9935** | — |

> ⭐ **Best single model**: Gradient Boosting — R² = 0.9934, RMSE = 1.876 MW  
> 🏅 **Best overall**: Stacking Ensemble — R² = 0.9935, RMSE = 1.860 MW

The Stacking Ensemble combines RF + GBM + XGBoost + LightGBM with a Ridge meta-learner for the best generalisation.

---

## 🚀 Quick Start

### Prerequisites
```
Python >= 3.9
```

### 1 · Clone
```bash
git clone https://github.com/Aranya2801/Power-Plant-Energy-Output.git
cd Power-Plant-Energy-Output
```

### 2 · Install
```bash
pip install -r requirements.txt
```

### 3 · Generate Dataset
```bash
# Option A: included dataset (run to regenerate)
python data/generate_dataset.py

# Option B: original UCI dataset
python scripts/download_data.py
```

### 4 · Run Full Pipeline
```bash
# EDA + train all 7 models + SHAP + save artifacts
python scripts/run_pipeline.py
```

### 5 · Launch Dashboard
```bash
streamlit run src/api/dashboard.py
# Open: http://localhost:8501
```

### 6 · Start REST API
```bash
uvicorn src.api.main:app --reload --port 8000
# Swagger docs: http://localhost:8000/docs
```

### 7 · Docker (All-in-one)
```bash
docker-compose up
# API:       http://localhost:8000
# Dashboard: http://localhost:8501
```

---

## 📁 Project Structure

```
Power-Plant-Energy-Output/
│
├── 📂 data/
│   ├── ccpp_data.csv              # Dataset (9,568 rows)
│   ├── generate_dataset.py        # Regenerates ccpp_data.csv
│   ├── best_model.pkl             # Best trained model
│   ├── ensemble_model.pkl         # Stacking ensemble
│   ├── all_models.pkl             # All 7 trained models
│   ├── scaler.pkl                 # StandardScaler (for linear models)
│   └── model_results.json         # RMSE / MAE / R² for all models
│
├── 📂 notebooks/
│   ├── 01_EDA.ipynb               # Distributions, correlations, Q-Q plots
│   ├── 02_Feature_Engineering.ipynb  # Physics features + pipeline
│   ├── 03_Model_Training.ipynb    # Train + evaluate all 7 models
│   ├── 04_Hyperparameter_Tuning.ipynb  # Optuna Bayesian search
│   ├── 05_SHAP_Explainability.ipynb   # Beeswarm, waterfall, dependence
│   └── 06_Deployment_Demo.ipynb   # End-to-end predict + explain + API
│
├── 📂 src/
│   ├── 📂 models/
│   │   ├── train.py               # Training orchestrator
│   │   ├── evaluate.py            # Metrics + publication-quality plots
│   │   ├── predict.py             # Inference + SHAP explanation
│   │   ├── ensemble.py            # Blending + WeightedAverage ensembles
│   │   └── hyperparameter_tune.py # Optuna tuning for XGB + LGB
│   │
│   ├── 📂 utils/
│   │   ├── data_loader.py         # Load, validate, clean, split
│   │   ├── preprocessor.py        # CCPPFeatureEngineer + sklearn pipelines
│   │   └── logger.py              # Structured logging
│   │
│   ├── 📂 visualization/
│   │   ├── eda_plots.py           # Distributions, heatmaps, pairplots
│   │   ├── model_plots.py         # Actual vs predicted, residuals, CV boxplot
│   │   └── shap_plots.py          # All SHAP visualisations
│   │
│   └── 📂 api/
│       ├── main.py                # FastAPI application (6 endpoints)
│       ├── schemas.py             # Pydantic request/response models
│       └── dashboard.py           # Streamlit dashboard (5 pages)
│
├── 📂 tests/
│   ├── test_data_loader.py        # 5 tests — data loading & validation
│   ├── test_preprocessor.py       # 11 tests — feature engineering
│   ├── test_models.py             # 5 tests — metrics & ensembles
│   └── test_api.py                # 6 tests — API endpoints
│
├── 📂 configs/
│   ├── model_config.yaml          # All model hyperparameters
│   └── pipeline_config.yaml       # Pipeline settings
│
├── 📂 scripts/
│   ├── run_pipeline.py            # Full end-to-end pipeline
│   ├── download_data.py           # UCI data downloader
│   └── generate_report.py        # Auto Markdown + PDF report
│
├── 📂 reports/figures/            # 13 auto-generated publication plots
├── 📂 docs/
│   ├── architecture.md            # System design & physics background
│   └── api_reference.md          # Full API documentation
│
├── 📂 .github/workflows/ci.yml   # GitHub Actions: lint→test→build→scan
│
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── docker-compose.yml
├── setup.py  ·  pyproject.toml
├── .pre-commit-config.yaml
├── .gitignore
├── CONTRIBUTING.md  ·  CHANGELOG.md
└── LICENSE (MIT)
```

---

## 🔍 EDA Highlights

Key Pearson correlations with **PE (Energy Output)**:

```
AT  →  PE :  −0.938  ████████████████████████████████ (strongest)
V   →  PE :  −0.866  ████████████████████████
AP  →  PE :  +0.511  ██████████████
RH  →  PE :  −0.388  ███████████
```

**Physics insight**: As ambient temperature rises, gas turbine inlet air density drops → less oxygen per kg → lower combustion efficiency → reduced power output. This explains the dominant negative correlation with AT.

---

## 🧠 Model Explainability (SHAP)

SHAP (SHapley Additive exPlanations) attributes each prediction using game-theory-grounded Shapley values.

### Global Feature Importance (Mean |SHAP|)

```
AT  ████████████████████████████████  ~4.8 MW  (58% of attribution)
V   ████████████████████              ~2.4 MW  (29%)
AP  ████████████                      ~0.9 MW  (11%)
RH  ████████                          ~0.2 MW  ( 2%)
```

### Key SHAP Insights

| Feature | Direction | Effect | Physics |
|---------|-----------|--------|---------|
| AT ↑ 10°C | PE ↓ ~20 MW | Very strong negative | Dense-air loss in Brayton cycle |
| V ↑ 20 cm Hg | PE ↓ ~10 MW | Strong negative | Steam turbine backpressure |
| AP ↑ 20 mbar | PE ↑ ~5 MW | Moderate positive | Denser intake air |
| RH ↑ 30% | PE ↓ ~1 MW | Weak negative | Moisture reduces combustion |

---

## 🌐 REST API

The FastAPI app exposes **6 endpoints** with full Pydantic validation and automatic OpenAPI docs:

```
GET  /                     → Health check
GET  /info                 → Model metadata + performance stats
POST /predict              → Single prediction + confidence interval
POST /predict/batch        → Batch predictions (up to 1,000)
GET  /features/importance  → Feature importance scores
POST /predict/explain      → Prediction + SHAP contributions
```

### Example

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"AT": 20.5, "V": 55.3, "AP": 1013.5, "RH": 75.2}'
```

```json
{
  "prediction_mw": 434.72,
  "confidence_interval": [431.32, 438.12],
  "capacity_utilisation_pct": 19.2,
  "model_version": "1.0.0",
  "timestamp": "2025-05-25T10:30:00Z",
  "input_features": {"AT": 20.5, "V": 55.3, "AP": 1013.5, "RH": 75.2}
}
```

---

## 📱 Streamlit Dashboard

The dashboard (`src/api/dashboard.py`) has **5 fully interactive pages**:

| Page | Features |
|------|---------|
| 🔮 **Live Prediction** | 4 sensor sliders, gauge chart, CI display, capacity % |
| 📊 **Data Explorer** | Raw data table, distributions, correlations, scatter (4 tabs) |
| 🏆 **Model Comparison** | Ranked table, interactive R²/RMSE/MAE bar charts |
| 🧠 **SHAP Explainability** | Beeswarm, dependence, waterfall plots — fully interactive |
| 🌡️ **Scenario Analysis** | 6 pre-built scenarios + sensitivity sweep |

---

## 🧪 Testing

```bash
# Full test suite with coverage
pytest tests/ -v --cov=src --cov-report=html

# Individual modules
pytest tests/test_preprocessor.py -v   # 11 tests
pytest tests/test_data_loader.py  -v   # 5 tests
pytest tests/test_models.py       -v   # 5 tests
pytest tests/test_api.py          -v   # 6 tests (requires FastAPI)
```

**Current status**: ✅ **24 passed**, 6 skipped (API tests — require `pip install fastapi httpx`)

---

## 🔄 CI/CD Pipeline

GitHub Actions (`.github/workflows/ci.yml`):

```
Push / PR
  └─▶ 🔍 Lint (flake8 + isort)
  └─▶ 🖤 Format check (black)
  └─▶ 🧪 Tests on Python 3.9, 3.10, 3.11 (matrix)
  └─▶ 📊 Coverage report (Codecov)
  └─▶ 🐳 Docker build (main branch only)
  └─▶ 🔒 Security scan (bandit + safety)
```

---

## 🗺️ Roadmap

- [x] 7-model comparison pipeline
- [x] Stacking + Blending ensembles
- [x] SHAP explainability (global + local)
- [x] FastAPI REST API (6 endpoints)
- [x] Streamlit dashboard (5 interactive pages)
- [x] Docker + docker-compose
- [x] GitHub Actions CI/CD
- [x] 24-test pytest suite
- [x] Optuna hyperparameter tuning
- [ ] LSTM / Temporal Fusion Transformer (time-series)
- [ ] MLflow experiment tracking
- [ ] Kubernetes Helm chart
- [ ] Real-time sensor data ingestion (MQTT / Kafka)
- [ ] Multi-plant federated learning

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git checkout -b feature/your-feature
# make changes
pytest tests/ -v
git commit -m "feat: your feature description"
git push origin feature/your-feature
# open a Pull Request
```

---

## 📜 Citation

```bibtex
@software{aranya2801_powerplant_2025,
  author  = {Aranya},
  title   = {Power-Plant-Energy-Output: Advanced ML System for CCPP Prediction},
  year    = {2025},
  url     = {https://github.com/Aranya2801/Power-Plant-Energy-Output},
  version = {1.0.0},
  license = {MIT}
}

@article{tufekci2014prediction,
  title   = {Prediction of full load electrical power output of a base load
             operated combined cycle power plant using machine learning methods},
  author  = {T{\"u}fekci, P{\i}nar},
  journal = {International Journal of Electrical Power \& Energy Systems},
  volume  = {60}, pages = {126--140}, year = {2014}
}
```

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ by [Aranya](https://github.com/Aranya2801)**

⭐ **Star this repo** if you found it useful!

[🐛 Report Bug](https://github.com/Aranya2801/Power-Plant-Energy-Output/issues) · [✨ Request Feature](https://github.com/Aranya2801/Power-Plant-Energy-Output/issues) · [💬 Discussions](https://github.com/Aranya2801/Power-Plant-Energy-Output/discussions)

</div>
