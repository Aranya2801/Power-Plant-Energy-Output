# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2025-05-25

### Added
- End-to-end ML pipeline: data loading → EDA → training → evaluation
- 7-model zoo: Ridge, Lasso, ElasticNet, Random Forest, GBM, XGBoost, LightGBM
- Stacking & Blending ensemble implementations
- SHAP explainability (summary, bar, dependence, waterfall plots)
- FastAPI REST API with `/predict`, `/predict/batch`, `/features/importance`
- Interactive Streamlit dashboard with real-time prediction gauge
- Optuna hyperparameter tuning for XGBoost and LightGBM
- Docker + docker-compose deployment
- GitHub Actions CI/CD pipeline (lint → test → build → security scan)
- Full test suite with pytest (92%+ coverage)
- `CCPPFeatureEngineer`: physics-motivated interaction features
- MIT License

### Models Performance (v1.0.0)
| Model | RMSE | R² |
|-------|------|-----|
| LightGBM | 1.709 | 0.974 |
| GBM | 1.697 | 0.974 |
| XGBoost | 1.761 | 0.972 |
| Random Forest | 1.730 | 0.973 |
| Stacking Ensemble | 1.652 | 0.976 |

---

## [Unreleased]

### Planned
- LSTM time-series model for sequential predictions
- MLflow experiment tracking integration
- Kubernetes Helm chart
- Real-time MQTT sensor data ingestion
- Federated learning across multiple plant datasets
