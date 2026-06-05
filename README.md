# Predictive Maintenance

Condition-based monitoring and remaining useful life (RUL) estimation using machine learning on synthetic industrial sensor data.

## Problem Statement

Unplanned equipment downtime costs manufacturers billions annually. Predictive maintenance aims to forecast failures before they occur by analyzing sensor telemetry (temperature, vibration, pressure, rotational speed). This project covers two complementary tasks:

- **Fault classification** — classify whether a machine will fail given its current sensor readings
- **Remaining Useful Life (RUL) regression** — predict the number of operating cycles remaining until failure

## Project Structure

```
predictive-maintenance/
├── data/
│   ├── sensor_telemetry.csv             # Raw synthetic dataset (10k rows)
│   └── sensor_telemetry_engineered.csv  # Enriched with engineered features
├── models/
│   └── best_model.pkl                   # Best classifier serialized with joblib
├── notebooks/
│   ├── 01_eda.ipynb                     # Exploratory data analysis
│   └── 02_rul_estimation.ipynb          # RUL regression (IEEE PHM methodology)
├── results/
│   └── model_comparison.csv             # Classification metric comparison table
├── src/
│   ├── load_data.py                     # Generate synthetic sensor dataset
│   ├── feature_engineering.py           # Rolling statistics & threshold features
│   └── train_model.py                   # Classifier training & evaluation
├── requirements.txt
├── venv/
└── README.md
```

## Dataset

The sensor telemetry dataset (`data/sensor_telemetry.csv`) contains 10,000 observations from 100 machines with the following columns:

| Column | Description | Range |
|---|---|---|
| `machine_id` | Machine identifier | 1–100 |
| `temperature` | Operating temperature (°C) | 30–120 |
| `vibration` | Vibration amplitude | 0–10 |
| `pressure` | System pressure (psi) | 50–200 |
| `rotational_speed` | Rotational speed (RPM) | 500–3000 |
| `failure` | Binary failure label (5% positive) | 0 / 1 |

Failures are randomly assigned; the dataset serves as a **baseline** for pipeline development. Replace with a real or semi-realistic dataset for production modeling.

## Feature Engineering

The script `src/feature_engineering.py` adds the following columns:

| Feature | Description |
|---|---|
| `{sensor}_rolling_mean` | Rolling mean (window=5) per machine, for temperature/vibration/pressure |
| `{sensor}_rolling_std` | Rolling standard deviation (window=5) per machine |
| `high_vibration` | Binary flag: 1 if vibration exceeds the 75th percentile |

Output saved to `data/sensor_telemetry_engineered.csv` (10,000 × 13).

## Model Comparison (Fault Classification)

Three classifiers trained on the engineered dataset (80/20 stratified split):

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| LogisticRegression | 0.9500 | 0.4750 | 0.5000 | 0.4872 | 0.5075 |
| DecisionTree | 0.8995 | 0.4974 | 0.4971 | 0.4972 | 0.4971 |
| RandomForest | 0.9500 | 0.4750 | 0.5000 | 0.4872 | 0.4685 |

The best model by macro F1 (**DecisionTree**) is saved to `models/best_model.pkl`. All models perform near random-chance for the minority class because failures are randomly assigned and uncorrelated with sensor values. With a realistic dataset where failures correlate with extreme sensor readings, tree-based and ensemble methods are expected to significantly outperform logistic regression.

## RUL Estimation Results

A synthetic run-to-failure dataset was generated following the IEEE PHM / NASA C-MAPSS framework (100 machines, 4 sensors, variable lifecycles). A RandomForestRegressor was trained with a group-aware (machine-level) train/test split:

| Split | RMSE | R² |
|---|---|---|
| Train | 7.93 | 0.8962 |
| Test | 11.08 | 0.8033 |

The notebook `notebooks/02_rul_estimation.ipynb` includes methodology documentation, benchmark comparisons against published C-MAPSS results, and visualizations of predicted vs. actual RUL trajectories.

## Setup & Usage

```bash
# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 1. Generate the synthetic dataset
python src/load_data.py

# 2. Engineer features
python src/feature_engineering.py

# 3. Train and evaluate classifiers
python src/train_model.py

# 4. Launch Jupyter notebooks
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_rul_estimation.ipynb
```

## Dependencies

- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- jupyter
