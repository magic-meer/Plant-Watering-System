# Plant Watering System

An AI-powered plant health monitoring and automated watering decision system developed as part of an AI/ML Fellowship program. This project uses machine learning models and rule-based decision logic to classify plant health status and recommend watering actions.

## Overview

The Plant Watering Intelligence System classifies plant health into three categories:
- **Healthy** - Optimal soil moisture and environmental conditions
- **Needs Water** - Low soil moisture requiring immediate irrigation
- **Overwatered** - Excess moisture requiring drainage intervention

## Features

- **Multi-Model ML Pipeline**: Logistic Regression, Random Forest, and XGBoost classifiers
- **Rule-Based Decision Engine**: Weighted rule system for interpretable decisions
- **Interactive Dashboard**: Streamlit-based web interface for real-time predictions
- **Model Comparison**: Comprehensive performance analysis with visualizations
- **Sensor Integration**: Support for soil moisture, temperature, humidity, and nutrient sensors

## Tech Stack

- **Frontend**: Streamlit, Plotly
- **Backend**: Python 3.10+
- **ML Libraries**: scikit-learn, XGBoost, pandas, numpy
- **Visualization**: Matplotlib, Seaborn, Plotly

## Project Structure

```
Plant-Watering-System/
├── app/                        # Streamlit application
│   ├── app.py                  # Main entry point
│   └── views/                  # Page views
│       ├── about.py
│       ├── dashboard.py
│       ├── dataset_visualization.py
│       ├── model_comparison.py
│       ├── predict.py
│       └── settings.py
├── src/                        # Source code
│   ├── backend/
│   │   └── predict.py          # ML prediction backend
│   ├── data/
│   │   ├── load_data.py        # Data loading utilities
│   │   ├── preprocess.py       # Data preprocessing
│   │   └── split.py            # Train-test split
│   ├── inference/
│   │   └── decision_logic.py   # Rule-based decision engine
│   ├── models/
│   │   ├── compare_models.py   # Model comparison script
│   │   ├── train_logistic.py   # Logistic Regression training
│   │   ├── train_rf.py         # Random Forest training
│   │   └── train_xgb.py        # XGBoost training
│   └── utils/
│       ├── config.py           # Central configuration
│       └── helpers.py          # Helper utilities
├── models/                     # Trained model files
├── data/                       # Dataset directory
│   ├── raw/                    # Raw data
│   └── processed/              # Processed data
├── reports/                    # Generated reports
│   └── figures/                # Visualization outputs
├── plots/                      # Static plots
├── notebooks/                  # Jupyter notebooks
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Plant-Watering-System
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

Start the Streamlit dashboard:
```bash
streamlit run app/app.py
```

The application will open in your browser at `http://localhost:8501`

### Training Models

1. **Preprocess Data**:
```bash
cd src/data
python preprocess.py
```

2. **Train Individual Models**:
```bash
cd src/models
python train_logistic.py
python train_rf.py
python train_xgb.py
```

3. **Compare Models**:
```bash
python compare_models.py
```

This generates:
- Performance metrics (JSON, CSV)
- Visualization charts (PNG)
- Comprehensive markdown report

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 71.25% | 71.49% | 71.25% | 71.36% |
| Random Forest | 100.00% | 100.00% | 100.00% | 100.00% |
| XGBoost | 99.58% | 99.59% | 99.58% | 99.58% |

**Recommended Model**: XGBoost (best balance of accuracy and generalization)

## Configuration

Edit `src/utils/config.py` to modify:
- File paths and directories
- Model hyperparameters
- Rule engine thresholds
- Class labels and colors

### Rule Engine Thresholds

```python
RULE_THRESHOLDS = {
    "soil_moisture_low": 30,    # Below -> Needs Water
    "soil_moisture_high": 70,   # Above -> Overwatered
    "temperature_high": 38,     # Above -> Stress flag
    "humidity_low": 35,         # Below -> Dry air
    "days_since_water": 3,      # Above -> Likely needs water
}
```

## Dataset

The dataset contains plant health sensor readings with the following features:

### Environmental Sensors
- Soil Moisture (%)
- Ambient Temperature (°C)
- Soil Temperature (°C)
- Humidity (%)
- Light Intensity (lux)
- Soil pH
- Nitrogen, Phosphorus, Potassium Levels
- Chlorophyll Content
- Electrochemical Signal

### Engineered Features
- days_since_last_watering
- watering_sma_3 (3-period moving average)
- Temporal features (Year, Month, Day, Hour)

## API Reference

### Backend Prediction

```python
from src.backend.predict import predict_all_models

sensor_input = {
    "Soil_Moisture": 25,
    "Ambient_Temperature": 28,
    "Humidity": 60,
    # ... other features
}

result = predict_all_models(sensor_input)
```

### Rule-Based Decision

```python
from src.inference.decision_logic import rule_based_decision, get_watering_action

decision = rule_based_decision(sensor_input)
action = get_watering_action(decision)
```

## Contributors

This project was developed as part of an AI/ML Fellowship program by a team of 5 members:

| Contributor | Role |
|-------------|------|
| Team Member 1 | ML Model Development |
| Team Member 2 | Data Preprocessing & EDA |
| Team Member 3 | Backend Development |
| Team Member 4 | Frontend Development |
| Team Member 5 | Model Evaluation & Testing |

> **Note**: Please update the contributor names above with the actual team member names from your GitHub repository.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- AI/ML Fellowship Program
- Streamlit community
- scikit-learn and XGBoost developers

## Contact

For questions or contributions, please open an issue on the GitHub repository.

---

*Plant Watering Intelligence System © 2026*
