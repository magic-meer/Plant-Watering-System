# Data Directory Guide

## Structure

- `raw/`: unmodified source datasets.
- `processed/`: cleaned data and model-ready train/test splits.

## Files and Naming

### Raw
- `raw/original_dataset.csv`
  - Original dataset used by the project.

### Processed
- `processed/cleaned_data.csv` — cleaned full dataset before split.
- `processed/X_train.csv` — training features.
- `processed/X_test.csv` — test features.
- `processed/y_train.csv` — training labels.
- `processed/y_test.csv` — test labels.

## Expected Core Columns

- `Timestamp`
- `Plant_ID`
- `Soil_Moisture`
- `Ambient_Temperature`
- `Soil_Temperature`
- `Humidity`
- `Light_Intensity`
- `Soil_pH`
- `Nitrogen_Level`
- `Phosphorus_Level`
- `Potassium_Level`
- `Chlorophyll_Content`
- `Electrochemical_Signal`
- `days_since_last_watering`
- `watering_sma_3`
- `Plant_Health_Status` (target)

## Sample Rows

Use this command to inspect sample rows:

```bash
python - <<'PY'
import pandas as pd
print(pd.read_csv('data/raw/original_dataset.csv').head(3).to_string(index=False))
PY
```

## Notes

- Keep raw files immutable.
- Only write transformed artifacts into `processed/`.
- If you add a new dataset, document source, license, and schema changes in this file.
