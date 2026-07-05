# Battery RUL Prediction Project

Predicting State of Health (SOH) and Remaining Useful Life (RUL) for lithium-ion
batteries using the NASA Battery Data Set (B0005, B0006, B0007, B0018).

## Pipeline

Run the notebooks in this order:

1. **`second_attempt.ipynb`**: Initial exploration of the raw `.mat` files.
   Cycle types (charge/discharge/impedance), field structure, value ranges,
   and data quality observations (outliers, NaNs, truncated cycles).

2. **`data_cleaning.ipynb`**: Cleans the raw data based on the issues found
   during exploration.
   - Drops the malformed final cycle (idx 615) for B0005/6/7
   - Drops the voltage spike outlier cycle (idx 84) for B0005/6/7
   - Fixes 2 NaN points in B0018 (cycle 114) via linear interpolation
   - Resamples every charge/discharge cycle to a fixed length (300 points)
   - Saves `clean_battery_data.pkl`

3. **`label_creation.ipynb`**: Computes the target variables.
   - **SOH** = capacity at cycle / initial rated capacity
   - **EOL** = first cycle where SOH drops to 70% or below
   - **RUL** = cycles remaining until EOL
   - Saves `labeled_battery_data.pkl`

4. **`feature_engineering.ipynb`**: Extracts scalar features from the
   voltage/current/temperature arrays of each cycle (mean, std, min, max,
   slope, range, temperature rise) plus cycle to cycle delta features.
   Saves `feature_battery_data.pkl`

5. **`baseline_models.ipynb`**: Trains 5 baseline regressors (Linear, Ridge,
   Lasso, Polynomial, SVR) to predict SOH and compares RMSE/MAE/R2.

6. **`ensemble_models.ipynb`**: Trains and tunes Random Forest and XGBoost
   (via GridSearchCV) for SOH, checks for feature leakage (`capacity_delta`),
   and repeats the full model comparison for the RUL target.

## Data flow

```
raw .mat files
   -> data_cleaning.ipynb       -> clean_battery_data.pkl
   -> label_creation.ipynb      -> labeled_battery_data.pkl
   -> feature_engineering.ipynb -> feature_battery_data.pkl
   -> baseline_models.ipynb     -> baseline_model_results.csv
   -> ensemble_models.ipynb     -> final_model_comparison.csv, rul_model_results.csv
```

## Notes

- Paths are currently hardcoded to `C:/Users/santh/battery-rul-project/...`.
  Update these if running on a different machine.
- `capacity_delta` was found to leak target information into the SOH model
  and was removed from the feature set used for RUL prediction.
