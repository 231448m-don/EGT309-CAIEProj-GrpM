#!/usr/bin/env bash
set -e  # stop if any script fails

echo "=== Running Logistic Regression ==="
python src/logistic_regression.py

echo "=== Running Random Forest ==="
python src/random_forest.py

echo "=== Running XGBoost Model ==="
python src/xgboostt.py

