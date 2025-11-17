#!/usr/bin/env bash
set -e


# Activate venv if you use one (optional)
# source .venv/bin/activate


# Ensure data exists
if [ ! -f data/raw/bmarket.db ]; then
echo "Place your bmarket.db file at data/raw/bmarket.db"
exit 1
fi


# Run the pipeline script
python src/bmarket_pipeline/scripts/run_pipeline.py


# Save a copy of important artifacts
echo "Pipeline finished. Saved model at saved_model/model.joblib"