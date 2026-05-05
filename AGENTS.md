# Repository Guidelines

## Project Structure & Module Organization
`data/` stores raw and processed macro/FX datasets plus ingestion and cleaning scripts in `data/process_scripts/`. `analysis/` contains the research code by stage: baseline and anomaly detection, SHAP and M2 component analysis, LSTM/Hybrid forecasting, and `fx_impact/` for downstream macro-impact modeling. `reports/` holds Markdown and Typst deliverables, presentation outlines, and exported PDFs. Root files such as `README.md` and `requirements.txt` describe the main workflow and Python dependencies.

## Build, Test, and Development Commands
Create an environment and install dependencies:
```bash
pip install -r requirements.txt
```
Run the main preprocessing pipeline from the repo root:
```bash
python data/process_scripts/process_all_indicators.py
python data/process_scripts/rebuild_daily_pipeline.py
```
Run representative analysis stages:
```bash
python analysis/baseline/analyze_factors.py
python analysis/anomaly/detect_anomaly_period.py
python analysis/LSTM/lstm_m2_demand_deposit/train_eval_extended.py
python analysis/fx_impact/predict_varx.py
```
Rebuild a report PDF when editing Typst sources:
```bash
typst compile reports/report_mid.typ
```

## Coding Style & Naming Conventions
This repository is Python-first. Use 4-space indentation, `snake_case` for functions/files, and clear column names that match dataset semantics. Prefer `pathlib.Path` and relative path derivation from `__file__` for new code; several legacy scripts still hardcode `/Applications/dollar_price`, and new changes should avoid expanding that pattern. Keep generated artifacts inside the relevant `analysis/*/`, `data/`, or `reports/` subdirectory.

## Testing Guidelines
There is no formal `pytest` suite or coverage gate. Validation is script-driven: rerun the pipeline or model you changed and inspect regenerated CSV, JSON, PNG, or PDF outputs. Files named `test_*.py` under `data/process_scripts/` are manual API/data probes, not CI-ready unit tests. For data changes, verify the affected processed dataset; for modeling changes, check `results.json`, `predictions.csv`, and evaluation plots.

## Commit & Pull Request Guidelines
Recent history uses short, scope-aware subjects such as `fx_impact analysis`, `causality analysis`, and `chore: reorganizing files and data`. Follow that pattern with concise imperative summaries, preferably `<scope>: <change>`. In PRs, state which pipeline or analysis stage changed, whether outputs were regenerated, and include updated plots or report screenshots when visual results moved. Never commit secrets from `.env`; ECOS API usage should stay local.
