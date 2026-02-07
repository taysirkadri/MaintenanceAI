# MaintenanceAI

CPU-friendly predictive-maintenance baseline for NASA C-MAPSS. It keeps the existing streaming services intact while providing one canonical, reproducible preprocessing/training/evaluation path.

## Quickstart

```powershell
python -m pip install -r requirements-mvp.txt
python -m cmapss_mvp.cli train --demo
uvicorn cmapss_mvp.api:app --host 0.0.0.0 --port 8000
```

Use a local, licensed C-MAPSS archive without downloading data:
`python -m cmapss_mvp.cli train --data-dir data/CMaps --dataset FD001`.
The loader expects `train_FD001.txt` (and optionally `test_FD001.txt`, `RUL_FD001.txt`). If absent, `--demo` generates deterministic synthetic trajectories and never writes a dataset.

API docs: `http://localhost:8000/docs`. POST `/predict/rul`, `/predict/failure`, and `/predict/anomaly` with `{"cycle": 80, "sensors": [0, ... 0]}` (exactly 21 values). `/health` is unauthenticated for container probes.

## Reproducibility and limitations

The baseline uses engine-grouped latest observations, capped RUL (125 cycles), ridge regression, a 30-cycle failure threshold, and standardized-distance anomaly detection. This is intentionally interpretable and CPU-friendly, not a production safety model. Evaluate on a held-out engine-grouped split before deployment; thresholds and synthetic fallback are not evidence of benchmark performance.

NASA C-MAPSS attribution: Saxena, Goebel, Simon and Eklund, PHM08, NASA Prognostics Data Repository. Restricted datasets and generated model artifacts are not committed.
