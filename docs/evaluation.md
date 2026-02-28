# Evaluation notes

`python -m cmapss_mvp.cli evaluate --demo` prints MAE, RMSE and R2 on the deterministic demo trajectories. For a meaningful report, keep one or more complete engine IDs out of training and evaluate on the supplied C-MAPSS test/RUL files. The MVP intentionally avoids downloading data and avoids claiming benchmark scores.
