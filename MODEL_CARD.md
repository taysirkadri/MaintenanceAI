# Model card: C-MAPSS MVP baseline

- **Intended use:** educational/demo RUL, failure-within-30-cycles and anomaly screening.
- **Data:** NASA C-MAPSS train files when supplied by the operator; deterministic synthetic fallback only for smoke tests.
- **Method:** latest engine observation, cycle plus 21 sensors, capped RUL, standardized ridge regression; failure probability is a calibrated-looking sigmoid heuristic, anomaly is standardized distance.
- **Limitations:** no claim of safety, calibration or benchmark performance; synthetic fallback is not representative; operating-condition shift and sensor missingness need validation.
- **Human oversight:** use predictions as decision support; inspect sensor quality and maintenance records before action.
- **License/attribution:** preserve the C-MAPSS/NASA terms supplied with the archive. This repository does not redistribute data.
