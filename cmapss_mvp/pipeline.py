"""Canonical C-MAPSS loading, labeling and engine-level feature preparation."""
from pathlib import Path
import numpy as np
import pandas as pd

COLUMNS = ["unit_id", "cycle", "setting_1", "setting_2", "setting_3"] + [f"sensor_{i}" for i in range(1, 22)]
SENSORS = COLUMNS[5:]

def _read(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS).dropna(how="all")

def load_cmapss(data_dir="data/CMaps", dataset="FD001", split="train"):
    """Load a NASA C-MAPSS split and attach cycle-based RUL labels."""
    root = Path(data_dir); name = root / f"{split}_{dataset}.txt"
    if not name.exists(): raise FileNotFoundError(f"Missing {name}; pass --data-dir or use --demo")
    frame = _read(name)
    if split == "train":
        frame["rul"] = frame.groupby("unit_id")["cycle"].transform("max") - frame["cycle"]
    else:
        rul_path = root / f"RUL_{dataset}.txt"
        if rul_path.exists():
            end_rul = pd.read_csv(rul_path, sep=r"\s+", header=None)[0].to_numpy()
            last = frame.groupby("unit_id")["cycle"].transform("max")
            frame["rul"] = frame["unit_id"].map({i + 1: int(v) for i, v in enumerate(end_rul)}) + last - frame["cycle"]
    return frame.reset_index(drop=True)

def make_demo_data(engines=24, seed=7):
    """Create deterministic, realistic-shaped data when C-MAPSS is unavailable."""
    rng = np.random.default_rng(seed); rows=[]
    for unit in range(1, engines + 1):
        life = int(rng.integers(55, 120))
        for cycle in range(1, life + 1):
            degradation = cycle / life
            settings = rng.normal(0, .1, 3)
            sensors = rng.normal(0, 1, 21)
            sensors[2] += 3 * degradation; sensors[6] -= 2 * degradation
            rows.append([unit, cycle, *settings, *sensors, life-cycle])
    return pd.DataFrame(rows, columns=COLUMNS + ["rul"])

def prepare_features(frame, cap=125):
    """Return one latest-observation row per engine, with capped RUL and labels."""
    frame = frame.sort_values(["unit_id", "cycle"]).copy()
    if "rul" not in frame: raise ValueError("frame must contain a rul column")
    frame["rul"] = frame["rul"].clip(upper=cap).astype(float)
    latest = frame.groupby("unit_id", as_index=False).tail(1).copy()
    latest["failure_within_30"] = (latest["rul"] <= 30).astype(int)
    features = ["cycle", *SENSORS]
    return latest, latest[features].to_numpy(dtype=float), latest["rul"].to_numpy(dtype=float)

def training_features(frame, cap=125):
    """Build row-level supervised samples while retaining engine chronology."""
    frame=frame.sort_values(["unit_id","cycle"]).copy()
    if "rul" not in frame: raise ValueError("frame must contain a rul column")
    frame["rul"]=frame["rul"].clip(upper=cap).astype(float)
    return frame, frame[["cycle", *SENSORS]].to_numpy(float), frame["rul"].to_numpy(float)
