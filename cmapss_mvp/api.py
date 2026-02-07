"""FastAPI contract for RUL, failure-risk and anomaly sample predictions."""
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .pipeline import make_demo_data, training_features, prepare_features, SENSORS
from .model import MaintenanceModel

app=FastAPI(title="Predictive Maintenance MVP",version="0.1.0",description="CPU-friendly C-MAPSS baseline API")
_MODEL=None

def get_model():
 global _MODEL
 if _MODEL is None:
  path=Path("artifacts/model.json")
  if path.exists(): _MODEL=MaintenanceModel.load(path)
  else:
   _,X,y=training_features(make_demo_data()); _MODEL=MaintenanceModel().fit(X,y)
 return _MODEL
class Sample(BaseModel):
 cycle: float = Field(..., ge=0)
 sensors: list[float] = Field(..., min_length=21, max_length=21)
 def vector(self): return np.asarray([self.cycle,*self.sensors],float).reshape(1,-1)
class RULResponse(BaseModel): rul: float; unit: str="cycles"; model_version: str="mvp-baseline"
@app.get("/health")
def health(): return {"status":"ok","model":"loaded" if _MODEL is not None or Path("artifacts/model.json").exists() else "demo-fallback"}
@app.get("/model-card")
def model_card(): return {"name":"cmapss-mvp-baseline","task":"RUL/failure/anomaly","features":["cycle",*SENSORS],"limitations":"Demo fallback is synthetic; validate on a licensed C-MAPSS split before production."}
@app.post("/predict/rul",response_model=RULResponse)
def predict_rul(sample:Sample): return {"rul":round(float(get_model().predict_rul(sample.vector())[0]),3)}
@app.post("/predict/failure")
def predict_failure(sample:Sample):
 label,p=get_model().predict_failure(sample.vector()); return {"failure_within_30_cycles":bool(label[0]),"probability":round(float(p[0]),4),"threshold_cycles":30}
@app.post("/predict/anomaly")
def predict_anomaly(sample:Sample):
 flag,d=get_model().predict_anomaly(sample.vector()); return {"anomaly":bool(flag[0]),"score":round(float(d[0]),4),"threshold":3.0}

