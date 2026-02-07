"""CPU-friendly deterministic baseline model and metrics."""
import json
from pathlib import Path
import numpy as np

class MaintenanceModel:
    def __init__(self): self.mean=None; self.scale=None; self.weights=None; self.bias=0.; self.anomaly_mean=None; self.anomaly_scale=None
    def fit(self, X, y):
        X=np.asarray(X,float); y=np.asarray(y,float)
        self.mean=X.mean(0); self.scale=X.std(0); self.scale[self.scale < 1e-8]=1
        Z=(X-self.mean)/self.scale; A=np.c_[np.ones(len(Z)), Z]
        # Ridge keeps the demo model stable even when sensors are collinear.
        self.bias, *self.weights = np.linalg.solve(A.T@A + .1*np.eye(A.shape[1]), A.T@y)
        self.weights=np.asarray(self.weights); self.anomaly_mean=Z.mean(0); self.anomaly_scale=Z.std(0); self.anomaly_scale[self.anomaly_scale<1e-8]=1
        return self
    def _z(self,X): return (np.asarray(X,float)-self.mean)/self.scale
    def predict_rul(self,X): return np.maximum(0, self.bias + self._z(X)@self.weights)
    def predict_failure(self,X, threshold=30):
        rul=self.predict_rul(X); probability=1/(1+np.exp(np.clip((rul-threshold)/8,-50,50)))
        return (probability >= .5).astype(int), probability
    def predict_anomaly(self,X, threshold=3.0):
        distance=np.sqrt(np.mean(((self._z(X)-self.anomaly_mean)/self.anomaly_scale)**2,axis=1))
        return distance >= threshold, distance
    def save(self,path):
        p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps({"mean":self.mean.tolist(),"scale":self.scale.tolist(),"weights":self.weights.tolist(),"bias":self.bias,"anomaly_mean":self.anomaly_mean.tolist(),"anomaly_scale":self.anomaly_scale.tolist()},indent=2))
    @classmethod
    def load(cls,path):
        d=json.loads(Path(path).read_text()); m=cls()
        for k in ("mean","scale","weights","anomaly_mean","anomaly_scale"): setattr(m,k,np.asarray(d[k],float))
        m.bias=float(d["bias"]); return m

def regression_metrics(y,p):
    y=np.asarray(y); p=np.asarray(p); e=y-p
    return {"mae":float(np.mean(np.abs(e))),"rmse":float(np.sqrt(np.mean(e**2))),"r2":float(1-np.sum(e**2)/(np.sum((y-y.mean())**2)+1e-12))}
