"""Train/evaluate the canonical baseline: python -m cmapss_mvp.cli train --demo"""
import argparse, json
from pathlib import Path
from .pipeline import load_cmapss, make_demo_data, training_features
from .model import MaintenanceModel, regression_metrics

def main():
 p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="command",required=True)
 for cmd in ("train","evaluate"):
  q=sub.add_parser(cmd); q.add_argument("--data-dir",default="data/CMaps"); q.add_argument("--dataset",default="FD001"); q.add_argument("--model",default="artifacts/model.json"); q.add_argument("--demo",action="store_true")
 a=p.parse_args(); frame=make_demo_data() if a.demo else load_cmapss(a.data_dir,a.dataset)
 latest,X,y=training_features(frame); model=MaintenanceModel().fit(X,y)
 if a.command=="train": model.save(a.model); print(json.dumps({"model":a.model,"engines":len(latest),"features":X.shape[1]},indent=2))
 else:
  print(json.dumps(regression_metrics(y,model.predict_rul(X)),indent=2))
if __name__ == "__main__": main()

