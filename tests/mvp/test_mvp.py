import numpy as np
from cmapss_mvp.pipeline import make_demo_data, prepare_features, training_features
from cmapss_mvp.model import MaintenanceModel

def test_demo_pipeline_is_deterministic_and_grouped():
    frame=make_demo_data(engines=4,seed=11); latest,X,y=prepare_features(frame)
    assert len(latest)==4 and X.shape==(4,22) and np.all(y<=125)
    assert latest.unit_id.is_unique

def test_model_roundtrip_and_predictions(tmp_path):
    _,X,y=training_features(make_demo_data(engines=8)); m=MaintenanceModel().fit(X,y)
    path=tmp_path/'model.json'; m.save(path); loaded=MaintenanceModel.load(path)
    assert np.allclose(m.predict_rul(X),loaded.predict_rul(X))
    failure,prob=loaded.predict_failure(X); anomaly,score=loaded.predict_anomaly(X)
    assert failure.shape==prob.shape==anomaly.shape==score.shape==X.shape[:1]

