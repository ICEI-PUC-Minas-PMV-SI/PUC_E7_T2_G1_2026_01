import joblib

MODEL_PATH = "pipeline_xgboost_v1.joblib"

model = joblib.load(MODEL_PATH)
model_columns = model.feature_names_in_