from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from app.schemas import PredictionRequest
from app.model import model, model_columns

app = FastAPI(
    title="API de Predição XGBoost",
    description="API para prever com modelo treinado",
    version="1.0.0"
)

origins = [
    "https://creditanalysis-t2-g1.netlify.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "API rodando 🚀"}

@app.post("/predict")
def predict(data: PredictionRequest):

    # transformar em dict
    input_dict = data.dict()

    # criando o DataFrame
    df = pd.DataFrame([input_dict])

    # preencher dados que nao vem no POST com 0
    df = df.reindex(columns=model_columns, fill_value=0)

    # predicao
    prediction = model.predict(df)[0]

    # 5. probabilidade
    try:
        proba = model.predict_proba(df)[0][1]
    except:
        proba = None

    return {
        "prediction": int(prediction),
        "probability": float(proba) if proba is not None else None
    }