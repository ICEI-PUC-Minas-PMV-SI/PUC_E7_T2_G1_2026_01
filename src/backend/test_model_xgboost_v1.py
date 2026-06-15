"""
Testes unitários do pipeline/modelo serializado (.joblib).

Como executar:
    pytest test_model_pipeline.py -v

Estes testes validam o artefato do modelo isoladamente, sem depender da API
ou do deploy.
"""

import joblib
import pandas as pd
import pytest

from app.model import model_columns

CAMINHO_MODELO = "pipeline_xgboost_v1.joblib"

FEATURES_ESPERADAS = [
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "DAYS_BIRTH",
    "AMT_CREDIT",
    "DAYS_EMPLOYED",
    "AMT_ANNUITY",
    "AMT_INCOME_TOTAL",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
]


@pytest.fixture(scope="module")
def modelo():
    return joblib.load(CAMINHO_MODELO)


@pytest.fixture
def cliente_exemplo():
    return pd.DataFrame([{
        "EXT_SOURCE_2": 0.60,
        "EXT_SOURCE_3": 0.55,
        "DAYS_BIRTH": -14000,
        "AMT_CREDIT": 400000,
        "DAYS_EMPLOYED": -2200,
        "AMT_ANNUITY": 22000,
        "AMT_INCOME_TOTAL": 180000,
        "DAYS_REGISTRATION": -3800,
        "DAYS_ID_PUBLISH": -1900,
        "DAYS_LAST_PHONE_CHANGE": -900,
    }]).reindex(columns=model_columns, fill_value=0)


def test_modelo_carrega_sem_erro(modelo):
    """O arquivo .joblib deve ser carregado sem lançar exceção."""
    assert modelo is not None


def test_predict_retorna_classe_valida(modelo, cliente_exemplo):
    """A predição deve retornar 0 (bom pagador) ou 1 (inadimplente)."""
    classe = modelo.predict(cliente_exemplo)
    assert classe[0] in (0, 1)


def test_predict_proba_soma_um(modelo, cliente_exemplo):
    """As probabilidades das duas classes devem somar 1."""
    proba = modelo.predict_proba(cliente_exemplo)[0]
    assert len(proba) == 2
    assert abs(sum(proba) - 1.0) < 1e-6


def test_predict_com_valores_nulos(modelo):
    """O pipeline deve lidar com valores ausentes (imputação pela mediana)."""
    cliente_com_nulos = pd.DataFrame([{
        "EXT_SOURCE_2": None,
        "EXT_SOURCE_3": 0.55,
        "DAYS_BIRTH": -14000,
        "AMT_CREDIT": 400000,
        "DAYS_EMPLOYED": -2200,
        "AMT_ANNUITY": 22000,
        "AMT_INCOME_TOTAL": 180000,
        "DAYS_REGISTRATION": -3800,
        "DAYS_ID_PUBLISH": -1900,
        "DAYS_LAST_PHONE_CHANGE": -900,
    }]).reindex(columns=model_columns, fill_value=0)

    # Não deve lançar exceção mesmo com valor ausente
    proba = modelo.predict_proba(cliente_com_nulos)
    assert 0.0 <= proba[0][1] <= 1.0

def test_coluna_faltante_lanca_erro(modelo, cliente_exemplo):
    """Remover uma coluna esperada deve causar erro (falha explícita, não silenciosa)."""
    cliente_incompleto = cliente_exemplo.drop(columns=["EXT_SOURCE_2"])
    with pytest.raises(Exception):
        modelo.predict_proba(cliente_incompleto)