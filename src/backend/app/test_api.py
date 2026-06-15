"""
Testes de integração da API FastAPI de avaliação de crédito.

Como executar (na mesma pasta do main.py e do .pkl):
    pytest test_api.py -v

Estes testes carregam a aplicação localmente e validam o contrato da API:
status codes, formato da resposta, consistência dos resultados e
validação de entrada.
"""
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# Payload de exemplo válido (mesmos campos esperados pelo pipeline)
PAYLOAD_VALIDO = {
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
}


def test_health_check():
    """A API deve responder 200 no endpoint raiz (usado para health check)."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API rodando 🚀"}


def test_predict_payload_valido():
    """Um payload válido deve retornar 200 e os campos esperados."""
    response = client.post("/predict", json=PAYLOAD_VALIDO)
    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert "probability" in data

    assert data["prediction"] in (0, 1)
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_determinismo():
    """A mesma entrada deve sempre gerar a mesma saída (modelo determinístico)."""
    r1 = client.post("/predict", json=PAYLOAD_VALIDO)
    r2 = client.post("/predict", json=PAYLOAD_VALIDO)
    assert r1.json() == r2.json()


def test_predict_tipo_invalido_retorna_422():
    """Um tipo de dado inválido (string em campo numérico) deve retornar 422."""
    payload_invalido = PAYLOAD_VALIDO.copy()
    payload_invalido["AMT_CREDIT"] = "não é um número"

    response = client.post("/predict", json=payload_invalido)
    assert response.status_code == 422


@pytest.mark.parametrize("score_externo", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_predict_faixa_de_scores_externos(score_externo):
    """A API deve responder corretamente para toda a faixa válida de EXT_SOURCE_2/3 (0 a 1)."""
    payload = PAYLOAD_VALIDO.copy()
    payload["EXT_SOURCE_2"] = score_externo
    payload["EXT_SOURCE_3"] = score_externo

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert 0.0 <= response.json()["probability"] <= 1.0