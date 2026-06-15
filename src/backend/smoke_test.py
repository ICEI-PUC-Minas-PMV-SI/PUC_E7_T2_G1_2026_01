"""
Smoke test do ambiente em produção (Azure App Service).

Faz chamadas reais à API já implantada e verifica se ela está respondendo
corretamente. Útil para validar o ambiente logo após cada deploy.

Como executar:
    python smoke_test.py https://creditanalysis-ayahd8b7bpehgmat.eastus-01.azurewebsites.net
"""

import sys
import time
import requests

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


def main(base_url):
    base_url = base_url.rstrip("/")
    falhas = []

    # 1. Health check
    print(f"[1/3] GET {base_url}/")
    inicio = time.time()
    r = requests.get(f"{base_url}/", timeout=30)
    duracao = time.time() - inicio
    print(f"      status={r.status_code} tempo={duracao:.3f}s")
    if r.status_code != 200:
        falhas.append("Health check não retornou 200")

    # 2. Predição com payload válido
    print(f"[2/3] POST {base_url}/predict (payload válido)")
    inicio = time.time()
    r = requests.post(f"{base_url}/predict", json=PAYLOAD_VALIDO, timeout=30)
    duracao = time.time() - inicio
    print(f"      status={r.status_code} tempo={duracao:.3f}s")
    print(f"      resposta={r.json() if r.status_code == 200 else r.text}")
    if r.status_code != 200:
        falhas.append("/predict não retornou 200 para payload válido")
    else:
        data = r.json()
        if not (0.0 <= data.get("probability", -1) <= 1.0):
            falhas.append("probability fora da faixa [0,1]")

    # 3. Validação de entrada inválida (campo obrigatório faltando)
    print(f"[3/3] POST {base_url}/predict (payload inválido)")
    payload_invalido = PAYLOAD_VALIDO.copy()
    del payload_invalido["AMT_CREDIT"]
    r = requests.post(f"{base_url}/predict", json=payload_invalido, timeout=30)
    print(f"      status={r.status_code}")
    if r.status_code != 422:
        falhas.append("Payload inválido deveria retornar 422")

    print("\n" + "=" * 40)
    if falhas:
        print(f"FALHOU - {len(falhas)} problema(s) encontrado(s):")
        for f in falhas:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("OK - todos os testes de smoke passaram")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python smoke_test.py https://creditanalysis-ayahd8b7bpehgmat.eastus-01.azurewebsites.net/")
        sys.exit(1)
    main(sys.argv[1])