"""
Teste de carga (load test) com Locust, para validar na prática as
premissas de capacidade do sistema (taxa de pico ~4,4 req/s e tempo de
resposta por requisição).

Como executar (interface web):
    locust -f locustfile.py --host https://creditanalysis-ayahd8b7bpehgmat.eastus-01.azurewebsites.net

Depois, abra http://localhost:8089 no navegador, defina o número de
usuários simulados e a taxa de spawn, e acompanhe em tempo real o
tthroughput (req/s), o tempo de resposta (p50/p95/p99) e a taxa de erros.

Como executar (linha de comando, sem interface, por X minutos):
    locust -f locustfile.py --host https://creditanalysis-ayahd8b7bpehgmat.eastus-01.azurewebsites.net \
           --users 20 --spawn-rate 2 --run-time 5m --headless
"""

from locust import HttpUser, task, between

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


class UsuarioSistemaInterno(HttpUser):
    """Simula um sistema interno do banco chamando a API de avaliação de crédito."""

    # Tempo de espera entre requisições de um mesmo "usuário" simulado
    wait_time = between(0.5, 2)

    @task(9)
    def predict(self):
        self.client.post("/predict", json=PAYLOAD_VALIDO)

    @task(1)
    def health_check(self):
        self.client.get("/")