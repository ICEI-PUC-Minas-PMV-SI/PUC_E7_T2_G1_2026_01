# Análise de Crédito — API de Predição XGBoost

API REST construída com **FastAPI** que expõe um modelo de Machine Learning (**XGBoost**)
para avaliação de risco de crédito. Dado um conjunto de características de um cliente, a API
retorna se ele tende a ser um **bom pagador (0)** ou **inadimplente (1)**, junto com a probabilidade
associada.

O modelo é servido a partir de um pipeline serializado (`pipeline_xgboost_v1.joblib`) que já
inclui as etapas de pré-processamento (imputação de valores ausentes) e a predição.

---

## 🧱 Stack

- **Python 3.11**
- **FastAPI** + **Uvicorn** — servidor da API
- **scikit-learn 1.6.1** + **XGBoost 3.2.0** — pipeline de ML
- **pandas** / **joblib** — manipulação de dados e carregamento do modelo
- **pytest** — testes unitários e de integração
- **Locust** — testes de carga
- Deploy contínuo no **Azure App Service** via GitHub Actions

---

## 📁 Estrutura do projeto

```
analise-credito-eixo7/
├── app/
│   ├── main.py            # Aplicação FastAPI (endpoints / e /predict)
│   ├── model.py           # Carregamento do modelo .joblib
│   ├── schemas.py         # Schema de entrada (Pydantic)
│   └── test_api.py        # Testes de integração da API
├── pipeline_xgboost_v1.joblib   # Modelo treinado (pipeline completo)
├── test_model_xgboost_v1.py     # Testes unitários do modelo/pipeline
├── smoke_test.py          # Smoke test contra o ambiente já implantado
├── locustfile.py          # Teste de carga (Locust)
├── startup.sh             # Comando de inicialização (usado no Azure)
├── requirements.txt
└── .github/workflows/     # Pipeline de build & deploy (Azure)
```

---

## 🚀 Como baixar e executar

### 1. Clonar o repositório

```bash
git clone https://github.com/systemagic-91/analise-credito-eixo7.git
cd analise-credito-eixo7
```

### 2. Criar e ativar um ambiente virtual

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows (Git Bash)
source .venv/Scripts/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Subir a API

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

> O script `startup.sh` executa esse mesmo comando (sem `--reload`) e é o ponto de entrada
> usado no deploy do Azure.

A API estará disponível em **http://127.0.0.1:8000**. A documentação interativa (Swagger UI)
é gerada automaticamente em **http://127.0.0.1:8000/docs**.

---

## 📡 Endpoints

### `GET /`
Health check. Retorna `200` quando a API está no ar.

```json
{ "message": "API rodando 🚀" }
```

### `POST /predict`
Recebe as características do cliente e retorna a predição.

**Exemplo de requisição:**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "EXT_SOURCE_2": 0.60,
    "EXT_SOURCE_3": 0.55,
    "DAYS_BIRTH": -14000,
    "AMT_CREDIT": 400000,
    "DAYS_EMPLOYED": -2200,
    "AMT_ANNUITY": 22000,
    "AMT_INCOME_TOTAL": 180000,
    "DAYS_REGISTRATION": -3800,
    "DAYS_ID_PUBLISH": -1900,
    "DAYS_LAST_PHONE_CHANGE": -900
  }'
```

**Exemplo de resposta:**

```json
{
  "prediction": 0,
  "probability": 0.087
}
```

| Campo         | Descrição                                                     |
|---------------|---------------------------------------------------------------|
| `prediction`  | `0` = bom pagador · `1` = inadimplente                         |
| `probability` | Probabilidade de inadimplência (classe `1`), entre `0` e `1`  |

Um payload com tipo inválido ou campo obrigatório faltando retorna **`422 Unprocessable Entity`**.

---

## 🧪 Como rodar os testes

Com o ambiente virtual ativado e as dependências instaladas:

### Testes unitários do modelo/pipeline

Validam o artefato `.joblib` isoladamente (carregamento, classes válidas, probabilidades,
tratamento de valores ausentes). Execute a partir da **raiz do projeto**:

```bash
pytest test_model_xgboost_v1.py -v
```

### Testes de integração da API

Validam o contrato da API (status codes, formato da resposta, determinismo e validação de
entrada) usando o `TestClient` do FastAPI. Execute a partir da pasta `app/`:

```bash
cd app
pytest test_api.py -v
```

### Rodar todos os testes de uma vez

```bash
pytest -v
```

---

## 🔥 Teste de carga (opcional)

Usando **Locust**, é possível simular múltiplos clientes chamando a API para validar
throughput e tempo de resposta.

**Modo interativo (interface web):**

```bash
locust -f locustfile.py --host http://127.0.0.1:8000
```

Depois abra **http://localhost:8089**, defina o número de usuários e a taxa de spawn.

**Modo headless (sem interface, por tempo definido):**

```bash
locust -f locustfile.py --host http://127.0.0.1:8000 \
       --users 20 --spawn-rate 2 --run-time 5m --headless
```

---

## ✅ Smoke test (ambiente implantado)

Valida rapidamente um ambiente já no ar (ex.: logo após um deploy), fazendo chamadas reais
de health check, predição e validação de entrada:

```bash
python smoke_test.py https://<sua-url-do-azure>
```

---

## ⚙️ Deploy

O deploy é automatizado via **GitHub Actions** (`.github/workflows/main_creditanalysis.yml`):
todo push na branch `main` dispara o build (Python 3.11 + instalação das dependências) e o
deploy no **Azure App Service**, que executa a aplicação através do `startup.sh`.

---

## 📄 Licença

Distribuído sob a licença **MIT**. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
