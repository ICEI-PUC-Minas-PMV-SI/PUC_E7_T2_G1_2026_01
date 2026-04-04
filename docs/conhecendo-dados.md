


---

## 📦 Dataset

| Campo | Detalhe |
|---|---|
| **Fonte** | [Kaggle – Credit Analysis Dataset](https://www.kaggle.com/datasets/kapoorshivam/credit-analysis/data) |
| **Registros** | ~307.511 clientes |
| **Colunas originais** | 122 features + 1 target |
| **Variável alvo** | `TARGET` (0 = Bom Pagador / 1 = Inadimplente) |
| **Desbalanceamento** | ~91,93% bons pagadores / ~8,07% inadimplentes |

> 📓 Este estudo foi desenvolvido originalmente no Google Colab. Você pode acessar o notebook interativo completo, com todas as células executadas e visualizações geradas, pelo link abaixo:
>
> 🔗 [Abrir no Google Colab](https://colab.research.google.com/drive/1p_bgm8KbK72PYLjoUXXOBhnsmbQv1uPS?usp=sharing)

### Colunas principais

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `TARGET` | int | Risco de crédito: `0` = Bom Pagador / `1` = Inadimplente |
| `EXT_SOURCE_2` | float | Score externo de crédito (fonte 2) |
| `EXT_SOURCE_3` | float | Score externo de crédito (fonte 3) |
| `DAYS_BIRTH` | int | Dias desde o nascimento (negativo — relativo à data da solicitação) |
| `AMT_CREDIT` | float | Valor do crédito solicitado |
| `DAYS_EMPLOYED` | int | Dias desde o início do emprego atual (negativo) |
| `AMT_ANNUITY` | float | Valor da anuidade do crédito |
| `AMT_INCOME_TOTAL` | float | Renda total declarada do cliente |
| `DAYS_REGISTRATION` | int | Dias desde o último registro (negativo) |
| `DAYS_ID_PUBLISH` | int | Dias desde a publicação do documento de identidade (negativo) |
| `DAYS_LAST_PHONE_CHANGE` | int | Dias desde a última troca de telefone (negativo) |

> **Nota sobre variáveis temporais negativas:** as colunas com prefixo `DAYS_` armazenam o tempo como número de dias relativos à data da solicitação, onde `hoje = 0`. Eventos passados são representados por valores negativos. Exemplo: `DAYS_BIRTH = -14.600` → cliente com ~40 anos de idade.

---

## 1. Coleta dos Dados

O download é realizado automaticamente via `kagglehub`, sem necessidade de baixar o arquivo manualmente:

```python
import kagglehub

path = kagglehub.dataset_download("kapoorshivam/credit-analysis")
df = pd.read_csv(os.path.join(path, 'current_app.csv'))
```

---
