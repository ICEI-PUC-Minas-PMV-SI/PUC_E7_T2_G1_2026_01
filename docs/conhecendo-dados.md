


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







---

### 2.6 Variance Inflation Factor (VIF)

O VIF mede o quanto a variância de um coeficiente é inflada devido à multicolinearidade.

| Variável           | VIF       | Decisão                                   |
| ------------------ | --------- | ----------------------------------------- |
| `AMT_GOODS_PRICE`  | **39.51** | ⚠️ Remover — multicolinearidade gravíssima |
| `AMT_CREDIT`       | **38.77** | ⚠️ Manter — variável central do problema   |
| `AMT_ANNUITY`      | 2.56      | ✅ Manter                                  |
| `DAYS_EMPLOYED`    | 1.64      | ✅ Manter                                  |
| `DAYS_BIRTH`       | 1.64      | ✅ Manter                                  |
| `AMT_INCOME_TOTAL` | 1.04      | ✅ Manter                                  |

---

## 3. Engenharia de Dados

### 3.1 Tratamento de Valores Nulos

O tratamento foi realizado em três etapas:

**Etapa 1 — Remoção de colunas com > 40% de nulos**

- Foram removidas **49 colunas**, em sua maioria relacionadas a características físicas dos imóveis dos clientes (`_AVG`, `_MODE`, `_MEDI`).
- O dataset reduziu de **123 para 74 colunas**.

**Etapa 2 — Imputação numérica pela mediana**

| Coluna                   | Mediana utilizada | Observação                             |
| ------------------------ | ----------------- | -------------------------------------- |
| `AMT_ANNUITY`            | 24.903,00         | Apenas 12 nulos — impacto mínimo       |
| `EXT_SOURCE_2`           | 0,57              | Score externo importante para o modelo |
| `EXT_SOURCE_3`           | 0,54              | Score externo importante para o modelo |
| `CNT_FAM_MEMBERS`        | 2,00              | Apenas 2 nulos                         |
| `DAYS_LAST_PHONE_CHANGE` | -757,00           | Apenas 1 nulo                          |

**Etapa 3 — Imputação categórica com `'desconhecido'`**

- `NAME_TYPE_SUITE`: tipo de acompanhante na solicitação.
- `OCCUPATION_TYPE`: ocupação do cliente — a categoria `'desconhecido'` preserva a informação de que o cliente não declarou ocupação, o que pode ser um sinal de risco.

> **Resultado: 0 valores nulos restantes. Shape final: 307.511 × 74 colunas.**

---

### 3.2 Registros Duplicados

```
Registros antes:          307.511
Duplicados encontrados:         0  (0,0000%)
Registros após remoção:   307.511
```

---

### 3.3 Anomalia em DAYS_EMPLOYED

A variável `DAYS_EMPLOYED` possui valores próximos a **365.243 dias** (~1.000 anos de emprego), o que é impossível na prática. Esse valor é uma **codificação especial** para representar clientes **aposentados ou sem vínculo empregatício formal**.

**Solução:** substituição por `NaN` e imputação pela mediana dos valores válidos.

```python
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
mediana = df['DAYS_EMPLOYED'].median()
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].fillna(mediana)
```

---

### 3.4 Remoção de Variável Redundante

Com base no VIF (**39.51**) e na correlação de **0.99** com `AMT_CREDIT`, a variável `AMT_GOODS_PRICE` foi removida por ser redundante, evitando multicolinearidade na modelagem.

```
Shape final após pré-processamento: 307.511 × 73 colunas
```

---
