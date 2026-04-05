


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

### 2.2 Pairplot

O pairplot compara as principais variáveis financeiras numéricas duas a duas, segmentado pela classificação de risco (`TARGET`).

![Gráficos Pairplot](https://github.com/ICEI-PUC-Minas-PMV-SI/PUC_E7_T2_G1_2026_01/raw/main/docs/img/Pairplot.png)

![Gráficos Pairplot “Balanceado”](https://github.com/ICEI-PUC-Minas-PMV-SI/PUC_E7_T2_G1_2026_01/raw/main/docs/img/Gr%C3%A1ficos%20Pairplot%20%E2%80%9CBalanceado%E2%80%9D.png)

| Variáveis comparadas | O que se observa |
|----------------------|-----------------|
| `AMT_CREDIT × AMT_GOODS_PRICE` | Correlação quase linear — o valor do crédito acompanha o preço do bem financiado |
| `AMT_CREDIT × AMT_ANNUITY` | Correlação positiva moderada — créditos maiores tendem a gerar anuidades maiores |
| `AMT_INCOME_TOTAL × AMT_CREDIT` | Fraca correlação — clientes de diferentes rendas solicitam valores similares de crédito |
| `DAYS_BIRTH × AMT_CREDIT` | Distribuição espalhada — a idade não determina diretamente o valor do crédito |
| `DAYS_EMPLOYED × demais` | Presença de valores extremos (~365.000 dias), indicando anomalia nos dados |

> Os inadimplentes não formam regiões claramente separadas, reforçando a necessidade de técnicas de balanceamento e modelos mais sofisticados.

---

### 2.3 Boxplots

![Boxplots](https://github.com/ICEI-PUC-Minas-PMV-SI/PUC_E7_T2_G1_2026_01/raw/main/docs/img/Boxplot1.png)

![Boxplots](https://github.com/ICEI-PUC-Minas-PMV-SI/PUC_E7_T2_G1_2026_01/raw/main/docs/img/Boxplot2.png)

![Boxplots](https://github.com/ICEI-PUC-Minas-PMV-SI/PUC_E7_T2_G1_2026_01/raw/main/docs/img/Boxplot3png)

![Boxplots](https://github.com/ICEI-PUC-Minas-PMV-SI/PUC_E7_T2_G1_2026_01/raw/main/docs/img/Boxplot4.png)

![Boxplots](https://github.com/ICEI-PUC-Minas-PMV-SI/PUC_E7_T2_G1_2026_01/raw/main/docs/img/Boxplot5.png)

| Variável | O que se observa |
|----------|-----------------|
| `AMT_CREDIT` | Medianas praticamente iguais entre as classes. Baixo poder discriminatório isolado. |
| `AMT_ANNUITY` | Distribuições muito similares. Pouco poder de separação isolado. |
| `AMT_INCOME_TOTAL` | Outlier extremo (~1,2 × 10⁸) na classe inadimplente. Exige tratamento. |
| `AMT_GOODS_PRICE` | Comportamento muito semelhante ao `AMT_CREDIT`. Confirma alta correlação. |
| `DAYS_BIRTH` | **Maior diferença visual entre as classes.** Inadimplentes são mais jovens (mediana ~38 anos vs ~44 anos). |
| `DAYS_EMPLOYED` | Confirma a anomalia: outliers em ~365.000 dias. Exige tratamento obrigatório. |

---




---

### 2.4 Histogramas

![Histogramas](https://github.com/ICEI-PUC-Minas-PMV-SI/PUC_E7_T2_G1_2026_01/raw/main/docs/img/Histograma1.png)

![Histogramas](https://github.com/ICEI-PUC-Minas-PMV-SI/PUC_E7_T2_G1_2026_01/raw/main/docs/img/Histograma2.png)

![Histogramas](https://github.com/ICEI-PUC-Minas-PMV-SI/PUC_E7_T2_G1_2026_01/raw/main/docs/img/Histograma3.png)


- `AMT_CREDIT` e `AMT_ANNUITY`: distribuições assimétricas à direita, com sobreposição quase completa entre as classes — **baixo poder discriminatório isolado**.
- `DAYS_BIRTH`: inadimplentes concentram-se em idades mais jovens.
- `AMT_INCOME_TOTAL`: inadimplentes tendem a se concentrar em faixas de renda mais baixa.
- `DAYS_EMPLOYED`: anomalia dos ~365.000 dias confirmada nas duas classes.

---

### 2.5 Matriz de Correlação

![Matriz de Correlação](https://github.com/ICEI-PUC-Minas-PMV-SI/PUC_E7_T2_G1_2026_01/raw/main/docs/img/Matriz%20de%20Correla%C3%A7%C3%A3o%20(Mapa%20de%20Calor).png)

| Par de Variáveis | Correlação | Interpretação |
|------------------|------------|---------------|
| `AMT_CREDIT × AMT_GOODS_PRICE` | **0.99** | Multicolinearidade gravíssima — variáveis praticamente idênticas |
| `AMT_CREDIT × AMT_ANNUITY` | **0.77** | Correlação forte — esperada no contexto de crédito |
| `AMT_ANNUITY × AMT_GOODS_PRICE` | **0.78** | Correlação forte pelo mesmo motivo |
| `DAYS_BIRTH × DAYS_EMPLOYED` | **-0.62** | Correlação negativa moderada — clientes mais velhos têm mais tempo de emprego |
| `DAYS_BIRTH × TARGET` | **+0.08** | Clientes mais jovens têm leve tendência à inadimplência |

> **Nenhuma variável apresenta correlação linear significativa com o TARGET**, indicando que os padrões de risco são **não lineares e complexos** — o que reforça a escolha de algoritmos como Random Forest e XGBoost.

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
