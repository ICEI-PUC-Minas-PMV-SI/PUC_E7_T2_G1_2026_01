# Predição de Inadimplência em Análise de Crédito

Este projeto aplica técnicas de ciência de dados e aprendizado de máquina para desenvolver
um modelo preditivo de risco de crédito, utilizando o dataset Credit Analysis, disponível
na plataforma Kaggle.

Vamos explorar, pré-processar e modelar dados do perfil financeiro de clientes para treinar
um modelo capaz de prever o risco de inadimplência com base em variáveis como:

- Idade do cliente em dias (`DAYS_BIRTH`)
- Tempo de emprego em dias (`DAYS_EMPLOYED`)
- Valor do crédito solicitado (`AMT_CREDIT`)
- Valor da anuidade (`AMT_ANNUITY`)
- Renda total declarada (`AMT_INCOME_TOTAL`)
- Scores externos de crédito (`EXT_SOURCE_2`, `EXT_SOURCE_3`)

Ao final queremos saber o seguinte: "Dado o perfil financeiro de um cliente, qual é a
probabilidade de ele ser classificado como bom ou mau pagador?"

> 📓 Este estudo foi desenvolvido originalmente no Google Colab. Você pode acessar o notebook interativo completo, com todas as células executadas e visualizações geradas, pelo link abaixo:
>
> 🔗 [Abrir no Google Colab](https://colab.research.google.com/drive/1p_bgm8KbK72PYLjoUXXOBhnsmbQv1uPS?usp=sharing)
---

## 🎯 Objetivo

> *"Dado o perfil financeiro de um cliente, qual é a probabilidade de ele ser classificado como bom ou mau pagador?"*

O projeto desenvolve um modelo preditivo capaz de identificar o risco de inadimplência com base em variáveis como:

- Idade do cliente (`DAYS_BIRTH`)
- Valor do crédito solicitado (`AMT_CREDIT`)
- Anuidade do crédito (`AMT_ANNUITY`)
- Renda total (`AMT_INCOME_TOTAL`)
- Tempo de emprego (`DAYS_EMPLOYED`)
- Scores externos (`EXT_SOURCE_2`, `EXT_SOURCE_3`)

---

## 📦 Dataset Original

| Campo | Detalhe |
|---|---|
| **Fonte** | [Kaggle – Credit Analysis Dataset](https://www.kaggle.com/datasets/kapoorshivam/credit-analysis/data) |
| **Registros** | ~307.511 clientes |
| **Colunas originais** | 121 features + 1 target  (122 colunas)|
| **Variável alvo** | `TARGET` (0 = Bom Pagador / 1 = Inadimplente) |
| **Desbalanceamento** | ~91,93% bons pagadores / ~8,07% inadimplentes |


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

## 2. Análise Exploratória dos Dados (EDA)

### Objetivos

- Entender os dados brutos recebidos
- Detectar valores ausentes, duplicados ou inconsistências
- Explorar a distribuição das variáveis (tendência, dispersão, outliers)
- Avaliar o balanceamento das classes no target
- Verificar correlações iniciais entre variáveis
- Identificar problemas de escala, unidades ou codificação de categorias

---

### 2.1 Balanceamento do Target

Os dados apresentam um conjunto **fortemente desequilibrado**:

| Classe | Registros | Percentual |
|--------|-----------|------------|
| `0` — Bom Pagador | 282.686 | 91,93% |
| `1` — Inadimplente | 24.825 | 8,07% |

Os dados apresentam um conjunto **fortemente desequilibrado**:

| Classe | Registros | Percentual |
|--------|-----------|------------|
| `0` — Bom Pagador | 282.686 | 91,93% |
| `1` — Inadimplente | 24.825 | 8,07% |

#### Observações para as etapas seguintes

Esta é apenas a etapa de conhecimento dos dados. As decisões sobre como lidar com o desbalanceamento — técnicas de reamostragem, ajuste de pesos, escolha de algoritmo, definição de métricas e calibração do limiar de decisão serão tratadas nas etapas posteriores de preparação e modelagem. Por ora, registramos apenas o diagnóstico:

- **Viés do modelo:** o algoritmo tende a prever sempre a classe majoritária (`0`), porque assim já alcança alta acurácia, ignorando os inadimplentes — justamente os que mais importa identificar.
- **Métricas enganosas:** um modelo que prevê `0` para todos acerta 91,93%, mas é completamente inútil para detectar inadimplência.
- **Custo assimétrico do erro:** não identificar um inadimplente (falso negativo) é geralmente mais custoso para a instituição do que recusar um bom pagador (falso positivo). Esse trade-off precisará ser considerado na escolha das métricas e na calibração do limiar de decisão.

---

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

### 2.4 Estatística Descritiva Interpretada

A tabela a seguir consolida as medidas de tendência central (média, mediana, moda) e medidas de dispersão (desvio-padrão, Q1, Q3, IQR, mínimo, máximo) das principais variáveis numéricas, calculadas separadamente por classe (Bom Pagador vs. Inadimplente) para evidenciar quais variáveis apresentam maior poder discriminatório.

> A análise por classe é especialmente relevante porque um modelo preditivo precisa encontrar diferenças sistemáticas entre os grupos. Variáveis com distribuições muito sobrepostas entre as classes têm pouco poder discriminatório isolado.

| Variável | Classe | Média | Mediana | Desvio-padrão | Q1 | Q3 | IQR | Mín | Máx |
|---|---|---|---|---|---|---|---|---|---|
| **AMT_INCOME_TOTAL** | Geral | 168.798 | 147.150 | 237.123 | 112.500 | 202.500 | 90.000 | 25.650 | 117.000.000 |
| | Bom Pagador | 168.866 | 148.500 | 241.455 | 112.500 | 202.500 | 90.000 | 25.650 | 117.000.000 |
| | Inadimplente | 168.136 | 135.000 | 196.960 | 108.000 | 202.500 | 94.500 | 27.000 | 18.000.000 |
| **AMT_CREDIT** | Geral | 599.026 | 513.531 | 402.491 | 270.000 | 808.650 | 538.650 | 45.000 | 4.050.000 |
| | Bom Pagador | 601.048 | 516.798 | 401.533 | 270.000 | 814.041 | 544.041 | 45.000 | 4.050.000 |
| | Inadimplente | 579.882 | 490.500 | 407.624 | 261.000 | 765.000 | 504.000 | 45.000 | 3.375.000 |
| **AMT_ANNUITY** | Geral | 27.109 | 24.903 | 14.493 | 16.524 | 34.596 | 18.072 | 1.615 | 258.026 |
| | Bom Pagador | 27.040 | 24.903 | 14.326 | 16.524 | 34.452 | 17.928 | 1.615 | 258.026 |
| | Inadimplente | 27.638 | 24.903 | 15.517 | 16.524 | 35.325 | 18.801 | 2.497 | 225.000 |
| **DAYS_BIRTH** | Geral | -16.037 | -15.775 | 4.363 | -19.682 | -12.413 | 7.269 | -25.229 | -7.489 |
| | Bom Pagador | -16.192 | -15.963 | 4.353 | -19.850 | -12.614 | 7.236 | -25.229 | -7.489 |
| | Inadimplente | -14.727 | -14.404 | 4.267 | -18.272 | -11.149 | 7.123 | -25.096 | -7.563 |
| **DAYS_EMPLOYED** | Geral | -2.385 | -1.648 | 2.338 | -3.531 | -767 | 2.764 | -17.912 | 365.243\* |
| | Bom Pagador | -2.418 | -1.680 | 2.355 | -3.589 | -775 | 2.814 | -17.912 | 365.243\* |
| | Inadimplente | -2.151 | -1.478 | 2.218 | -3.214 | -700 | 2.514 | -15.945 | 365.243\* |
| **EXT_SOURCE_2** | Geral | 0,5365 | 0,5658 | 0,1939 | 0,3918 | 0,6881 | 0,2963 | 0,0001 | 0,8551 |
| | Bom Pagador | 0,5492 | 0,5789 | 0,1901 | 0,4087 | 0,7003 | 0,2916 | 0,0001 | 0,8551 |
| | Inadimplente | 0,4459 | 0,4453 | 0,1949 | 0,2945 | 0,5980 | 0,3035 | 0,0003 | 0,8499 |
| **EXT_SOURCE_3** | Geral | 0,5102 | 0,5351 | 0,1948 | 0,3693 | 0,6666 | 0,2973 | 0,0005 | 0,8960 |
| | Bom Pagador | 0,5233 | 0,5486 | 0,1921 | 0,3851 | 0,6791 | 0,2940 | 0,0005 | 0,8960 |
| | Inadimplente | 0,3985 | 0,3968 | 0,1882 | 0,2596 | 0,5344 | 0,2748 | 0,0019 | 0,8655 |

> \* O valor 365.243 em `DAYS_EMPLOYED` é uma anomalia (sentinela para aposentados/desempregados) tratada na seção 3.3. As estatísticas acima refletem o dataset pré-tratamento.

#### Interpretação

| Variável | Observação principal |
|---|---|
| **AMT_INCOME_TOTAL** | Médias praticamente idênticas entre as classes; o desvio-padrão é muito alto em ambas devido a outliers extremos. **Pouco discriminatória isoladamente.** |
| **AMT_CREDIT** | A mediana dos inadimplentes é ligeiramente menor. IQR similar — **pouca capacidade discriminatória direta.** |
| **AMT_ANNUITY** | Distribuições muito próximas. **Pouco poder de separação isolado.** |
| **DAYS_BIRTH** | **Variável mais informativa entre as numéricas analisadas:** inadimplentes têm mediana ~-14.404 (≈ 39 anos) vs. bons pagadores ~-15.963 (≈ 44 anos). **Clientes mais jovens apresentam maior risco.** |
| **DAYS_EMPLOYED** | Distorcida pela anomalia 365.243. Após tratamento, espera-se maior tempo de emprego entre bons pagadores (fator protetor). |
| **EXT_SOURCE_2** | **Diferença consistente:** bons pagadores têm mediana 0,58 vs. inadimplentes 0,45. Uma das variáveis de maior poder preditivo. |
| **EXT_SOURCE_3** | **Maior diferença entre as classes:** medianas de 0,55 (bons pagadores) vs. 0,40 (inadimplentes). Variável mais discriminatória do dataset. |

**Variáveis com maior potencial discriminatório:** `EXT_SOURCE_3`, `EXT_SOURCE_2`, `DAYS_BIRTH` e `DAYS_EMPLOYED` (após tratamento da anomalia).

**Variáveis com baixa capacidade discriminatória isolada:** `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE` e `AMT_INCOME_TOTAL` — distribuições muito sobrepostas entre as classes. Podem agregar valor quando **combinadas** (ex.: razão crédito/renda, razão anuidade/renda) em engenharia de features nas etapas posteriores.

---

### 2.5 Histogramas

![Histogramas](https://github.com/ICEI-PUC-Minas-PMV-SI/PUC_E7_T2_G1_2026_01/raw/main/docs/img/Histograma1.png)

![Histogramas](https://github.com/ICEI-PUC-Minas-PMV-SI/PUC_E7_T2_G1_2026_01/raw/main/docs/img/Histograma2.png)

![Histogramas](https://github.com/ICEI-PUC-Minas-PMV-SI/PUC_E7_T2_G1_2026_01/raw/main/docs/img/Histograma3.png)


- `AMT_CREDIT` e `AMT_ANNUITY`: distribuições assimétricas à direita, com sobreposição quase completa entre as classes — **baixo poder discriminatório isolado**.
- `DAYS_BIRTH`: inadimplentes concentram-se em idades mais jovens.
- `AMT_INCOME_TOTAL`: inadimplentes tendem a se concentrar em faixas de renda mais baixa.
- `DAYS_EMPLOYED`: anomalia dos ~365.000 dias confirmada nas duas classes.

---

### 2.6 Matriz de Correlação

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


### 2.7 Variance Inflation Factor (VIF)

O VIF mede o quanto a variância de um coeficiente é inflada devido à multicolinearidade.

**Observação:** o VIF é especialmente relevante para modelos lineares (como regressão logística), onde a multicolinearidade distorce diretamente os coeficientes e seus intervalos de confiança. Para modelos baseados em árvores (como Random Forest), o impacto é menor, árvores de decisão selecionam uma variável por vez em cada divisão e são naturalmente mais tolerantes à colinearidade. Ainda assim, a remoção de variáveis altamente redundantes reduz a dimensionalidade e melhora a interpretabilidade do modelo.

| Variável           | VIF       | Decisão                                   |
| ------------------ | --------- | ----------------------------------------- |
| `AMT_GOODS_PRICE`  | **39.51** | ⚠️ Remover — multicolinearidade gravíssima |
| `AMT_CREDIT`       | **38.77** | ⚠️ Manter — variável central do problema   |
| `AMT_ANNUITY`      | 2.56      | ✅ Manter                                  |
| `DAYS_EMPLOYED`    | 1.64      | ✅ Manter                                  |
| `DAYS_BIRTH`       | 1.64      | ✅ Manter                                  |
| `AMT_INCOME_TOTAL` | 1.04      | ✅ Manter                                  |

---

### 2.8 Análise dos Atributos Categóricos

Além das variáveis numéricas, o dataset contém variáveis categóricas relevantes que precisam ser exploradas: tipo de contrato, gênero, posse de carro/imóvel, tipo de renda, escolaridade, estado civil, tipo de moradia, ocupação e tipo de organização.

Para cada variável analisamos a taxa de inadimplência por categoria (% de `TARGET = 1` dentro de cada nível), que é a leitura mais informativa — revela quais grupos apresentam maior risco relativo.

> **Taxa média geral de inadimplência: 8,07%** (usada como referência para identificar categorias com risco acima ou abaixo da média.)

| Variável | Achado relevante |
|---|---|
| **NAME_CONTRACT_TYPE** | A maioria das solicitações é do tipo *Cash loans*. *Revolving loans* (crédito rotativo) apresentam taxa de inadimplência ligeiramente menor que a média geral. |
| **CODE_GENDER** | Há proporção maior de mulheres na base. Clientes do sexo masculino apresentam **taxa de inadimplência consistentemente mais alta** que a média geral; clientes do sexo feminino ficam abaixo da média. |
| **FLAG_OWN_CAR** | Clientes **sem carro** registram taxa de inadimplência ligeiramente maior do que clientes com carro próprio. |
| **FLAG_OWN_REALTY** | Diferença pequena entre quem possui ou não imóvel; clientes **sem imóvel** registram taxa levemente mais alta. |
| **NAME_INCOME_TYPE** | *Pensioner* (aposentados) apresenta **taxa abaixo da média**; *Working* e *Commercial associate* concentram a maior parte da base. Categorias como *Maternity leave* e *Unemployed* aparecem com taxas elevadas, mas com poucos registros. |
| **NAME_EDUCATION_TYPE** | **Gradiente claro:** *Lower secondary* e *Secondary / secondary special* têm as **maiores taxas** de inadimplência; *Higher education* e *Academic degree*, as **menores**. Variável com boa capacidade discriminatória. |
| **NAME_FAMILY_STATUS** | *Civil marriage* e *Single / not married* tendem a taxas **acima da média**; *Widow* e *Married* tendem a ficar **abaixo da média**. |
| **NAME_HOUSING_TYPE** | *Rented apartment* e *With parents* apresentam taxas **acima da média**; *House / apartment* (categoria majoritária) fica próximo da média geral. |
| **OCCUPATION_TYPE** | *Low-skill Laborers*, *Drivers* e *Security staff* estão entre as ocupações com **maior risco**; *Accountants*, *High skill tech staff* e *Managers*, entre as de **menor risco**. Atenção: muitos registros têm ocupação ausente (codificada como `'desconhecido'`). |
| **ORGANIZATION_TYPE** | Variável de **alta cardinalidade** (mais de 50 categorias). Setores de transporte e construção tendem a taxas maiores; setores governamentais e de educação, menores. Recomenda-se agrupamento de categorias raras antes da modelagem. |

**Resumo:** escolaridade (`NAME_EDUCATION_TYPE`), gênero (`CODE_GENDER`), tipo de renda (`NAME_INCOME_TYPE`) e ocupação (`OCCUPATION_TYPE`) são as variáveis categóricas com maior diferença de taxa de inadimplência entre seus níveis, sendo candidatas a boas preditoras na etapa de modelagem. Variáveis com muitos níveis e baixa frequência por categoria (`OCCUPATION_TYPE`, `ORGANIZATION_TYPE`) provavelmente exigirão **agrupamento de categorias raras** e **codificação** (one-hot ou ordinal encoding) em etapa posterior.

---

### 2.9 Análise de Outliers

A identificação de outliers foi realizada pelo critério IQR de Tukey (limites = Q1 − 1,5 × IQR e Q3 + 1,5 × IQR), complementado pela inspeção visual dos boxplots. A tabela abaixo sistematiza os resultados para as principais variáveis numéricas:

| Variável | Critério | Limite inferior | Limite superior | Registros afetados | % da base | Decisão |
|---|---|---|---|---|---|---|
| `AMT_INCOME_TOTAL` | IQR Tukey | −22.500 | 337.500 | ~20.000 | ~6,5% | Capping no limite superior (winsorização) |
| `AMT_CREDIT` | IQR Tukey | −541.975 | 1.620.616 | ~15.000 | ~4,9% | Manter — variação real no mercado de crédito |
| `AMT_ANNUITY` | IQR Tukey | −10.584 | 61.704 | ~12.000 | ~3,9% | Manter — variação real |
| `DAYS_BIRTH` | IQR Tukey | −30.594 | −1.501 | ~0 | ~0,0% | Nenhuma ação necessária |
| `DAYS_EMPLOYED` | Valor sentinela | — | 365.243 | 55.374 | 18,0% | Substituição por `NaN` + imputação pela mediana (tratado em 3.3) |
| `EXT_SOURCE_2` | IQR Tukey | −0,048 | 1,127 | ~0 | ~0,0% | Nenhuma ação necessária (variável normalizada 0–1) |
| `EXT_SOURCE_3` | IQR Tukey | −0,076 | 1,110 | ~0 | ~0,0% | Nenhuma ação necessária (variável normalizada 0–1) |

O outlier extremo de `AMT_INCOME_TOTAL` (~1,2 × 10⁸, identificado no boxplot) afeta principalmente a classe inadimplente e distorce significativamente a média. A decisão de winsorização preserva o registro sem inflar artificialmente as estatísticas. O valor sentinela em `DAYS_EMPLOYED` (365.243 dias ≈ 1.000 anos) não é um outlier estatístico no sentido estrito, mas sim uma codificação intencional que representa clientes aposentados ou sem vínculo formal, tratada separadamente.

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

**Resultado: 0 valores nulos restantes. Shape final: 307.511 × 74 colunas.**

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

# 4 Ética, Privacidade e LGPD
 
> Esta seção atende a uma exigência *obrigatória* da etapa de conhecimento dos dados:
discutir as implicações éticas, de privacidade e de proteção de dados envolvidas em um
projeto de *predição de inadimplência em análise de crédito*. Por se tratar de um caso
de uso que toma — ou apoia — *decisões automatizadas* sobre pessoas, com consequências
financeiras concretas, a reflexão ética não é acessória; é parte do projeto.
 
## 4.1 Sensibilidade dos dados utilizados
 
O dataset utilizado contém um conjunto de informações **pessoais e financeiras de alta
sensibilidade**:
> * *Dados financeiros*: renda total declarada (AMT_INCOME_TOTAL), valor do crédito
solicitado (AMT_CREDIT), anuidade (AMT_ANNUITY), valor de bens financiados
(AMT_GOODS_PRICE);
> * *Dados demográficos*: idade (DAYS_BIRTH), gênero (CODE_GENDER), estado civil
(NAME_FAMILY_STATUS), número de filhos (CNT_CHILDREN);
> * *Dados socioeconômicos*: escolaridade (NAME_EDUCATION_TYPE), tipo de renda
(NAME_INCOME_TYPE), ocupação (OCCUPATION_TYPE), tipo da organização
(ORGANIZATION_TYPE), tipo de moradia (NAME_HOUSING_TYPE);
> * *Dados comportamentais*: scores externos (EXT_SOURCE_1/2/3), tempo desde o último
contato (DAYS_LAST_PHONE_CHANGE), tempo de registro (DAYS_REGISTRATION).
 
Embora o SK_ID_CURR seja apenas um identificador interno e não revele identidade direta,
a combinação dessas variáveis pode permitir reidentificação indireta dos clientes,
especialmente quando cruzadas com bases externas. Esse é um risco clássico de
*reidentificação* que exige cuidado especial no manuseio dos dados.
 
## 4.2 LGPD — Lei Geral de Proteção de Dados (Lei 13.709/2018)
 
No contexto brasileiro, um projeto desse tipo está sujeito à *LGPD*. Os principais
pontos a observar são:
 
> * *Base legal de tratamento (art. 7º)*: dados pessoais só podem ser tratados com
amparo em uma das hipóteses legais. Em concessão de crédito, a base mais comum é
*execução de contrato* ou *legítimo interesse*, sempre respeitando o teste de
proporcionalidade.
> * *Finalidade (art. 6º, I): os dados devem ser usados **apenas* para a finalidade
informada — analisar risco de crédito. Não podem ser reaproveitados para marketing,
venda de outros produtos, ou cessão a terceiros sem nova base legal.
> * *Adequação e necessidade (art. 6º, II e III)*: só podem ser coletadas variáveis
estritamente necessárias para a análise. Variáveis demográficas redundantes ou que não
contribuem para o desempenho do modelo devem ser descartadas (princípio da minimização).
> * *Transparência e informação (art. 6º, VI; art. 9º)*: o cliente deve saber **quais
dados** são tratados, *com que finalidade, **por quanto tempo* e quais são seus direitos.
> * *Direito à revisão de decisões automatizadas (art. 20)*: o titular tem o **direito
expresso** de solicitar revisão humana de decisões tomadas por sistemas automatizados que
afetem seus interesses. Em um modelo de crédito, isso significa que **toda decisão de
recusa baseada apenas no modelo deve poder ser revista por uma pessoa**.
> * *Segurança e prevenção (art. 46)*: os dados devem ser protegidos por medidas
técnicas e administrativas adequadas (criptografia, controle de acesso, registro de
operações), inclusive durante o treinamento e a operação do modelo.
> * *Dados sensíveis (art. 5º, II; art. 11)*: a LGPD trata de forma especial dados
sobre origem racial/étnica, convicção religiosa, saúde, orientação sexual, entre outros.
Este dataset *não contém* explicitamente essas categorias — mas algumas variáveis (como
ORGANIZATION_TYPE) podem, em casos limítrofes, atuar como *proxy* para dados
sensíveis e exigir avaliação cuidadosa.
 
## 4.3 Vieses, discriminação e variáveis sensíveis
 
Mesmo sem dados sensíveis explícitos, **um modelo de crédito pode reproduzir e
amplificar desigualdades históricas**. Os principais riscos identificados nesta base são:
 
> * *Gênero (CODE_GENDER)*: incluir gênero como variável preditiva pode levar o modelo
a tratar clientes de forma diferente apenas pelo sexo — uma prática que pode configurar
*discriminação direta*. Mesmo retirando a coluna, o modelo pode aprender o gênero por
*variáveis correlacionadas* (proxy discrimination), efeito conhecido na literatura de
"fairness in ML".
> * *Idade (DAYS_BIRTH)*: a EDA mostrou que clientes mais jovens têm taxa de
inadimplência maior. Usar idade como preditor pode penalizar sistematicamente jovens —
inclusive aqueles com bom histórico individual. É preciso avaliar se essa diferenciação é
justificável e proporcional.
> * *Estado civil e número de filhos (NAME_FAMILY_STATUS, CNT_CHILDREN)*: podem
penalizar mães solo, casais sem filhos, divorciados, etc., reforçando estereótipos
sociais.
> * *Tipo de moradia / ocupação / organização*: podem atuar como **proxy de classe
social, território (CEP de origem dos dados em outros datasets), ou raça**. Mesmo sem
intenção, um modelo treinado nesses dados pode reproduzir desigualdades estruturais.
 
> *Recomendação prática:* na etapa de modelagem, devem ser executados **testes de
fairness** comparando taxas de aprovação, falsos positivos e falsos negativos entre
subgrupos demográficos (gênero, faixa etária, escolaridade), e o resultado deve ser
documentado.
 
## 4.4 Explicabilidade do modelo
 
A LGPD garante o direito do titular obter explicações sobre decisões automatizadas
(art. 20). Por isso, modelos "caixa-preta" puros não atendem aos requisitos legais. Em
um cenário de produção, deve-se:
 
> * *Preferir modelos interpretáveis* quando o desempenho for comparável (regressão
logística, árvores rasas, modelos aditivos);
> * Quando forem usados modelos complexos (Random Forest, XGBoost, redes neurais),
*aplicar técnicas de explicabilidade* como *SHAP, **LIME* ou **importância por
permutação**, capazes de gerar explicações por decisão individual;
> * *Manter um canal de revisão humana* para clientes que tiverem o crédito recusado;
> * *Documentar* as principais variáveis usadas e o impacto de cada uma na decisão,
em linguagem clara para o titular.
 
## 4.5 Impacto dos falsos positivos e falsos negativos
 
Erros de classificação têm **consequências assimétricas e diferentes para cada parte
envolvida**:
 
| Tipo de erro                | Definição                                      | Impacto para o cliente                                                         | Impacto para a instituição                                |
| --------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------- |
| *Falso Positivo (FP)*     | Modelo classifica um *bom pagador* como inadimplente | Crédito recusado indevidamente, cliente penalizado financeiramente, possível dano à reputação cadastral, *exclusão financeira* | Perda de receita; cliente migra para concorrente          |
| *Falso Negativo (FN)*     | Modelo classifica um *inadimplente* como bom pagador | (Nenhum impacto direto sobre o cliente — recebe o crédito)                      | *Perda financeira direta* com inadimplência; aumento do provisionamento |
 
Do ponto de vista da instituição, FN tende a ser mais custoso. Mas do ponto de vista do cliente e da sociedade, FP pode ser igualmente ou mais
grave: significa negar acesso a crédito a quem teria honrado o compromisso, com
impactos potencialmente sérios em vida pessoal, saúde mental, capacidade de empreender e
mobilidade social. Um modelo "tecnicamente bom" para a instituição pode ser socialmente
nocivo se concentrar FPs em determinados grupos demográficos.
 
Por isso, a escolha do limiar de decisão e o monitoramento de disparidades entre
grupos não é apenas uma questão técnica, é uma questão ética que precisa ser tomada de
forma transparente e justificada, idealmente com a participação de áreas de
compliance, jurídico e DPO (Encarregado pela Proteção de Dados).
 
## 4.6 Boas práticas adotadas 
 
Algumas decisões já tomadas neste trabalho, mesmo sem rotulação explícita, contribuem
para a conformidade e a ética:
> * *Anonimização*: o dataset utilizado já vem anonimizado (apenas SK_ID_CURR como
identificador, sem nome, CPF ou contato);
> * *Minimização*: variáveis com mais de 40% de nulos foram removidas, reduzindo a
quantidade de dados pessoais armazenados desnecessariamente;
> * *Diagnóstico de redundância*: o uso de matriz de correlação e VIF para descartar
variáveis redundantes (AMT_GOODS_PRICE) também atende ao princípio da minimização;
> * *Documentação*: este notebook documenta o que é feito com cada variável, o que
favorece auditoria e prestação de contas (accountability).
 
## 4.7 Próximos passos recomendados
 
Para que o projeto esteja em plena conformidade ética e legal antes de qualquer uso em
produção, recomenda-se:
> 1. Realizar *avaliação de impacto à proteção de dados (RIPD/DPIA)*, conforme art. 38
da LGPD;
> 2. Implementar *testes de fairness* por gênero, faixa etária e escolaridade na etapa
de avaliação do modelo;
> 3. Implementar *explicabilidade individual* (ex.: SHAP) para suportar o direito à
revisão de decisões automatizadas;
> 4. Definir *políticas de retenção e descarte* dos dados de treinamento;
> 5. Estabelecer *processo de monitoramento contínuo* do modelo em produção, incluindo
detecção de drift e auditoria de viés.