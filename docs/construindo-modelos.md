# Predição de Inadimplência em Análise de Crédito
## Resumo

Este relatório documenta de forma completa e detalhada o desenvolvimento de um modelo de aprendizado de máquina para predição de inadimplência em análise de crédito. O projeto utilizou o dataset Credit Analysis , disponível no Kaggle, composto por 307.511 registros de clientes e 122 variáveis, com o objetivo de construir um sistema capaz de identificar, antes da concessão do crédito, clientes com alta probabilidade de inadimplência.

A questão central do projeto é: *Dado o perfil financeiro de um cliente, qual é a probabilidade de ele ser classificado como bom ou mau pagador?* Para respondê-la, o grupo executou um pipeline completo de ciência de dados desde a coleta até a modelagem e avaliação.

### Resultados Alcançados

| **Métrica** | **Meta do Projeto** | **Resultado Obtido** | **Status** |
|---|---|---|---|
| Recall (Inadimplente) | ≥ 0,50 | 0,537 (53,7%) | Meta atingida |
| Balanced Accuracy | ≥ 0,60 | 0,657 (65,7%) | Meta atingida |
| FP Rate (Falsos Positivos) | ≤ 30% | 22% | Dentro do limite |
| Accuracy Geral | — | 0,76 (76%) | Aceitável |
| F1-Score (Inadimplente) | — | 0,26 | Referência |

**Modelo selecionado:** Random Forest Classifier com `class_weight='balanced'`, 200 árvores, profundidade máxima de 12 níveis e 10 features selecionadas.

---

## 1. Preparação dos Dados (Pré-processamento)

A qualidade do modelo depende diretamente da qualidade dos dados. Um dado mal tratado introduz viés, ruído e distorções que nenhum algoritmo consegue corrigir por conta própria. Por isso, o pré-processamento é a etapa mais crítica e trabalhosa de qualquer projeto de ciência de dados.

### Panorama Inicial dos Dados

Ao carregar o dataset, a base apresentava as seguintes características:

- 307.511 linhas (cada linha representa um cliente solicitante de crédito)
- 122 colunas (121 variáveis explicativas + 1 variável-alvo `TARGET`)
- 65 colunas do tipo `float64` (números decimais)
- 41 colunas do tipo `int64` (números inteiros)
- 16 colunas do tipo `object` (textos variáveis categóricas)
- 24,20% das células continham valores nulos equivalendo a 9.152.465 células vazias no total

Esse volume massivo de dados ausentes — quase um quarto de toda a base — foi o primeiro e mais urgente problema a ser enfrentado no pré-processamento. A estratégia de tratamento foi dividida em três etapas distintas, cada uma com sua própria lógica de decisão.

---
## 1.2 Análise Exploratória dos Dados (1ª Parte)

O dataset foi explorado em profundidade para compreender a estrutura dos dados, identificar problemas e levantar hipóteses sobre quais variáveis teriam maior poder preditivo. Esta seção resume os principais achados das análises realizadas.

### Desbalanceamento das Classes

A primeira e mais importante constatação foi o **forte desbalanceamento entre as classes**:

- **Bom Pagador (TARGET = 0): 282.686 clientes — 91,93% da base**

- **Inadimplente (TARGET = 1): 24.825 clientes — apenas 8,07% da base**

Esse desequilíbrio tem implicações diretas para a modelagem: um classificador ingênuo que previsse sempre Bom Pagador atingiria 91,93% de acurácia sem identificar nenhum inadimplente — resultado tecnicamente correto, mas operacionalmente inútil. Por isso, a **acurácia foi descartada como métrica principal** já nesta etapa exploratória.

### Estatística Descritiva por Classe

A análise estatística comparou as distribuições das principais variáveis entre bons pagadores e inadimplentes, revelando quais variáveis têm maior poder discriminatório:

|                             |                                                                                                                            |                                            |
|-----------------------------|----------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|
| **Variável**                | **Observação Principal**                                                                                                   | **Poder Discriminatório**                  |
| EXT_SOURCE_2 e EXT_SOURCE_3 | Bons pagadores têm medianas claramente mais altas (~0,60) que inadimplentes (~0,45)                                        | ALTO — diferença consistente entre classes |
| DAYS_BIRTH (idade)          | Mediana dos inadimplentes menos negativa (~-14.000, ~38 anos) vs bons pagadores (~-16.000, ~44 anos)                       | ALTO — jovens concentram maior risco       |
| DAYS_EMPLOYED               | Distorção severa pelo valor 365.243 — requer tratamento obrigatório antes da análise                                       | MÉDIO — após tratamento                    |
| AMT_CREDIT                  | Medianas praticamente iguais entre as classes (~513k). IQR similar.                                                        | BAIXO isoladamente                         |
| AMT_ANNUITY                 | Distribuições muito sobrepostas, medianas próximas (~25k)                                                                  | BAIXO isoladamente                         |
| AMT_INCOME_TOTAL            | Média e mediana similares entre classes. Outlier extremo (~R\$117M) apenas na classe inadimplente distorce o desvio-padrão | BAIXO isoladamente                         |

### Análise de Variáveis Categóricas — Taxa de Inadimplência por Categoria

Para cada variável categórica, foi calculada a taxa de inadimplência (% de TARGET=1) dentro de cada categoria, comparada com a taxa geral de 8,07%. Os principais achados foram:

|                     |                                                                     |                                                   |
|---------------------|---------------------------------------------------------------------|---------------------------------------------------|
| **Variável**        | **Categorias de Maior Risco**                                       | **Categorias de Menor Risco**                     |
| NAME_EDUCATION_TYPE | Lower secondary: 10,93% \| Secondary/secondary special: 8,94%       | Academic degree: 1,83% \| Higher education: 5,36% |
| CODE_GENDER         | Masculino (M): 10,14% — acima da média geral                        | Feminino (F): 7,00% — abaixo da média geral       |
| NAME_INCOME_TYPE    | Desempregados: 36,36% \| Licença maternidade: 40,00% (poucos casos) | Aposentados: 5,39% \| Servidores: 5,75%           |
| NAME_HOUSING_TYPE   | Apartamento alugado: 12,31% \| Mora com pais: 11,70%                | Apartamento de escritório: 6,57%                  |
| OCCUPATION_TYPE     | Low-skill Laborers: 17,15% \| Motoristas: 11,33%                    | Contadores: baixo \| High skill tech: baixo       |
| NAME_CONTRACT_TYPE  | Cash loans: 8,35%                                                   | Revolving loans: 5,48%                            |

**Conclusão da análise categórica:** escolaridade, gênero, tipo de renda e tipo de moradia são as variáveis categóricas com maior diferença de taxa de inadimplência entre categorias — candidatas a boas preditoras na modelagem.

### Análises Visuais — Pairplot, Boxplot e Histogramas

Foram geradas três famílias de visualizações para explorar as relações entre variáveis numéricas e o TARGET:

**Pairplot (amostra balanceada, 250 por classe):** com amostras iguais das duas classes para dar visibilidade equivalente aos inadimplentes, os principais padrões observados foram:

- **AMT_CREDIT × AMT_GOODS_PRICE:** correlação linear quase perfeita em ambas as classes — as duas variáveis carregam informação praticamente idêntica.

- **DAYS_BIRTH:** distribuição deslocada para idades mais jovens (valores menos negativos) na classe inadimplente — sinal mais claro de separação entre as classes nas variáveis temporais.

- **DAYS_EMPLOYED:** grande concentração de pontos em ~365.000 nas duas classes, confirmando a anomalia que exige tratamento.

- **AMT_INCOME_TOTAL:** inadimplentes ligeiramente mais concentrados em rendas baixas, mas com forte sobreposição com bons pagadores.

**Boxplots por TARGET:** confirmaram visualmente as conclusões da estatística descritiva:

- **DAYS_BIRTH:** única variável com diferença visual clara entre as medianas das duas classes — caixa dos inadimplentes deslocada para valores menos negativos (clientes mais jovens).

- **AMT_INCOME_TOTAL:** outlier extremo (~R\$117M) apenas na classe inadimplente comprime completamente a caixa, tornando o boxplot ilegível — confirmando a necessidade de tratamento.

- **DAYS_EMPLOYED:** outliers em ~365.000 aparecem nas duas classes, confirmando que o valor é uma codificação especial, não um outlier estatístico legítimo.

**Histogramas com KDE:** AMT_CREDIT e AMT_ANNUITY apresentam distribuições **assimétricas à direita** com múltiplos picos, sugerindo que os valores seguem padrões discretos (produtos financeiros com valores pré-definidos). As distribuições das duas classes se sobrepõem quase completamente nessas variáveis — confirma que isoladamente têm baixo poder discriminatório.

### Matriz de Correlação e VIF

A matriz de correlação de Pearson entre as variáveis numéricas revelou dois pontos críticos:

- **AMT_CREDIT × AMT_GOODS_PRICE: correlação de 0,99** — multicolinearidade gravíssima. As duas variáveis são praticamente a mesma informação.

- **AMT_CREDIT × AMT_ANNUITY: 0,77** e **AMT_ANNUITY × AMT_GOODS_PRICE: 0,78** — correlações fortes, mas com informação complementar relevante.

- **DAYS_BIRTH × DAYS_EMPLOYED: -0,62** — correlação negativa moderada esperada: clientes mais velhos tendem a ter mais tempo de emprego.

- **Correlações com TARGET: todas fracas** — a maior é DAYS_BIRTH com apenas 0,08. Confirma que os padrões de risco são não-lineares e exigem um modelo que capture interações entre variáveis.
- O VIF quantificou a multicolinearidade: AMT_GOODS_PRICE (39,51) e AMT_CREDIT (38,77) com valores gravíssimos; todas as demais variáveis com VIF próximo de 1 (independentes). Isso fundamentou a decisão de remover AMT_GOODS_PRICE.
---

### 1.3 Tratamento de Valores Ausentes

A presença de valores ausentes em um dataset é bem comum em problemas reais. Eles surgem por falhas no sistema de coleta, recusa do cliente em fornecer informações, campos opcionais não preenchidos ou inconsistências na integração de diferentes fontes de dados. A decisão sobre como tratar cada caso é uma das mais importantes do projeto, pois afeta diretamente a qualidade do modelo.

#### Etapa 1: Remoção de Colunas com Mais de 40% de Nulos

A primeira ação foi identificar e remover as colunas com proporção excessiva de valores ausentes. O critério adotado foi o limiar de 40%: qualquer coluna em que mais de 40% dos valores fossem nulos seria descartada da análise.

**Por que 40%?**: Esse limiar é amplamente utilizado na literatura de ciência de dados como ponto de equilíbrio entre preservar informação e evitar ruído. Quando mais de 40% dos valores de uma variável estão ausentes, qualquer técnica de imputação estaria, essencialmente, inventando dados para mais de um a cada três registros, o que introduziria mais viés do que valor preditivo.

**Resultado:** 49 colunas removidas (de 122, ou seja, 39,8% das colunas da base original). Essas colunas eram, em sua maioria, relacionadas a características físicas dos imóveis dos clientes, organizadas em três sufixos distintos: `_AVG` (média), `_MODE` (moda) e `_MEDI` (mediana). Após essa etapa, a base foi reduzida de **122 para 73 colunas**, sem perda significativa de poder preditivo.

#### Etapa 2 — Imputação de Colunas Numéricas pela Mediana

Para as colunas numéricas que ainda apresentavam valores ausentes, a estratégia adotada foi a imputação pela **mediana**. A mediana foi escolhida em vez da média por sua robustez a outliers — em variáveis financeiras como `AMT_INCOME_TOTAL`, um único cliente com renda de R$ 117 milhões distorceria a média de forma desproporcional.

No total, 315.116 valores numéricos foram imputados pela mediana, representando 0,83% do total de células. Os casos mais relevantes foram:

| **Coluna** | **Mediana Utilizada** | **Nulos Imputados** | **Observação** |
|---|---|---|---|
| AMT_ANNUITY | R$ 24.903,00 | 12 | Impacto mínimo — poucos nulos |
| AMT_GOODS_PRICE | R$ 450.000,00 | 278 | Posteriormente removida por multicolinearidade |
| EXT_SOURCE_2 | 0,57 | ~660.000 | Score externo — variável de alto impacto preditivo |
| EXT_SOURCE_3 | 0,54 | ~820.000 | Score externo — variável de alto impacto preditivo |
| CNT_FAM_MEMBERS | 2,00 | 2 | Impacto mínimo |
| DAYS_LAST_PHONE_CHANGE | -757 | 1 | Impacto irrelevante |

A imputação de `EXT_SOURCE_2` e `EXT_SOURCE_3` pela mediana merece atenção especial: essas são as variáveis que o modelo acabou identificando como as mais importantes para a predição. Em versões futuras, recomenda-se uma imputação mais sofisticada, como KNN Imputer, que usa os valores de vizinhos semelhantes para estimar o valor ausente.

#### Etapa 3 — Imputação de Colunas Categóricas com `'desconhecido'`

Para as variáveis categóricas com valores nulos, a estratégia foi criar uma nova categoria explícita chamada `'desconhecido'`. As duas variáveis tratadas desta forma foram:

- **NAME_TYPE_SUITE**: tipo de acompanhante no momento da solicitação de crédito. Há 97.683 registros com este dado ausente.
- **OCCUPATION_TYPE**: tipo de ocupação profissional declarada pelo cliente. A criação da categoria `'desconhecido'` preserva a informação de que esses clientes não forneceram sua ocupação, o que pode ser em si mesmo um sinal de risco.

**Resultado final:** 0 valores nulos restantes. A base passou de 307.511 × 122 para **307.511 × 73 colunas** — todas as linhas foram preservadas.

---

### 1.4 Verificação de Registros Duplicados

A verificação foi feita ignorando a coluna `SK_ID_CURR` (identificador único do cliente), para identificar linhas com exatamente os mesmos valores em todas as demais variáveis.

**Resultado: nenhum registro duplicado foi encontrado.** Todos os 307.511 clientes são únicos.

---

### 1.5 Tratamento da Anomalia em DAYS_EMPLOYED

Durante a análise exploratória, foi identificada uma anomalia crítica na variável `DAYS_EMPLOYED`: 55.374 registros (18% da base) apresentavam o valor 365.243 — equivalente a mais de 1.000 anos de emprego, fisicamente impossível. Trata-se de um valor sentinela utilizado para representar clientes aposentados ou sem vínculo empregatício formal.

Decisão de tratamento: substituir 365.243 por `NaN` e imputar pela mediana dos valores válidos restantes (-1.648 dias ≈ 4,5 anos de emprego). Essa abordagem foi escolhida porque:

- Remover os 55.374 registros implicaria perda de 18% da base — volume significativo com risco de viés de seleção.
- A mediana dos valores válidos representa o comportamento típico dos clientes com emprego formal.
- A variável `NAME_INCOME_TYPE` já captura explicitamente a categoria `'Pensioner'`, reduzindo a dependência exclusiva do `DAYS_EMPLOYED` como sinalizador.

---

### 1.6 Remoção de Variável Redundante (Multicolinearidade)

A multicolinearidade ocorre quando duas ou mais variáveis explicativas são tão fortemente correlacionadas que se tornam essencialmente redundantes. Para detectá-la de forma quantitativa, foi utilizado o **VIF (Variance Inflation Factor)**. A regra geral é que VIF > 10 indica multicolinearidade problemática, e VIF > 30 indica multicolinearidade gravíssima.

| **Variável** | **VIF Calculado** | **Interpretação** | **Decisão** |
|---|---|---|---|
| AMT_GOODS_PRICE | 39,51 | Multicolinearidade gravíssima | REMOVER |
| AMT_CREDIT | 38,77 | Multicolinearidade gravíssima | MANTER (variável central) |
| AMT_ANNUITY | 2,56 | Sem multicolinearidade relevante | MANTER |
| DAYS_EMPLOYED | 1,64 | Sem multicolinearidade | MANTER |
| DAYS_BIRTH | 1,64 | Sem multicolinearidade | MANTER |
| AMT_INCOME_TOTAL | 1,04 | Praticamente independente | MANTER |

A correlação de 0,99 entre `AMT_CREDIT` e `AMT_GOODS_PRICE` confirmou o que o VIF já indicava. Foi decidido remover `AMT_GOODS_PRICE` e manter `AMT_CREDIT`, por ser a variável central do problema de análise de risco.

**Resultado:** a base passou de 307.511 × 73 para 307.511 × 72 colunas.

---

### 1.7 Codificação de Variáveis Categóricas

A técnica utilizada foi o **Label Encoding**: cada categoria recebe um número inteiro. Essa abordagem é adequada para algoritmos baseados em árvores de decisão, como o Random Forest, que não interpreta os números como ordenação — ele trata os valores apenas como critérios de divisão binária.

Para modelos lineares (como Regressão Logística), o Label Encoding seria problemático, sendo necessário o One-Hot Encoding. Como o modelo escolhido é o Random Forest, o Label Encoding é a opção mais eficiente.

---

### 1.8 Seleção de Features — Da Base de 72 para 10 Variáveis

Após todo o pré-processamento, a base ainda continha 71 variáveis explicativas. Incluir todas elas seria subótimo por motivos de overfitting, custo computacional, interpretabilidade e boas práticas de privacidade por design (LGPD).

#### Etapa 1: Random Forest Exploratório com Threshold de 1%

Foi treinado um Random Forest exploratório (100 árvores, profundidade máxima 10, `class_weight='balanced'`, amostra de 50.000 registros). O critério de corte foi: variáveis com importância abaixo de 1% seriam candidatas ao descarte.

#### Etapa 2: Seleção Manual entre as Variáveis do Limiar

Com base nos resultados do RF exploratório e na análise exploratória, foram selecionadas 10 features finais:

| **Feature** | **Descrição** | **Justificativa da Seleção** |
|---|---|---|
| EXT_SOURCE_2 | Score externo de crédito (fonte 2) | 1ª em importância — captura histórico em outras instituições |
| EXT_SOURCE_3 | Score externo de crédito (fonte 3) | 2ª em importância — complementar à EXT_SOURCE_2 |
| DAYS_BIRTH | Idade do cliente em dias negativos | Clientes mais jovens têm maior taxa de inadimplência |
| AMT_CREDIT | Valor do crédito solicitado | Variável central do problema — exposição financeira |
| DAYS_EMPLOYED | Tempo de emprego em dias negativos | Estabilidade profissional — sinal de capacidade de pagamento |
| AMT_ANNUITY | Valor da anuidade do empréstimo | Compromisso mensal — relação com capacidade de pagamento |
| AMT_INCOME_TOTAL | Renda total declarada | Base financeira do cliente — poder de pagamento |
| DAYS_REGISTRATION | Dias desde o último registro/endereço | Estabilidade residencial — proxy de risco |
| DAYS_ID_PUBLISH | Dias desde emissão do documento | Indicador complementar de perfil |
| DAYS_LAST_PHONE_CHANGE | Dias desde a última troca de telefone | Comportamento potencialmente anômalo |

**Resultado final do pipeline de pré-processamento:** 307.511 clientes × 10 features, sem valores nulos, com variáveis numéricas tratadas e codificação categórica aplicada.

---

### 1.9 Separação em Treino e Teste

O dataset foi dividido em dois conjuntos independentes:

- **Treino (80%):** 246.008 registros — usado para ajustar os parâmetros do modelo.
- **Teste (20%):** 61.503 registros — reservado exclusivamente para avaliação em dados nunca vistos.

O parâmetro `stratify=y` foi aplicado para garantir que a proporção entre bons pagadores (91,93%) e inadimplentes (8,07%) fosse mantida identicamente nos dois conjuntos.

---

## 2. Descrição do Modelo de Aprendizado de Máquina

### 2.1 O Algoritmo Escolhido: Random Forest Classifier

Random Forest é um algoritmo de aprendizado de máquina supervisionado pertencente à família dos métodos ensemble baseados em árvores de decisão. Em vez de construir uma única árvore de decisão, o Random Forest constrói centenas de árvores diferentes e combina suas previsões para produzir uma resposta mais robusta e precisa.

---

### 2.2 Justificativa da Escolha do Random Forest

#### Adequação ao Tipo de Dado

O dataset é composto por dados tabulares mistos: variáveis numéricas em escalas muito diferentes, variáveis categóricas de alta cardinalidade e forte assimetria nas distribuições. O Random Forest é naturalmente robusto a essas características:

- **Não exige normalização:** árvores de decisão trabalham com limiares de divisão, não com distâncias.
- **Robusto a outliers:** a divisão binária em cada nó considera apenas "maior ou menor que X?".
- **Lida com não-linearidades e interações:** captura automaticamente interações entre variáveis.

#### Desbalanceamento de Classes

O parâmetro `class_weight='balanced'` aumenta automaticamente o peso dos inadimplentes durante o treinamento na proporção inversa à sua frequência — cada inadimplente recebe peso ≈ 11,4× maior que cada bom pagador.

#### Interpretabilidade e Auditabilidade

O Random Forest suporta três mecanismos complementares de explicabilidade:

- **Feature Importance (Gini):** indica quais variáveis contribuíram mais para as divisões nas árvores.
- **Permutation Importance:** mede o impacto direto de cada variável no desempenho do modelo.
- **SHAP (SHapley Additive exPlanations):** gera explicações individuais por cliente — ideal para atender ao art. 20 da LGPD.

#### Alinhamento com o Estado da Arte

- **CHANG et al. (2024):** Random Forest e XGBoost são as melhores opções em termos de Recall e AUC-PR em datasets desbalanceados.
- **KONATHAM et al. (2025):** técnicas de balanceamento são avaliadas primariamente pelo ganho em Recall da classe minoritária.
- **SHI et al. (2022):** em revisão sistemática de 76 estudos em credit risk, modelos baseados em árvores são os mais robustos.

---

### 2.3 Parâmetros do Modelo e Justificativa

| **Parâmetro** | **Valor** | **Justificativa Detalhada** |
|---|---|---|
| n_estimators | 200 | Mais árvores = maior estabilidade e menor variância. 200 é suficiente para convergência. |
| max_depth | 12 | Limita a complexidade — árvores muito profundas "decoram" clientes específicos do treino. |
| min_samples_split | 10 | Impede divisões baseadas em poucos clientes, reduzindo ruído. |
| min_samples_leaf | 5 | Impede decisões baseadas em 1–2 clientes — evita especificidade excessiva. |
| max_features | 'sqrt' | Avalia apenas √10 ≈ 3 variáveis por divisão. Garante diversidade entre árvores. |
| random_state | 42 | Garante reprodutibilidade — o mesmo resultado a cada execução. |
| n_jobs | -1 | Paraleliza o treinamento usando todos os núcleos da CPU. |
| class_weight | 'balanced' | Atribui peso 11,4× maior aos inadimplentes — sem isso, o modelo prevê 0 para quase todo mundo. |

---

### 2.4 Comparação com SMOTE e Calibração

#### SMOTE (Synthetic Minority Over-sampling Technique)

O SMOTE cria registros sintéticos da classe minoritária por interpolação entre vizinhos próximos. Neste experimento, o SMOTE piorou todos os indicadores em relação ao modelo base:

- **Recall (Inadimplente):** caiu de 0,537 para 0,457 (−15%)
- **Balanced Accuracy:** caiu de 0,657 para 0,605 (−8%)
- **Macro-F1:** caiu de 0,559 para 0,525 (−6%)

**Hipótese explicativa:** com 246.008 registros no treino e o `class_weight='balanced'` já compensando o desbalanceamento, o SMOTE criou amostras sintéticas com combinações irrealistas de valores que introduziram ruído e prejudicaram a generalização.

#### CalibratedClassifierCV (Sigmoid e Isotonic)

A calibração padrão eliminou completamente a capacidade de detectar inadimplentes:

- **Recall (Inadimplente):** despencou de 0,537 para praticamente 0,000
- **Balanced Accuracy:** caiu de 0,657 para 0,500 (equivalente a chute aleatório)

**Por que isso aconteceu?** O `CalibratedClassifierCV(cv=5)` retreina o modelo internamente em 5 folds, neutralizando o efeito do `class_weight='balanced'` e fazendo o modelo classificar praticamente todos como bons pagadores.

---

## 3. Avaliação do Desempenho do Modelo

### 3.1 Definição da Métrica Principal e Justificativa

**Métrica principal escolhida: Recall da classe Inadimplente (Sensibilidade / TPR).**

O Recall responde à pergunta: *"De todos os clientes que realmente se tornaram inadimplentes, quantos o modelo conseguiu identificar antes da concessão do crédito?"*

```
Recall = TP / (TP + FN)
```

**Justificativas:**

- **Custo assimétrico dos erros:** um Falso Negativo (inadimplente aprovado) significa perda direta do principal + custos de cobrança. A literatura estima que FNs custam 5 a 10 vezes mais que FPs.
- **Insensibilidade ao desbalanceamento:** um modelo trivial que prevê sempre `0` teria Recall = 0% na classe minoritária, expondo imediatamente sua inutilidade.
- **Aderência regulatória:** FNs geram exposição de risco de carteira com impacto direto nas provisões exigidas pelo Banco Central.

#### Régua Quantitativa de Sucesso (definida inicialmente)

| **Patamar** | **Critério** | **Justificativa** |
|---|---|---|
| Mínimo aceitável | Recall ≥ 0,40 | 5× melhor que o modelo trivial (~2%); mínimo para utilidade prática |
| Meta do projeto | Recall ≥ 0,50 E FP rate ≤ 30% E Balanced Accuracy ≥ 0,60 | Capturar metade dos inadimplentes mantendo recusas indevidas gerenciáveis |
| Excelência | Recall ≥ 0,65 E FP rate ≤ 25% | Patamar competitivo conforme literatura recente |
| Monitoramento em produção | AUC-PR como métrica de drift | Insensível ao limiar; capta mudanças no comportamento da classe minoritária |

---

### 3.2 Relatório Completo de Classificação

| **Métrica** | **Bom Pagador (0)** | **Inadimplente (1)** | **Macro Avg** | **Weighted Avg** |
|---|---|---|---|---|
| Precision | 0,95 | 0,17 | 0,56 | 0,89 |
| Recall | 0,78 | 0,54 | 0,66 | 0,76 |
| F1-Score | 0,85 | 0,26 | 0,56 | 0,81 |
| Support | 56.538 | 4.965 | 61.503 | 61.503 |
| Accuracy | — | — | 0,76 (76%) | — |

#### Interpretação Detalhada de Cada Métrica

- **Recall (Inadimplente) = 0,537:** o modelo identificou corretamente 2.668 dos 4.965 inadimplentes reais. Supera a meta do projeto (≥ 0,50) e representa melhoria de 26× em relação ao modelo trivial.
- **Precision (Inadimplente) = 0,17:** de todos classificados como inadimplentes, 17% realmente eram. Os outros 83% eram FPs — esperado e aceitável dado o desbalanceamento e a priorização do recall.
- **Recall (Bom Pagador) = 0,78:** o modelo aprovou corretamente 78% dos bons pagadores. FP rate = 22% — dentro do limite de 30%.
- **Balanced Accuracy = 0,657:** supera o patamar mínimo de 0,60. Indica que o modelo está genuinamente aprendendo padrões de risco.
- **Accuracy = 0,76:** a queda de 92% (trivial) para 76% é o "preço" pago pelo `class_weight='balanced'` — o modelo renuncia a acertos fáceis na classe majoritária para capturar mais inadimplentes.
- **F1-Score (Inadimplente) = 0,26:** reflexo do trade-off entre Precision baixa e Recall alto. Métrica complementar, não a principal.

---

### 3.3 Matriz de Confusão — Análise Detalhada

|  | **Previsto: Bom Pagador (0)** | **Previsto: Inadimplente (1)** |
|---|---|---|
| **Real: Bom Pagador (0)** | 43.905 (TN) | 12.633 (FP) |
| **Real: Inadimplente (1)** | 2.297 (FN) | 2.668 (TP) |

| **Quadrante** | **Quantidade** | **Significado em Crédito** | **Custo de Negócio** |
|---|---|---|---|
| TN (Verdadeiro Negativo) | 43.905 | Bom pagador aprovado corretamente | Sem custo — cenário ideal |
| FP (Falso Positivo) | 12.633 | Bom pagador indevidamente recusado | Perda da receita marginal de juros (recuperável) |
| FN (Falso Negativo) | 2.297 | Inadimplente aprovado indevidamente | Perda direta do principal + custos — **ERRO MAIS CARO** |
| TP (Verdadeiro Positivo) | 2.668 | Inadimplente corretamente identificado | Sem perda direta — proteção da carteira |

---

### 3.4 Análise de Calibração das Probabilidades

| **Modelo** | **Brier Score** | **LogLoss** | **ECE** | **Recall (Inadimp.)** | **Balanced Accuracy** |
|---|---|---|---|---|---|
| Base (class_weight) | 0,1705 | 0,5184 | 0,0766 | 0,537 | 0,657 |
| SMOTE | 0,1705 | 0,5293 | 0,0189 | 0,457 | 0,605 |
| Calibrado (sigmoid) | 0,0697 | 0,2550 | 0,0012 | 0,000 | 0,500 |
| Calibrado (isotonic) | 0,0697 | 0,2550 | 0,0017 | 0,004 | 0,502 |

Para uso em produção, recomenda-se calibração a posteriori sem retreinamento do modelo — abordagem mais controlada que preserva o sinal de risco.

---

### 3.5 Importância das Features — Três Métodos Complementares

#### Feature Importance (Gini)

| **Posição** | **Feature** | **Importância (Gini)** | **Interpretação** |
|---|---|---|---|
| 1º | EXT_SOURCE_2 | 25,6% | Score externo — variável mais decisiva do modelo |
| 2º | EXT_SOURCE_3 | 24,9% | Segundo score externo — complementar |
| 3º | DAYS_EMPLOYED | 8,9% | Tempo de emprego — estabilidade profissional |
| 4º | DAYS_BIRTH | 7,8% | Idade — jovens têm maior risco |
| 5º | AMT_CREDIT | 6,3% | Valor do crédito — exposição financeira |
| 6º | AMT_ANNUITY | 6,2% | Valor da anuidade — compromisso mensal |
| 7º | DAYS_ID_PUBLISH | 5,8% | Tempo desde emissão do documento |
| 8º | DAYS_LAST_PHONE_CHANGE | 5,4% | Tempo desde troca de telefone |
| 9º | DAYS_REGISTRATION | 5,3% | Tempo desde último registro |
| 10º | AMT_INCOME_TOTAL | 3,8% | Renda — surpreendentemente, a menos importante isolada |

#### Permutation Importance por Classe (foco na Inadimplência)

- **EXT_SOURCE_2:** +0,0514 — maior impacto na detecção de inadimplentes
- **EXT_SOURCE_3:** +0,0511 — impacto similar ao EXT_SOURCE_2
- **DAYS_BIRTH:** +0,0050
- **AMT_CREDIT:** +0,0038
- **DAYS_EMPLOYED:** +0,0031
- **AMT_ANNUITY:** +0,0023
- **DAYS_LAST_PHONE_CHANGE:** +0,0019
- **DAYS_REGISTRATION:** +0,0011
- **AMT_INCOME_TOTAL:** +0,0004 — isolada, quase irrelevante
- **DAYS_ID_PUBLISH:** ~0,0000 — contribuição nula para classe 1

#### SHAP Values — Explicabilidade por Cliente

Para um perfil de alto risco (`EXT_SOURCE_2 = 0,10`, `EXT_SOURCE_3 = 0,15`):

| **Feature** | **Valor** | **Contribuição SHAP** | **Efeito** |
|---|---|---|---|
| EXT_SOURCE_2 | 0,10 | +0,1615 | ↑ Score muito baixo aumenta risco fortemente |
| EXT_SOURCE_3 | 0,15 | +0,1488 | ↑ Segundo score muito baixo confirma risco |
| DAYS_EMPLOYED | -3.100 | -0,0215 | ↓ ~8,5 anos de emprego reduz risco (protetor) |
| DAYS_BIRTH | -14.200 | +0,0200 | ↑ Idade ~39 anos adiciona risco moderado |
| AMT_CREDIT | 500.000 | +0,0196 | ↑ Crédito alto contribui para risco |
| AMT_ANNUITY | 27.000 | +0,0115 | ↑ Anuidade elevada contribui para risco |
| AMT_INCOME_TOTAL | 150.000 | +0,0040 | ≈ Renda contribui muito pouco |

**Resultado:** P(Inadimplente) = 0,50 + 0,1615 + 0,1488 − 0,0215 + ... ≈ 0,847

Aplicação LGPD (art. 20): *"A análise indicou alto risco principalmente porque seus scores externos de crédito estão abaixo do esperado. O tempo no seu emprego atual foi o fator positivo mais importante. Você pode solicitar revisão e fornecer informações complementares."*

---

## 4. Pipeline de Pesquisa e Análise de Dados

### 4.1 Especificação Formal do Problema

| **Dimensão** | **Definição** |
|---|---|
| Domínio | Análise de crédito ao consumidor — concessão de empréstimos pessoais |
| Tipo de problema | Classificação binária supervisionada |
| Objetivo de negócio | Identificar clientes com alta probabilidade de inadimplência ANTES da concessão do crédito |
| Variável-alvo (TARGET) | 0 = Bom Pagador (adimplente) / 1 = Inadimplente |
| Dataset | Home Credit Default Risk (Kaggle) — `current_app.csv` |
| Volume de dados | 307.511 clientes × 122 colunas originais |
| Distribuição das classes | 91,93% Bom Pagador / 8,07% Inadimplente — fortemente desbalanceado |
| Métrica principal | Recall da classe Inadimplente |
| Métrica secundária | Balanced Accuracy |
| Meta quantitativa | Recall ≥ 0,50 E FP rate ≤ 30% E Balanced Accuracy ≥ 0,60 |
| Restrições regulatórias | LGPD art. 20 — direito a explicação de decisões automatizadas |
| Resultado alcançado | Recall = 0,537 / FP rate = 22% / Balanced Accuracy = 0,657 |

---

### 4.2 Diagrama do Fluxo do Pipeline

| **Etapa** | **Fase** | **Entrada** | **Saída** | **Ferramentas Principais** |
|---|---|---|---|---|
| 1 | Coleta de Dados | — | 307.511 × 122 DataFrame | kagglehub, pandas |
| 2 | EDA — 1ª Parte | DataFrame bruto | Diagnóstico completo dos dados | matplotlib, seaborn, statsmodels |
| 3 | Pré-processamento | DataFrame bruto diagnosticado | 307.511 × 10 features limpas | pandas, numpy, sklearn |
| 4 | EDA — 2ª Parte | DataFrame tratado | 10 features validadas e selecionadas | sklearn RF exploratório, SHAP |
| 5 | Modelagem | X_train (246k), y_train | Modelo treinado (Random Forest) | sklearn RandomForestClassifier |
| 6 | Avaliação | X_test (61k), y_test | Relatório completo de métricas | sklearn metrics, SHAP, calibration |
| 7 | Inferência | Novos clientes (X_new) | Probabilidade + explicação SHAP | model.predict_proba(), shap |

---

### 4.3 Tabela Consolidada das Decisões Técnicas por Fase

#### Fase 1 — Coleta de Dados

| **Decisão** | **Técnica / Ferramenta** | **Justificativa** |
|---|---|---|
| Fonte dos dados | Kaggle (Home Credit Default Risk) | Dataset público, versionado, amplamente referenciado na literatura de credit scoring |
| Forma de acesso | kagglehub.dataset_download | Acesso programático e rastreável |
| Arquivo utilizado | current_app.csv | Contém o TARGET necessário para aprendizado supervisionado |

#### Fase 2 — Análise Exploratória (EDA — 1ª Parte)

| **Decisão** | **Técnica / Ferramenta** | **Justificativa** |
|---|---|---|
| Análise estrutural | df.shape, df.info(), dtypes, head/tail | Entender dimensão, tipos de dados e primeiros valores |
| Estatística descritiva por classe | describe() + agrupamento por TARGET | Identificar quais variáveis têm distribuições diferentes entre as classes |
| Análise de outliers | Critério IQR de Tukey (1,5×IQR) | Padrão estatístico robusto para identificar valores extremos |
| Análise de correlação | Matriz de Pearson + heatmap | Identificar multicolinearidade entre variáveis financeiras |
| VIF (Variance Inflation Factor) | statsmodels.variance_inflation_factor | Quantificar multicolinearidade — VIF > 30 justifica remoção |
| Visualizações bivariadas | Pairplot (seaborn), boxplot por TARGET | Verificar separabilidade visual e identificar anomalias |
| Análise categórica | Tabela de taxa de inadimplência por categoria | Escolaridade, gênero e tipo de renda se destacaram |

#### Fase 3 — Pré-processamento

| **Decisão** | **Técnica / Ferramenta** | **Justificativa** |
|---|---|---|
| Remoção de colunas com >40% nulos | Filtragem por df.isnull().mean() > 0.4 | Variáveis muito esparsas introduzem mais ruído que sinal |
| Imputação numérica | Mediana (SimpleImputer strategy='median') | Robusta a outliers nas variáveis financeiras |
| Imputação categórica | Categoria 'desconhecido' (fill_value) | Preserva todos os registros e transforma a ausência em categoria informativa |
| Tratamento de anomalia DAYS_EMPLOYED | 365.243 → NaN → mediana (-1.648) | Valor fisicamente impossível (~1000 anos) |
| Verificação de duplicados | drop_duplicates() excluindo SK_ID_CURR | Garantir que o modelo não "memorize" registros repetidos |
| Remoção de variável redundante | AMT_GOODS_PRICE removida (VIF=39,5; ρ=0,99) | Multicolinearidade gravíssima com AMT_CREDIT |
| Codificação categórica | LabelEncoder | Compatível com Random Forest |
| Não normalização | Escalas originais preservadas | Random Forest é insensível a escala |

#### Fase 4 — EDA 2ª Parte e Seleção de Features

| **Decisão** | **Técnica / Ferramenta** | **Justificativa** |
|---|---|---|
| Validação pós-tratamento | Recálculo de VIF, df.isnull().sum() | Confirmar que todas as transformações produziram o resultado esperado |
| RF exploratório para seleção | RandomForest (n=100, depth=10, balanced, n=50k) | Critério objetivo de seleção baseado no próprio tipo de modelo |
| Limiar de corte | 1% de feature importance | Equilíbrio entre poder preditivo e parcimônia |
| Seleção final manual | Análise de interpretabilidade + LGPD | Garantir variáveis explicáveis sem proxies de discriminação |
| Redução final | 122 → 73 → 72 → 10 features | Cada etapa documentada com justificativa específica |

#### Fase 5 — Modelagem

| **Decisão** | **Técnica / Ferramenta** | **Justificativa** |
|---|---|---|
| Divisão treino/teste | 80/20 com stratify=y | Preserva a proporção de classes — crítico em dados desbalanceados |
| Reprodutibilidade | random_state=42 em todas as etapas | Permite replicar exatamente os mesmos resultados |
| Algoritmo | RandomForestClassifier | Robusto, interpretável, suporta class_weight |
| Balanceamento de classes | class_weight='balanced' | Solução eficiente sem inflação artificial do dataset |
| n_estimators=200 | 200 árvores na floresta | Convergência estável do ensemble |
| max_depth=12 | Profundidade máxima das árvores | Previne overfitting |
| min_samples_leaf=5 | Mínimo de amostras por folha | Impede decisões baseadas em pouquíssimos clientes |
| n_jobs=-1 | Paralelismo total | Treinar 200 árvores em 246k registros demanda paralelização |

#### Fase 6 — Avaliação

| **Decisão** | **Técnica / Ferramenta** | **Justificativa** |
|---|---|---|
| Métrica principal | Recall classe Inadimplente | Custo de FN é 5–10× maior que FP em crédito |
| Métrica secundária | Balanced Accuracy | Insensível ao desbalanceamento |
| Métricas complementares | Precision, F1, Brier, LogLoss, ECE | Visão multidimensional do desempenho |
| Teste SMOTE | imblearn SMOTE no conjunto de treino apenas | Comparação empírica — resultado: SMOTE piorou |
| Calibração | CalibratedClassifierCV (sigmoid e isotonic) | Verificar calibração das probabilidades — resultado: destruiu a detecção |
| Feature Importance | Gini, Permutation Importance, SHAP | Três perspectivas complementares da relevância de cada variável |
| SHAP por cliente | TreeExplainer + Waterfall + Force Plot | Atende ao art. 20 da LGPD |

#### Fase 7 — Inferência e Produção

| **Decisão** | **Técnica / Ferramenta** | **Justificativa** |
|---|---|---|
| Predição de classe | model.predict(X_new) | Saída binária direta para integração com sistemas de crédito |
| Predição probabilística | model.predict_proba(X_new) | Permite ordenação por risco e definição de limiar customizado |
| Simulação de uso | 6 perfis fictícios com decisões documentadas | Demonstra o comportamento em cenários reais |
| Zona cinzenta | Probabilidade entre 0,4 e 0,6 → revisão humana | Clientes próximos do limiar devem ter decisão complementada por analista |
| Persistência do modelo | sklearn.pipeline.Pipeline + joblib.dump | Reuso em produção sem retreinamento |
| Pipeline encapsulado | sklearn Pipeline (preprocess + classifier) | Pré-processamento e modelo em um único objeto reprodutível |

---

### 4.4 Pipeline Executável em Sklearn

O pipeline foi encapsulado em um objeto `sklearn.pipeline.Pipeline`, combinando pré-processamento e modelagem em um único objeto serializado:

- **Passo 1 — preprocess (ColumnTransformer):** aplica imputação pela mediana nas 10 features numéricas selecionadas.
- **Passo 2 — classifier (RandomForestClassifier):** treinado com os parâmetros documentados na Fase 5.

A avaliação do pipeline encapsulado produziu resultados idênticos ao modelo treinado manualmente: **Recall = 0,537, Balanced Accuracy = 0,657** — confirmando que a encapsulação preservou todas as decisões técnicas.

---

### 4.5 Pontos Fortes e Limitações do Pipeline

#### Pontos Fortes

- **Encapsulação completa:** pré-processamento e modelo em um único objeto — reduz risco de erros em produção.
- **Reprodutibilidade total:** `random_state=42` em todas as etapas garante resultados idênticos a cada execução.
- **Modularidade:** cada etapa pode ser substituída independentemente.
- **Auditabilidade:** todas as decisões técnicas documentadas com justificativas explícitas.
- **Replicabilidade:** o pipeline salvo em joblib pode ser carregado em qualquer ambiente Python.
- **Aderência à LGPD:** explicações SHAP por cliente integradas ao fluxo de inferência.

#### Limitações Conhecidas

- **Seleção de features fora do pipeline:** a redução 122 → 10 features ocorreu antes da encapsulação. Em produção, recomenda-se incluir um `SelectFromModel` como passo do pipeline.
- **Tratamento de outliers manual:** o critério IQR foi aplicado ao DataFrame diretamente. Em produção, deveria ser um `FunctionTransformer`.
- **Calibração não incluída:** a tentativa de calibração padrão falhou — a calibração a posteriori ainda precisa ser implementada.
- **Dependência de dados externos:** `EXT_SOURCE_2` e `EXT_SOURCE_3` respondem por ~50% do poder preditivo — a indisponibilidade dessas fontes deterioraria drasticamente o modelo.
- **Limiar de decisão fixo em 0,5:** o limiar ótimo deveria ser calibrado com base na matriz de custos real da instituição.

---

## 5. Considerações Finais e Recomendações

### 5.1 Resumo dos Resultados

O projeto desenvolveu com sucesso um modelo de predição de inadimplência baseado em Random Forest, superando todas as metas quantitativas definidas a priori.

| **Objetivo** | **Meta** | **Resultado** | **Status** |
|---|---|---|---|
| Recall (Inadimplente) | ≥ 0,50 | 0,537 | Superado |
| Balanced Accuracy | ≥ 0,60 | 0,657 | Superado |
| FP Rate | ≤ 30% | 22% | Dentro do limite |
| Modelo operacional | Treinado e avaliado | Pipeline v1 salvo e documentado | Concluído |
| Explicabilidade (LGPD) | SHAP por cliente | Implementado (Waterfall + Force Plot) | Concluído |
| SMOTE testado | Comparação empírica | SMOTE piorou — base adequada mantida | Concluído |
| Calibração testada | Comparação empírica | Calibração padrão inadequada — identificado | Concluído |

---

### 5.2 Pontos a serem observados

- A seleção criteriosa de 10 features a partir de 122 variáveis produziu um modelo mais simples, interpretável e operacional.
- `class_weight='balanced'` superou SMOTE neste contexto: com 246.000 registros de treino, o balanceamento por pesos já era suficiente. O SMOTE introduziu ruído desnecessário.
- Calibração padrão pode ser mais prejudicial que útil: o `CalibratedClassifierCV` destruiu a capacidade de detecção ao retreinar o modelo sem preservar o `class_weight`.
- EXT_SOURCE domina o modelo: os scores externos responderam por ~50% do poder preditivo — risco operacional caso essas fontes falhem. Diversificação de features é essencial em versões futuras.
- AMT_INCOME_TOTAL isolada é praticamente irrelevante: a renda declarada sozinha pouco distingue as classes. Seu poder emerge ao ser combinada (ex.: razão AMT_CREDIT/AMT_INCOME_TOTAL) — ganho claro para engenharia de features futura.

---

### 5.4 Conclusão

O modelo construído neste projeto atingiu todos os objetivos propostos. Com Recall de 53,7% na classe inadimplente, Balanced Accuracy de 65,7% e FP rate de 22%, o modelo é operacionalmente viável para aplicação em análise de crédito real.

A transparência garantida pelo SHAP, combinada com a documentação detalhada de cada decisão técnica ao longo do pipeline, torna o modelo não apenas eficaz, mas também auditável, explicável e aderente à LGPD. O pipeline encapsulado está pronto para ser carregado em outros ambientes, aplicado a novas safras de clientes e adaptado a outros produtos de crédito, estabelecendo uma base sólida para iterações futuras.
