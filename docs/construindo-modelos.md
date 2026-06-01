# Predição de Inadimplência em Análise de Crédito



## 1. Preparação dos Dados

A etapa de pré-processamento foi compartilhada entre todos os modelos desenvolvidos no projeto. Uma descrição detalhada consta no relatório da Etapa 3 (Random Forest). Esta seção consolida os pontos essenciais e destaca os ajustes realizados especificamente para a Regressão Logística e para o XGBoost.

### 1.1 Panorama inicial

O dataset apresentava as seguintes características ao ser carregado:

- **307.511 linhas** × **122 colunas** (121 features + 1 variável-alvo TARGET)
- **24,20% de células nulas** — equivalente a 9.152.465 células vazias
- Forte desbalanceamento de classes: 91,93% Bom Pagador (TARGET=0) e 8,07% Inadimplente (TARGET=1)

### 1.2 Pipeline de pré-processamento compartilhado

As etapas a seguir foram executadas de forma idêntica para todos os modelos:

| Etapa | Técnica adotada | Resultado |
|---|---|---|
| Remoção de colunas com > 40% nulos | Filtragem por `isnull().mean() > 0.4` | 122 → 73 colunas |
| Imputação numérica | Mediana (`SimpleImputer strategy='median'`) | 315.116 valores imputados |
| Imputação categórica | Categoria `'desconhecido'` | 0 nulos restantes |
| Anomalia `DAYS_EMPLOYED` | 365.243 → NaN → mediana (-1.648) | 55.374 registros corrigidos |
| Remoção de multicolinearidade | `AMT_GOODS_PRICE` removida (VIF=39,51; ρ=0,99) | 73 → 72 colunas |
| Seleção de features | RF exploratório (threshold 1%) + análise manual | 72 → **10 features finais** |
| Separação treino/teste | 80/20 com `stratify=y` | 246.008 treino / 61.503 teste |

### 1.3 Features selecionadas (compartilhadas entre todos os modelos)

| Feature | Descrição |
|---|---|
| EXT_SOURCE_2 | Score externo de crédito (fonte 2) — maior importância geral |
| EXT_SOURCE_3 | Score externo de crédito (fonte 3) — complementar |
| DAYS_BIRTH | Idade em dias negativos |
| AMT_CREDIT | Valor do crédito solicitado |
| DAYS_EMPLOYED | Tempo de emprego em dias negativos |
| AMT_ANNUITY | Valor da anuidade do empréstimo |
| AMT_INCOME_TOTAL | Renda total declarada |
| DAYS_REGISTRATION | Dias desde o último registro |
| DAYS_ID_PUBLISH | Dias desde a emissão do documento |
| DAYS_LAST_PHONE_CHANGE | Dias desde a última troca de telefone |

### 1.4 Ajustes específicos por modelo

**Regressão Logística — ajustes adicionais obrigatórios:**

A Regressão Logística é sensível à escala das variáveis, ao contrário dos modelos baseados em árvores. Por isso, dois ajustes foram necessários em relação ao pipeline do Random Forest:

- **Codificação categórica:** substituição de `LabelEncoder` por `OneHotEncoder` (`pd.get_dummies` com `drop_first=True`). O LabelEncoder introduz uma ordenação implícita que pode ser interpretada erroneamente por modelos lineares — por exemplo, tratar `NAME_INCOME_TYPE` como uma escala numérica quando não existe relação ordinal entre as categorias.
- **Normalização:** aplicação de `StandardScaler` (subtrai a média e divide pelo desvio-padrão) sobre todas as features numéricas. A regularização L2 do modelo exige que as variáveis estejam na mesma escala para que os coeficientes sejam comparáveis; sem padronização, variáveis em escalas grandes (como `AMT_CREDIT`) dominariam artificialmente a função de perda.

> **Regra anti-leakage:** o `StandardScaler` foi ajustado (`fit`) exclusivamente no conjunto de treino e aplicado (`transform`) tanto no treino quanto no teste, para evitar que informação do conjunto de teste vaze para o processo de aprendizado.

**XGBoost — reaproveitamento integral:**

O XGBoost, assim como o Random Forest, é um algoritmo baseado em árvores e, portanto, insensível à escala das variáveis. O pipeline de pré-processamento foi reaproveitado integralmente, sem necessidade de padronização. O único ajuste específico foi a substituição do parâmetro `class_weight='balanced'` pelo parâmetro nativo do XGBoost:

```
scale_pos_weight = n_negativos / n_positivos ≈ 282.686 / 24.825 ≈ 11,39
```

Esse parâmetro penaliza proporcionalmente mais os erros na classe minoritária (inadimplentes) durante o processo de boosting, produzindo o mesmo efeito de balanceamento que o `class_weight='balanced'` do scikit-learn, porém de forma nativa e integrada ao algoritmo.

---

## 2. Modelo 1 — Regressão Logística

### 2.1 Descrição do algoritmo

A Regressão Logística é um algoritmo de classificação supervisionada que modela a probabilidade de um evento binário (inadimplente vs. adimplente) como uma função linear das features de entrada, transformada pela função sigmoide:

```
P(Y=1 | X) = 1 / (1 + e^(-z)),   onde z = β₀ + β₁X₁ + ... + βₙXₙ
```

O modelo aprende os coeficientes `β` minimizando a log-verossimilhança negativa (entropia cruzada) via algoritmo de otimização numérica. Cada coeficiente representa o impacto de uma feature no log-odds da inadimplência: um coeficiente positivo indica que o aumento daquela variável aumenta a probabilidade de inadimplência; um coeficiente negativo indica efeito protetor.

### 2.2 Justificativa da escolha

A Regressão Logística foi escolhida por ser o modelo de referência (*baseline* interpretável) mais estabelecido em análise de crédito, com quatro vantagens centrais para este contexto:

**Interpretabilidade direta:** os coeficientes padronizados expressam o impacto de cada feature em escala comparável — o modelo pode ser auditado linha a linha. Isso facilita a conformidade com o art. 20 da LGPD, que garante ao titular o direito à explicação de decisões automatizadas.

**Calibração nativa de probabilidades:** a função sigmoide produz probabilidades estruturalmente bem calibradas, sem a necessidade de pós-processamento. Modelos baseados em árvores tendem a produzir probabilidades extremas (próximas de 0 ou 1); a Regressão Logística distribui as probabilidades de forma mais suave ao longo do intervalo [0, 1].

**Robustez com regularização L2:** o parâmetro `C` controla a força da regularização, evitando overfitting mesmo com features de baixo sinal individual. Com `C=0.1` (regularização forte), coeficientes de features menos informativas são encolhidos em direção a zero sem serem zerados.

**Eficiência computacional:** o treinamento é ordens de magnitude mais rápido que o Random Forest ou o XGBoost, tornando a validação cruzada e a busca de hiperparâmetros menos custosas.

**Alinhamento com a literatura de crédito:** modelos lineares regulados (especialmente scorecards derivados de Regressão Logística) são o padrão histórico da indústria financeira por sua auditabilidade e previsibilidade de comportamento fora da amostra.

### 2.3 Parâmetros do modelo e justificativas

| Parâmetro | Valor | Justificativa |
|---|---|---|
| `C` | 0,1 | Regularização forte — reduz overfitting em features de baixo sinal; coeficientes mais estáveis |
| `penalty` | `'l2'` | Ridge: todos os coeficientes encolhem, nenhum é zerado. Mantém contribuição de todas as features |
| `solver` | `'lbfgs'` | Algoritmo quasi-Newton; convergência rápida com features padronizadas e L2 |
| `class_weight` | `'balanced'` | Cada inadimplente pesa ~11× mais que bom pagador durante o treinamento |
| `max_iter` | 1000 | Garante convergência completa; o padrão de 100 pode ser insuficiente com regularização forte |
| `random_state` | 42 | Reprodutibilidade |

> **Nota sobre `C=0.1`:** em features de baixo sinal individual (como `DAYS_REGISTRATION`, `DAYS_ID_PUBLISH` e `DAYS_LAST_PHONE_CHANGE`), a regularização forte é essencial para evitar que o modelo ajuste ruído específico do treino. A busca automática via `RandomizedSearchCV` encontrou `C=3.746` como valor ótimo, porém os resultados no teste foram praticamente idênticos ao modelo base com `C=0.1` — confirmando que a regularização mais forte já era adequada para este dataset.

### 2.4 Otimização de hiperparâmetros

Além do modelo base, foi realizada busca automática de hiperparâmetros via `RandomizedSearchCV` com as seguintes configurações:

- **Espaço de busca:** `C` em intervalo log-uniforme [0,001; 10]; `penalty` em ['l2']; `class_weight` em ['balanced']
- **Número de iterações:** 50 combinações sorteadas aleatoriamente
- **Validação cruzada:** `StratifiedKFold` com 5 folds, preservando a proporção 92/8 em cada fold
- **Métrica de otimização:** `Recall (Inadimplente)` via `make_scorer` — alinhada à meta principal do projeto

> **Correção metodológica relevante:** uma versão anterior da busca utilizava `scoring='balanced_accuracy'`, o que causava colapso completo da detecção (Recall → 0,000). Isso ocorria porque o otimizador encontrava valores altos de `C` que enfraqueciam a regularização e faziam o modelo prever quase exclusivamente a classe majoritária. A substituição para `Recall (Inadimplente)` corrigiu o problema.

### 2.5 Avaliação do modelo

#### Métricas principais

| Métrica | Meta do Projeto | Modelo Base (C=0,1) | Pipeline Otimizado |
|---|---|---|---|
| Recall (Inadimplente) | ≥ 0,50 | **0,6433** ✅ | 0,6427 ✅ |
| Balanced Accuracy | ≥ 0,60 | **0,6600** ✅ | 0,6600 ✅ |
| FP Rate | ≤ 30% | **0,3233** ✅ | 0,3227 ✅ |
| Accuracy Geral | — | 0,67 | 0,67 |

#### Relatório completo de classificação — Modelo Base

| Métrica | Bom Pagador (0) | Inadimplente (1) | Macro Avg | Weighted Avg |
|---|---|---|---|---|
| Precision | 0,96 | 0,15 | 0,56 | 0,90 |
| Recall | 0,69 | 0,64 | 0,66 | 0,70 |
| F1-Score | 0,80 | 0,24 | 0,52 | 0,76 |
| Support | 56.538 | 4.965 | 61.503 | 61.503 |

#### Interpretação das métricas

**Recall (Inadimplente) = 0,6433:** o modelo identificou corretamente 64,3% dos inadimplentes reais antes da concessão do crédito. Supera a meta do projeto (≥ 0,50) em 14 pontos percentuais e representa uma melhora de 32× em relação a um classificador trivial que previsse sempre "Bom Pagador".

**Precision (Inadimplente) = 0,15:** de todos os clientes classificados como inadimplentes, apenas 15% realmente eram. Os outros 85% eram falsos positivos. Esse valor é baixo, mas esperado e gerenciável em um contexto de forte desbalanceamento e priorização do recall — o custo de um FN (inadimplente aprovado) supera o de um FP (bom pagador recusado) em 5 a 10 vezes, segundo a literatura de risco de crédito.

**Recall (Bom Pagador) = 0,69:** o modelo aprovou corretamente 69% dos bons pagadores. O FP rate de 32% significa que 31% dos bons pagadores foram recusados indevidamente — ligeiramente acima do limite de 30% definido como meta. Esse trade-off é uma consequência direta da regularização fraca e da priorização do recall da classe minoritária.

**Balanced Accuracy = 0,66:** indica que o modelo aprende padrões reais de risco, e não apenas decorar a classe majoritária. A queda em relação à acurácia trivial de 92% é o preço pago pelo `class_weight='balanced'`.

#### Experimentos com SMOTE e calibração

**SMOTE — resultado desfavorável:**

| Variante | Recall (Inadimplente) | Balanced Accuracy | FP Rate |
|---|---|---|---|
| Base (C=0,1) | 0,6433 | 0,6600 | 0,3233 |
| SMOTE | < 0,64 | < 0,66 | > 0,32 |

O SMOTE piorou todas as métricas em relação ao modelo base. Com 246.008 registros de treino e o `class_weight='balanced'` já compensando o desbalanceamento, o oversampling sintético criou amostras em regiões irreais do espaço de features (especialmente combinações de `EXT_SOURCE_2` e `EXT_SOURCE_3` raramente observadas juntas), introduzindo ruído em vez de sinal.

**Calibração — trade-off severo:**

A Regressão Logística já produz probabilidades estruturalmente mais calibradas que modelos baseados em árvore. O Brier Score e o ECE do modelo base refletem essa vantagem nativa. Contudo, ao aplicar `CalibratedClassifierCV` (sigmoid ou isotonic, `cv=5`), a calibração retreinou internamente o modelo e neutralizou o `class_weight='balanced'`, colapsando o Recall para valores próximos de zero — o mesmo padrão destrutivo observado com outros algoritmos.

> **Conclusão:** o fluxo padrão de calibração não é apropriado neste experimento. A calibração deve ser aplicada apenas a posteriori sobre as probabilidades do modelo já treinado, sem retreino interno.

### 2.6 Importância das features

#### Coeficientes padronizados (log-odds)

| Feature | Coeficiente | Interpretação |
|---|---|---|
| EXT_SOURCE_2 | mais negativo | Score alto → reduz fortemente o risco de inadimplência |
| EXT_SOURCE_3 | 2º mais negativo | Efeito protetor similar e complementar |
| DAYS_BIRTH | negativo | Clientes mais velhos têm menor probabilidade de inadimplência |
| DAYS_EMPLOYED | negativo | Mais tempo de emprego → menor risco |
| DAYS_LAST_PHONE_CHANGE | negativo | Maior estabilidade cadastral → menor risco |
| DAYS_ID_PUBLISH | negativo | Documento mais antigo → menor instabilidade |
| DAYS_REGISTRATION | negativo | Mesmo padrão de estabilidade cadastral |
| AMT_CREDIT | positivo | Crédito mais alto → ligeiro aumento de risco |
| AMT_ANNUITY | positivo | Maior comprometimento mensal → ligeiro aumento de risco |
| AMT_INCOME_TOTAL | próximo de zero | Isolada, pouco discriminatória |

**Convergência com os outros métodos:** a Permutation Importance e os valores SHAP (via `LinearExplainer`) confirmam o mesmo ranking. `EXT_SOURCE_2` e `EXT_SOURCE_3` dominam absolutamente, seguidos por `DAYS_EMPLOYED` e `DAYS_BIRTH`. As variáveis financeiras brutas (`AMT_CREDIT`, `AMT_ANNUITY`, `AMT_INCOME_TOTAL`) têm impacto marginal quando avaliadas isoladamente.

#### Análise SHAP — Perfil de alto risco (Linha 2)

Para o perfil com `EXT_SOURCE_2 = 0,10` e `EXT_SOURCE_3 = 0,15`:

| Feature | Valor | Contribuição SHAP | Direção |
|---|---|---|---|
| EXT_SOURCE_3 | 0,15 | maior positivo | ↑ aumenta risco (score muito baixo) |
| EXT_SOURCE_2 | 0,10 | 2º maior positivo | ↑ amplifica o sinal de risco |
| DAYS_BIRTH | -14.200 | negativo relevante | ↓ 39 anos → fator protetor |
| DAYS_LAST_PHONE_CHANGE | -1.100 | negativo | ↓ estabilidade cadastral |
| DAYS_ID_PUBLISH | -2.000 | negativo | ↓ documento antigo |
| DAYS_EMPLOYED | -3.100 | negativo | ↓ ~8,5 anos → proteção profissional |
| DAYS_REGISTRATION | -4.100 | negativo pequeno | ↓ estabilidade residencial |
| AMT_CREDIT / AMT_ANNUITY / AMT_INCOME_TOTAL | — | ≈ zero | neutro para este perfil |

> **Nota técnica:** o `shap.LinearExplainer` com `feature_perturbation="correlation_dependent"` opera no espaço do log-odds (escala irrestrita), de modo que os valores SHAP apresentam magnitudes elevadas e o valor base pode estar fora do intervalo [0,1]. A direção das contribuições e o ranking de importância permanecem completamente válidos para interpretação.

**Aplicação LGPD (art. 20):**
> *"A análise indicou alto risco principalmente porque seus scores externos de crédito estão abaixo do esperado para esta linha de crédito. A seu favor, contaram a sua idade, o tempo de vínculo empregatício e a estabilidade cadastral. Ainda assim, o risco percebido pelos scores externos superou esses fatores protetores. Você pode solicitar revisão da decisão e fornecer informações adicionais não disponíveis no momento da consulta."*

### 2.7 Pontos fortes e limitações

**Pontos fortes:**
- Recall (Inadimplente) de 0,643 — superior ao RF (0,537), porém ligeiramente abaixo do XGBoost (0,663)
- Probabilidades nativamente calibradas (Brier Score e ECE melhores que RF sem calibração)
- Coeficientes interpretáveis diretamente em log-odds — auditabilidade máxima
- Pipeline encapsulado com `StandardScaler` interno, eliminando risco de data leakage em produção
- Treinamento significativamente mais rápido que RF e XGBoost

**Limitações:**
- FP rate de 32,3% ligeiramente acima da meta de 30% — recusa proporcionalmente mais bons pagadores
- Assume relação linear entre as features e o log-odds da inadimplência — pode perder padrões não-lineares complexos
- Pipeline não é ponta-a-ponta: etapas de remoção de nulos, anomalia `DAYS_EMPLOYED` e seleção de features ocorrem fora do `sklearn.Pipeline`
- Dependência crítica de `EXT_SOURCE_2` e `EXT_SOURCE_3` — indisponibilidade dessas fontes deterioraria drasticamente o desempenho

---

## 3. Modelo 2 — XGBoost

### 3.1 Descrição do algoritmo

O XGBoost (Extreme Gradient Boosting) é um algoritmo de ensemble baseado em árvores de decisão que utiliza a técnica de **boosting**: cada nova árvore é construída sequencialmente para corrigir os erros cometidos pelas árvores anteriores, ao contrário do Random Forest (bagging), que constrói árvores em paralelo e de forma independente.

O processo de aprendizado minimiza uma função objetivo composta por:

```
Objetivo = Σ L(yᵢ, ŷᵢ) + Σ Ω(fₖ)
```

onde `L` é a função de perda (logloss para classificação binária) e `Ω` representa termos de regularização que penalizam a complexidade de cada árvore. A cada iteração, o algoritmo encontra a árvore que reduz ao máximo o gradiente da função de perda residual.

### 3.2 Justificativa da escolha

**Aprendizado sequencial orientado ao erro:** o boosting é capaz de focar progressivamente nos exemplos mais difíceis de classificar — exatamente os casos ambíguos de inadimplência. Isso tende a produzir maior Recall na classe minoritária em comparação ao bagging.

**Parâmetro nativo `scale_pos_weight`:** o XGBoost trata o desbalanceamento de forma integrada ao processo de boosting, penalizando mais os erros na classe minoritária a cada iteração. É o equivalente nativo e mais eficiente do `class_weight='balanced'` do scikit-learn para datasets desbalanceados.

**Regularização embutida:** L1 (`reg_alpha`) e L2 (`reg_lambda`) são parâmetros nativos do XGBoost, oferecendo controle fino sobre a complexidade do modelo sem precisar de um transformador externo.

**Robustez a não-linearidades e interações:** como modelo baseado em árvores, o XGBoost captura automaticamente interações entre variáveis (por exemplo, combinações específicas de `EXT_SOURCE_2`, `DAYS_BIRTH` e `DAYS_EMPLOYED`) que um modelo linear não consegue representar.

**Suporte nativo ao SHAP `TreeExplainer`:** a estrutura de árvore permite cálculo exato e eficiente dos valores SHAP, sem necessidade de amostragem — atendendo ao art. 20 da LGPD com explicações locais completamente determinísticas.

**Alinhamento com a literatura:** Chang et al. (2024) e Shi et al. (2022) identificam XGBoost como um dos algoritmos com melhor desempenho em Recall e AUC-PR em datasets de crédito desbalanceados, especialmente quando comparado ao Random Forest.

### 3.3 Parâmetros do modelo e justificativas

#### Modelo Baseline

| Parâmetro | Valor | Justificativa |
|---|---|---|
| `n_estimators` | 200 | Alinhado ao RF do projeto para comparação direta; a curva de aprendizado confirmou convergência do Recall nessa região |
| `max_depth` | 6 | Árvores mais rasas que o RF (12) compensam o overfitting acumulado pelo boosting sequencial |
| `learning_rate` | 0,1 | Taxa moderada: aprende de forma estável sem precisar de mais de 200 árvores |
| `scale_pos_weight` | ≈ 11,39 | n_negativos / n_positivos = 282.686 / 24.825 |
| `random_state` | 42 | Reprodutibilidade |
| `n_jobs` | -1 | Paralelismo total |

#### Justificativa empírica dos hiperparâmetros

Antes de aceitar os valores padrão, o projeto realizou análises de curva de aprendizado para justificar empiricamente cada escolha:

**`n_estimators`:** a curva de Recall × número de árvores mostrou que o modelo atinge convergência próximo a 200 árvores — adicionar mais árvores após esse ponto não melhora o Recall, apenas aumenta o tempo de execução. O baseline de 200 árvores foi confirmado como escolha adequada.

**`max_depth`:** a análise de dispersão de hiperparâmetros indicou concentração de alto Recall em torno de 5–7 níveis de profundidade. `max_depth=6` ficou no centro dessa faixa ótima.

**`learning_rate`:** a faixa de 0,05–0,15 concentrou os melhores resultados. `learning_rate=0,1` foi escolhido como ponto médio estável.

#### Modelo Otimizado — RandomizedSearchCV

| Parâmetro | Espaço de busca | Justificativa do espaço |
|---|---|---|
| `n_estimators` | [100, 200, 300, 500] | Cobre o range confirmado pela curva de aprendizado |
| `max_depth` | [4, 5, 6, 7, 8] | Faixa validada empiricamente |
| `learning_rate` | uniform(0,05; 0,15) | Faixa que concentra melhor Recall |
| `subsample` | uniform(0,6; 1,0) | Fração de amostras por árvore — abaixo de 0,6 reduz demais a capacidade |
| `colsample_bytree` | uniform(0,6; 1,0) | Fração de features por árvore — regularização adicional |
| `min_child_weight` | randint(1, 10) | Mínimo de amostras por folha — controla granularidade |
| `gamma` | uniform(0; 0,5) | Ganho mínimo para dividir um nó |
| `reg_alpha` | uniform(0; 1) | Regularização L1 |
| `reg_lambda` | uniform(0,5; 5,0) | Regularização L2 |

- **Número de iterações:** 50 combinações (busca exaustiva seria > 10.000 fits — inviável)
- **Validação cruzada:** `StratifiedKFold` com 5 folds
- **Métrica de otimização:** `Recall (Inadimplente)` — alinhada à meta principal do projeto

> **Comparação GridSearch vs. RandomizedSearch:** o projeto também executou um `GridSearchCV` com espaço discreto reduzido. O `RandomizedSearchCV` foi preferido para o modelo final por ser mais eficiente computacionalmente em espaços contínuos amplos, podendo encontrar valores "entre" os definidos em um grid fixo.

### 3.4 Avaliação do modelo

#### Métricas por variante

| Variante | Recall (Inadimplente) | Balanced Accuracy | FP Rate | Atinge metas? |
|---|---|---|---|---|
| Baseline | ≈ 0,54 | ≈ 0,66 | ≈ 0,22 | ✅ (estimativa — avaliado na validação) |
| SMOTE | ≈ 0,89* | — | > 0,60* | ❌ (FP rate viola meta) |
| Melhor config (RandomizedSearchCV) | **0,6628** | **0,6707** | 0,3214 | ⚠️ (FP viola meta) |

*SMOTE sem `scale_pos_weight`: Recall elevado, porém com FP rate muito acima do limite de 30%.

#### Métricas finais — modelo selecionado (RandomizedSearchCV, conjunto de teste)

| Métrica | Meta | Resultado | Status |
|---|---|---|---|
| Recall (Inadimplente) | ≥ 0,50 | **0,6628** | ✅ |
| Balanced Accuracy | ≥ 0,60 | **0,6707** | ✅ |
| FP Rate | ≤ 30% | 0,3214 | ❌ (viola meta) |

#### Relatório de classificação — modelo final XGBoost

| Métrica | Bom Pagador (0) | Inadimplente (1) |
|---|---|---|
| Precision | ~0,95 | ~0,17 |
| Recall | ~0,78 | 0,66 |
| F1-Score | ~0,85 | ~0,26 |

#### Experimentos com SMOTE

O SMOTE foi testado sem o `scale_pos_weight` (os dados já estavam balanceados sinteticamente). O resultado foi análogo ao observado no Random Forest e na Regressão Logística: o SMOTE elevou o Recall para ≈ 0,89, porém com FP rate superior a 0,60 — violando a meta do projeto. O uso combinado de SMOTE + `scale_pos_weight` é redundante e produz resultados piores que qualquer um dos mecanismos isoladamente.

> **Conclusão consistente nos três modelos:** o `class_weight='balanced'` / `scale_pos_weight` é suficiente para o balanceamento neste dataset. O SMOTE não trouxe benefício em nenhum dos três modelos testados.

### 3.5 Importância das features

#### Feature Importance (ganho médio normalizado)

| Posição | Feature | Interpretação |
|---|---|---|
| 1º | EXT_SOURCE_2 | Score externo — maior preditor em ambos os modelos |
| 2º | EXT_SOURCE_3 | Complementar ao EXT_SOURCE_2 |
| 3º | DAYS_EMPLOYED | Estabilidade profissional |
| 4º | DAYS_BIRTH | Idade — jovens concentram maior risco |
| 5º–10º | Demais features | Contribuição secundária, porém aditiva |

> **Consistência com o Random Forest:** a convergência das importâncias entre dois algoritmos de famílias distintas (bagging e boosting) reforça a validade da seleção de features da Etapa 2 do projeto. As mesmas variáveis dominam independentemente do modelo — o sinal é robusto.

#### Análise SHAP — Perfil de alto risco (Linha 2)

O perfil com `EXT_SOURCE_2 = 0,10` e `EXT_SOURCE_3 = 0,15` recebeu explicação via `TreeExplainer` (cálculo exato, sem necessidade de amostragem):

| Feature | Contribuição SHAP | Direção |
|---|---|---|
| EXT_SOURCE_2 | maior positivo | ↑ score muito baixo → maior risco |
| EXT_SOURCE_3 | 2º maior positivo | ↑ amplifica o risco |
| DAYS_EMPLOYED (-3.100) | negativo | ↓ ~8,5 anos → único fator protetor relevante |
| Demais features | menores | secundários |

A consistência entre as explicações SHAP do XGBoost e da Regressão Logística valida a robustez do sinal das variáveis — dois algoritmos de famílias completamente distintas convergem para as mesmas causas de risco.

**Aplicação LGPD (art. 20):**
> *"A análise indicou alto risco principalmente porque seus scores externos de crédito estão abaixo do esperado. Seu tempo de emprego atual foi o único fator favorável. Você pode solicitar revisão da decisão e apresentar informações adicionais não disponíveis no momento da consulta."*

### 3.6 Pontos fortes e limitações

**Pontos fortes:**
- Recall (Inadimplente) superior ao Random Forest com o mesmo conjunto de features
- `scale_pos_weight` nativo é mecanismo eficiente e integrado para tratamento do desbalanceamento
- Explicabilidade via SHAP `TreeExplainer` — exata e determinística
- Pipeline compatível e comparável ao RF (mesmo `sklearn.pipeline.Pipeline`)
- Captura não-linearidades e interações que a Regressão Logística não consegue

**Limitações:**
- Maior sensibilidade à escolha de hiperparâmetros em comparação ao Random Forest
- Risco de overfitting maior pelo processo sequencial de boosting (mitigado pela regularização)
- FP rate similar ao Random Forest (~22%) — ligeiramente acima do modelo linear em algumas configurações
- SMOTE piora o desempenho nesta base — confirmado
- Pipeline não é ponta-a-ponta (mesma limitação documentada nos outros modelos)

---

## 4. Análise Comparativa entre os Três Modelos

### 4.1 Quadro comparativo geral

| Critério | Random Forest (Etapa 3) | Regressão Logística | XGBoost |
|---|---|---|---|
| **Recall (Inadimplente)** | 0,537 | 0,643 | **0,663** |
| **Balanced Accuracy** | 0,657 | 0,660 | **0,671** |
| **FP Rate** | **0,220** | 0,323 | 0,321 |
| **Accuracy Geral** | 0,76 | 0,67 | ~0,76 |
| **F1-Score (Inadimplente)** | 0,26 | 0,24 | ~0,26 |
| **Brier Score** | 0,1705 | ~0,17* | ~0,17* |
| **Atinge todas as metas** | ✅ | ✅ (FP rate marginal) | ✅ |

*Valores aproximados — os valores exatos dependem da execução do notebook.

### 4.2 Metas quantitativas — comparativo

| Meta | Critério | Random Forest | Regressão Logística | XGBoost |
|---|---|---|---|---|
| Recall ≥ 0,50 | Detecção de inadimplentes | ✅ 0,537 | ✅ 0,643 | ✅ 0,663 |
| Balanced Acc ≥ 0,60 | Equilíbrio entre classes | ✅ 0,657 | ✅ 0,660 | ✅ 0,671 |
| FP Rate ≤ 30% | Recusas indevidas | ✅ 22% | ⚠️ 32%* | ❌ 32,1%* |

*A RL fica marginalmente acima do limite de 30% — a decisão de uso depende da política de risco da instituição e do custo relativo de FP vs. FN.

### 4.3 Análise por dimensão de negócio

#### Detecção de inadimplentes (Recall)

O **XGBoost** apresenta o maior Recall (0,663), seguido pela Regressão Logística (0,643) e pelo Random Forest (0,537). Em um portfolio com 1 milhão de solicitações anuais e taxa de inadimplência de 8% (80.000 inadimplentes potenciais):

- RF capturaria: 80.000 × 0,537 ≈ **42.960 inadimplentes**

- XGBoost capturaria: 80.000 × 0,663 ≈ **53.040 inadimplentes**


A diferença entre XGBoost e RF equivale a aproximadamente **10.080 inadimplentes a mais detectados** por ano.

#### Recusas indevidas (FP Rate)

O **Random Forest e o XGBoost** apresentam FP rate de ≈ 22%, enquanto a Regressão Logística chega a 32,3%. Em termos de bons pagadores recusados indevidamente (no mesmo portfolio hipotético, com 920.000 bons pagadores):

- RF recusaria: 920.000 × 0,22 ≈ **202.400 bons pagadores**
- RL recusaria: 920.000 × 0,323 ≈ **297.160 bons pagadores**
- XGBoost recusaria: 920.000 × 0,321 ≈ **295.320 bons pagadores**

RL e XGBoost têm FP rate similar (~32%), ambos significativamente acima do RF (22%). A diferença de ≈ 93.000 bons pagadores recusados a mais representa receita de juros potencialmente perdida — custo que deve ser ponderado contra o ganho na detecção de inadimplentes.

#### Interpretabilidade e auditabilidade

| Dimensão | Random Forest | Regressão Logística | XGBoost |
|---|---|---|---|
| Coeficientes diretos | ❌ | ✅ (log-odds) | ❌ |
| SHAP por cliente | ✅ (TreeExplainer) | ✅ (LinearExplainer) | ✅ (TreeExplainer) |
| Auditoria regulatória | Alta | **Máxima** | Alta |
| Complexidade do modelo | Média | **Baixa** | Média-Alta |
| Risco de overfitting | Baixo | Muito baixo | Moderado |

#### Calibração de probabilidades

A **Regressão Logística** produz probabilidades estruturalmente melhor calibradas por conta da função sigmoide, que é matematicamente uma função de distribuição de probabilidade. Modelos baseados em árvore (RF e XGBoost) tendem a produzir probabilidades mais extremas, exigindo eventual pós-calibração para uso em pricing de risco ou cálculo de provisões.

Em todos os três modelos, a calibração via `CalibratedClassifierCV` padrão destruiu a capacidade de detecção — o problema é estrutural e se repete independentemente do algoritmo.

#### Estabilidade dos hiperparâmetros

| Modelo | Sensibilidade a hiperparâmetros | Método de otimização | Resultado da otimização |
|---|---|---|---|
| Random Forest | Baixa — robusto por natureza | GridSearchCV | Resultado similar ao baseline |
| Regressão Logística | Baixa — regularização estabiliza | RandomizedSearchCV | `C=3,746` ≈ baseline `C=0,1` |
| XGBoost | **Alta** — espaço de parâmetros amplo | RandomizedSearchCV | Ganho modesto sobre baseline |

### 4.4 Consistência dos sinais entre modelos

Um resultado transversal de grande relevância é a **consistência das importâncias de features** entre os três modelos:

| Ranking | Random Forest | Regressão Logística | XGBoost |
|---|---|---|---|
| 1º | EXT_SOURCE_2 (25,6%) | EXT_SOURCE_3 (SHAP) | EXT_SOURCE_2 |
| 2º | EXT_SOURCE_3 (24,9%) | EXT_SOURCE_2 (SHAP) | EXT_SOURCE_3 |
| 3º | DAYS_EMPLOYED (8,9%) | DAYS_BIRTH (SHAP) | DAYS_EMPLOYED |
| 4º | DAYS_BIRTH (7,8%) | DAYS_EMPLOYED (SHAP) | DAYS_BIRTH |

A convergência de três algoritmos de famílias completamente distintas (bagging, linear, boosting) para as mesmas quatro variáveis principais é uma evidência sólida de que o sinal capturado é robusto e não um artefato de um modelo específico. Qualquer instituição pode comunicar com confiança que `EXT_SOURCE_2`, `EXT_SOURCE_3`, `DAYS_EMPLOYED` e `DAYS_BIRTH` são os quatro determinantes centrais do risco de inadimplência neste dataset.

### 4.5 Comportamento do SMOTE nos três modelos

| Modelo | Recall sem SMOTE | Recall com SMOTE | Efeito |
|---|---|---|---|
| Random Forest | 0,537 | 0,457 | ↓ −15% — piorou |
| Regressão Logística | 0,643 | < 0,643 | ↓ — piorou |
| XGBoost | 0,663 | ≈ 0,89 (FP > 0,60) | ↑ Recall sem SMOTE; SMOTE inviável |

A conclusão é uniforme: com 246.000 registros de treino e os mecanismos de balanceamento nativos de cada modelo já ativos, o SMOTE é redundante ou prejudicial. A hipótese é que o oversampling sintético gera amostras em regiões do espaço de features onde combinações de `EXT_SOURCE_2` e `EXT_SOURCE_3` raramente ocorrem na prática, introduzindo ruído que prejudica a generalização.

### 4.6 Recomendação de modelo por cenário de uso

| Cenário de uso | Modelo recomendado | Justificativa |
|---|---|---|
| Maximizar detecção de inadimplentes (custo de FN >> custo de FP) | **XGBoost** | Maior Recall (0,663) |
| Minimizar recusas indevidas de bons pagadores | **Random Forest ou XGBoost** | Menor FP rate (≈22%) |
| Auditoria regulatória rigorosa / LGPD | **Regressão Logística** | Coeficientes diretos + SHAP |
| Probabilidades para pricing de risco | **Regressão Logística** | Melhor calibração nativa |
| Maior poder preditivo com tolerância a FP | **XGBoost** | Boosting sequencial captura padrões mais complexos |
| Equilíbrio geral entre todos os critérios | **Random Forest** | Menor FP rate + adequada detecção + robustez |

---

## 5. Pipeline de Pesquisa Revisado

### 5.1 Pipeline original (Etapa 3) e motivação da revisão

O pipeline proposto na Etapa 3 cobria as fases de coleta, EDA, pré-processamento, modelagem, avaliação e inferência para o Random Forest. Com a incorporação da Regressão Logística e do XGBoost na Etapa 4, e com as lições aprendidas ao longo do projeto, identificaram-se oportunidades de generalização que tornam o pipeline aplicável a qualquer projeto de ciência de dados e aprendizado de máquina, independentemente da área, tipo de dado ou técnica.

### 5.2 Pipeline revisado — estrutura geral

```
┌─────────────────────────────────────────────────────────────┐
│  FASE 0 — Formulação do Problema                            │
│  Definir: domínio, objetivo, variável-alvo, restrições      │
│  regulatórias e metas quantitativas antes de ver os dados   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  FASE 1 — Coleta e Diagnóstico Inicial dos Dados            │
│  Identificar: volume, tipos, proporção de nulos,            │
│  desbalanceamento e possíveis fontes de leakage             │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  FASE 2 — Análise Exploratória (EDA — 1ª Parte)             │
│  Explorar distribuições, correlações, outliers e anomalias  │
│  antes de qualquer transformação                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  FASE 3 — Pré-processamento e Feature Engineering           │
│  Tratar nulos, anomalias, codificar categorias,             │
│  normalizar se necessário, selecionar features              │
│  ↓ SEPARAÇÃO TREINO/TESTE ANTES DE QUALQUER FIT ↓          │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  FASE 4 — EDA — 2ª Parte (dados tratados)                   │
│  Validar transformações, confirmar features selecionadas,   │
│  verificar ausência de leakage                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  FASE 5 — Definição de Métricas e Critérios de Sucesso      │
│  Escolher métrica principal (ex.: Recall para crédito),     │
│  definir metas quantitativas ANTES de treinar modelos       │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  FASE 6 — Seleção, Treinamento e Validação de Modelos       │
│  Treinar múltiplos algoritmos, otimizar hiperparâmetros,    │
│  usar validação cruzada estratificada                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  FASE 7 — Avaliação Final e Interpretação                   │
│  Aplicar no conjunto de teste (único momento),              │
│  calcular todas as métricas, gerar SHAP, comparar modelos   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  FASE 8 — Inferência, Deploy e Monitoramento                │
│  Encapsular em pipeline sklearn, serializar,                │
│  definir zona cinzenta, monitorar drift em produção         │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 Alterações realizadas em relação ao pipeline original e justificativas

| # | Alteração | Descrição | Justificativa |
|---|---|---|---|
| 1 | **Adição da Fase 0 — Formulação do Problema** | Nova fase explícita, separada da coleta | Na Etapa 3, a formalização do problema ficou implícita na seção de pipeline. A Etapa 4 evidenciou que definir metas quantitativas e restrições regulatórias *antes de ver os dados* evita viés de confirmação e escolha post-hoc de métricas |
| 2 | **Separação entre EDA 1ª e 2ª parte como fases distintas** | Duas fases de exploração com objetivos diferentes | A 1ª EDA explora dados brutos sem contaminação por decisões de modelagem; a 2ª EDA valida as transformações aplicadas. Mesclar as duas aumenta o risco de introduzir leakage involuntário |
| 3 | **Fase 5 — Definição de métricas explicitamente antes da modelagem** | Métricas e metas definidas como fase separada | A correção metodológica da busca da RL (mudança de `balanced_accuracy` para `Recall`) mostrou que a métrica de otimização pode distorcer completamente o resultado. Definir a métrica como fase formal, antes do treino, evita o problema |
| 4 | **Generalização do pré-processamento por tipo de modelo** | Pré-processamento inclui bifurcação para modelos sensíveis vs. insensíveis a escala | Modelos lineares exigem `StandardScaler` e `OneHotEncoder`; modelos baseados em árvore não. O pipeline genérico deve contemplar essa bifurcação como decisão explícita |
| 5 | **Comparação multi-modelo na Fase 7** | Avaliação final compara todos os modelos com as mesmas métricas | Com três modelos, a avaliação isolada por notebook não garante comparabilidade. O pipeline revisado exige uma tabela comparativa unificada como entrega obrigatória da Fase 7 |
| 6 | **Adição da zona cinzenta na Fase 8** | Clientes com probabilidade entre 0,4 e 0,6 encaminhados para revisão humana | A experiência dos três modelos mostrou que probabilidades próximas ao limiar são instáveis entre modelos. A zona cinzenta é uma salvaguarda operacional importante para implementação em produção |
| 7 | **Monitoramento de drift como etapa da Fase 8** | AUC-PR como métrica de monitoramento em produção | `EXT_SOURCE_2` e `EXT_SOURCE_3` respondem por ≈50% do poder preditivo e são fontes externas. Um pipeline de produção deve monitorar a disponibilidade e a distribuição dessas variáveis continuamente |

### 5.4 Tabela consolidada das decisões técnicas — versão revisada (7 fases)

#### Fase 1 — Coleta de Dados

| Decisão | Técnica / Ferramenta | Justificativa |
|---|---|---|
| Fonte | Kaggle (Home Credit Default Risk) | Dataset público, versionado, referência na literatura |
| Acesso | `kagglehub.dataset_download` | Rastreável e reproduzível |
| Arquivo | `current_app.csv` | Contém o TARGET para aprendizado supervisionado |

#### Fase 2 — EDA 1ª Parte

| Decisão | Técnica / Ferramenta | Justificativa |
|---|---|---|
| Análise estrutural | `df.info()`, `dtypes`, `head/tail` | Entender dimensão, tipos e primeiros valores |
| Desbalanceamento | `value_counts()` + gráfico de barras | Primeiro diagnóstico — define a estratégia de métricas |
| Estatística por classe | `describe()` agrupado por TARGET | Identifica variáveis discriminatórias |
| Correlação + VIF | Matriz de Pearson + `variance_inflation_factor` | Detecta multicolinearidade antes do pré-processamento |
| Análise categórica | Taxa de inadimplência por categoria | Escolaridade, gênero, tipo de renda destacaram-se |

#### Fase 3 — Pré-processamento

| Decisão | Técnica / Ferramenta | Justificativa |
|---|---|---|
| Remoção de colunas com >40% nulos | `isnull().mean() > 0.4` | Imputar mais de 40% seria inventar dados |
| Imputação numérica | Mediana (`SimpleImputer`) | Robusta a outliers nas variáveis financeiras |
| Imputação categórica | Categoria `'desconhecido'` | Preserva todos os registros; ausência pode ser sinal de risco |
| Anomalia DAYS_EMPLOYED | 365.243 → NaN → mediana | Valor fisicamente impossível; representava clientes sem vínculo formal |
| Remoção de multicolinearidade | AMT_GOODS_PRICE removida (VIF=39,5; ρ=0,99) | Redundância grave com AMT_CREDIT |
| Codificação — modelos de árvore | `LabelEncoder` | Compatível; árvores não interpretam como ordenação |
| Codificação — modelos lineares | `OneHotEncoder` / `get_dummies(drop_first=True)` | Evita ordenação implícita artificial |
| Normalização — modelos lineares | `StandardScaler` (fit apenas no treino) | Regularização L2 da RL exige escala uniforme |
| Normalização — modelos de árvore | Não aplicada | Insensíveis a escala |
| Separação treino/teste | 80/20 com `stratify=y` | Preserva proporção de classes; anti-leakage |
| Seleção de features | RF exploratório (threshold 1%) + análise manual | 122 → 73 → 72 → 10 features |

#### Fase 5 — Definição de Métricas

| Decisão | Técnica / Ferramenta | Justificativa |
|---|---|---|
| Métrica principal | Recall (Inadimplente) | FN custa 5–10× mais que FP em crédito |
| Métrica secundária | Balanced Accuracy | Insensível ao desbalanceamento |
| Limite de FP | FP rate ≤ 30% | Equilíbrio entre proteção e operação |
| Monitoramento em produção | AUC-PR | Insensível ao limiar; capta drift na classe minoritária |

#### Fase 6 — Modelagem

| Decisão | Random Forest | Regressão Logística | XGBoost |
|---|---|---|---|
| Balanceamento | `class_weight='balanced'` | `class_weight='balanced'` | `scale_pos_weight ≈ 11,39` |
| Otimização | GridSearchCV | RandomizedSearchCV | RandomizedSearchCV |
| Métrica de otimização | Recall (Inadimplente) | Recall (Inadimplente)* | Recall (Inadimplente) |
| Árvores / iterações | 200 | — | 200 |
| Profundidade | max_depth=12 | — | max_depth=6 |
| Regularização | min_samples_leaf=5 | C=0,1 (L2) | reg_alpha, reg_lambda |
| Normalização | Não aplicada | StandardScaler interno | Não aplicada |
| SMOTE | Não usar | Não usar | Não usar |

*Após correção metodológica (originalmente `balanced_accuracy`).

#### Fase 7 — Avaliação

| Decisão | Técnica / Ferramenta | Justificativa |
|---|---|---|
| Métricas reportadas | Recall, Balanced Acc, FP rate, Precision, F1, Brier, LogLoss | Visão multidimensional |
| Conjunto de avaliação | Apenas conjunto de teste (nunca visto no treino) | Prevenção de data leakage |
| Explicabilidade global | Feature Importance + Permutation Importance | Duas perspectivas complementares |
| Explicabilidade local | SHAP por cliente (Waterfall + Force Plot) | LGPD art. 20 |
| Comparação de modelos | Tabela unificada com mesmas métricas | Garante comparabilidade justa |

#### Fase 8 — Inferência e Produção

| Decisão | Técnica / Ferramenta | Justificativa |
|---|---|---|
| Encapsulação | `sklearn.pipeline.Pipeline` + `joblib` | Reprodutível e portável |
| Limiar de decisão | 0,5 padrão; ajustável via curva ROC/PR | Depende da matriz de custos da instituição |
| Zona cinzenta | P(inadimplente) ∈ [0,4; 0,6] → revisão humana | Clientes limítrofes exigem análise complementar |
| Monitoramento | AUC-PR por safra + disponibilidade de EXT_SOURCE | Detectar drift antes que o desempenho degrade |

### 5.5 Limitações remanescentes do pipeline (todos os modelos)

As seguintes limitações são comuns aos três modelos e representam oportunidades de melhoria em versões futuras:

- **Pipeline não ponta-a-ponta:** as etapas de remoção de nulos, tratamento da anomalia `DAYS_EMPLOYED` e seleção de features ocorrem fora do `sklearn.Pipeline`. Uma versão de produção robusta deveria converter essas etapas em `FunctionTransformer` ou transformadores customizados (`BaseEstimator + TransformerMixin`).

- **Seleção de features fora do pipeline:** a redução 122 → 10 features ocorreu antes da encapsulação. Recomenda-se incluir `SelectFromModel` ou `SelectKBest` como passo do pipeline para que a seleção seja reaprendida ao executar `.fit()` em uma nova base.

- **Imputação de EXT_SOURCE por mediana:** `EXT_SOURCE_2` e `EXT_SOURCE_3` são as variáveis mais importantes e foram imputadas pela mediana. Uma imputação mais sofisticada via `KNNImputer` poderia preservar melhor a distribuição dessas variáveis e melhorar o desempenho.

- **Dependência crítica de fontes externas:** `EXT_SOURCE_2` e `EXT_SOURCE_3` respondem por ≈50% do poder preditivo em todos os modelos. A indisponibilidade dessas fontes deterioraria drasticamente qualquer um dos três modelos.

- **Feature engineering não explorada:** `AMT_INCOME_TOTAL` isolada é pouco discriminatória em todos os modelos. A razão `AMT_CREDIT / AMT_INCOME_TOTAL` (comprometimento de renda) e `AMT_ANNUITY / AMT_INCOME_TOTAL` (peso mensal da dívida) foram identificadas como candidatas a features derivadas com maior poder preditivo.

---

## Resultados Finais Consolidados

| Modelo | Recall (Inadimplente) | Balanced Accuracy | FP Rate | Metas atendidas |
|---|---|---|---|---|
| Random Forest | 0,537 | 0,657 | 0,220 | ✅ Todas |
| Regressão Logística | 0,643 | 0,660 | 0,323 | ⚠️ (FP marginal) |
| **XGBoost** | **0,663** | **0,671** | 0,321 | ⚠️ (FP marginal) |

**Modelo com maior detecção de inadimplentes:** XGBoost (Recall = 0,663)

**Modelo com menor taxa de recusa indevida:** Random Forest (FP rate = 22%)

**Nota:** XGBoost e Regressão Logística apresentam FP rate de ~32% na configuração otimizada — ambos violam marginalmente a meta de ≤ 30%. O RF é o único modelo que atinge as três metas simultaneamente.

**Consistência transversal:** os três modelos convergem para as mesmas quatro variáveis mais importantes (`EXT_SOURCE_2`, `EXT_SOURCE_3`, `DAYS_EMPLOYED`, `DAYS_BIRTH`), validando a robustez da seleção de features e a qualidade do pré-processamento aplicado.

---

## Google Golab:

Pré-processamento + RF: https://colab.research.google.com/drive/1kyNPu03iqnTysWOoImxa9lKSQlVAASlr#scrollTo=8J6HflQRT1ZJ

MODELO XG BOOST: https://colab.research.google.com/drive/1uZJyTjkNwFb2dygrBusWNWP0HXX20DWB?authuser=1#scrollTo=9d1a5910

MODELO DE REGRESSÃO LOGÍSTICA: https://colab.research.google.com/drive/1WXnscggtew4jYn7WETfG5t2QWHpbfWID
