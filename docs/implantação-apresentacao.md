# Implantação da solução

##  Descrição do processo de implantação

### 1.1 Visão geral da solução

O sistema implantado é uma [API de avaliação de risco de crédito](https://creditanalysis-ayahd8b7bpehgmat.eastus-01.azurewebsites.net/docs), construída em Python usando FastAPI, que expõe um endpoint REST capaz de receber os dados de um cliente e retornar a probabilidade estimada de inadimplência, calculada por um modelo XGBoost previamente treinado. A aplicação não possui banco de dados: ela é stateless, ou seja, cada requisição é processada de forma independente, sem persistência de histórico. Planejamos que o consumo será feito por dois tipos de clientes: o frontend hospedado na Vercel (uma interface de análise para colaboradores do banco) e outros sistemas internos do banco que integram essa API ao fluxo de concessão de crédito (por exemplo, um sistema de originação de propostas de renegociação de dividas).

### 1.2 Preparação do artefato do modelo

O modelo foi treinado em um notebook Jupyter (XGBoost.ipynb), utilizando um pipeline scikit-learn que encapsula tanto o pré-processamento (imputação de valores nulos pela mediana) quanto o classificador XGBoost. Ao final do treinamento, esse pipeline foi serializado em um arquivo `pipeline_xgboost_v1.joblib`, que passou a ser versionado junto ao código-fonte da API no repositório Git. Esse arquivo é o artefato central da implantação: ele é carregado uma única vez, em memória, no momento em que a aplicação inicia, e permanece disponível para todas as requisições subsequentes enquanto o processo estiver ativo.

### 1.3 Empacotamento da aplicação

A aplicação foi estruturada com três componentes principais: o `main.py`, contendo a definição da API FastAPI, o schema de entrada via Pydantic e o endpoint `/predict`; o `requirements.txt`, com as dependências fixadas em versões específicas (Python 3.11, scikit-learn 1.6.1, xgboost 3.2.0, fastapi, uvicorn, joblib, pandas) para garantir reprodutibilidade entre o ambiente de desenvolvimento e o ambiente de produção; e o `startup.sh`, que define o comando de inicialização do servidor ASGI via uvicorn.

### 1.4 Provisionamento da infraestrutura no Azure

A infraestrutura foi provisionada manualmente pelo portal do Azure, utilizando a conta de estudante (Azure for Students). Primeiro foi criado um Resource Group dedicado ao projeto, que agrupa os recursos relacionados. Dentro desse grupo, foi criado um Azure App Service (Web App) com sistema operacional Linux, runtime Python 3.11 e plano de hospedagem F1 (Free Tier). A escolha do App Service em vez de uma abordagem containerizada simplificou o processo, eliminando a necessidade de criar, versionar e manter uma imagem Docker, o que foi adequado para o estágio atual do projeto.

### 1.5 Configuração da aplicação no App Service

Após o provisionamento, foi configurado o comando de inicialização (`bash startup.sh`) na seção Configuration > General Settings do App Service, instruindo a plataforma a executar o uvicorn apontando para a instância `app` definida em `main.py`. 

### 1.6 Pipeline de integração e entrega contínua (CI/CD)

A entrega contínua foi configurada através do Deployment Center do App Service, integrado diretamente ao repositório no GitHub. Essa integração gera automaticamente um workflow de GitHub Actions, que é disparado a cada push na branch principal. O workflow realiza três etapas: build (instalação das dependências listadas no `requirements.txt` em um ambiente Python 3.11), empacotamento dos artefatos da aplicação (incluindo o arquivo `.joblib` do modelo) e deploy desse pacote para o App Service, que reinicia o processo para carregar a nova versão. Não há etapa de treinamento de modelo nesse pipeline: o modelo é tratado como um artefato estático, versionado junto ao código.

### 1.7 Validação pós-implantação

Após cada deploy, a validação é feita em três níveis: primeiro, verificação do endpoint de health check ([GET /](https://creditanalysis-ayahd8b7bpehgmat.eastus-01.azurewebsites.net/)), que confirma que a aplicação subiu e respondeu; segundo, um teste funcional do endpoint `POST /predict`, enviando um payload de exemplo e confirmando que a resposta contém a classificação e a probabilidade esperadas ([API de Predição XGBoost - Swagger UI](https://creditanalysis-ayahd8b7bpehgmat.eastus-01.azurewebsites.net/docs)); terceiro, inspeção do Log Stream do App Service, para identificar eventuais erros de inicialização, falhas de importação de bibliotecas ou problemas ao carregar o `.joblib`. 

### 1.8 Ciclo de atualização

Qualquer alteração, segue o mesmo fluxo: commit no repositório, push para a branch principal, execução automática do workflow de CI/CD e substituição da versão em produção. É importante apontar que, no Plano F1, esse processo não conta com deployment slots (ambientes de staging para troca sem downtime), de forma que durante o redeploy a aplicação fica temporariamente indisponível.

### 1.9 Arquitetura da solução

![arquitetura](https://github.com/ICEI-PUC-Minas-PMV-SI/PUC_E7_T2_G1_2026_01/blob/main/docs/img/arquitetura_implantacao_api_credito.png)

## Planejamento e avaliação de capacidade

### 2.1 Cenário  hipotético de Produção

O sistema consiste em um serviço interno de análise de crédito utilizado por um banco com atuação em todo o território brasileiro, cujo objetivo é apoiar a tomada de decisão na concessão de crédito a clientes. A partir de requisições enviadas por outros sistemas (como plataformas de solicitação de empréstimos ou cartões), a API recebe dados cadastrais e financeiros do cliente e aplica um modelo preditivo treinado para estimar a probabilidade de inadimplência. Com base nesse resultado, o sistema retorna uma resposta no formato:

```json
{
	"prediction": 0,
	"probability": 0.4004741907119751
}
```

Essa resposta auxilia analistas e regras automatizadas a decidir se o crédito deve ser aprovado, recusado ou encaminhado para uma análise mais detalhada, garantindo maior eficiência, padronização e redução de riscos nas operações financeiras.

### 2.1 Objetivo e escopo

O objetivo desta avaliação é verificar se a infraestrutura atual (Azure App Service, Plano F1) é adequada para o cenário de uso descrito acima e, a partir dessa verificação, propor o dimensionamento adequado para o cenário de produção descrito. Como não há dados reais de tráfego disponíveis, as premissas de demanda apresentadas a seguir são estimativas também hipotéticas.

### 2.2 Condição de uso

Vamos considerar um cenário em que o sistema de crédito do banco realiza, em escala nacional, uma chamada para cada proposta de crédito analisada (cartão de crédito, empréstimo pessoal, financiamento). Vamos assumir um volume diário de 80.000 propostas analisadas, concentradas no horário comercial. Vamos considerar, também, que 80% desse volume ocorre em uma periodo de 12 horas (8h às 20h), e o restante é distribuído ao longo da madrugada por processos automatizados (como reavaliações periódicas de limite). 

Na prática o uso da API não é constante, existem momentos que muita gente usa ao mesmo tempo então vamos assumir um fator de pico de 3x sobre a média de uso no horário comercial, para simular momentos mais movimentados, ou seja, o sistema pode receber 3 vezes mais requisições que a média nos horarios de pico. Isso é importante pois se o sistema for diemensionado pela média a latencia pode aumentar e requisições criticas podem falhar resultando em um sistema indisponivel no horario de maior movimentação.

### 2.3 Estimativa de demanda

A partir dos que foi dito acima, é possível estimar a taxa de requisições por segundo em alguns cenários:

| Cenário                           | Cálculo                  | Resultado aproximado |
| --------------------------------- | ------------------------ | -------------------- |
| Volume diário total               | 80.000 requisições/dia   |                      |
| Volume em horário comercial (80%) | 64.000 requisições / 12h |                      |
| Taxa média (horário comercial)    | 64.000 / 43.200 s        | ≈ 1,5 req/s          |
| Taxa de pico (fator 3x)           | 1,5 × 3                  | ≈ 4,4 req/s          |
| Taxa fora do horário comercial    | 16.000 / 43.200 s        | ≈ 0,4 req/s          |

Esses números mostram que, mesmo em um cenário de escala nacional para um banco de médio/grande porte, a taxa de requisições por segundo é relativamente modesta sob a ótica de throughput puro. O ponto crítico, como será visto a seguir, não está na taxa de requisições por segundo, e sim em como essa demanda se traduz em consumo acumulado de recursos ao longo do dia frente às cotas do plano Free.

A análise da estimativa de demanda mostra que, apesar da taxa de requisições por segundo ser não ser tão alta, inclusive nos momentos de pico, o principal problema não está no volume de acessos, e sim no consumo contínuo de recursos ao longo do dia. Como o sistema permanece em execução constante e está hospedado em um ambiente com recursos limitados (Plano Free), o acúmulo desse processamento ao longo do tempo pode gerar impactos. Então mesmo com uma quantidade moderada de req/s, o uso de CPU e memória pode levar à perda de desempenho ou até indisponibilidade. 

A seguir a tabela de recursos do plano gratuito:

|        ACU/vCPU         | vCPU | Memoria(GB) | Armazenamento(GB) | Escala | SLA  |
| :---------------------: | :--: | :---------: | :---------------: | :----: | :--: |
| 60min/dia de computacao |  NA  |      1      |         1         |   NA   |  NA  |

- ACU/vCPU: 60 minutos usando CPU de verdade
- vCPU: Não há garantia de CPU dedicada. CPU é compartilhada com outros usuários
- Memoria: Limite máximo de RAM
- Armazenamento: Espaço para código, logs, etc
- Escala: Sem multiplas escalas  (só existe 1 instancia do sistema rodando)
- SLA: Não tem garantia de disponibilidade, pode cair ou reiniciar.

Para os proximos calculos vamos assumir: 

- Tempo de resposta medio de requisições: 500ms (0,5s)
- Uso de CPU por requisição: ~100% de 1 core durante 0,5s
- Mem[oria da aplicação: 300MB + 50 MB por requisição concorrente
- Plano Free Tier da Azure:
  - 1 core compartilhado
  - 1 GB de RAM

### Calculo da concorrencia: 

*Concorrência = Taxa (req/s) × Tempo de resposta (s)*

|                 | Taxa(req/s) | Tempo(s) | Requisições Simultaneas |
| --------------- | :---------: | :------: | :---------------------: |
| Cenario Médio   |     1,5     |   0,5    |          0,75           |
| Cenário de Pico |     4,4     |   0,5    |           2,2           |

Vemos que, no cenário médio temos menos de uma requisição por simultanea por segundo e cada request usa praticamente o unico core inteiro por 0,5s. Já no cenário de pico estariamos recebendo 2,2 requisições simultaneas a cada 0,5s mas como só temos 1 core de capacidade, e o uso necessario para processamento seria de 220% de CPU, isso gera fila de requisições, aumento de latencia e timeout. O sistema precisa de mais CPU do que tem disponível.

### Impacto na Memória: 

- APP carregada: 300 MB
- Concorrencia no Pico: 2,2 req × 50 MB ≈ 110 MB
- Total:  410 MB

Ainda cabe nos 1GB disponiveis mas o Plano Free não significa que teremos memoria dedicada estável. Se no futuro o tamanho do nosso modelo aumentar, vamos supor que para 500 MB o novo total seria de: 

```
300 + 500 + 110 = 910 MB
```

Um valor muito proximo do limite de 1GB. Nesse caso teriamos o risco de sofrer com OOM(Out of Memory), restarts da aplicação e indisponibilidade.

### 2.4 Consumo de recursos por requisição

Em produção nos teriamos:

- 80.000 requisições/dia
- Cada uma usa 0,5s de CPU

Total de CPU por dia:

```
80.000 × 0,5 = 40.000 segundos de CPU
```

Convertendo:

```
40.000 / 3600 ≈ 11,1 horas de CPU
```

O sistema não quebra por volume de requisiçõers, mas porque no plano FREE ele não tem recurso suficiente para sustentar a carga ao longo do tempo. O sistema precisa de 11 horas de CPU por dia e só tem disponivel 1 hora.

### 2.5 Avaliação da infraestrutura atual (Plano F1)

| Característica       | Plano F1 (Free)           | Adequação ao cenário do banco                                |
| -------------------- | ------------------------- | ------------------------------------------------------------ |
| Cota de CPU          | 60 min/dia                | Insuficiente - consumo estimado muito abaixo do necessario   |
| Número de instâncias | 1 (sem escala horizontal) | Inadequado - sem redundância, qualquer falha derruba o serviço |
| SLA                  | Não há SLA                | Inadequado para um sistema que apoia decisão de crédito      |
| Deployment slots     | Não disponível            | Gera indisponibilidade durante deploys                       |
| Armazenamento        | 1 GB                      | Suficiente para o Modelo atual                               |
| Memória              | 1 GB                      | Suficiente para o Modelo atual                               |

### Conclusão

O Plano F1 não atende aos requisitos mínimos de um sistema interno de produção em um banco, principalmente pela ausência de SLA, pela falta de redundância e pela cota de CPU que seria rapidamente esgotada em uso real.

### 2.6 Dimensionamento recomendado para produção

Para um ambiente de produção, será necessário migrar para um plano que remova a cota de CPU diária e suporte escala horizontal, idealmente o Plano Premium v3 (P1v3 ou superior), que oferece suporte a zonas de disponibilidade.

| Caracteristica   | Plano Premium v3 P1V3 | Adequação ao cenário do banco |
| ---------------- | --------------------- | ----------------------------- |
| Cota de CPU      | ~195 ACU por vCPU     | Adequado                      |
| Instancias       | 30                    | Adequado                      |
| SLA              | 99,5%                 | Adequado                      |
| Deployment Slots | Até ~20 slots         | Adequado                      |
| Armazenamento    | 250                   | Adequado                      |
| Memória          | 8                     | Adequado                      |

O número de instâncias necessárias pode ser estimado pela fórmula:

```
instâncias = arredondar_para_cima(taxa_de_pico / throughput_por_instância)
```

Considerando a taxa de pico estimada (≈4,4 req/s) e assumindo que uma instância com 1 vCPU sustenta algo entre 20 e 50 req/s, uma única instância já cobriria o cenário de pico estimado em termos de throughput. Mas, para um sistema bancário interno, o dimensionamento não deve ser guiado apenas pelo throughput médio, mas também por: Redundância e alta disponibilidade: manter no mínimo 2 instâncias ativas simultaneamente, para que a falha de uma instância não interrompa o serviço.

Escalonamento (autoscale): configurar regras de autoscale baseadas em métricas de CPU e memória (por exemplo, adicionar uma instância quando a CPU média ultrapassar 70% por 5 minutos, e remover quando cair abaixo de 30%), garantindo capacidade extra em picos inesperados, como em eventos sazonais (Black Friday, datas de pagamento de salário).

### 2.7 Auditoria 

Um ponto importante para sistemas bancários, mesmo não sendo o foco técnico inicial do projeto (que não utiliza banco de dados), é a questão das auditorias. Em sistemas de análise de crédito, geralmente existe a necessidade de registrar informações como o resultado retornado pelo modelo, o momento da requisição e qual versão do modelo foi utilizada. Atualmente, como o sistema não possui persistência de dados, esse tipo de controle não é possível. Caso o sistema evolua para um ambiente de produção, será necessário incluir uma camada de armazenamento para esses registros, como por exemplo o uso de um serviço de storage ou banco de dados. Além disso, o planejamento de capacidade também deverá considerar o crescimento desses dados ao longo do tempo, com base na quantidade de requisições diárias e no tamanho médio de cada log gerado.

### 2.8 Conclusão

A implantação atual, no Plano F1 do Azure App Service, não cumpre adequadamente os requisitos para um cenário de produção. A avaliação de capacidade mostra que, para o cenário descrito de uso interno em um banco com operação nacional, essa infraestrutura apresenta limitações de cota de CPU diária, ausência de redundância e a falta de SLA. O que torna necessária a migração para um plano de produção com múltiplas instâncias e autoscale.

## Estratégia de testes e resultados do teste de carga

### 1. Testes implementados na API

A correção e a robustez da API foram verificadas com 4 tipos de testes. Os **testes unitários** do modelo (`test_model_xgboost_v1.py`), que carregam o arquivo `.joblib` diretamente e verificam o modelo isoladamente, sem depender da API. Esses testes confirmam que o modelo carrega sem erro, que a predição retorna sempre 0 ou 1, que o pipeline trata corretamente valores ausentes e que a ausência de uma coluna obrigatória gera um em vez de uma falha silenciosa. 

Os **testes de integração** da API (`test_api.py`), que sobem a aplicação FastAPI em memória e testam o contrato exposto: o endpoint de health check responde 200, o `/predict` retorna os dois campos esperados (`prediction`, `probability`), a mesma entrada produz sempre a mesma saída (determinismo) e entradas inválidas retornam 422 em vez de um erro genérico. 

O **smoke test** em produção (`smoke_test.py`), que faz chamadas reais contra a URL do Azure, repetindo as verificações essenciais (health check, predição válida, rejeição de payload inválido) e medindo o tempo de resposta de cada chamada. 

O **teste de carga** com Locust (`locustfile.py`), que simula múltiplos sistemas internos chamando `/predict` simultaneamente, para responder à pergunta: o sistema continua funcionando dentro de um tempo de resposta aceitável quando vários sistemas chamam a API ao mesmo tempo?

### 2. Resultados do teste de carga - caso 1 (pico de 1000 usuários, spawn rate 10)

#### Contexto do teste

Com uma taxa de criação (spawn rate) de 10 usuários por segundo, o Locust levou aproximadamente 100 segundos para atingir o total de 1000 usuários simulados. Cada usuário realizava requisições alternadas entre `GET /` e `POST /predict`, seguindo a proporção de 9(POST) para 1(GET), conforme definido no arquivo `locustfile.py`. Houve um intervalo de espera entre 0,5 e 2 segundos entre cada requisição.

#### Análise dos resultados

A primeira informação a observar é a taxa de falhas: foram registradas 0 falhas  tanto para o endpoint `GET /` quanto para o `POST /predict` o que indica que o sistema se comportou corretamente mesmo sob carga, sem retornar erros do tipo 4xx ou 5xx. 

A mediana do tempo minimo de resposta foi de 5,5 segundos, a média de 6,09 segundos, o percentil 95 (p95) chegou a 12 segundos e o percentil 99 (p99) a 14 segundos, com tempo máximo de 14,49 segundos. Isso significa que metade das requisições levou mais de 5,5 segundos para responder, e uma pequena parte demorou até cerca de 14 segundos.

A taxa de processamento (Current RPS) atingiu 62,3 req/s. Estima-se que cerca de 379 requisições estavam sendo processadas simultaneamente (ou aguardando em fila). Esse valor sugere que o servidor não consegue acompanhar a quantidade de requisições, gerando acúmulo e filas.

#### Comparação com o planejamento de capacidade

Comparando os resultados com o planejamento inicial, temos uma grande diferença. A taxa de pico esperada era de aproximadamente 4,4 requisições por segundo, enquanto no teste foram observamos 62,3 requisições por segundo, ou seja, uma taxa 14 vezes maior. O tempo de resposta esperado era de poucos milissegundos, mas no teste ficou na faixa de segundos.

Esse teste representa um teste de estresse, com carga muito acima do esperado.

#### Diagnóstico provável

- O plano F1 utilizado possui recursos compartilhados e uma limitação diária de CPU. Durante o teste, é provável que essa cota tenha sido atingida, causando redução de desempenho (throttling). 
- O sistema roda em apenas uma instância e, por padrão, o servidor uvicorn utiliza apenas um worker. Como o processamento do modelo (XGBoost) exige CPU, as requisições acabam sendo tratadas de forma sequencial, gerando filas.

#### Conclusão e próximos passos

O teste com 1000 usuários mostrou que o sistema continua funcionando corretamente, mesmo sob alta carga, mas também mostrou que o plano F1 não suporta esse volume de requisições com tempos de resposta adequados. Isso está de acordo com as limitações já previstas no planejamento.

### 3. Resultados do teste de carga - caso 2 (pico de 15 usuários, ramp-up 2, 5 minutos)

#### Contexto do teste

Realizamos uma simulação mais próxima do ambiente de produção hipotético, com 15 usuários simultâneos no pico, ramp-up de 2 usuários por segundo (levando cerca de 7,5 segundos para atingir o total) e duração de 5 minutos. Esse cenário está alinhado com a taxa de pico estimada no planejamento de capacidade (~4,4 requisições por segundo). Durante o teste observamos uma taxa agregada de aproximadamente 10,8 req/s, mais que o dobro do valor inicialmente previsto.

#### Análise dos resultados

Os resultados mostram um bom desempenho do sistema na maior parte do tempo. A mediana foi de aproximadamente 140 ms e o percentil 95 ficou em torno de 190 ms. Isso indica que mais de 95% das requisições foram respondidas em menos de 200 ms. Mesmo acima da taxa de pico, o sistema conseguiu manter o desempenho ao longo dos 5 minutos de execução.

Sobre os tempos de resposta: o percentil 99 chegou a aproximadamente 2,7 segundos, e o tempo máximo observado foi de cerca de 37,9 segundos. indicando que uma parcela das requisições apresentou tempos muito elevados. Cerca de 1% das requisições (~27 a 28 chamadas) levaram mais de 2,7 segundos para responder. Em um cenário real de produção, com um volume maior de requisições diárias, esse percentual pode representar centenas de chamadas lentas, impactando a experiência do usuário.

![REPORT](https://github.com/ICEI-PUC-Minas-PMV-SI/PUC_E7_T2_G1_2026_01/blob/main/docs/img/report_c2.png)
![GRAFICOS](https://github.com/ICEI-PUC-Minas-PMV-SI/PUC_E7_T2_G1_2026_01/blob/main/docs/img/total_requests_per_second_C2.png)

#### Diagnóstico provável

No teste de estresse (com 1000 usuários), o tempo máximo observado foi menor do que neste teste com apenas 15 usuários. Isso sugere que os tempos elevados não estão relacionados ao volume de requisições. É provável que esses valores extremos estejam associados a eventos pontuais do ambiente de execução, como cold start da aplicação, reinicialização da instância ou limitações de CPU do plano utilizado (como o plano F1, que possui recursos compartilhados e restrições de processamento).

#### Conclusão

Para uso em produção, seria importante eliminar esses picos de latência, o que pode ser feito com a migração para um plano mais robusto, que ofereça maior estabilidade e recursos dedicados.

### 3. Resultados do teste de carga - caso 3 (Plano B1, 15 usuários, ramp-up 2, 5 minutos)

#### Contexto e Conclusão

Mesma configuração de carga do Teste 2 (15 usuários, ramp-up de 2/s, 5 minutos), mas agora com a aplicação rodando no Plano B1, com CPU dedicada e Always On ativado. Segue a tabela de comparação dos 3 testes:

| Métrica        | Teste 1 — F1, 1000 users | Teste 2 — F1, 15 users | Teste 3 — B1, 15 users |
| -------------- | ------------------------ | ---------------------- | ---------------------- |
| Mediana        | 5.500 ms                 | 140 ms                 | 190 ms                 |
| P95            | 12.000 ms                | 190 ms                 | 1.000 ms               |
| P99            | 14.000 ms                | 2.800 ms               | 1.700 ms               |
| Máximo         | 14.488 ms                | 37.925 ms              | 2.213 ms               |
| Falhas         | 0                        | 0                      | 3 (0,116%)             |
| RPS sustentado | 62,3                     | 10,8                   | 8,9                    |

A migração para o B1 fez com que o sistema operasse dentro de uma latência mais controlada. o response time máximo caiu de 37.925 ms para 2.213 ms, uma redução de 94%. Isso pode indicar que os  extremos do Teste 2 talvez fossem causados pelo cold start e pelo throttling de CPU do Plano F1, ambos eliminados no B1 com Always On. A mediana do tempo de resposta ficou um pouco acima 190 ms vs. 140 ms do Teste 2. O P95 subiu de 190 ms para 1.000 ms, e o P99 de 2.800 ms para 1.700 ms, ou seja, o P99 na verdade melhorou, mas o P95 piorou. 

### 4. Resultados do teste de carga - caso 4 (Plano B2, 2 workers, 15 usuários, 2ramp up, 5 minutos)

#### Contexto e Analise

Mesma configuração de carga do Teste 2 (15 usuários, ramp-up de 2/s, 5 minutos), mas agora com a aplicação rodando no Plano B2, com 2 vCPUs, 2 workers, CPU dedicada e Always On ativado.  Adicionar 2 workers significa permitir que a aplicação processe mais de uma requisição ao mesmo tempo, utilizando melhor o poder da CPU disponível.

Caracteristicas do plano B2:

| Categoria | Característica | Detalhes | O que significa na prática                     |
| --------- | -------------- | -------- | ---------------------------------------------- |
| Compute   | vCPU           | 2 vCPUs  | Define quantas tarefas podem rodar em paralelo |
| Memória   | RAM            | ~3.5 GB  | Espaço para armazenar dados temporários        |
| CPU       | Dedicada       | Sim      | Sem disputa com outros clientes                |
| Always On | Disponível     | Sim      | Evita lentidão inicial (cold start)            |
| Custo     | Aproximado     | Médio    | Mais caro que F1/B1, mas com mais estabilidade |

Este teste teve o melhor resultado. Tivemos zero falhas: mediana do tempo de resposta ficou em 160 ms, do P95 em 230 ms, P99 em 350 ms e máximo de tempo de resposta em apenas 678 ms para 3.130 requisições no `POST /predict`, rodando a 10,4 req/s (mais que o dobro do pico estimado de 4,4 req/s). Com relação ao Teste 3 o P95 caiu 77%, P99 caiu 79% e o máximo caiu 69% e o throughput ficou semelhante de 8,9 para10,4 req/s. Ou seja: o B2 entregou mais throughput, menos latência e zero falhas.

#### Conclusão

O Plano F1 é inadequado para produção, ele possui cold starts severos (de até 37 s) e entra em colapso sob estresse. A migração para o B1 eliminou os cold starts mas ainda deixou as requisições mais lentas do sistema na casa dos segundos (P99 de 1,7 s, máximo de 2,2 s) e produziu 3 falhas de timeout. Houve uma tentativa de adicionar um segundo worker no B1 que ajudou a confirmar que o gargalo era a vCPU única, porém mais workers competindo pelo mesmo núcleo pioraram deixaram algumas requisições mais lentas, piorando o que se chama de Cauda de latência. A mudança para o plano B2 com 2 workers resolveu esse gargalo: com 2 vCPUs disponíveis, os processos tem recursos suficientes para processar cada requisição sem competição, esse cenário não apresentou falhas, e apresentou P99 abaixo de 400 ms.

##### Graficos gerados no Locust
![LOCUST](https://github.com/ICEI-PUC-Minas-PMV-SI/PUC_E7_T2_G1_2026_01/blob/main/docs/img/total_requests_per_second_C5.png)
##### Comparativo testes
![RESULTADOS](https://github.com/ICEI-PUC-Minas-PMV-SI/PUC_E7_T2_G1_2026_01/blob/main/docs/img/reports_C5_comparativo.png)



# Apresentação da solução

Vídeo de até 15 minutos apresentando:
 - o escopo geral do projeto
 - um resumo das etapas desenvolvidas 
 - demonstração da solução publicada
 - conclusões finais, destacando aprendizados, impacto e possibilidades de melhorias.
