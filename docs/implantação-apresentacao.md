# Implantação da solução

##  Descrição do processo de implantação

### 1.1 Visão geral da solução

O sistema implantado é uma API de avaliação de risco de crédito, construída em Python usando FastAPI, que expõe um endpoint REST capaz de receber os dados de um cliente e retornar a probabilidade estimada de inadimplência, calculada por um modelo XGBoost previamente treinado. A aplicação não possui banco de dados: ela é stateless, ou seja, cada requisição é processada de forma independente, sem persistência de histórico. Planejamos que o consumo será feito por dois tipos de clientes: o frontend hospedado na Vercel (uma interface de análise para colaboradores do banco) e outros sistemas internos do banco que integram essa API ao fluxo de concessão de crédito (por exemplo, um sistema de originação de propostas de renegociação de dividas).

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


# Apresentação da solução

Nesta seção, deve ser produzido um vídeo de até 15 minutos apresentando o escopo geral do projeto, um resumo das etapas desenvolvidas, a demonstração da solução publicada e as conclusões finais, destacando aprendizados, impacto e possibilidades de melhorias.

# É IMPRESCINDÍVEL: 
* Atualizar o arquivo **CITATION.cff** disponível no diretório raiz do repositório
* Atualizar as **Instruções de utilização** no arquivo read.me



