# Embeddings de notícias financeiras e comunicados do Copom em português na previsão do retorno diário do Ibovespa: uma avaliação com validação walk-forward

**Hiago Lacerda**
Universidade Federal de Pernambuco (UFPE)

---

## Resumo

Este trabalho avalia se embeddings textuais de notícias financeiras e o
sinal de política monetária do Copom, ambos em língua portuguesa, melhoram
a previsão do retorno diário (D+1) do Ibovespa, em comparação a baselines
financeiros clássicos (passeio aleatório, ARIMA/ARIMAX) e a um modelo
tabular treinado exclusivamente com dados de preço (XGBoost). Manchetes do
InfoMoney (2018–2024, seções mercados e economia, 57.611 manchetes) foram
convertidas em representações vetoriais por dois modelos de linguagem em
português — BERTimbau e FinBERT-PT-BR — e combinadas a uma rede LSTM de
arquitetura parcimoniosa, treinada sobre variáveis de preço. A validação
segue um protocolo walk-forward expansível, com embargo de 21 dias entre
treino e teste, e todo ajuste dependente dos dados é realizado
exclusivamente no conjunto de treino de cada partição (fold). Nenhum
modelo, de preço ou de texto, supera o acaso na previsão da direção do
retorno de forma estatisticamente significativa (teste de
Pesaran-Timmermann, p > 0,05 após correção de Holm em todos os casos). O
sinal textual mostra-se reativo, e não preditivo: a polaridade de
sentimento correlaciona-se fortemente com o retorno do mesmo dia (r ≈ 0,20),
mas não com o retorno do dia seguinte (r entre 0,00 e −0,04). Testes de
Diebold-Mariano indicam que um modelo tabular relativamente simples
(XGBoost) é significativamente mais preciso do que diversas variantes de
LSTM com texto, e que a inclusão de sentimento textual piora a previsão da
LSTM de forma estatisticamente significativa. Sob custo de transação de 5
pontos-base por operação — estimativa condizente com o mercado brasileiro
—, dez das onze estratégias mecânicas derivadas das previsões apresentam
índice de Sharpe negativo ou nulo, e a única exceção observada é
estatisticamente indistinguível de zero e instável entre partições. Uma
verificação de robustez, retreinando todos os modelos de texto sobre o
corpo completo das matérias (56.819 textos, em vez de apenas a manchete),
reproduz o mesmo padrão nulo, descartando a hipótese de que o resultado
decorra do escopo textual restrito. Conclui-se que, sob validação
rigorosa e sem vazamento temporal de informação, embeddings de notícias
financeiras em português não melhoram a previsão do retorno diário do
Ibovespa, constituindo um resultado nulo obtido com rigor metodológico, e
não uma limitação de implementação.

**Palavras-chave:** previsão de séries financeiras, embeddings de texto,
BERTimbau, FinBERT, validação walk-forward, Ibovespa.

---

## 1. Introdução

A hipótese de que notícias financeiras carregam informação capaz de
antecipar movimentos de preço é conhecida na literatura de finanças e
ganhou renovado interesse com o desenvolvimento de modelos de linguagem
baseados em arquiteturas Transformer. Modelos como o BERT tornaram viável a
extração de representações vetoriais densas a partir de texto em
português, superando a dependência de léxicos de sentimento manualmente
curados e permitindo sua incorporação a modelos de séries temporais. Este
trabalho investiga se embeddings textuais em português, extraídos de
notícias financeiras, melhoram a previsão do retorno diário (D+1) do
Ibovespa em relação a baselines financeiros clássicos e a um modelo
tabular robusto treinado exclusivamente com dados de preço.

O desenho experimental adota uma progressão incremental de complexidade —
passeio aleatório, ARIMA/ARIMAX, XGBoost, LSTM apenas com preço, e LSTM com
embeddings textuais ou sinal do Copom —, na qual cada nível é validado
isoladamente antes de ser adotado como referência para o nível seguinte.
Essa progressão responde a um propósito metodológico específico: caso a
inclusão de texto produza ganho real, este deve manifestar-se como melhora
mensurável e estatisticamente significativa sobre o melhor modelo de preço
já construído, e não como resultado favorável obtido apenas em comparação
com um baseline artificialmente fraco.

O alvo de todos os modelos é o log-retorno de D+1,
$$y_t = \ln(\text{close}_{t+1}/\text{close}_t),$$
avaliado sob três dimensões: magnitude do erro (MAE, RMSE), acerto
direcional (Directional Accuracy, F1, MCC) e calibração probabilística
(Brier Score). A validação adota sempre o esquema walk-forward temporal,
nunca partição aleatória, e toda transformação dependente dos dados —
normalização, PCA, calibração isotônica, seleção da ordem do modelo ARIMA
— é ajustada exclusivamente no conjunto de treino de cada fold.

As contribuições deste trabalho são as seguintes:

1. Um corpus de 57.611 manchetes do InfoMoney (2018–2024), coletado e
   temporalmente atribuído a cada pregão mediante corte explícito às 17h
   (horário de fechamento da B3), respeitando o fuso horário histórico,
   inclusive o horário de verão — cuidado frequentemente omitido em
   trabalhos que combinam dados textuais e de preço.
2. Uma comparação simétrica entre dois modelos de linguagem em português —
   BERTimbau, de propósito geral, e FinBERT-PT-BR, especializado em
   finanças —, avaliando tanto seus embeddings quanto o sentimento
   explícito produzido pelo FinBERT-PT-BR.
3. Evidência direta, obtida por meio de análise de correlação contemporânea
   versus defasada, de que o sinal textual presente nas notícias
   financeiras é reativo, isto é, descreve o passado, e não preditivo, não
   antecipando o futuro — explicação de natureza mecanística para a
   ausência de ganho, e não apenas sua constatação empírica.
4. Um protocolo de avaliação estatística abrangente (Diebold-Mariano,
   Pesaran-Timmermann, correção de Holm para múltiplas comparações,
   avaliação multi-semente) aplicado uniformemente a onze modelos,
   incluindo uma via adicional com comunicados do Copom, coletados via API
   de dados abertos do Banco Central e classificados por zero-shot, com
   alinhamento cuidadoso dos períodos de teste entre modelos com e sem
   embargo temporal.
5. Uma avaliação de custo de oportunidade financeiro (Sharpe, drawdown
   máximo, custo de transação) que demonstra que nenhum modelo produz
   vantagem econômica estatisticamente robusta sob custos de transação
   realistas, incluindo a análise criteriosa do único índice de Sharpe
   líquido positivo observado, estatisticamente indistinguível de zero e
   instável entre partições.

---

## 2. Trabalhos relacionados

A literatura de previsão de retornos financeiros com aprendizado profundo e
dados textuais pode ser organizada em três frentes que orientam diretamente
as escolhas metodológicas deste trabalho.

**Combinação de LSTM com embeddings financeiros.** Trabalhos recentes que
combinam arquiteturas LSTM com representações textuais de notícias
financeiras (GU et al., 2024) reportam ganhos modestos e inconsistentes
entre mercados, frequentemente sem controle rigoroso de vazamento temporal
na etapa de normalização ou seleção de variáveis — lacuna metodológica que
este trabalho procura sanar ao exigir que todo ajuste estatístico (scaler,
PCA, calibração) ocorra exclusivamente dentro do conjunto de treino de cada
fold do walk-forward.

**Comparações de modelos de sentimento em finanças.** Estudos comparativos
sobre o uso de sentimento textual em previsão financeira (DAHAL et al.,
2023) documentam que o poder preditivo de sentimento agregado tende a ser
pequeno e sensível à forma de agregação temporal, o que reforça a
necessidade de uma regra de corte temporal explícita e validada empiricamente
— adotada, neste trabalho, por meio da função `assign_trading_day`, com
corte às 17h America/Sao_Paulo, validada por suíte de testes automatizados
(ver Seção 3.3).

**FinBERT e modelos de linguagem especializados em finanças.** A literatura
de FinBERT (ARACI, 2019) estabelece que modelos de linguagem ajustados
(finetunados) em corpora financeiros produzem sentimento mais alinhado ao
domínio do que modelos de sentimento genéricos, mas não garante, por si
só, que esse sentimento seja preditivo, apenas que constitui descrição mais
precisa do texto. Este trabalho distingue explicitamente essas duas
propriedades (Seção 5.5): o FinBERT-PT-BR aqui utilizado produz sentimento
fortemente correlacionado ao retorno contemporâneo, mas não ao retorno
futuro, o que é consistente com a hipótese de eficiência informacional de
mercado, segundo a qual a informação relevante já se encontra precificada
no momento em que a notícia é publicada e indexada.

---

## 3. Dados

### 3.1 Preços

A série diária do Ibovespa (^BVSP), de julho de 2018 a dezembro de 2024,
foi obtida junto ao Stooq (símbolo `^bvp`), após tentativas frustradas de
acesso ao Yahoo Finance/yfinance (limitação de taxa de requisições) e ao
SGS/BCB. A série foi validada contra referências públicas conhecidas — o
ponto mínimo da crise da COVID-19, em 23/03/2020 (aproximadamente 63.570
pontos), e o fechamento do último pregão de 2024 (aproximadamente 120.283
pontos) —, com desvio de 0,00% em ambos os casos. O período de julho a
dezembro de 2018 funciona como intervalo de aquecimento (*buffer*), de
modo que variáveis calculadas sobre janelas longas (por exemplo,
volatilidade de 21 dias) não produzam valores ausentes no início do
período de estudo. O período de estudo propriamente dito compreende
janeiro de 2019 a dezembro de 2024, totalizando 1.492 pregões com alvo
definido.

### 3.2 Notícias

Manchetes do InfoMoney, seções *mercados* e *economia*, referentes ao
mesmo intervalo de aquecimento e período de estudo (julho de 2018 a
dezembro de 2024), foram coletadas por meio dos sitemaps do WordPress/Yoast
e da extração dos campos `datePublished` e `headline` do bloco JSON-LD de
cada página. Após deduplicação e nova tentativa de coleta para os registros
com falha de rede, o corpus final totaliza 57.611 manchetes, cobrindo 100%
dos pregões do período de estudo. A seção *política* foi coletada em
paralelo como reserva para eventual estudo de ablação, mas não compõe o
corpus principal reportado neste trabalho.

Cabe destacar um aspecto metodológico relevante: os timestamps do JSON-LD
do InfoMoney foram verificados empiricamente como UTC verdadeiro,
comparando o horário do JSON-LD com o horário exibido na própria página
para uma amostra de notícias (ver `scripts/verifica_timezone.py`), e não
como um deslocamento fixo pré-aplicado. A conversão para hora local emprega
o módulo `zoneinfo` com o fuso `America/Sao_Paulo`, que trata corretamente
o horário de verão histórico, vigente no Brasil até fevereiro de 2019,
período compreendido no intervalo de estudo. Uma implementação que
simplesmente subtraísse três horas fixas teria classificado incorretamente
as notícias publicadas durante o período de horário de verão.

### 3.3 Alvo e corte temporal (anti-vazamento)

O alvo é gerado em um único ponto do código (`make_target`,
`utils/tcc_utils.py`): $y_t = \ln(\text{close}_{t+1}/\text{close}_t)$, com
o último dia da série descartado por não possuir alvo definido. Cada
notícia é atribuída ao pregão cujo fechamento ela pode informar por meio da
função `assign_trading_day`, com corte inclusivo às 17h America/Sao_Paulo:
uma notícia publicada até as 17h00, inclusive, do dia D é atribuída ao
pregão D; publicada depois desse horário, é atribuída ao pregão seguinte no
calendário da B3, que já exclui fins de semana e feriados, uma vez que o
calendário deriva do próprio índice de preços validado, e não de lista
manual de feriados. Essa função é coberta por uma suíte de 14 testes
automatizados (`utils/test_tcc_utils.py`, dos quais dez versam sobre o
corte temporal), contemplando corte exatamente às 17h, transição de sexta
para segunda-feira, feriados, horário de verão histórico e instantes UTC
reais — este último desenhado especificamente para detectar o erro
clássico de tratar o fuso horário como deslocamento fixo, em vez de
consultar a regra de horário de verão vigente na data.

### 3.4 Comunicados do Copom

Os 49 comunicados de decisão de política monetária do Copom (reuniões 219
a 267, de dezembro de 2018 a dezembro de 2024) foram obtidos por meio da
API pública de dados abertos do Banco Central (`dadosabertos.bcb.gov.br`,
dataset `atas-comunicados-copom`), com texto integral e data de cada
reunião. Trata-se de abordagem mais robusta do que a extração direta de
páginas HTML, por prescindir das fragilidades inerentes a esse método sobre
uma fonte que já disponibiliza dados estruturados em formato JSON. O
comunicado é liberado no mesmo dia da decisão, após o fechamento do pregão
(por volta das 18h30) — tratado, para fins de atribuição temporal, como
publicado nesse horário e atribuído ao pregão seguinte pela mesma função
`assign_trading_day` empregada para as notícias, o que já produzia
resultado correto para decisões do Copom mesmo antes desta implementação
específica. A reunião 219 (dezembro de 2018), anterior ao período de
estudo, funciona como semente, de modo que os primeiros pregões de 2019 não
permaneçam sem estado de política monetária conhecido; sem essa semente, os
aproximadamente 25 primeiros pregões de 2019 permaneceriam sem valor
definido até a primeira reunião do ano, em 06/02.

Optou-se pelo comunicado — nota breve, liberada no mesmo dia — em
detrimento da ata detalhada, publicada cerca de seis dias úteis depois, com
timestamp de publicação próprio: o comunicado é o documento que estava
genuinamente disponível em tempo real no fechamento do dia da decisão,
evitando a complexidade adicional de rastrear dois eventos textuais com
atribuição temporal distinta para a mesma reunião.

---

## 4. Metodologia

### 4.1 Variáveis de preço

O arquivo `features_preco.parquet` contém 18 variáveis, todas calculadas
com dados disponíveis até o dia $t$, inclusive, para prever $t+1$: retornos
defasados (lags 0, 1, 2, 3, 5, 10), volatilidade realizada (janelas de 5,
10 e 21 dias), momento ou retorno acumulado (5, 10, 21 dias), distância à
média móvel (10, 21 dias), variáveis de calendário (dia da semana, mês) e
duas variáveis de volume: `vol_rel` (volume relativo à média móvel de 21
dias) e `dvol` (variação logarítmica do volume). O nível absoluto de
volume nunca é incluído como variável, por carregar tendência estrutural
(crescimento do mercado ao longo dos anos) que constituiria sinal espúrio.
A maior correlação absoluta entre uma variável e o alvo é 0,159 (retorno do
próprio dia), o que é consistente com a ausência de vazamento. O volume foi
validado como sinal financeiramente plausível antes de sua inclusão:
correlação entre retorno absoluto e volume de +0,18, consistente com
Karpoff (1987), e persistência estatisticamente significativa (Ljung-Box,
p ≈ 0). A decisão de incluir o volume, mesmo sem garantia prévia de que
contribuiria para o desempenho preditivo, foi deliberada, com o objetivo de
constituir o baseline de preço mais robusto possível, de modo que qualquer
ganho atribuído ao texto precisasse superar o melhor modelo de preço
construível, e não um baseline artificialmente fraco.

### 4.2 Variáveis de texto

**BERTimbau** (`neuralmind/bert-base-portuguese-cased`): embeddings
congelados, de 768 dimensões, obtidos por mean-pooling mascarado sobre cada
manchete e agregados por pregão mediante média simples dos vetores do dia,
acrescidos de duas variáveis agregadas de baixa dimensão: `n_noticias`
(contagem) e `emb_dispersao` (distância média ao centroide do dia, medida
de heterogeneidade do noticiário).

**FinBERT-PT-BR** (`lucas-leme/FinBERT-PT-BR`): modelo
`BertForSequenceClassification` ajustado para classificação de sentimento
(positivo, negativo, neutro) em português no domínio financeiro. A
extração ocorreu em uma única passada, produzindo simultaneamente (i) as
probabilidades de sentimento, por meio do método forward oficial do
classificador (`model(**enc).logits`), e (ii) os embeddings do codificador
(`model.bert(**enc)`, com mean-pooling mascarado) — as duas vias pelas
quais o modelo expõe informação, sem reconstrução manual da camada de
classificação, cuja tentativa inicial produziu logits sem interpretação
plausível e foi descartada. Os embeddings do FinBERT-PT-BR foram agregados
por pregão com a mesma metodologia aplicada ao BERTimbau (mean-pooling e
dispersão), resultando no arquivo `features_texto_finbert.parquet`, de
mesmo formato e período de estudo (1.492 pregões, cobertura textual de
100%), o que permite comparação direta entre as duas fontes de embedding.

Para a redução de dimensionalidade (de 768 para $k$ dimensões), aplicou-se
PCA ajustado exclusivamente no conjunto de treino de cada fold do
walk-forward, nunca sobre a série completa, com dois valores de $k$ por
fonte de embedding: $k=8$ (compressão mais agressiva) e $k=30$
(desambiguação, isto é, verificação de que um eventual resultado nulo não
decorre de perda de sinal na compressão). A variância explicada difere
consideravelmente entre as duas fontes: o BERTimbau retém aproximadamente
45% da variância em $k=8$ e 69% em $k=30$; o FinBERT-PT-BR, por tratar-se de
modelo mais especializado, com saída mais concentrada em razão do ajuste
para um espaço de decisão de três classes, retém aproximadamente 86% em
$k=8$ e 92% em $k=30$. Trata-se de diferença estrutural relevante entre os
dois espaços de embedding que, como se discute adiante, não se traduz em
vantagem preditiva para nenhuma das duas fontes.

O sentimento explícito do FinBERT-PT-BR, isto é, as probabilidades de
classe (e não os embeddings), foi agregado em quatro variáveis densas por
pregão: `sent_polaridade` (média de P(positivo) menos P(negativo) no dia),
`sent_dispersao` (desvio-padrão da polaridade no dia), `sent_pos_frac` e
`sent_neg_frac` (frações de manchetes classificadas como positivas e
negativas, respectivamente).

### 4.2.1 Variáveis do Copom

Cada comunicado foi classificado por zero-shot
(`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, modelo de inferência em
linguagem natural multilíngue, por meio do pipeline
`zero-shot-classification` da biblioteca Hugging Face) entre dois rótulos —
viés de aperto monetário (*hawkish*) e viés de afrouxamento monetário
(*dovish*) —, gerando a variável `copom_hawkish_score` = P(hawkish) −
P(dovish), no intervalo [−1, +1].

**Truncagem e seleção de janela textual.** Os comunicados possuem entre
1.219 e 1.876 tokens (mediana de 1.482), o que excede, em todos os casos, o
limite de 512 tokens do mDeBERTa, que trunca a entrada de forma silenciosa.
Em razão disso, classificaram-se duas janelas textuais — o início do texto,
correspondente ao truncamento natural do pipeline, e o trecho final,
compreendendo os últimos 1.500 caracteres, onde a decisão costuma ser
enunciada, visto que apenas 2 dos 49 comunicados iniciam com a frase "O
Copom decidiu..." —, selecionando-se a janela de maior correlação com a
direção real da decisão. O início do texto apresentou melhor desempenho
(r=+0,615, contra r=+0,424 do trecho final, o qual é diluído por listas de
votantes e texto de natureza burocrática). O critério de seleção emprega
um rótulo auxiliar (a direção real da Selic), sem qualquer contato com o
alvo de previsão, de modo que não há vazamento de informação.

**Validação empírica do classificador.** A correlação entre o score obtido
e a direção real da decisão de Selic — alta, queda ou manutenção, extraída
diretamente do título oficial do comunicado por meio de expressão regular,
não pelo modelo — é de +0,615, o que confirma que o classificador zero-shot
capta viés real de política monetária, e não ruído, antes mesmo de
qualquer teste de natureza preditiva.

Como o regime de política monetária persiste entre reuniões, com intervalo
médio de aproximadamente 45 dias, e o Copom não altera sua posição a cada
pregão, o score, a direção da última decisão (`direcao_ultima`) e o número
de pregões decorridos desde a última reunião (`dias_desde_copom`) são
propagados por preenchimento antecedente (*forward-fill*) diário —
convenção padrão em finanças macroeconômicas, segundo a qual o mercado
precifica o último viés conhecido do Banco Central até a decisão seguinte,
sem descartar essa informação no intervalo entre reuniões. O nível absoluto
da Selic é excluído das variáveis, pela mesma razão que motivou a exclusão
do nível de volume (Seção 4.1): trata-se de variável que carrega tendência
estrutural do ciclo econômico — a Selic caiu de 6,5% em 2019 para 2% em
2020–2021 e subiu para 13,75% em 2022 —, e não sinal de curto prazo; sua
inclusão incorreria no mesmo tipo de sinal espúrio por tendência que
motivou a exclusão do nível de volume.

### 4.3 Modelos

O desenho experimental segue uma progressão de complexidade, na qual cada
nível é validado antes de o seguinte ser interpretado:

- **Nível 1 — Passeio aleatório.** Regressão: previsão de retorno igual a
  zero. Direção: persistência, isto é, repetição do sinal do retorno do
  próprio dia. Probabilidade: frequência climatológica de alta observada no
  treino.
- **Nível 2 — ARIMA/ARIMAX.** Ordem selecionada por critério de informação
  de Akaike (AIC) via `auto_arima`, reajustada a cada fold, com $d=0$
  justificado por teste de Dickey-Fuller aumentado (estatística de
  −11,86, p ≈ 7×10⁻²², confirmando estacionariedade). O diagnóstico de
  Ljung-Box indica estrutura linear fraca, porém estatisticamente
  significativa, no retorno (estatística entre aproximadamente 57 e 133,
  conforme o número de defasagens, p ≈ 0) e estrutura consideravelmente
  mais forte no quadrado do retorno (estatística entre aproximadamente
  1.750 e 2.250, mesmas defasagens) — evidência de agrupamento
  (*clustering*) de volatilidade, e não de previsibilidade direcional. A
  validação segue esquema walk-forward estrito, de um passo à frente: o
  modelo é reajustado a cada novo dia de teste, utilizando `serie[:t+1]`
  para prever `serie[t+1]`. O modelo ARIMAX emprega como variáveis
  exógenas `n_noticias` e `emb_dispersao` (BERTimbau), padronizadas por
  escore-z ajustado no treino de cada fold.
- **Nível 3 — XGBoost.** `XGBRegressor` e `XGBClassifier` com
  hiperparâmetros fixos e conservadores (profundidade máxima de 3, taxa de
  aprendizado de 0,02, 300 estimadores, regularização forte, adequados a
  um conjunto de dados de pequeno porte) e calibração isotônica via
  hold-out temporal interno, correspondente aos 20% finais do treino de
  cada fold, sem contato com o conjunto de teste. O regressor é treinado
  sobre o conjunto de treino completo do fold, por não requerer
  calibração; o classificador é treinado em uma fração do treino e
  calibrado na fração restante — assimetria deliberada e documentada, não
  uma omissão. Avaliado com N=10 sementes aleatórias, dado que o XGBoost é
  estocástico em razão da subamostragem de observações e de variáveis.
- **Nível 4 — LSTM apenas com preço.** Arquitetura parcimoniosa (uma
  camada, 16 unidades ocultas, dropout de 0,4, camada de saída linear),
  adequada ao tamanho reduzido do conjunto de dados (entre
  aproximadamente 230 e 1.050 sequências por fold). Janela de defasagem
  (*lookback*) de 21 dias, alinhada à extensão das variáveis de
  volatilidade e à autocorrelação observada na série. Otimizador Adam, 75
  épocas fixas, sem parada antecipada, de modo a não consumir dados em uma
  validação adicional, e função de perda MSE. O `StandardScaler` é
  ajustado por fold, com achatamento da dimensão temporal. Avaliado com
  N=5 sementes aleatórias; o determinismo do PyTorch é mantido fixo
  (`cudnn.deterministic=True`), de modo que, fixada uma semente, o
  resultado seja reprodutível de forma exata.
- **Nível 5 — LSTM com embeddings textuais ou sinal do Copom.** Idêntico
  ao nível 4 em arquitetura, sementes, folds, calibração e normalização —
  a única diferença consiste na adição das variáveis textuais ao tensor
  de entrada. Essa simetria é deliberada, de modo a isolar o efeito do
  texto como a diferença entre o nível 5 (ou Via 1) e o nível 4, sem
  introduzir variáveis de confusão metodológica. Avaliado para as duas
  fontes de embedding (BERTimbau: nível 5/5b; FinBERT: Via 1/1b), para o
  sentimento explícito do FinBERT (Via 2, quatro variáveis densas, sem
  PCA, por serem de baixa dimensão e diretamente interpretáveis), e para
  o sinal do Copom (Via 3, três variáveis densas — `copom_hawkish_score`,
  `direcao_ultima`, `dias_desde_copom` —, também sem PCA).

Um aspecto crítico de alinhamento merece registro: modelos com janela de
defasagem (LSTM) utilizam `embargo=21` no esquema walk-forward, de modo que
os primeiros 21 dias de cada bloco de teste sejam descartados, garantindo
que nenhuma sequência de teste contenha dias pertencentes ao treino. Os
baselines, por não possuírem janela de defasagem, utilizam `embargo=0`.
Para que os dois grupos de modelos sejam comparáveis nos testes
estatísticos pareados (Seção 4.4), ambos empregam a mesma fórmula de
particionamento (`walk_forward_splits`, com
$\text{fold\_size}=(n-\text{min\_train})/(n_{\text{splits}}+1)$), de modo
que os limites de treino de cada fold coincidam exatamente entre famílias
de modelos; o teste da LSTM constitui, por construção, um subconjunto
deslocado, em razão do embargo, do teste do baseline no mesmo fold. Uma
consequência dessa fórmula é que os últimos 210 pregões da série (março a
dezembro de 2024) não integram nenhum bloco de teste: a avaliação
fora da amostra cobre o período de janeiro de 2020 a fevereiro de 2024.

### 4.4 Validação estatística

**Pesaran-Timmermann (1992).** Avalia se a acurácia direcional de um
modelo excede o valor esperado sob independência entre o sinal previsto e
o sinal realizado. Aplicado por modelo, com os cinco folds concatenados em
ordem cronológica; como os dados de teste são temporalmente disjuntos entre
folds, a concatenação preserva a estrutura temporal sem introduzir
sobreposição.

**Diebold-Mariano (1995), com correção de Harvey-Leybourne-Newbold
(1997).** Avalia se a acurácia de previsão, medida pelo erro quadrático,
de dois modelos difere de forma estatisticamente significativa, por meio
da estatística
$$DM = \frac{\bar d}{\sqrt{\text{Var}(d)/n}}, \quad d_t = e_{1,t}^2 - e_{2,t}^2,$$
ajustada pelo fator de pequena amostra
$\sqrt{(n+1-2h+h(h-1)/n)/n}$, com $h=1$ (previsão de um passo à frente), e
comparada à distribuição t de Student com $n-1$ graus de liberdade —
correção recomendada para amostras de tamanho pequeno a moderado, como as
deste trabalho (aproximadamente 1.030 observações pareadas por
comparação).

Três aspectos de protocolo merecem registro: (i) para modelos
estocásticos, os testes de Diebold-Mariano e os testes de rentabilidade
simulada utilizam a previsão média entre as sementes, constituindo um
pequeno ensemble, e não uma semente individual; (ii) a Directional Accuracy
é calculada sobre o sinal da previsão de regressão, ao passo que F1, MCC e
Brier Score utilizam a probabilidade calibrada com limiar de 0,5 — regras
distintas, que podem divergir em casos-limite, razão pela qual a DA e o
MCC ocasionalmente indicam direções diferentes na tabela comparativa; e
(iii) o calibrador isotônico é ajustado sobre as previsões de um modelo
treinado nos primeiros 80% do conjunto de treino, sendo então aplicado às
previsões do modelo treinado sobre o conjunto de treino completo —
aproximação deliberada, adotada para não descartar 20% dos dados na etapa
de regressão.

Duas famílias de comparações pareadas foram testadas: (i) LSTM apenas com
preço em relação a cada uma das seis variantes de texto ou Copom, nos
mesmos dias de teste, dada a mesma geometria de fold, sem necessidade de
interseção; e (ii) XGBoost, o baseline tabular mais robusto, em relação a
cada variante de LSTM, na interseção dos dias de teste entre as duas
famílias (185 de 206 dias por fold, em média, em razão do deslocamento
produzido pelo embargo). Os 13 p-valores resultantes foram corrigidos para
múltiplas comparações pelo método de Holm, de controle da taxa de erro
familiar, evitando a inflação de falsos positivos decorrente da simples
multiplicação de testes. Uma ressalva sobre a definição dessa família de
testes é apresentada na Seção 5.2, junto aos resultados correspondentes.

### 4.5 Métricas financeiras

Adotou-se estratégia mecânica long/short: a posição no dia $t$ corresponde
ao sinal da previsão do modelo ($+1$, $-1$ ou $0$, caso a previsão seja
exatamente zero, como ocorre trivialmente no passeio aleatório). O retorno
líquido diário é dado por
$$r_t = \text{posição}_t \cdot y_t - \text{custo} \cdot |\text{posição}_t - \text{posição}_{t-1}|,$$
em que o termo de custo é proporcional ao turnover, contemplando o
fechamento da posição anterior e a abertura da nova posição em uma
reversão completa. Reportam-se o índice de Sharpe anualizado
($\overline{r}/\sigma_r \times \sqrt{252}$) e o drawdown máximo sobre a
curva de patrimônio acumulado, tanto sem custo de transação (0 pontos-base,
referência teórica) quanto com custo de 5 pontos-base por operação —
estimativa conservadora de corretagem e *slippage* para operações em
índice ou futuro do Ibovespa na B3.

---

## 5. Resultados

### 5.1 Tabela comparativa consolidada

| Modelo | MAE | RMSE | DA | MCC | MCC (± dp entre sementes) | Brier | PT p (Holm) |
|---|---|---|---|---|---|---|---|
| Random Walk | 0,0116 | 0,0159 | 0,485 | −0,033 | — | 0,250 | 1,000 |
| ARIMA | 0,0120 | 0,0166 | 0,522 | +0,039 | — | — | 1,000 |
| ARIMAX (texto linear) | 0,0124 | 0,0181 | 0,511 | +0,006 | — | — | 1,000 |
| XGBoost (preço) | 0,0117 | 0,0161 | 0,515 | +0,040 | ±0,049 | 0,266 | 0,577 |
| LSTM (preço) | 0,0127 | 0,0172 | 0,494 | −0,009 | ±0,057 | 0,273 | 1,000 |
| LSTM + BERTimbau k=8 | 0,0132 | 0,0181 | 0,494 | −0,018 | ±0,054 | 0,273 | 1,000 |
| LSTM + BERTimbau k=30 | 0,0133 | 0,0178 | 0,508 | −0,009 | ±0,060 | 0,265 | 1,000 |
| LSTM + FinBERT sentimento | 0,0136 | 0,0182 | 0,498 | +0,004 | ±0,055 | 0,273 | 1,000 |
| LSTM + FinBERT emb. k=8 | 0,0136 | 0,0183 | 0,491 | −0,017 | ±0,051 | 0,276 | 1,000 |
| LSTM + FinBERT emb. k=30 | 0,0131 | 0,0178 | 0,502 | −0,012 | ±0,055 | 0,271 | 1,000 |
| LSTM + Copom | 0,0140 | 0,0186 | 0,507 | +0,006 | ±0,069 | 0,263 | 1,000 |

*(Fonte: `reports/tabela_comparativa_final.csv`. PT = Pesaran-Timmermann,
p já corrigido por Holm sobre a família de 11 testes de habilidade
direcional.)*

Nenhum modelo atinge significância direcional a 5% após a correção; o
p-valor corrigido mais baixo corresponde ao XGBoost (0,577), estando os
demais acima de 0,9. A magnitude do MCC de todos os modelos situa-se
tipicamente dentro de um desvio-padrão entre sementes em relação a zero,
quando esse desvio está disponível, como no caso do XGBoost e das
variantes de LSTM — evidência direta de que as diferenças observadas são
compatíveis com ruído de amostragem e de inicialização, não constituindo
sinal real. Em termos de poder estatístico, com n=1.030 observações
agregadas, o intervalo de 95% para a DA de um classificador sem habilidade
direcional é de 50% ± 3,1 pontos percentuais; a maior DA observada (52,5%,
XGBoost) situa-se dentro dessa faixa. Os dados possuem poder para detectar
habilidade direcional a partir de aproximadamente 53,1%, patamar que
nenhum modelo atinge.

### 5.2 Testes pareados (Diebold-Mariano)

Das 13 comparações pareadas testadas (seis do tipo LSTM-preço versus
variantes de texto ou Copom; sete do tipo XGBoost versus variantes de
LSTM), quatro permanecem significativas a 5% após a correção de Holm —
as mesmas quatro já identificadas antes da inclusão do Copom, sendo que a
comparação entre XGBoost e LSTM+Copom situa-se próxima do limiar, sem
contudo ultrapassá-lo (p=0,055 após Holm):

| Comparação | DM (estat.) | p (Holm) | Direção |
|---|---|---|---|
| LSTM-preço vs. LSTM+FinBERT-sentimento | −3,12 | 0,015 | texto piora a previsão |
| XGBoost vs. LSTM+BERTimbau k=30 | −3,52 | 0,005 | XGBoost mais preciso |
| XGBoost vs. LSTM+FinBERT-sentimento | −3,90 | 0,001 | XGBoost mais preciso |
| XGBoost vs. LSTM+FinBERT-emb. k=30 | −3,18 | 0,017 | XGBoost mais preciso |

*(Fonte: `reports/testes_dm_pareados.parquet`. A comparação entre XGBoost
e LSTM+Copom não figura na tabela por não ser significativa, mas situa-se
próxima do limiar: DM=−2,75, p=0,055 após Holm.)*

Em nenhuma das 13 comparações a adição de texto ou de sinal do Copom
reduz o erro de forma significativa. Nos casos em que há significância,
esta aponta na direção oposta: (i) a adição de sentimento explícito à
LSTM aumenta o erro quadrático de forma estatisticamente significativa em
relação à LSTM apenas com preço; e (ii) um modelo tabular relativamente
simples (XGBoost, apenas com preço) mostra-se significativamente mais
preciso do que três das sete variantes de LSTM com texto ou Copom. Esse
segundo achado corrobora, com respaldo estatístico formal, um padrão
qualitativo já observável na progressão de complexidade: o aumento da
sofisticação do modelo (passeio aleatório → ARIMA → XGBoost → LSTM) não
melhora a previsão; em diversas comparações diretas, o modelo mais simples
apresenta desempenho mensuravelmente superior.

Duas ressalvas de natureza metodológica são pertinentes a esses testes.
Em primeiro lugar, a família de 13 comparações corrigida por Holm foi
definida com base na estrutura do desenho experimental (nível 4 versus
variantes de texto ou Copom; XGBoost versus variantes de LSTM), e não
registrada previamente: a correção protege contra a inflação de
falso-positivo dentro dessa família específica, mas não contra o risco
mais amplo de múltiplas comparações não declaradas (*garden of forking
paths*), caso outras comparações tivessem sido exploradas antes da
definição dessas 13. Em segundo lugar, e de maior relevância,
significância estatística não implica relevância econômica: as
diferenças de erro quadrático médio subjacentes aos quatro resultados
significativos (o parâmetro `dbar` da tabela de Diebold-Mariano) são da
ordem de 10⁻⁵, detectáveis em razão de n ≈ 1.030 conferir poder
estatístico suficiente para diferenças de pequena magnitude, e não porque
tais diferenças sejam economicamente relevantes. A Seção 5.6 (Métricas
financeiras) trata desse ponto diretamente: mesmo o modelo de melhor
desempenho relativo apresenta índice de Sharpe líquido próximo de zero ou
negativo.

### 5.3 Robustez: defasagem temporal e janela de lookback

Duas verificações adicionais testam se o resultado nulo depende de
escolhas arbitrárias de projeto.

**Defasagem temporal (D versus D−1).** A substituição do texto do próprio
dia $D$ pelo texto do dia anterior ($D-1$) como variável do modelo
LSTM+BERTimbau $k=8$ não altera o quadro geral: obtém-se MCC de
−0,016 ± 0,054 para o texto de $D-1$, valor intermediário entre o modelo
apenas com preço (−0,009) e o texto do dia $D$ (−0,018 ± 0,054). A
correlação preditiva entre o texto de $D-1$ e o retorno de $D+1$ é ainda
mais próxima de zero (`n_noticias`: −0,023; `emb_dispersao`: −0,036) do
que a já nula correlação entre o texto de $D$ e o retorno de $D+1$,
resultado consistente com a hipótese de que notícias mais antigas
carregam, quando muito, sinal ainda menor sobre o retorno futuro.

**Janela de lookback.** O nível 4/5 fixou lookback de 21 dias, escolhido
por alinhamento às variáveis de volatilidade de janela mais longa e à
função de autocorrelação observada. A tabela a seguir apresenta o MCC
para lookback pertencente a {1, 3, 5, 10, 21}, tanto para o modelo apenas
com preço quanto para o modelo com texto (BERTimbau k=8), com
embargo igual ao valor do lookback em cada linha.

| Lookback | MCC só-preço | MCC + BERTimbau k=8 |
|---|---|---|
| 1  | −0,014 ± 0,068 | +0,001 ± 0,060 |
| 3  | −0,006 ± 0,042 | −0,021 ± 0,053 |
| 5  | −0,003 ± 0,043 | −0,012 ± 0,052 |
| 10 | +0,005 ± 0,045 | −0,006 ± 0,055 |
| 21 | −0,009 ± 0,057 | −0,018 ± 0,054 |

*(Fonte: `reports/sweep_lookback_consolidado.parquet`.)*

Em nenhuma janela o MCC ultrapassa o intervalo de aproximadamente um
desvio-padrão entre sementes em torno de zero, e não se observa tendência
monotônica discernível conforme a janela aumenta. O resultado nulo não
constitui, portanto, artefato da escolha específica de 21 dias, tampouco
da escolha de utilizar o texto do próprio dia em vez do dia anterior.

### 5.4 Robustez: anisotropia dos embeddings e qualidade textual

Embeddings de BERT são conhecidamente anisotrópicos (ETHAYARAJH, 2019):
textos sem relação semântica tendem a exibir similaridade de cosseno bem
acima de zero, pois parte substancial da variância do espaço vetorial
concentra-se em uma direção genérica compartilhada, não informativa para
distinguir um texto de outro. Verificou-se empiricamente que a similaridade
de cosseno média entre pares aleatórios de manchetes do corpus é 0,659; em
um espaço isotrópico, esperar-se-ia valor próximo de zero. O próprio teste
de sanidade original do pipeline de extração já continha essa evidência,
sem que tivesse sido interpretada: a similaridade entre a manchete "Ibovespa
fecha em alta" e a manchete de teste "Receita de bolo de cenoura fica
pronta em 40 minutos" era de 0,505 — valor que, embora menor do que a
similaridade entre duas manchetes financeiras relacionadas (0,899), não é
baixo em termos absolutos.

Como o pipeline principal aplica PCA diretamente sobre os embeddings crus,
sem correção prévia de anisotropia, testou-se se essa característica
estaria suprimindo sinal preditivo real, hipótese diretamente testável.
Aplicou-se *whitening* (decorrelação e equalização de variância entre
direções, na linha da técnica difundida como BERT-flow) sobre os embeddings
crus, antes da agregação diária e do PCA, de forma exploratória — ajustado
sobre a série completa, apenas para fins de diagnóstico, sem uso em nenhum
modelo cujo resultado seja reportado como conclusão deste trabalho (o que
constituiria vazamento de informação).

| Fonte | Preditivo (cru) | Preditivo (whitened) | Contemporâneo (cru) | Contemporâneo (whitened) |
|---|---|---|---|---|
| BERTimbau | 0,085 | 0,090 | 0,088 | 0,068 |
| FinBERT | 0,082 | 0,079 | 0,215 | 0,189 |

*(Máximo de |r| entre os k=300 componentes de maior variância, sobre a
série completa; fonte: `reports/ablacao_anisotropia_whitening.parquet`.)*

Os resultados diferem entre as duas fontes de embedding. Para o BERTimbau,
modelo de propósito geral, o whitening eleva e estabiliza a correlação
contemporânea, que varia entre 0,03 e 0,09 conforme k nos embeddings crus e
passa a variar entre 0,067 e 0,077, de forma estável, após a correção —
indício de que a correção recupera sinal real antes diluído pela
anisotropia. Para o FinBERT, especializado em sentimento financeiro, a
correlação contemporânea já é alta e estável mesmo nos embeddings crus
(0,215, para todo k testado), aproximando-se da correlação de +0,201 já
obtida pela via independente da cabeça de sentimento dedicada (Seção 5.5),
o que constitui validação cruzada: duas vias distintas de extrair
sentimento do mesmo modelo convergem para magnitude semelhante de
correlação contemporânea. O FinBERT, por já ser ajustado especificamente
para essa tarefa, não se beneficia da correção de anisotropia da mesma
forma que o BERTimbau.

Em ambas as fontes, contudo, a correlação **preditiva** permanece na faixa
de ruído para todo k ∈ {8, 30, 100, 300} testado, com ou sem correção de
anisotropia. Um teste de permutação (500 repetições, com o alvo embaralhado
a cada repetição) confirma isso formalmente: o maior valor observado entre
todas as configurações testadas (BERTimbau whitened, k=300, r=0,090) é
estatisticamente indistinguível de ruído (distribuição nula empírica com
média 0,082 e percentil 95 em 0,101; p=0,178).

Testaram-se ainda duas hipóteses concretas sobre qualidade textual, não
relacionadas à compressão dimensional: (i) remoção de manchetes-template
recorrentes — colunas fixas do InfoMoney como "Os 5 assuntos que vão
movimentar o mercado nesta [dia da semana]", que correspondem a 1,5% do
corpus e têm embedding quase idêntico a cada repetição semanal,
independentemente do conteúdo real do dia; e (ii) isolar manchetes de
linguagem prospectiva (termos como "deve", "espera-se", "previsão";
9,6% do corpus) em oposição a manchetes de linguagem retrospectiva (termos
como "fechou", "subiu", "registrou"; 0,4% do corpus), sob a hipótese de que
a maioria retrospectiva das manchetes estaria diluindo um sinal prospectivo
minoritário. Nenhuma das intervenções produziu correlação preditiva
estatisticamente distinguível de ruído: a remoção de templates não alterou
a correlação de forma significativa (p=0,347 com templates, p=0,430 sem);
o subconjunto prospectivo apresentou correlação preditiva menor do que a
base completa, não maior (p=0,867); e o subconjunto retrospectivo, apesar
de exibir correlação nominal mais alta (r=0,173, com apenas 215 dias), não
resistiu ao teste de permutação (p=0,290) — resultado consistente com
efeito de estatística de ordem sobre amostra pequena e múltiplos
componentes testados, não com sinal real.

O resultado nulo mostra-se, portanto, robusto tanto à anisotropia dos
embeddings quanto a intervenções razoáveis de limpeza e filtragem textual.

### 5.5 Sentimento reativo versus sentimento preditivo

A correlação entre o sentimento do FinBERT-PT-BR e o retorno depende
substancialmente da direção temporal considerada:

- Sentimento do dia $t$ em relação ao retorno contemporâneo do dia $t$: r = +0,201
- Sentimento do dia $t$ em relação ao retorno de D+1, o alvo de todos os modelos: r = −0,039

Todas as quatro variáveis de sentimento agregado (`sent_polaridade`,
`sent_dispersao`, `sent_pos_frac`, `sent_neg_frac`) apresentam correlação
preditiva entre −0,04 e +0,03 com o alvo, valores indistinguíveis de zero,
ao passo que a correlação contemporânea de `sent_polaridade` é
aproximadamente cinco vezes maior em magnitude. Essa assimetria constitui
a explicação de natureza mecanística proposta neste trabalho para a
ausência de ganho preditivo do texto em todas as arquiteturas testadas: as
notícias financeiras descrevem com fidelidade o que já ocorreu no mercado
no mesmo dia, mas não antecipam o que ocorrerá no dia seguinte. Tal
resultado é consistente com a hipótese de eficiência informacional de
mercado, segundo a qual, caso uma notícia contenha informação genuinamente
nova e acionável, essa informação tende a já estar precificada no
fechamento do próprio dia de publicação, antes que um modelo treinado no
dia $t$ possa explorá-la para o dia $t+1$.

Cabe registrar uma limitação do FinBERT-PT-BR utilizado: o modelo exibe
viés para classificações negativas (53% das manchetes classificadas como
negativas, contra 21% como positivas, em período de alta acumulada do
índice), plausivelmente em razão de associar vocabulário técnico neutro
(por exemplo, "juros mantidos inalterados") a tom negativo. Esse viés,
contudo, não explica o resultado nulo: a correlação preditiva foi
calculada sobre a polaridade contínua, P(positivo) menos P(negativo), e
não sobre a classe discreta, permanecendo estatisticamente nula ainda que
essa ressalva seja considerada.

### 5.6 Métricas financeiras

| Modelo | Sharpe (0 bps) | Sharpe (5 bps) | MaxDD (5 bps) | Nº trades |
|---|---|---|---|---|
| Buy-and-hold (referência) | 0,091 | — | −46,8% | — |
| Random Walk | 0,000 | 0,000 | 0,0% | 0 |
| ARIMA | −0,035 | −0,445 | −45,2% | 469 |
| ARIMAX | +0,149 | −0,258 | −34,7% | 467 |
| XGBoost | +0,002 | −0,303 | −49,2% | 350 |
| LSTM (preço) | −0,277 | −0,570 | −55,3% | 335 |
| LSTM + BERTimbau k=8 | −1,017 | −1,251 | −76,2% | 267 |
| LSTM + BERTimbau k=30 | +0,175 | −0,019 | −41,1% | 221 |
| LSTM + FinBERT sentimento | −0,308 | −0,584 | −60,8% | 316 |
| LSTM + FinBERT emb. k=8 | +0,063 | −0,196 | −38,9% | 296 |
| LSTM + FinBERT emb. k=30 | −0,312 | −0,537 | −58,1% | 258 |
| LSTM + Copom | +0,688 | +0,424 | −37,7% | 301 |

*(Fonte: `reports/metricas_financeiras.csv`. Estratégia long/short
mecânica; posição igual ao sinal da previsão; custo por operação igual ao
turnover multiplicado pela taxa. Na linha do Random Walk, a previsão de
regressão é zero, de modo que a estratégia nunca opera, constituindo piso
trivial de comparação. Os períodos de teste diferem levemente entre
famílias de modelos, pois o embargo desloca os dias do LSTM; o
buy-and-hold é calculado sobre o período dos baselines. O drawdown máximo
das famílias de LSTM atravessa os intervalos de embargo na concatenação
dos folds.)*

Sem custo de transação, 5 dos 11 modelos exibem índice de Sharpe bruto
positivo. Sob custo realista de 5 pontos-base por operação — valor
condizente com a faixa observada de corretagem e *slippage* para operações
em índice na B3 —, dez dos onze modelos apresentam Sharpe negativo ou
nulo, incluindo o LSTM+BERTimbau k=30, cujo Sharpe cai de +0,175 para
−0,019. A única exceção é o modelo LSTM+Copom, com Sharpe líquido de
+0,42, resultado que requer interpretação criteriosa por três razões: (i)
o retorno médio diário da estratégia apresenta estatística t=0,86
(p≈0,39), sendo estatisticamente indistinguível de zero; (ii) o
desempenho é instável entre folds — Sharpe por fold de +0,84, +1,78,
−1,69, +0,57 e +0,18, com perda de 23,6% no terceiro fold; e (iii)
trata-se do valor máximo observado entre 11 estratégias derivadas de
modelos sem habilidade direcional detectável (PT p=0,27 sem correção; MCC
+0,006±0,069), de modo que o máximo de diversos modelos nulos tende a
apresentar-se como positivo em decorrência de efeito de seleção, e não de
capacidade preditiva genuína. Não há, portanto, evidência de vantagem
econômica exploravel associada a esse resultado. O Random Walk, por nunca
assumir posição, dado que a previsão de regressão é sempre igual a zero,
constitui piso trivial de comparação, com retorno, risco e drawdown
idênticos a zero. Esse conjunto de resultados corrobora, sob métrica de
utilidade econômica direta, a conclusão obtida pelos testes estatísticos
de acurácia: nenhum modelo produz vantagem operacionalizável.

### 5.7 Robustez: corpo completo das matérias

A análise apresentada nas seções anteriores utiliza exclusivamente a
manchete de cada matéria, uma vez que o corpus original foi coletado por
requisição em streaming interrompida antes do corpo do texto (Seção 7).
Para testar diretamente se essa restrição de escopo explica a ausência de
sinal preditivo, coletou-se o corpo completo das 57.611 matérias do
corpus, via os mesmos parágrafos filhos diretos do elemento
`article.im-article` do HTML, obtendo-se texto com sucesso para 56.819
matérias (98,6%; mediana de 2.584 caracteres, percentil 90 em 7.540
caracteres). Repetiu-se, sobre esse corpo completo, a extração de
embeddings BERTimbau e FinBERT-PT-BR (mesmo *mean-pooling* mascarado,
comprimento máximo de 512 tokens — o limite do modelo, contra os 64 usados
para manchetes), a agregação diária por pregão e o retreinamento de todas
as sete variantes de LSTM com texto (níveis 5 e 5b, vias 1, 1b e 2), sob
protocolo idêntico ao original: mesmas sementes, mesmos folds
walk-forward, mesmo embargo de 21 dias, PCA e normalização ajustados
exclusivamente no treino de cada fold.

| Modelo | MCC (manchete) | MCC (corpo completo) | Δ |
|---|---|---|---|
| LSTM + BERTimbau k=8 | −0,018 | −0,003 | +0,015 |
| LSTM + BERTimbau k=30 | −0,009 | +0,001 | +0,009 |
| LSTM + FinBERT emb. k=8 | −0,017 | −0,006 | +0,011 |
| LSTM + FinBERT emb. k=30 | −0,012 | −0,004 | +0,008 |
| LSTM + FinBERT sentimento | +0,004 | +0,002 | −0,002 |

*(Fontes: `reports/lstm_*_full.parquet` em cada diretório. O desvio-padrão
entre as cinco sementes, para cada modelo e em ambas as versões, situa-se
entre 0,04 e 0,07 — uma ordem de grandeza acima de todos os Δ reportados.)*

Os deltas são pequenos, sistematicamente positivos e, em todos os casos,
inferiores ao desvio-padrão entre sementes (0,04–0,07), portanto
estatisticamente indistinguíveis de ruído de amostragem. O teste de
Pesaran-Timmermann pooled permanece não significativo para os onze
modelos após correção de Holm, com acurácia direcional agregada
praticamente idêntica à obtida com manchetes (por exemplo, LSTM-preço:
DA=0,4864 em ambas as versões, por não depender do texto). No teste de
Diebold-Mariano pareado, o padrão qualitativo se mantém: nenhuma
comparação em que o texto (manchete ou corpo completo) supera
significativamente o preço isolado ou o XGBoost; ao contrário, das
comparações significativas após Holm-Bonferroni — quatro de treze com
manchetes, duas de treze com corpo completo —, todas favorecem o modelo
mais simples (XGBoost) sobre a variante de LSTM com texto, nunca o
inverso.

No plano financeiro, o LSTM+BERTimbau k=30 exibe Sharpe líquido de custos
substancialmente mais alto com o corpo completo (+0,59, contra −0,02 com
manchetes); dado que o MCC subjacente desse mesmo modelo variou apenas de
−0,009 para +0,001 — modificação trivial, dentro do ruído entre sementes
—, essa oscilação de Sharpe ilustra a sensibilidade já discutida na Seção
5.6: métricas financeiras calculadas sobre uma única trajetória de
backtest amplificam diferenças pequenas e estatisticamente não
significativas nas previsões, e não devem ser lidas como evidência de
ganho real proveniente do corpo do texto.

Conclui-se que o resultado nulo não constitui artefato do uso exclusivo
de manchetes: ampliar o texto disponível em cerca de trinta vezes (de
dezenas de caracteres para milhares) não desloca o MCC, a acurácia
direcional ou os testes de significância de forma perceptível. Esse
achado reforça a interpretação mecanicista já estabelecida na Seção 5.5 —
o texto financeiro carrega sinal contemporâneo, não preditivo — como
explicação mais provável para a ausência de ganho do que a insuficiência
de informação textual disponível ao modelo.

---

## 6. Discussão

Os resultados deste trabalho convergem para uma conclusão consistente sob
três perspectivas independentes — acurácia estatística de magnitude
(MAE/RMSE), habilidade direcional formal (Pesaran-Timmermann) e utilidade
econômica (Sharpe líquido de custos): nenhum modelo, com ou sem texto,
produz previsão de retorno diário do Ibovespa estatisticamente
distinguível do acaso, uma vez aplicado o rigor estatístico apropriado.

O achado mais relevante não reside na ausência de sinal em si —
resultado já esperado à luz da hipótese de eficiência de mercado —, mas na
explicação de natureza mecanística para essa ausência, discutida na Seção
5.4: o texto financeiro em português carrega sinal genuíno sobre o
mercado, porém esse sinal é contemporâneo, não preditivo. Esse aspecto
distingue o presente trabalho de resultados nulos não explicados, que
apenas reportam ausência de ganho, e de trabalhos que reportam ganhos sem
controle adequado de vazamento temporal, problema recorrente na literatura
de FinBERT-LSTM discutida na Seção 2, na qual a normalização ou a seleção
de variáveis frequentemente não respeita a fronteira entre treino e teste
em validações walk-forward.

O sinal do Copom estende esse padrão a uma fonte estruturalmente distinta
do noticiário: trata-se de evento raro (49 reuniões em seis anos, contra
57.611 manchetes), de caráter oficial, cujo viés é diretamente verificável
em relação à decisão real de Selic (r=+0,615). Ou seja, o classificador
zero-shot mostra-se eficaz, mas o sinal de política monetária assim
capturado tampouco antecipa o retorno de D+1 (MCC +0,006 ± 0,069, Seção
5.1). Esse resultado sugere que a ausência de ganho preditivo não constitui
peculiaridade do noticiário do InfoMoney nem da extração de sentimento,
mas padrão de caráter mais geral: informação pública relevante, qualquer
que seja sua fonte, parece já estar precificada no fechamento do dia em
que se torna conhecida.

O achado secundário — superioridade estatisticamente significativa do
XGBoost em relação a diversas variantes de LSTM no teste de
Diebold-Mariano — requer interpretação criteriosa: não significa que o
XGBoost preveja o mercado, uma vez que seu MCC de +0,040 ± 0,049 também
não é distinguível de zero, e seu p-valor de Pesaran-Timmermann, 0,577
após correção, está distante da significância. Significa, antes, que,
entre modelos igualmente desprovidos de habilidade preditiva real, os
modelos mais simples produzem erros de magnitude menor e mais estável —
argumento favorável à parcimônia como princípio de projeto mesmo em
cenários nos quais nenhum modelo alcança sucesso preditivo, e reafirmação
empírica de que o aumento de complexidade, seja da LSTM em relação ao
XGBoost, seja do texto em relação ao preço, não apenas deixa de contribuir
para o desempenho, como ocasionalmente o prejudica de forma mensurável.

Esses resultados devem ser interpretados à luz da hipótese de mercados
eficientes: se o mercado brasileiro incorpora rapidamente informação
pública ao preço, o intervalo em que notícias financeiras poderiam ser
exploradas por um modelo supervisionado, treinado com base no fechamento
do dia anterior, tende a já ter se encerrado no momento em que a notícia
se torna texto indexável. Isso não implica que o texto seja desprovido de
utilidade para todos os fins de modelagem financeira, mas apenas que, no
horizonte e na frequência aqui testados — diário, D+1, sem acesso a dados
intradiários —, o canal de informação já foi consumido pelo mercado antes
que um modelo baseado em fechamentos diários pudesse monetizá-lo.

Um resultado adicional merece registro: o MCC do passeio aleatório
direcional, isto é, da estratégia de persistência que repete o sinal do
retorno de hoje, é negativo (−0,033), e não próximo de zero. A correlação
negativa entre persistência e o retorno seguinte sugere leve reversão à
média de curtíssimo prazo no Ibovespa diário, padrão que nenhum dos
modelos testados, do ARIMA à LSTM, converteu em habilidade preditiva
estatisticamente significativa. Esse resultado reforça a leitura de
eficiência informacional: ainda que exista alguma estrutura fraca na
série, também observável no teste de Ljung-Box do retorno (Seção 4.3), tal
estrutura não é suficientemente robusta para ser explorada de forma
consistente por nenhuma das arquiteturas aqui avaliadas.

---

## 7. Limitações

- **Frequência diária.** O uso de fechamentos diários, sem dados
  intradiários, pode subestimar a velocidade real de incorporação de
  notícias ao preço; um horizonte de minutos ou horas poderia, em
  princípio, capturar janela de informação ainda não precificada, o que,
  no entanto, está fora do escopo deste trabalho.
- **Viés de classificação do FinBERT-PT-BR.** O modelo utilizado
  sobrerrepresenta sentimento negativo (53% contra 21% positivo) em
  período de alta do índice, plausivelmente por interpretar vocabulário
  técnico neutro como negativo. Embora a análise de correlação preditiva
  (Seção 5.5) utilize a polaridade contínua e não seja afetada por esse
  viés na conclusão central, tal viés limita a interpretabilidade direta
  das frações `sent_pos_frac` e `sent_neg_frac` como medidas de sentimento
  de mercado; não foi possível mitigá-lo, por exemplo mediante
  reponderação de classes ou comparação com outro modelo de sentimento,
  em razão de restrição de tempo.
- **Análise principal restrita à manchete; corpo completo truncado em 512
  tokens na verificação de robustez.** Por desenho da coleta original —
  requisição em streaming, interrompida a aproximadamente 200KB para
  capturar somente o bloco JSON-LD do cabeçalho HTML — o corpus usado nas
  Seções 5.1 a 5.6 contém exclusivamente a manchete de cada matéria. A
  Seção 5.7 testou diretamente se essa restrição de escopo explicava o
  resultado nulo, recoletando o corpo completo de 56.819 matérias (98,6%
  do corpus) e retreinando todas as variantes de LSTM com texto sob
  protocolo idêntico; os deltas de MCC observados (todos entre −0,002 e
  +0,015) permaneceram muito inferiores ao desvio-padrão entre sementes, o
  que indica que a ausência de sinal preditivo não decorre da limitação de
  escopo do texto. Uma limitação residual permanece: o corpo completo,
  com mediana de 2.584 caracteres, foi truncado em 512 tokens — o limite
  de comprimento de contexto do BERTimbau e do FinBERT-PT-BR —, de modo
  que artigos mais longos (o percentil 90 chega a 7.540 caracteres) têm
  apenas sua porção inicial representada nos embeddings. Estratégias de
  particionamento em blocos (*chunking*) com agregação posterior poderiam,
  em princípio, capturar o texto integral, mas não foram testadas em razão
  de restrição de tempo.
- **Cobertura editorial única.** O corpus está restrito às seções
  *mercados* e *economia* de um único veículo (InfoMoney); vieses
  editoriais específicos desse veículo não foram controlados nem
  comparados a outras fontes.
- **Edição de matérias após a publicação original.** Verificação direta no
  corpus, comparando os campos `datePublished` e `dateModified`, indicou
  que 4,3% das matérias foram editadas mais de 24 horas após a publicação
  original, e 2,6% mais de 7 dias depois — risco real, ainda que reduzido,
  de que o texto coletado anos depois não coincida exatamente com o texto
  publicado em tempo real, por exemplo em razão de reescrita de manchete
  para fins de otimização em mecanismos de busca.
- **Redes sociais não implementadas.** O sinal de redes sociais (X/Twitter)
  não foi coletado nesta versão do trabalho.
- **Ausência de baseline não recorrente.** Não foi testado um perceptron
  multicamadas (MLP) simples sobre a mesma janela achatada (21×18), o que
  impediria isolar se a recorrência da LSTM, em si, agrega desempenho em
  relação ao XGBoost neste conjunto de dados; os resultados aqui
  apresentados não permitem distinguir entre "arquitetura recorrente não
  contribui" e "nenhuma arquitetura testada contribui".
- **Componentes de PCA não interpretados.** O PCA é ajustado por fold, o
  que é necessário para evitar vazamento de informação, mas impede leitura
  estável de quais dimensões do embedding cada componente captura entre
  folds; o sinal textual permanece, portanto, uma caixa-preta mesmo nos
  casos em que poderia ter contribuído para o desempenho preditivo.
- **Migração de ambiente computacional.** Os resultados aqui reportados
  foram obtidos localmente em GPU RTX 3070, após terem sido originalmente
  produzidos em GPU T4 do Google Colab. Os valores reproduziram-se dentro
  da variação esperada entre sementes, uma vez que diferenças de ponto
  flutuante entre GPUs não são idênticas bit a bit, sem alteração de
  nenhuma conclusão qualitativa.
- **Estratégia financeira estilizada.** A avaliação de Sharpe e drawdown
  emprega estratégia mecânica simples, baseada no sinal da previsão, sem
  dimensionamento de posição, sem mecanismo de *stop-loss* e sem
  consideração de liquidez ou custo de financiamento de posições vendidas;
  não pretende constituir estratégia de investimento realista, mas apenas
  um teste de utilidade econômica mínima das previsões.

---

## 8. Conclusão

Sob protocolo de validação walk-forward rigoroso — com embargo temporal
apropriado à arquitetura de cada modelo, ajuste de todo componente
dependente de dados exclusivamente dentro do conjunto de treino de cada
fold, avaliação multi-semente para todo modelo estocástico, e testes
formais de significância com correção para múltiplas comparações —,
nenhum modelo testado neste trabalho, com ou sem embeddings de notícias
financeiras em português ou sinal de política monetária do Copom, prevê o
retorno diário do Ibovespa de forma estatisticamente distinguível do
acaso. O sinal textual presente nas manchetes financeiras é genuíno,
porém reativo: descreve o retorno do próprio dia com correlação de +0,20,
e não o retorno do dia seguinte, cuja correlação com o sentimento é
estatisticamente nula. O mesmo padrão repete-se para o Copom: um
classificador validado (r=+0,615 com a decisão real de Selic) cujo sinal,
ainda assim, não antecipa o retorno de D+1, generalizando a conclusão para
além do noticiário de um único veículo.

Trata-se de resultado nulo obtido com rigor metodológico, e não de
limitação de implementação. Em um cenário de pesquisa no qual resultados
positivos tendem a ser sobrerrepresentados na literatura, frequentemente
às custas de validação temporal inadequada, demonstrar de forma
controlada e reprodutível que um canal de informação amplamente
considerado útil — notícias financeiras — não produz vantagem preditiva
mensurável, e explicar por que esse resultado ocorre, constitui, em si,
contribuição científica válida.

---

## Referências

- ARACI, D. FinBERT: Financial Sentiment Analysis with Pre-trained Language
  Models. *arXiv preprint arXiv:1908.10063*, 2019.
- DIEBOLD, F. X.; MARIANO, R. S. Comparing Predictive Accuracy. *Journal of
  Business & Economic Statistics*, v. 13, n. 3, p. 253–263, 1995.
- HARVEY, D.; LEYBOURNE, S.; NEWBOLD, P. Testing the Equality of Prediction
  Mean Squared Errors. *International Journal of Forecasting*, v. 13, n. 2,
  p. 281–291, 1997.
- KARPOFF, J. M. The Relation Between Price Changes and Trading Volume: A
  Survey. *Journal of Financial and Quantitative Analysis*, v. 22, n. 1, p.
  109–126, 1987.
- PESARAN, M. H.; TIMMERMANN, A. A Simple Nonparametric Test of Predictive
  Performance. *Journal of Business & Economic Statistics*, v. 10, n. 4, p.
  461–465, 1992.
- DAHAL, K. R.; POKHREL, N. R.; GAIRE, S.; MAHATARA, S.; JOSHI, R. P.;
  GUPTA, A.; BANJADE, H. R.; JOSHI, J. A comparative study on effect of
  news sentiment on stock price prediction with deep learning architecture.
  *PLOS ONE*, v. 18, n. 4, e0284695, 2023. DOI: 10.1371/journal.pone.0284695.
- GU, W.; ZHONG, Y.; LI, S.; WEI, C.; DONG, L.; WANG, Z.; YAN, C. Predicting
  Stock Prices with FinBERT-LSTM: Integrating News Sentiment Analysis. In:
  *Proc. 8th International Conference on Cloud and Big Data Computing
  (ICCBDC)*, Oxford, UK, 2024. DOI: 10.1145/3694860.3694870.
- ETHAYARAJH, K. How Contextual are Contextualized Word Representations?
  Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings. In:
  *Proceedings of the 2019 Conference on Empirical Methods in Natural
  Language Processing and the 9th International Joint Conference on
  Natural Language Processing (EMNLP-IJCNLP)*, p. 55–65, 2019.

---

## Apêndice A — Reprodutibilidade

Todo o código, os dados brutos, as variáveis derivadas e os resultados
reportados neste trabalho estão disponíveis no repositório do projeto
(`D:\tcc_ibov`), com instruções de execução em `README.md` e ambiente
fixado em `requirements.txt`. A suíte de testes automatizados
`utils/test_tcc_utils.py` (14 casos, cobrindo corte temporal, horário de
verão, embargo e geração do alvo) documenta e verifica formalmente as
garantias anti-vazamento descritas na Seção 3.3.
