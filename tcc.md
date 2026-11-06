# Embeddings de notícias financeiras e comunicados do Copom em português na previsão do retorno diário do Ibovespa: uma avaliação com validação walk-forward

**Hiago Willyam Araújo Lacerda**
Universidade Federal de Pernambuco (UFPE)

---

## Resumo

Este trabalho avalia se embeddings de manchetes financeiras em português
(BERTimbau e FinBERT-PT-BR; 57.611 manchetes do InfoMoney, 2018–2024) e o
viés dos comunicados do Copom melhoram a previsão do retorno diário (D+1)
do Ibovespa frente a passeio aleatório, ARIMA/ARIMAX, XGBoost e LSTM apenas
com preço, em validação walk-forward com embargo de 21 dias e todo ajuste
restrito ao treino de cada partição (fold). Nenhum modelo supera o passeio
aleatório em erro (R² fora da amostra de −0,01 a −0,37) nem o acaso em
direção (Pesaran-Timmermann, menor p após Holm de 0,73), o ganho do texto
não se distingue de zero em 26 combinações de desenho e, sob custo de 5
pontos-base por operação, dez das onze estratégias têm índice de Sharpe
negativo ou nulo. Os intervalos de 95% excluem acurácia direcional acima de
aproximadamente 55% em qualquer modelo (5 pontos percentuais sobre o acaso;
Sharpe bruto de aproximadamente 1,1) e ganho do texto sobre o preço acima
de aproximadamente 0,13 de MCC, mas não efeitos menores: com n = 1.030, o
teste só atinge 80% de poder a partir de 54,4% de acerto (Sharpe de
aproximadamente 0,9). O sentimento acompanha o retorno do próprio dia (r ≈
0,18), sobretudo em manchetes que relatam preços (+0,25, contra +0,07 nas
de eventos), e a parte não explicada pelo preço não prevê D+1 (r = −0,007,
IC [−0,07; +0,05]). Com aberturas reais, o texto publicado antes da
abertura relaciona-se com o gap noturno (r ≈ 0,13), mas não prevê o retorno
abertura→fechamento (r = 0,00, IC [−0,05; +0,05]); o classificador do Copom
lê a direção da decisão (r = +0,615; +0,25 sem a frase do anúncio), sem
relação detectável com o retorno nem na janela de reação (48 eventos).

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

1. **Protocolo anti-vazamento testável:** regra de atribuição temporal ao
   fechamento efetivo da B3 (17h ou 18h, conforme o período), com fuso
   histórico, inclusive o horário de verão; 25 testes automatizados,
   inclusive sobre os próprios dados; todo ajuste restrito ao treino de
   cada fold; análise de poder do desenho.
2. **Evidência de que o texto descreve o retorno do dia**, e de que o
   encoder de domínio faz isso melhor (correlação contemporânea máxima de
   0,19 no FinBERT-PT-BR, contra 0,08 no BERTimbau), em comparação simétrica
   de embeddings e sentimento explícito.
3. **Caracterização do que o desenho consegue e não consegue detectar:**
   intervalos por block bootstrap, margem de equivalência, cadeia de
   robustez em tabela única, testes formais (Diebold-Mariano,
   Pesaran-Timmermann, Holm) sobre onze modelos, viabilidade econômica sob
   custo de transação e o teste da janela de reação com aberturas reais.
4. **Corpus e variáveis reconstituíveis:** código de coleta e de geração
   das variáveis publicado para as 57.611 manchetes e os 49 comunicados; os
   dados podem ser disponibilizados sob solicitação.

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
corte no fechamento efetivo da B3, validada por suíte de testes
automatizados (ver Seção 3.3).

**FinBERT e modelos de linguagem especializados em finanças.** A literatura
de FinBERT (ARACI, 2019) estabelece que modelos de linguagem ajustados
(finetunados) em corpora financeiros produzem sentimento mais alinhado ao
domínio do que modelos de sentimento genéricos, mas não garante, por si
só, que esse sentimento seja preditivo, apenas que constitui descrição mais
precisa do texto. Este trabalho distingue explicitamente essas duas
propriedades (Seção 5.5): o FinBERT-PT-BR aqui utilizado produz sentimento
correlacionado ao retorno contemporâneo, mas não ao retorno futuro, porque
acompanha o preço do dia, sobretudo nas manchetes que relatam o próprio
mercado. A dificuldade de extrair sentimento genuinamente informativo de
texto financeiro remonta a Loughran e McDonald (2011), que demonstraram que
léxicos de sentimento genéricos classificam erroneamente parte substancial
do vocabulário técnico financeiro como negativo. Nesta amostra, porém, o
viés negativo do FinBERT-PT-BR não parece ter essa origem: um modelo
genérico independente é ainda mais negativo (Seção 7).

**Onde a previsão funciona: regimes estruturalmente distintos.** A
literatura de aprendizado de máquina em finanças documenta sucesso
preditivo genuíno e robusto, mas tipicamente fora do regime testado neste
trabalho. Em corte transversal, com milhares de ativos e centenas de
características fundamentalistas, árvores e redes neurais alcançam poder
explicativo estatística e economicamente significativo sobre retornos
mensais (GU; KELLY; XIU, 2020). Em alta frequência, tendo o livro de
ofertas como fonte de informação, redes neurais preveem a direção do preço
em horizonte de segundos, com acurácia estável entre ativos de diferentes
setores (SIRIGNANO; CONT, 2019). Em ambos os casos, o que viabiliza a
previsibilidade é estruturalmente distinto do problema aqui tratado: poder
estatístico massivo obtido por meio de seção transversal, no primeiro
caso; horizonte curto o bastante para que a informação pública ainda não
tenha sido precificada, no segundo. Prever a direção de um único índice,
em horizonte diário, a partir de texto de notícias públicas, combina
justamente as três características menos favoráveis à previsibilidade sob
a hipótese de eficiência de mercado — o que contextualiza o resultado
nulo deste trabalho como esperado pela teoria, e não como anomalia a ser
explicada por falha metodológica.

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

Cabe destacar um aspecto metodológico relevante: até a migração do site do
InfoMoney (24/09/2019), o `datePublished` do JSON-LD traz a hora local
rotulada como UTC; depois dela, UTC verdadeiro. A verificação amostral
inicial (`scripts/verifica_timezone.py`) não detectou essa diferença, e o
erro afetava 5.356 manchetes (9,3% do corpus; 3.033 no período de estudo).
Relógios independentes do carimbo o confirmam: os relatos de fechamento do
Ibovespa aparecem, em mediana, às 14h19 no carimbo original e às 17h19 no
corrigido (100% e 12% antes do fechamento, respectivamente; de 0% a 2%
depois da migração); a decisão do Fed de 31/07/2019, divulgada às 15h em
Brasília, às 12h01 e às 15h01; e a coluna diária "Os 5 assuntos..." deixa
de saltar de 04h53 para 07h44 na migração. A própria manchete "Novo site do
InfoMoney estreia nesta quarta-feira", de 24/09/2019, registra a causa. A
função `corrige_carimbo_infomoney` (`utils/tcc_utils.py`) trata os dois
regimes.

A conversão para hora local emprega o módulo `zoneinfo` com o fuso
`America/Sao_Paulo`, que trata corretamente o horário de verão histórico,
vigente no Brasil até fevereiro de 2019, período compreendido no intervalo
de estudo. Uma implementação que simplesmente subtraísse três horas fixas
teria classificado incorretamente as notícias publicadas durante o período
de horário de verão.

### 3.3 Alvo e corte temporal (anti-vazamento)

O alvo é gerado em um único ponto do código (`make_target`,
`utils/tcc_utils.py`): $y_t = \ln(\text{close}_{t+1}/\text{close}_t)$, com
o último dia da série descartado por não possuir alvo definido. Cada
notícia é atribuída ao pregão cujo fechamento ela pode informar por meio da
função `assign_trading_day`, com corte inclusivo no fechamento efetivo da
B3, que acompanha o relógio de Nova York: 18h entre o fim e o início do
horário de verão americano (34% dos pregões da amostra) e 17h no restante.
As datas de transição foram conferidas nos relatos de fechamento do próprio
corpus e em notícias da época. Uma notícia publicada até o fechamento do
dia D é atribuída ao pregão D; publicada depois, ao pregão seguinte no
calendário da B3, que já exclui fins de semana e feriados, uma vez que
deriva do próprio índice de preços validado, e não de lista manual de
feriados.

Essa regra é coberta por uma suíte de 25 testes automatizados
(`utils/test_tcc_utils.py`): 18 sobre corte e carimbo, contemplando corte
exatamente no fechamento, transição de sexta para segunda-feira, feriados,
horário de verão histórico, datas de transição do fechamento e os dois
regimes do carimbo, e 3 sobre os próprios dados (relatos de fechamento
precisam sair depois do fechamento; a rotina editorial não pode saltar na
migração; nenhuma manchete pode entrar na linha de um pregão já encerrado),
dos quais os dois primeiros falham com o carimbo original.

Todos os resultados deste trabalho usam o carimbo corrigido, que muda a
linha de 1.742 manchetes (3%): 918 pelo erro de fuso e 974 publicadas entre
17h e 18h com o pregão aberto, com sobreposição entre os dois grupos. O
erro de fuso era vazamento no treino: até 24/09/2019, a linha D recebia
manchetes publicadas até as 20h de D, com o pregão encerrado. Nenhum dia de
teste era afetado (o primeiro é 08/01/2020), mas o treino sim (73%, 40%,
28%, 21% e 17% das linhas de treino dos folds 1 a 5). Retreinadas com os
dois carimbos, na mesma máquina e nas mesmas sementes, as cinco variantes
de texto mudam o MCC agrupado em −0,01 a +0,03, com todos os intervalos
contendo zero, e o R² em 0,00 a +0,04, sem alterar as conclusões
(`scripts/sensibilidade_carimbo.py`).

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
detrimento da ata detalhada, publicada às 8h, em até seis dias úteis, com
timestamp de publicação próprio: o comunicado é o documento que estava
genuinamente disponível em tempo real no fechamento do dia da decisão,
evitando a complexidade adicional de rastrear dois eventos textuais com
atribuição temporal distinta para a mesma reunião.

### 3.5 Aberturas reais e expectativas de mercado

O `Open` do índice não é utilizável em nenhuma fonte testada: no Stooq
(`^BVP`) e no Yahoo (`^BVSP`), a abertura fica praticamente no fechamento
anterior (gap noturno com desvio-padrão de 0,009% no Stooq, contra da ordem
de 0,5% em uma abertura real). Para o teste da janela de reação (Seção
5.9), usou-se o ETF BOVA11 (Yahoo Finance, `BOVA11.SA`,
`scripts/coleta_janela_reacao.py`), com correlação de 0,988 entre seu
retorno fechamento a fechamento e o do índice. Vinte dias têm volume zero e
abertura = máxima = mínima = fechamento, 18 deles seguidos em março e abril
de 2021, dentro do período de teste; nesses dias a abertura é falsa, de
modo que abertura→fechamento e gap são tratados como ausentes, assim como o
gap do pregão seguinte. Os fechamentos desses dias acompanham o índice
(diferença média de 0,2%) e são mantidos.

As expectativas de Selic por reunião vêm do Focus (API Olinda do BCB), com
a última expectativa disponível antes de cada decisão; a base de 30 dias
cobre todas as reuniões, e a de 5 dias, mais fresca, existe só desde 2021.

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
construível, e não um baseline artificialmente fraco. Essa garantia, de
fato, não se confirmou para o XGBoost especificamente: comparando a versão
com volume (18 features) contra a versão sem `vol_rel`/`dvol` (16
features), a acurácia direcional melhora discretamente (DA +0,5189 vs.
+0,5153), mas F1 (+0,3852 vs. +0,4099) e MCC (+0,0280 vs. +0,0401) pioram.
Ainda assim, o conjunto de 18 variáveis foi mantido como definitivo para
todos os modelos de preço (XGBoost, ARIMA/ARIMAX e LSTM), por dois motivos.
Primeiro, a validade do volume como variável foi estabelecida por um
critério independente do desempenho de qualquer modelo específico —
plausibilidade financeira (correlação com retorno absoluto, Karpoff 1987)
e persistência estatística (Ljung-Box) —, o que evita selecionar a
composição do baseline a posteriori, a partir do resultado que mais o
favorece; essa prática enviesaria exatamente a comparação que esse
baseline deveria arbitrar de forma justa (mesma disciplina anti-*forking
paths* discutida na Seção 5.2 para a família de 13 testes de
Diebold-Mariano). Segundo, o mesmo conjunto de features precisa ser
compartilhado entre os três modelos de preço para que a comparação entre
eles (Seção 5.1) isole o efeito do algoritmo, e não de qual subconjunto de
variáveis cada um recebeu.

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
para um espaço de decisão de três classes, retém aproximadamente 85% em
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
compreendendo os últimos 1.500 caracteres —, selecionando-se a janela de
maior correlação com a direção real da decisão. O início do texto
apresentou melhor desempenho (r=+0,615, contra r=+0,424 do trecho final, o
qual é diluído por listas de votantes e texto de natureza burocrática). O
critério de seleção emprega um rótulo auxiliar (a direção real da Selic),
sem qualquer contato com o alvo de previsão, de modo que não há vazamento
de informação.

**Validação empírica do classificador.** A correlação entre o score obtido
e a direção real da decisão de Selic — alta, queda ou manutenção, extraída
do título oficial do comunicado por expressão regular, não pelo modelo — é
de +0,615 (IC de 95% [+0,44; +0,75]; 49 comunicados), o que indica que o
classificador lê a direção da decisão. Essa validade vem em boa parte do
próprio anúncio: em 34 dos 49 comunicados, a frase "o Copom decidiu..."
está dentro da janela classificada (r=+0,72, contra +0,16 nos 15 demais).
Retiradas as frases do anúncio e reclassificados os textos, r=+0,25 (IC
[−0,05; +0,50]), o que não estabelece que o escore meça tom além da decisão
(`scripts/diagnostico_copom_validade.py`, que reproduz o escore salvo com
diferença 0). É a mesma ressalva que vale para as manchetes (Seção 5.5):
parte do sinal é o rótulo escrito no texto.

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

| Modelo | MAE | RMSE | DA agrup. | MCC | MCC (± dp entre sementes) | Brier | PT p (Holm) |
|---|---|---|---|---|---|---|---|
| Random Walk | 0,0116 | 0,0159 | 0,485 | −0,033 | — | 0,250 | 1,000 |
| ARIMA | 0,0120 | 0,0166 | 0,522 | +0,039 | — | — | 1,000 |
| ARIMAX (texto linear) | 0,0126 | 0,0189 | 0,505 | −0,005 | — | — | 1,000 |
| XGBoost (preço) | 0,0117 | 0,0161 | 0,523 | +0,028 | ±0,038 | 0,265 | 0,727 |
| LSTM (preço) | 0,0127 | 0,0172 | 0,486 | −0,009 | ±0,057 | 0,273 | 1,000 |
| LSTM + BERTimbau k=8 | 0,0132 | 0,0177 | 0,479 | −0,023 | ±0,059 | 0,290 | 1,000 |
| LSTM + BERTimbau k=30 | 0,0134 | 0,0182 | 0,512 | −0,005 | ±0,057 | 0,279 | 1,000 |
| LSTM + FinBERT sentimento | 0,0135 | 0,0181 | 0,489 | +0,006 | ±0,060 | 0,273 | 1,000 |
| LSTM + FinBERT emb. k=8 | 0,0136 | 0,0185 | 0,496 | −0,011 | ±0,054 | 0,269 | 1,000 |
| LSTM + FinBERT emb. k=30 | 0,0130 | 0,0175 | 0,500 | +0,008 | ±0,054 | 0,269 | 1,000 |
| LSTM + Copom | 0,0140 | 0,0186 | 0,513 | +0,006 | ±0,069 | 0,263 | 1,000 |

*(Fontes: `reports/tabela_comparativa_final.csv` e
`reports/testes_pt_pooled.parquet`. DA agrup.: acerto de sinal da previsão
média entre sementes, com os cinco folds concatenados (n = 1030); as demais
métricas são médias por semente e por fold. PT = Pesaran-Timmermann, p já
corrigido por Holm sobre a família de 11 testes de habilidade
direcional.)*

Em erro de previsão, nenhum modelo supera o passeio aleatório, que tem o
menor RMSE: o R² fora da amostra, 1 − EQM(modelo)/EQM(passeio aleatório), é negativo nos dez demais,
com intervalo inteiramente abaixo de zero em nove (exceção: XGBoost,
−0,01), de −0,08 a −0,20 nas variantes com texto ou Copom e −0,07 na LSTM
apenas com preço. Mais insumos significam mais variância: o desvio-padrão
das previsões (média entre sementes) sobe de 0,41% na LSTM apenas com preço
para 0,49%–0,72% com texto ou Copom, e o erro piora. O R² de −0,37 do ARIMAX reflete sobretudo quatro
previsões extremas (de +6% a +20%, de setembro de 2021 a janeiro de 2022):
sem elas, −0,05. As datas dessas previsões mudam com pequenas alterações
nas variáveis exógenas, e o ARIMA puro também tem uma (+8,8%, em novembro
de 2021); a causa não foi diagnosticada. Quanto ao acerto de sinal, nenhum
modelo atinge significância direcional a 5% após a correção; o menor
p-valor corrigido é o do XGBoost (0,727).

| Modelo | R² fora da amostra (IC) | DA agrupada (IC) | MCC agrupado (IC) | ΔMCC vs. LSTM preço (IC) | ΔMCC vs. passeio aleatório (IC) |
|---|---|---|---|---|---|
| Random Walk | 0 (referência) | 48,5% [45,4; 51,6] | −0,03 [−0,09; +0,03] | — | — |
| ARIMA | −0,06 [−0,17; −0,01] | 52,2% [49,1; 55,3] | +0,04 [−0,02; +0,10] | — | +0,07 [−0,04; +0,18] |
| ARIMAX (texto linear) | −0,37 [−1,20; −0,04] | 50,5% [47,7; 53,6] | 0,00 [−0,06; +0,06] | — | +0,03 [−0,07; +0,14] |
| XGBoost (preço) | −0,01 [−0,04; +0,02] | 52,3% [49,4; 55,0] | +0,05 [−0,01; +0,10] | — | +0,08 [−0,01; +0,16] |
| LSTM (preço) | −0,07 [−0,15; −0,02] | 48,6% [45,4; 51,6] | −0,03 [−0,09; +0,03] | — | 0,00 [−0,10; +0,09] |
| LSTM + BERTimbau k=8 | −0,10 [−0,20; −0,04] | 47,9% [44,9; 51,0] | −0,05 [−0,11; +0,02] | −0,02 [−0,08; +0,06] | −0,02 [−0,11; +0,08] |
| LSTM + BERTimbau k=30 | −0,12 [−0,23; −0,06] | 51,2% [48,3; 54,2] | +0,03 [−0,03; +0,08] | +0,05 [−0,01; +0,13] | +0,05 [−0,03; +0,14] |
| LSTM + FinBERT sentimento | −0,20 [−0,48; −0,07] | 48,9% [46,1; 51,7] | −0,02 [−0,07; +0,04] | +0,01 [−0,04; +0,08] | +0,01 [−0,08; +0,10] |
| LSTM + FinBERT emb. k=8 | −0,10 [−0,23; −0,03] | 49,6% [46,5; 52,3] | −0,01 [−0,07; +0,05] | +0,02 [−0,04; +0,08] | +0,02 [−0,07; +0,11] |
| LSTM + FinBERT emb. k=30 | −0,08 [−0,17; −0,03] | 50,0% [46,8; 52,8] | 0,00 [−0,06; +0,06] | +0,03 [−0,04; +0,09] | +0,02 [−0,07; +0,11] |
| LSTM + Copom | −0,16 [−0,37; −0,04] | 51,3% [48,3; 54,3] | +0,02 [−0,04; +0,08] | +0,05 [−0,01; +0,12] | +0,05 [−0,05; +0,14] |

*(Fontes: `reports/ic_mcc_bootstrap.json` e `reports/ic_da_bootstrap.json`.
Intervalos de 95% por block bootstrap sobre dias, blocos de 10 dias,
B = 10.000, com a previsão média entre sementes e os cinco folds
concatenados. ΔMCC nos mesmos dias de teste; o passeio aleatório é a
persistência nos dias de teste de cada modelo.)*

O desvio entre sementes não é a incerteza relevante para essa comparação, e
sim a amostral. Com ela, o MCC agrupado de nenhum dos onze modelos se
distingue de zero (erro-padrão de aproximadamente 0,03). Frente à LSTM
apenas com preço, nenhuma variante com texto ou Copom tem ΔMCC distinguível
de zero, mas nenhum intervalo cabe na margem de equivalência de ±0,05 de
MCC (aproximadamente ±2,5 pontos percentuais de DA): os limites superiores
vão de +0,06 a +0,13. Frente ao passeio aleatório, nenhum dos dez modelos
se distingue de zero, com limites superiores de +0,08 a +0,18. O desenho
exclui, portanto, ganhos do texto sobre o preço acima de aproximadamente
0,13 de MCC, não efeitos menores.

Em termos de poder estatístico, com n=1.030 observações agregadas, o
intervalo de 95% para a DA de um classificador sem habilidade direcional é
de 50% ± 3,1 pontos percentuais; a maior DA observada (52,3%, XGBoost)
situa-se dentro dessa faixa. O limite superior dessa faixa (53,1%)
corresponde a apenas 50% de poder; 80% de poder exige DA verdadeira de
aproximadamente 54,4%. Em Sharpe equivalente, sem custos e supondo acerto
de sinal independente da magnitude do retorno, com E|r|/√E[r²] = 0,66
medido nos dias de teste, esses patamares correspondem a 0,64 e 0,91. As
onze DAs observadas (47,9% a 52,3%) caem todas nessa região de baixo poder.
O que os dados excluem, pelo maior limite superior de IC de 95% da DA, é
acurácia acima de aproximadamente 55% em qualquer modelo (55,3%, ARIMA;
Sharpe bruto de aproximadamente 1,1).

### 5.2 Testes pareados (Diebold-Mariano)

Das 13 comparações pareadas testadas (seis do tipo LSTM-preço versus
variantes de texto ou Copom; sete do tipo XGBoost versus variantes de
LSTM), duas permanecem significativas a 5% após a correção de Holm, ambas
contra a LSTM+FinBERT-sentimento; outras quatro ficam com p corrigido de
0,06 a 0,08, todas a favor do XGBoost (BERTimbau k=8, BERTimbau k=30,
FinBERT emb. k=30 e Copom):

| Comparação | DM (estat.) | d̄ | p (Holm) | Direção |
|---|---|---|---|---|
| LSTM-preço vs. LSTM+FinBERT-sentimento | −2,96 | −0,000040 | 0,037 | texto piora a previsão |
| XGBoost vs. LSTM+FinBERT-sentimento | −3,68 | −0,000058 | 0,003 | XGBoost mais preciso |

*(Fonte: `reports/testes_dm_pareados.parquet`.)*

Em nenhuma das 13 comparações a adição de texto ou de sinal do Copom
reduz o erro de forma significativa. Nos casos em que há significância,
esta aponta na direção oposta: (i) a adição de sentimento explícito à
LSTM aumenta o erro quadrático de forma estatisticamente significativa em
relação à LSTM apenas com preço; e (ii) um modelo tabular relativamente
simples (XGBoost, apenas com preço) mostra-se significativamente mais
preciso do que a LSTM+FinBERT-sentimento, com vantagem próxima da
significância sobre outras quatro variantes. Esse segundo achado corrobora
um padrão qualitativo já observável na progressão de complexidade: o
aumento da sofisticação do modelo (passeio aleatório → ARIMA → XGBoost →
LSTM) não melhora a previsão; nas comparações significativas, o modelo mais
simples apresenta desempenho mensuravelmente superior.

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
diferenças de erro quadrático médio subjacentes aos dois resultados
significativos (o parâmetro `dbar` da tabela de Diebold-Mariano) são da
ordem de 10⁻⁵, detectáveis em razão de n ≈ 1.030 conferir poder
estatístico suficiente para diferenças de pequena magnitude, e não porque
tais diferenças sejam economicamente relevantes. A Seção 5.6 (Métricas
financeiras) trata desse ponto diretamente: mesmo o modelo de melhor
desempenho relativo apresenta índice de Sharpe líquido próximo de zero ou
negativo.

### 5.3 Robustez: defasagem temporal, janela de lookback e agregação por recência

Três verificações adicionais testam se o resultado nulo depende de
escolhas arbitrárias de projeto.

**Defasagem temporal (D versus D−1).** A substituição do texto do próprio
dia $D$ pelo texto do dia anterior ($D-1$) como variável do modelo
LSTM+BERTimbau $k=8$ não altera o quadro geral: obtém-se MCC de −0,012 ±
0,043 para o texto de $D-1$, próximo do modelo apenas com preço (−0,009) e
do texto do dia $D$ (−0,023 ± 0,059), e ΔMCC agrupado de −0,01 (IC [−0,08;
+0,06]) frente à LSTM apenas com preço, nos dias em comum. A correlação
preditiva entre o texto de $D-1$ e o retorno de $D+1$ é próxima de zero
(`n_noticias`: −0,022; `emb_dispersao`: −0,034).

**Janela de lookback.** O nível 4/5 fixou lookback de 21 dias, escolhido
por alinhamento às variáveis de volatilidade de janela mais longa e à
função de autocorrelação observada. A tabela a seguir apresenta o MCC
para lookback pertencente a {1, 3, 5, 10, 21}, tanto para o modelo apenas
com preço quanto para o modelo com texto (BERTimbau k=8), com
embargo igual ao valor do lookback em cada linha.

| Lookback | MCC só-preço | MCC + BERTimbau k=8 |
|---|---|---|
| 1  | −0,014 ± 0,068 | −0,005 ± 0,061 |
| 3  | −0,006 ± 0,042 | −0,016 ± 0,033 |
| 5  | −0,003 ± 0,043 | −0,019 ± 0,059 |
| 10 | +0,005 ± 0,045 | −0,004 ± 0,050 |
| 21 | −0,009 ± 0,057 | −0,023 ± 0,059 |

*(Fonte: `reports/sweep_lookback_consolidado.parquet`.)*

Não se observa tendência monotônica discernível conforme a janela aumenta,
e o ΔMCC agrupado frente ao modelo apenas com preço de mesmo lookback
contém zero em todas as janelas (Seção 5.8). O resultado nulo não
constitui, portanto, artefato da escolha específica de 21 dias, tampouco da
escolha de utilizar o texto do próprio dia em vez do dia anterior.

**Agregação por recência (última manchete versus média do dia).** O
pipeline principal agrega o dia por média de todas as manchetes até o
corte. Testou-se uma agregação alternativa, substituindo a média do dia
pela manchete de maior timestamp dentro do próprio `trading_day` (a mais
recente antes do corte), sem necessidade de nova coleta, sob a hipótese de
que a média diluísse a informação mais próxima do fechamento. O resultado
não confirma a hipótese: MCC de −0,011 ± 0,061 para a última manchete,
contra −0,023 ± 0,059 para a média do dia, e ΔMCC agrupado de 0,00 (IC
[−0,06; +0,07]) frente à LSTM apenas com preço. A correlação preditiva
direta entre a última manchete (projetada em 8 componentes principais) e o
retorno de D+1 tampouco se eleva (máximo de |r| = 0,050). Concentrar a
agregação na informação mais recente do dia não destrava sinal preditivo
adicional.

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
| BERTimbau | 0,087 | 0,088 | 0,076 | 0,076 |
| FinBERT | 0,092 | 0,082 | 0,193 | 0,125 |

*(Máximo de |r| entre os k=300 componentes de maior variância, sobre a
série completa; fontes: `reports/reexecucao/07_ablacoes.log` e
`reports/ablacao_anisotropia_whitening.parquet`.)*

Os resultados diferem entre as duas fontes de embedding. Para o BERTimbau,
modelo de propósito geral, o whitening eleva e estabiliza a correlação
contemporânea, que varia entre 0,03 e 0,08 conforme k nos embeddings crus e
passa a variar entre 0,065 e 0,076, de forma estável, após a correção —
indício de que a correção recupera sinal real antes diluído pela
anisotropia. Para o FinBERT, especializado em sentimento financeiro, a
correlação contemporânea já é alta e estável mesmo nos embeddings crus
(0,193, para todo k testado), aproximando-se da correlação de +0,179 obtida
pela via independente da cabeça de sentimento dedicada (Seção 5.5), o que
constitui validação cruzada: duas vias distintas de extrair sentimento do
mesmo modelo convergem para magnitude semelhante de correlação
contemporânea. O FinBERT, por já ser ajustado especificamente para essa
tarefa, não se beneficia da correção de anisotropia da mesma forma que o
BERTimbau; com ela, sua correlação contemporânea cai para 0,08–0,13.

Em ambas as fontes, contudo, a correlação **preditiva** permanece na faixa
de ruído para todo k ∈ {8, 30, 100, 300} testado, com ou sem correção de
anisotropia. Um teste de permutação (500 repetições, com o alvo embaralhado
a cada repetição), aplicado ao BERTimbau com whitening e k=300 (r=0,088),
não distingue esse valor de ruído (distribuição nula empírica com média
0,082 e percentil 95 em 0,101; p=0,226). O maior valor entre todas as
configurações (FinBERT cru, k=300, r=0,092) também fica abaixo desse
percentil.

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
estatisticamente distinguível de ruído (permutação com 300 repetições): a
remoção de templates não alterou a correlação de forma significativa
(r=0,050 e p=0,800 com templates; r=0,042 e p=0,970 sem); o subconjunto
prospectivo tampouco (r=0,061, p=0,533); e o subconjunto retrospectivo,
apesar de exibir correlação nominal mais alta (r=0,153, com apenas 218
dias), não resistiu ao teste (p=0,517) — resultado consistente com efeito
de estatística de ordem sobre amostra pequena e múltiplos componentes
testados, não com sinal real.

Testou-se, por fim, uma hipótese estrutural mais radical: seria o próprio
uso de uma representação densa e contextual — e não a anisotropia
especificamente — o fator que suprime sinal? A literatura recente de
avaliação de embeddings no domínio financeiro documenta que, em tarefas de
similaridade semântica financeira, uma representação lexical simples de
*bag-of-words* (BoW) supera embeddings densos sofisticados (TANG; YANG,
2025), achado que motivou testar diretamente se o mesmo padrão se repete
na tarefa de previsão de retorno deste trabalho. Substituiu-se o embedding
do BERTimbau por uma representação TF-IDF do texto agregado do dia,
reduzida por decomposição em valores singulares truncada (SVD) a $k$
componentes — análogo esparso ao PCA já aplicado aos embeddings densos —,
sob o mesmo protocolo (vocabulário do TF-IDF e SVD ajustados
exclusivamente no treino de cada fold). Testaram-se $k=8$ e $k=30$, este
último por precaução análoga à já aplicada ao BERTimbau: o SVD sobre o
espaço lexical (vocabulário de até 5.000 termos, após remoção de
*stopwords* do português) reteve apenas 6,6% ($k=8$) e 16,1% ($k=30$) da
variância, proporção muito inferior à retida pelo BERTimbau nos mesmos $k$
(45–55% e 65–70%, respectivamente) — reflexo da dispersão estrutural
maior do espaço lexical frente ao denso.

| Representação | MCC ($k=8$) | MCC ($k=30$) |
|---|---|---|
| BERTimbau (denso) | −0,023 ± 0,059 | −0,005 ± 0,057 |
| BoW/TF-IDF (esparso) | −0,032 ± 0,047 | +0,002 ± 0,057 |

*(Fonte: `reports/lstm_bow_k8_full.parquet`, `reports/lstm_bow_k30_full.parquet`.)*

Em nenhuma configuração a representação lexical esparsa produziu MCC
distinguível de zero, tampouco diferente do já obtido pela representação
densa — o achado de Tang e Yang (2025) não se replica na tarefa de
previsão de retorno diário aqui tratada. Combinado ao teste de anisotropia
já apresentado, esse resultado indica que a escolha específica de
representação textual — densa ou esparsa, corrigida ou não quanto à
anisotropia — não constitui o fator limitante: em nenhuma das
representações testadas o texto produz sinal preditivo detectável (Seção
5.5).

O resultado nulo mostra-se, portanto, robusto tanto à anisotropia dos
embeddings quanto a intervenções razoáveis de limpeza e filtragem textual,
e também à substituição da própria representação densa por uma
representação lexical esparsa.

### 5.5 Sentimento reativo versus sentimento preditivo

A correlação entre o sentimento do FinBERT-PT-BR e o retorno depende
substancialmente da direção temporal considerada:

- Sentimento do dia $t$ em relação ao retorno contemporâneo do dia $t$: r = +0,179
- Sentimento do dia $t$ em relação ao retorno de D+1, o alvo de todos os modelos: r = −0,036

Essas duas correlações compartilham a variável de sentimento e usam
retornos adjacentes da mesma série ($t$ e $t+1$, com autocorrelação de
lag-1 de $-0{,}159$) — não são independentes, de modo que um teste de
diferença entre correlações independentes (ex.: Fisher z simples) não se
aplica. Testou-se formalmente se a diferença $\Delta = r_{\text{contemp}}
- r_{\text{predit}}$ é distinguível de zero via bootstrap (reamostragem de
dias com reposição, $B=20\,000$, IC percentil): para `sent_polaridade`,
$\Delta=+0{,}214$, IC 95% $[+0{,}095,\ +0{,}337]$, $p=0{,}0005$ (bicaudal,
$p_{\text{Holm}}=0{,}002$). O mesmo teste, repetido para as quatro
variáveis de sentimento agregado, permanece significativo a 5% mesmo após
correção de Holm para essa família de 4 comparações — `sent_dispersao`
($\Delta=+0{,}126$, $p=0{,}036$, $p_{\text{Holm}}=0{,}036$),
`sent_pos_frac` ($\Delta=+0{,}157$, $p=0{,}0006$,
$p_{\text{Holm}}=0{,}002$) e `sent_neg_frac` ($\Delta=-0{,}184$,
$p=0{,}006$, $p_{\text{Holm}}=0{,}013$). A assimetria entre sentimento
contemporâneo e preditivo é, portanto, uma diferença estatisticamente
robusta. O que ela significa é outra questão, tratada a seguir: manchetes
financeiras descrevem o pregão, e parte da correlação contemporânea é o
rótulo escrito no texto, não evidência de eficiência.

*(Fonte: `scripts/teste_reativo_vs_preditivo.py`, saída em
`reports/teste_reativo_vs_preditivo.{parquet,json}`; `ret_lag0` como
retorno contemporâneo, `y` como alvo D+1.)*

**Reversão de preço e ortogonalização.** As quatro variáveis de sentimento
agregado têm correlação preditiva entre −0,04 e +0,03, e a maior parte dela
é reversão de preço transmitida pelo sentimento: o retorno do dia
correlaciona-se com o de D+1 em r = −0,16, e o produto dessa reversão pela
correlação contemporânea (−0,028) responde por cerca de 80% da correlação
preditiva bruta de `sent_polaridade` (−0,036). Retirada do sentimento a
parte explicada pelo retorno do mesmo dia (correlação semiparcial), a
correlação com D+1 é −0,007, IC de 95% [−0,07; +0,05] por block bootstrap
sobre dias; controlando pelas 18 variáveis de preço, |r| ≤ 0,03 nas quatro
variáveis.

**Relato de mercado versus evento.** Para separar o que a imprensa relata
do que ela informa, cada manchete foi classificada por regra explícita como
*relato de mercado* (verbo de movimento de preço, como "sobe", "fecha" ou
"despenca", junto a um ativo, índice ou código de negociação) ou *evento*
(as demais). Os relatos são 21% do corpus; a auditoria manual de 40
manchetes sorteadas por classe confirmou 38 relatos e 39 eventos
(`scripts/diagnostico_relato_evento.py`). A correlação contemporânea
concentra-se nos relatos (+0,25, IC [+0,22; +0,29]) e é pequena nos eventos
(+0,07, IC [+0,01; +0,13]).

**Horário de publicação.** Manchetes publicadas após o fechamento
correlacionam-se com o pregão que acabou de ocorrer (+0,40) e não com o
seguinte (−0,04). As publicadas antes das 10h correlacionam-se com o pregão
do mesmo dia (+0,19), mas por meio de relatos de mercados externos e do
Ibovespa futuro (+0,25), não de eventos (+0,07): descrevem movimento de
preço já em formação antes da abertura, que integra o retorno fechamento a
fechamento do dia. Em nenhum recorte o sentimento prevê D+1: o menor p
corrigido por Holm entre 15 testes preditivos (classes, faixas de horário,
com e sem ortogonalização) é 0,28.

A correlação contemporânea mede, portanto, sobretudo a imprensa descrevendo
o preço do dia, e não informação nova que o mercado incorpore antes do
fechamento. Esse padrão é compatível com eficiência informacional, mas não
a demonstra: a reação à informação publicada fora do pregão recai no gap
noturno, que o alvo fechamento a fechamento não isola (Seção 5.9).

Cabe registrar uma limitação do FinBERT-PT-BR utilizado: o modelo exibe
viés para classificações negativas (53% das manchetes classificadas como
negativas, contra 21% como positivas, em período de alta acumulada do
índice). A hipótese inicial de que esse viés decorreria de o modelo
associar vocabulário técnico neutro (por exemplo, "juros mantidos
inalterados") a tom negativo foi testada diretamente na Seção 7, mediante
comparação com um modelo de sentimento genérico e independente: o resultado
não confirma a hipótese — o modelo genérico classifica fração ainda maior
das manchetes como negativas, sugerindo que a assimetria reflete antes o
estilo editorial das manchetes financeiras em português do que uma
característica específica do FinBERT-PT-BR. De todo modo, esse viés,
qualquer que seja sua origem, não explica o resultado nulo: a correlação
preditiva foi calculada sobre a polaridade contínua, P(positivo) menos
P(negativo), e não sobre a classe discreta.

### 5.6 Métricas financeiras

| Modelo | Sharpe (0 bps) | Sharpe (5 bps) | MaxDD (5 bps) | Nº trades |
|---|---|---|---|---|
| Buy-and-hold (referência) | 0,091 | — | −46,8% | — |
| Random Walk | 0,000 | 0,000 | 0,0% | 0 |
| ARIMA | −0,035 | −0,445 | −45,2% | 469 |
| ARIMAX | +0,215 | −0,212 | −42,4% | 489 |
| XGBoost | −0,083 | −0,406 | −50,6% | 371 |
| LSTM (preço) | −0,277 | −0,570 | −55,3% | 335 |
| LSTM + BERTimbau k=8 | −0,413 | −0,640 | −59,5% | 259 |
| LSTM + BERTimbau k=30 | −0,087 | −0,302 | −39,0% | 245 |
| LSTM + FinBERT sentimento | +0,039 | −0,246 | −42,6% | 326 |
| LSTM + FinBERT emb. k=8 | +0,215 | −0,045 | −37,2% | 298 |
| LSTM + FinBERT emb. k=30 | +0,174 | −0,059 | −45,2% | 266 |
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
em índice na B3 —, dez dos onze modelos apresentam Sharpe negativo ou nulo,
incluindo o LSTM+FinBERT emb. k=8, cujo Sharpe cai de +0,215 para −0,045. A única exceção é o modelo LSTM+Copom, com Sharpe líquido de
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
| LSTM + BERTimbau k=8 | −0,023 | −0,007 | +0,016 |
| LSTM + BERTimbau k=30 | −0,005 | +0,009 | +0,014 |
| LSTM + FinBERT emb. k=8 | −0,011 | −0,003 | +0,008 |
| LSTM + FinBERT emb. k=30 | +0,008 | −0,005 | −0,012 |
| LSTM + FinBERT sentimento | +0,006 | +0,007 | 0,000 |

*(Fontes: `reports/lstm_*_full.parquet` em cada diretório. O desvio-padrão
entre as cinco sementes, para cada modelo e em ambas as versões, situa-se
entre 0,05 e 0,08, acima de todos os Δ reportados.)*

Os deltas são pequenos, de sinais mistos e, em todos os casos, inferiores
ao desvio-padrão entre sementes; o ΔMCC agrupado frente à LSTM apenas com
preço contém zero em todas as variantes (Seção 5.8). O teste de
Pesaran-Timmermann pooled permanece não significativo para os onze
modelos após correção de Holm, com acurácia direcional agregada
praticamente idêntica à obtida com manchetes (por exemplo, LSTM-preço:
DA=0,4864 em ambas as versões, por não depender do texto). No teste de
Diebold-Mariano pareado, o padrão qualitativo se mantém: nenhuma
comparação em que o texto (manchete ou corpo completo) supera
significativamente o preço isolado ou o XGBoost; ao contrário, das
comparações significativas após Holm-Bonferroni — duas de treze com
manchetes, duas de treze com corpo completo —, todas favorecem o modelo
mais simples (XGBoost ou LSTM apenas com preço) sobre a variante de LSTM
com texto, nunca o inverso.

No plano financeiro, o LSTM+BERTimbau k=30 exibe Sharpe líquido de custos
mais alto com o corpo completo (+0,27, contra −0,30 com manchetes); dado
que o MCC subjacente desse mesmo modelo variou apenas de −0,005 para +0,009
— modificação trivial, dentro do ruído entre sementes —, essa oscilação de
Sharpe ilustra a sensibilidade já discutida na Seção 5.6: métricas
financeiras calculadas sobre uma única trajetória de backtest amplificam
diferenças pequenas e estatisticamente não significativas nas previsões, e
não devem ser lidas como evidência de ganho real proveniente do corpo do
texto.

Conclui-se que o resultado nulo não constitui artefato do uso exclusivo
de manchetes: ampliar o texto disponível em cerca de trinta vezes (de
dezenas de caracteres para milhares) não desloca o MCC, a acurácia
direcional ou os testes de significância de forma perceptível. A
insuficiência de informação textual disponível ao modelo não explica,
portanto, a ausência de ganho.

O corpo completo, contudo, ainda era truncado em 512 tokens (o limite de
contexto do BERTimbau e do FinBERT-PT-BR), de modo que artigos mais longos
que essa janela tinham apenas sua porção inicial representada. Para
eliminar também essa fonte residual de perda de informação, o texto de
cada matéria foi dividido em blocos (*chunking*) de até 512 tokens, cada
bloco processado separadamente pelo modelo e os embeddings resultantes
agregados por média — cobrindo o artigo inteiro, não apenas seu início
(56.819 artigos, 123.570 blocos, média de 2,17 blocos por artigo). As três
variantes de texto foram retreinadas sob o mesmo protocolo:

| Modelo | Manchete | Corpo (truncado) | Corpo (*chunking*) |
|---|---|---|---|
| LSTM + BERTimbau k=8 | −0,023 | −0,007 | −0,020 |
| LSTM + FinBERT emb. k=8 | −0,011 | −0,003 | −0,025 |
| LSTM + FinBERT sentimento | +0,006 | +0,007 | −0,014 |

*(Desvio-padrão entre sementes entre 0,04 e 0,07 em todas as células,
acima das diferenças entre colunas.)*

Cobrir o artigo inteiro, sem qualquer truncamento, não desloca o MCC de
forma perceptível em relação ao corpo truncado — nos três casos, o valor
com *chunking* fica, se algo, ligeiramente mais distante de zero, e não
mais próximo, ainda que a diferença permaneça dentro do ruído entre
sementes. Não há, portanto, indício de que a fração de texto não
representada pelo truncamento continha sinal preditivo recuperável: o
resultado nulo é robusto tanto à restrição de escopo (manchete versus
corpo) quanto à restrição de comprimento de contexto dentro do próprio
corpo completo.

### 5.8 Cadeia de robustez em tabela única

Cada escolha de desenho das Seções 5.3, 5.4 e 5.7 e do texto acoplado ao
XGBoost (Seção 7) foi comparada ao modelo sem texto na mesma configuração:
LSTM apenas com preço, mesmo lookback e mesmos dias; XGBoost apenas com
preço na última linha.

| Verificação | BERTimbau k=8 | BERTimbau k=30 | FinBERT sent. | FinBERT emb. k=8 | FinBERT emb. k=30 |
|---|---|---|---|---|---|
| Principal (manchete, LSTM, lookback 21) | −0,02 [−0,08; +0,06] | +0,05 [−0,01; +0,13] | +0,01 [−0,04; +0,08] | +0,02 [−0,04; +0,08] | +0,03 [−0,04; +0,09] |
| Texto do dia anterior (D−1) | −0,01 [−0,08; +0,06] | — | — | — | — |
| Última manchete antes do corte | 0,00 [−0,06; +0,07] | — | — | — | — |
| Lookback de 1 dia | −0,02 [−0,08; +0,05] | — | — | — | — |
| Lookback de 3 dias | −0,03 [−0,10; +0,04] | — | — | — | — |
| Lookback de 5 dias | −0,02 [−0,09; +0,04] | — | — | — | — |
| Lookback de 10 dias | −0,03 [−0,10; +0,05] | — | — | — | — |
| TF-IDF+SVD no lugar do embedding | 0,00 [−0,07; +0,07] | +0,02 [−0,05; +0,10] | — | — | — |
| Corpo completo (512 tokens) | +0,01 [−0,05; +0,07] | +0,03 [−0,05; +0,11] | 0,00 [−0,06; +0,06] | 0,00 [−0,06; +0,06] | +0,02 [−0,04; +0,10] |
| Corpo completo (chunking) | −0,01 [−0,08; +0,06] | — | −0,01 [−0,07; +0,06] | +0,01 [−0,05; +0,08] | — |
| XGBoost no lugar da LSTM | −0,05 [−0,10; +0,01] | −0,04 [−0,10; +0,03] | −0,03 [−0,06; +0,01] | −0,01 [−0,06; +0,04] | −0,04 [−0,10; +0,02] |

*(ΔMCC agrupado com IC de 95% por block bootstrap sobre dias; fonte:
`reports/ic_mcc_bootstrap.json`, chave `robustez`. "—": combinação não
testada.)*

Nenhum dos 26 intervalos exclui zero, com a mesma resolução da comparação
principal: os limites superiores vão de +0,01 a +0,13 (mediana +0,06). A
cadeia mostra que nenhuma escolha de desenho destrava sinal, não que
efeitos pequenos estejam excluídos. O nível de `n_noticias`, que cresce com
o tempo (r = 0,80) e fica, no teste, de 1,3 a 2,3 desvios-padrão acima da
média do treino, permanece nas LSTMs com embeddings e no ARIMAX, embora o
nível do volume e da Selic seja excluído por carregar tendência estrutural.
Retirá-lo das quatro LSTMs com embeddings não altera o resultado (ΔMCC de
−0,04 a +0,01, IC contendo zero; o R² melhora em duas e piora em duas;
`scripts/sensibilidade_n_noticias.py`). O ARIMAX não foi testado sem essa
variável.

### 5.9 Janela de reação: texto antes da abertura

Uma notícia publicada após o fechamento de D entra na linha D+1, cujo alvo
é o retorno D+1→D+2; o retorno que reage a ela, D→D+1, é o contemporâneo
dessa linha. O alvo fechamento a fechamento não isola essa janela, e o gap
noturno responde por 47% da variância do retorno diário. O desenho
preditivo adequado à informação publicada fora do pregão prevê o retorno
abertura→fechamento com o texto disponível antes da abertura, o que exige
aberturas reais (Seção 3.5).

| Sentimento publicado antes da abertura | Gap (já ocorreu) | Abertura→fechamento (preditivo) | Parte ortogonal ao gap |
|---|---|---|---|
| Todas | +0,127 [+0,08; +0,18] | 0,000 [−0,05; +0,05] | −0,004 [−0,06; +0,05] |
| Relato | +0,162 [+0,12; +0,22] | −0,002 [−0,06; +0,05] | −0,009 [−0,06; +0,04] |
| Evento | +0,046 [−0,01; +0,10] | +0,012 [−0,04; +0,07] | +0,011 [−0,04; +0,07] |
| Noite (após o fechamento até 24h, fins de semana) | −0,042 [−0,10; 0,00] | −0,038 [−0,09; +0,01] | −0,037 [−0,09; +0,01] |
| Manhã (0h–10h) | +0,244 [+0,18; +0,31] | +0,031 [−0,02; +0,09] | +0,023 [−0,03; +0,07] |

*(Correlações de Pearson, IC de 95% por block bootstrap sobre dias; fonte:
`reports/diagnostico_janela_abertura.json`.)*

O sentimento pré-abertura correlaciona-se com o gap, sobretudo o da manhã,
mas não com abertura→fechamento: relaciona-se com o preço que já se moveu.
Nos 10 testes preditivos, o menor p corrigido por Holm é 1,00. A mesma
LSTM, com as variáveis de preço até t−1, o gap de t e o texto pré-abertura,
foi treinada nesse desenho, com os mesmos folds, sementes e embargo:

| Modelo | DA | MCC (IC) | ΔMCC vs. só preço (IC) | R² | Sharpe bruto / 10 bps ao dia |
|---|---|---|---|---|---|
| Só preço + gap | 50,8% | +0,013 [−0,05; +0,07] | — | −0,20 | +0,16 / −1,14 |
| FinBERT sentimento | 48,9% | −0,038 [−0,10; +0,03] | −0,051 [−0,12; +0,02] | −0,19 | −0,04 / −1,34 |
| BERTimbau k=8 | 49,3% | −0,026 [−0,09; +0,05] | −0,039 [−0,09; +0,02] | −0,22 | −0,17 / −1,47 |
| BERTimbau k=30 | 50,6% | +0,009 [−0,05; +0,06] | −0,004 [−0,09; +0,07] | −0,22 | +0,49 / −0,81 |
| FinBERT emb. k=8 | 50,0% | −0,012 [−0,07; +0,05] | −0,025 [−0,08; +0,04] | −0,24 | +0,36 / −0,93 |
| FinBERT emb. k=30 | 49,7% | −0,020 [−0,08; +0,04] | −0,033 [−0,10; +0,04] | −0,20 | +0,17 / −1,12 |

*(Fonte: `reports/modelo_abertura/resumo.json`; 1.011 dias de teste, sem os
pregões sem negociação.)*

O modelo com texto não supera o modelo sem texto (Holm: menor p do ΔMCC de
0,84), nenhum se distingue do acaso (Holm: 1,00 em todos), e nenhuma
variante supera ficar sempre vendido, cujo Sharpe bruto é +0,68, porque o
retorno intradiário foi negativo na amostra (comprar na abertura rende
−0,68). A 10 pontos-base ao dia, com entrada e saída a 5 pontos-base cada,
todos têm Sharpe negativo.

**Copom.** Divulgado às 18h30, o comunicado tem a reação fora do alvo por
construção: testa-se se ele prevê o dia seguinte à reação. Abrindo a
janela, em 48 eventos:

| Sinal (r) | Gap | Abertura→fechamento | Reação (D→D+1, índice) | Alvo (D+1→D+2) |
|---|---|---|---|---|
| Escore (nível) | −0,03 | −0,16 | −0,17 | +0,03 |
| Escore sem anúncio | −0,16 | +0,06 | +0,03 | +0,20 |
| Δ do escore | +0,06 | −0,07 | −0,07 | −0,14 |
| Δ do escore sem anúncio | +0,10 | +0,07 | +0,15 | +0,07 |
| Direção da decisão | −0,07 | −0,16 | −0,21 | +0,08 |
| Variação da Selic | −0,12 | −0,16 | −0,21 | +0,08 |
| Surpresa, média Focus (30 dias) | −0,05 | −0,23 | −0,28 | −0,04 |
| Surpresa, mediana Focus (7 não nulas) | −0,05 | −0,18 | −0,23 | −0,12 |
| Surpresa, média Focus (5 dias, n = 30) | −0,35 | −0,20 | −0,26 | −0,16 |
| Surpresa, mediana Focus (5 dias, n = 30; 3 não nulas) | −0,30 | −0,17 | −0,20 | −0,20 |

*(Fonte: `reports/diagnostico_copom_reacao.json`. Com regressor esparso, os
ICs por bootstrap não são confiáveis; os p-valores são por permutação.)*

Nenhum escore textual, em nível ou em variação entre comunicados, tem
relação detectável com a surpresa frente ao Focus (escore, r = +0,19,
p = 0,19 por permutação; variação do escore, r = +0,12, p = 0,43; menor p entre
os oito cruzamentos, 0,13), o gap, o pregão seguinte ou o alvo. Já a surpresa
aponta na direção esperada na reação (alta acima do esperado → queda;
r = −0,28, p = 0,05) e quase some no alvo
(−0,04), mas nenhuma das 40 correlações resiste à correção de Holm (menor p
corrigido de 1,00): é indício, e só |r| ≳ 0,3 seria detectável com 48
eventos. O |retorno| médio nos dias de reação é de 1,19%, contra 1,03% nos
demais (diferença de +0,16 ponto percentual, IC [−0,07; +0,39]).

---

## 6. Discussão

Os resultados deste trabalho convergem para uma conclusão consistente sob
três perspectivas — acurácia estatística de magnitude (MAE/RMSE, R²),
habilidade direcional formal (Pesaran-Timmermann) e utilidade econômica
(Sharpe líquido de custos), todas derivadas das mesmas previsões: nenhum
modelo, com ou sem texto, produz previsão de retorno diário do Ibovespa
estatisticamente distinguível do acaso. O que se pode afirmar é
quantificado: os intervalos excluem acurácia acima de aproximadamente 55%,
não efeitos menores (Seção 5.1).

O resultado mais informativo não é a ausência de sinal em si, mas a
caracterização de onde está o sinal textual (Seção 5.5): o sentimento
acompanha o retorno do dia, sobretudo nas manchetes que relatam preços e
nas publicadas após o fechamento, e, descontada essa parte, nada sobra para
D+1 nem para o retorno abertura→fechamento (Seção 5.9). Esse aspecto
distingue o presente trabalho de resultados nulos não explicados e de
trabalhos que reportam ganhos sem controle adequado de vazamento temporal,
problema recorrente na literatura de FinBERT-LSTM discutida na Seção 2. O
próprio erro de carimbo encontrado neste trabalho (Seção 3.3), detectado
apenas ao confrontar os dados com relógios independentes, ilustra como esse
tipo de problema escapa a testes que verificam só o código.

O classificador do Copom é o único sinal textual com validade verificável
contra um critério externo: o escore correlaciona-se com a decisão real de
Selic em r=+0,615 (IC de 95% [+0,44; +0,75]; 49 comunicados), de modo que o
nulo não decorre de um instrumento alheio à decisão. Essa validade, porém,
é sobretudo a leitura da decisão anunciada: sem as frases do anúncio,
r=+0,25 (IC [−0,05; +0,50]), o que não estabelece que o escore meça tom
além dela, coerente com a ausência de relação na janela de reação (Seção
5.9). A surpresa frente à expectativa, medida fora do texto, é o único
sinal com indício de reação, e não resiste à correção para múltiplas
comparações.

A vantagem do XGBoost sobre as variantes de LSTM com texto, significativa
após Holm apenas frente à LSTM+FinBERT-sentimento, requer interpretação
criteriosa: não significa que o XGBoost preveja o mercado, uma vez que seu
MCC de +0,028 ± 0,038 também não é distinguível de zero, e seu p-valor de
Pesaran-Timmermann, 0,727 após correção, está distante da significância.
Significa, antes, que, entre modelos igualmente desprovidos de habilidade
preditiva real, os modelos mais simples produzem erros de magnitude menor e
mais estável — argumento favorável à parcimônia como princípio de projeto
mesmo em cenários nos quais nenhum modelo alcança sucesso preditivo.

Esses resultados são compatíveis com a hipótese de mercados eficientes, mas
não a demonstram. O texto relaciona-se com o preço que já se moveu, e o
resultado nulo tem a resolução descrita na Seção 5.1: um efeito pequeno,
abaixo do limiar de poder, passaria despercebido. Isso não implica que o
texto seja desprovido de utilidade para todos os fins de modelagem
financeira, mas apenas que, no horizonte e na frequência aqui testados —
diário, sem acesso a dados intradiários —, não se detecta habilidade
atribuível a ele.

Um resultado adicional merece registro: o MCC do passeio aleatório
direcional, isto é, da estratégia de persistência que repete o sinal do
retorno de hoje, é negativo (−0,033), e não próximo de zero. A correlação
negativa entre persistência e o retorno seguinte sugere leve reversão à
média de curtíssimo prazo no Ibovespa diário, padrão que nenhum dos modelos
testados, do ARIMA à LSTM, converteu em habilidade preditiva
estatisticamente significativa — compatível com a leitura de eficiência
informacional, sem demonstrá-la.

Torres, Bonomo e Fernandes (2002) rejeitaram a hipótese de passeio
aleatório para a Bovespa diária e semanal entre 1986 e 1998, mas
registraram que o grau de rejeição diminuía ao longo da amostra, sobretudo
após o Plano Real. A estrutura linear fraca (e não ausente) encontrada
nesta amostra, duas décadas mais recente, é compatível com a continuidade
dessa tendência.

---

## 7. Limitações

- **Frequência diária.** Sem dados intradiários, a velocidade de
  incorporação de notícias ao preço não é observada. O teste com aberturas
  reais (Seção 5.9) separa gap e pregão, mas um horizonte de minutos ou
  horas poderia, em princípio, capturar janela de informação ainda não
  precificada, o que está fora do escopo deste trabalho.
- **Viés de classificação do FinBERT-PT-BR.** O modelo utilizado
  sobrerrepresenta sentimento negativo (53% contra 21% positivo) em
  período de alta do índice. Testou-se diretamente se esse viés é
  específico do FinBERT-PT-BR, comparando-o a um segundo modelo de
  linhagem independente e sem ajuste financeiro
  (`nlptown/bert-base-multilingual-uncased-sentiment`, multilíngue,
  escala de 1 a 5 estrelas) aplicado ao mesmo corpus de manchetes. O
  resultado é contrário à hipótese original de que o viés decorreria da
  interpretação equivocada de vocabulário técnico financeiro pelo
  FinBERT-PT-BR: o modelo genérico classifica uma fração **ainda maior**
  das manchetes como negativas (67,7%, contra 53% do FinBERT-PT-BR),
  sugerindo que a assimetria reflete antes o estilo editorial das
  manchetes financeiras em português — que tende a enfatizar riscos,
  quedas e alertas mesmo em períodos de alta — do que uma característica
  específica do FinBERT-PT-BR. A correlação entre as polaridades dos dois
  modelos é moderada (r = +0,348), indicando concordância parcial sobre
  quais dias são relativamente mais positivos ou negativos, apesar da
  calibração absoluta distinta. Como verificação adicional, a correlação
  preditiva da polaridade do modelo alternativo com o retorno de D+1
  também é estatisticamente nula (r = +0,008, contra r = +0,074 com o
  retorno contemporâneo) — o mesmo padrão reativo já documentado para o
  FinBERT-PT-BR (Seção 5.5), obtido agora por uma via de sentimento
  totalmente independente. A análise de correlação preditiva reportada
  como resultado principal utiliza, em todo caso, a polaridade contínua
  do FinBERT-PT-BR, e não a classe discreta, permanecendo válida
  independentemente deste viés.
- **Análise principal restrita à manchete.** Por desenho da coleta
  original — requisição em streaming, interrompida a aproximadamente
  200KB para capturar somente o bloco JSON-LD do cabeçalho HTML — o
  corpus usado nas Seções 5.1 a 5.6 contém exclusivamente a manchete de
  cada matéria. A Seção 5.7 testou diretamente se essa restrição de
  escopo explicava o resultado nulo, recoletando o corpo completo de
  56.819 matérias (98,6% do corpus) e retreinando todas as variantes de
  LSTM com texto sob protocolo idêntico; os deltas de MCC observados
  (todos entre −0,012 e +0,016) permaneceram inferiores ao
  desvio-padrão entre sementes. Testou-se ainda se o truncamento em 512
  tokens do próprio corpo completo (limite de contexto do BERTimbau e do
  FinBERT-PT-BR, que deixava artigos mais longos representados apenas por
  sua porção inicial) escondia sinal recuperável: o texto de cada matéria
  foi dividido em blocos de até 512 tokens (*chunking*), cada bloco
  processado separadamente e os embeddings agregados por média, cobrindo
  o artigo inteiro (56.819 artigos, 123.570 blocos). O MCC com *chunking*
  permaneceu dentro da mesma faixa de ruído observada com o corpo
  truncado (por exemplo, LSTM+BERTimbau k=8: −0,020 com *chunking* contra
  −0,007 truncado e −0,023 com manchete), sem melhora atribuível à
  cobertura textual mais completa. A ausência de sinal preditivo não
  decorre, portanto, nem do escopo (manchete versus corpo) nem do
  comprimento de contexto disponível ao modelo dentro do corpo completo.
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
- **Recorrência versus capacidade.** Testou-se um perceptron multicamadas
  (MLP) simples — uma camada oculta de 32 unidades, mesmo otimizador,
  número de épocas, dropout e regularização da LSTM — sobre a mesma janela
  de 21 dias, apenas achatada (21×18 = 378 entradas) em vez de sequencial,
  sob protocolo idêntico (walk-forward, embargo, `StandardScaler` por
  fold, calibração isotônica, 5 sementes). O MLP obteve MCC de
  −0,004 ± 0,046, estatisticamente indistinguível do MCC da LSTM
  (−0,009 ± 0,057) e, como este, também indistinguível de zero — ambos
  permanecem distantes do XGBoost (+0,028 ± 0,038). A recorrência da LSTM
  não produz, portanto, vantagem detectável sobre um MLP raso neste
  conjunto de dados; a limitação de arquitetura não explica o resultado
  nulo.
- **Eficiência em dado pequeno: texto acoplado a um modelo não
  recorrente.** Questionamento de banca: dado o tamanho reduzido da
  amostra (≈1.492 pregões), uma rede recorrente como a LSTM —
  arquitetura notoriamente ávida por dados — poderia ser um veículo
  estruturalmente fraco para o sinal textual, de modo que o resultado
  nulo refletisse a escolha do modelo, e não a ausência de sinal no
  próprio texto. Testou-se essa hipótese diretamente: o mesmo sinal
  textual usado no nível 5/via 1/via 2 (BERTimbau k=8/k=30, FinBERT
  embeddings k=8/k=30, FinBERT sentimento; PCA ajustado exclusivamente
  no treino de cada fold, protocolo idêntico) foi acoplado ao XGBoost em
  vez da LSTM, mantendo os hiperparâmetros e o protocolo do nível 3 (10
  sementes, calibração isotônica via hold-out interno, embargo=0). Em
  todas as cinco variantes, o MCC do XGBoost com texto foi inferior ao do
  XGBoost apenas-preço (+0,028 ± 0,038): BERTimbau k=8, +0,015 ± 0,041;
  BERTimbau k=30, +0,025 ± 0,036; FinBERT emb. k=8, +0,013 ± 0,045; FinBERT
  emb. k=30, +0,016 ± 0,037; FinBERT sentimento, +0,006 ± 0,038 — o mesmo
  padrão nulo observado com a LSTM, com todos os ΔMCC agrupados contendo
  zero (Seção 5.8). Ainda assim, o XGBoost com texto supera a LSTM com o
  mesmo texto em quatro das cinco comparações diretas, empatando na quinta
  (FinBERT sentimento, +0,006 em ambos) — indício de que a arquitetura mais
  eficiente em dado pequeno extrai algo a mais do sinal textual do que a
  LSTM, mas não o suficiente para superar sequer o próprio baseline de preço
  do XGBoost. O resultado nulo, portanto, não é artefato de ter acoplado o
  texto a uma arquitetura inadequada ao volume de dados disponível.

  *(Fonte:
  `reports/xgboost_{bertimbau_k8,bertimbau_k30,finbert_k8,finbert_k30,finbert_sent}_full.parquet`,
  notebook `07_ablacoes.ipynb`, célula "Ablação — XGBoost + embeddings de
  texto".)*
- **Interpretação dos componentes de PCA.** Ajustou-se, para fins
  exclusivamente diagnósticos, um PCA (k=8) sobre a série completa de
  embeddings do BERTimbau (não o PCA por fold usado nos modelos
  reportados, que constituiria vazamento se empregado para esse fim). A
  leitura qualitativa (manchetes nos extremos de cada componente) não
  revela eixos semânticos nitidamente interpretáveis — padrão esperado
  para embeddings de propósito geral, cujas dimensões não correspondem a
  conceitos isolados. A leitura quantitativa, por correlação de cada
  componente agregado por dia com metadados conhecidos, é mais reveladora:
  o segundo componente (5,9% da variância) correlaciona-se com o **volume
  editorial** do dia (`n_noticias`, r = +0,44; r = −0,58 com
  `emb_dispersao`), o primeiro (11,8%) quase não (r = −0,10), e a polaridade
  de sentimento aparece com |r| de até 0,16. Essa correlação com o volume é
  sobretudo tendência comum: `n_noticias` tem r = 0,80 com o tempo, e,
  retirada a tendência, a correlação do segundo componente cai para +0,16
  (IC [+0,06; +0,25]). Nenhum componente, entre os oito testados,
  correlaciona-se de forma perceptível com o retorno contemporâneo ou com o
  retorno de D+1 (|r| ≤ 0,04 em todos os casos). Isto é, o sinal capturado
  pelos primeiros componentes principais descreve o noticiário do dia, não
  a direção do mercado.
- **Ambiente computacional.** Os resultados aqui reportados foram gerados
  em GPU RTX 3070. Os modelos que não dependem do carimbo corrigido reproduzem os
  resultados anteriores: XGBoost, LSTM apenas com preço em todos os
  lookbacks e LSTM+Copom são idênticos, e o ARIMA difere em no máximo
  2×10⁻¹⁰.
- **Estratégia financeira estilizada.** A avaliação de Sharpe e drawdown
  emprega estratégia mecânica simples, baseada no sinal da previsão, sem
  dimensionamento de posição, sem mecanismo de *stop-loss* e sem
  consideração de liquidez ou custo de financiamento de posições vendidas;
  não pretende constituir estratégia de investimento realista, mas apenas
  um teste de utilidade econômica mínima das previsões.
- **BOVA11 como aproximação do índice.** O teste com aberturas reais usa um
  ETF, não o índice (correlação de 0,988 no retorno fechamento a
  fechamento), com dados do Yahoo Finance que têm falhas conhecidas (Seção
  3.5). A série oficial da B3 (COTAHIST) é a alternativa mais citável.
- **Escolha dos modelos de embedding.** BERTimbau e FinBERT-PT-BR foram
  escolhidos por serem, à época do desenho deste trabalho, referências
  estabelecidas para texto em português geral e financeiro,
  respectivamente. Não foram comparados contra alternativas mais recentes
  de representação textual em português (modelos de sentence-embedding
  mais novos, por exemplo); um resultado nulo obtido com esses dois
  modelos específicos não descarta que uma representação mais moderna
  capture um sinal diferente — a robustez demonstrada neste trabalho
  (anisotropia, BoW/TF-IDF, corpo completo) é robustez *dentro* da família
  de modelos testada, não uma prova de que nenhum embedding jamais
  funcionaria.

---

## 8. Conclusão

Sob protocolo de validação walk-forward rigoroso — com embargo temporal
apropriado à arquitetura de cada modelo, ajuste de todo componente
dependente de dados exclusivamente dentro do conjunto de treino de cada
fold, avaliação multi-semente para todo modelo estocástico, e testes
formais de significância com correção para múltiplas comparações —, nenhum
modelo testado neste trabalho, com ou sem embeddings de notícias
financeiras em português ou sinal de política monetária do Copom, prevê o
retorno diário do Ibovespa de forma estatisticamente distinguível do acaso.
Os intervalos de 95% excluem acurácia direcional acima de aproximadamente
55% em qualquer modelo (Sharpe bruto de aproximadamente 1,1), mas não
efeitos menores, que este desenho não distingue de zero com poder adequado.
O sentimento das manchetes acompanha o retorno do próprio dia (r = +0,18),
sobretudo nas que relatam preços, e a parte não explicada pelo preço não
prevê o dia seguinte (r = −0,007).

O resultado caracteriza, portanto, o que o desenho consegue e não consegue
detectar. O texto não acrescenta habilidade detectável nem no alvo
fechamento a fechamento nem no retorno abertura→fechamento, com o texto
disponível antes da abertura: relaciona-se com o preço que já se moveu. No
Copom, a surpresa frente à expectativa é indício a verificar com amostra
maior ou dados intradiários.

O passo seguinte é o teste da variância, onde está a dependência serial da
série: baseline HAR (CORSI, 2009) com perda QLIKE, volatilidade por
amplitude (Parkinson, Garman-Klass) e volume de notícias como candidato a
preditor. Os estimadores devem usar o BOVA11, porque a amplitude do índice
no Stooq incorpora o gap noturno (correlação de 0,99 com a do BOVA11
estendida ao fechamento anterior, contra 0,89 com a do pregão; cerca de um
terço maior em dias de gap acima de 1%), e Garman-Klass ainda depende da
abertura, que no índice é o fechamento anterior. O indício a favor do
volume de notícias é modesto: sem tendência, sua correlação com a
volatilidade do dia seguinte é +0,08 (IC [+0,01; +0,15]), e +0,03 (IC
[−0,03; +0,10]) descontada a do dia
(`scripts/diagnostico_ingredientes_volatilidade.py`). A persistência da
volatilidade, por sua vez, é clara: o log da volatilidade de Parkinson do
BOVA11 tem autocorrelação de 0,54, 0,47 e 0,32 nas defasagens 1, 5 e 21.

---

## Referências

- ARACI, D. FinBERT: Financial Sentiment Analysis with Pre-trained Language
  Models. *arXiv preprint arXiv:1908.10063*, 2019.
- CORSI, F. A Simple Approximate Long-Memory Model of Realized Volatility.
  *Journal of Financial Econometrics*, v. 7, n. 2, p. 174–196, 2009.
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
- LOUGHRAN, T.; MCDONALD, B. When Is a Liability Not a Liability? Textual
  Analysis, Dictionaries, and 10-Ks. *The Journal of Finance*, v. 66, n. 1,
  p. 35–65, 2011. DOI: 10.1111/j.1540-6261.2010.01625.x.
- GU, S.; KELLY, B.; XIU, D. Empirical Asset Pricing via Machine Learning.
  *The Review of Financial Studies*, v. 33, n. 5, p. 2223–2273, 2020.
- SIRIGNANO, J.; CONT, R. Universal features of price formation in
  financial markets: perspectives from Deep Learning. *Quantitative
  Finance*, v. 19, n. 9, p. 1449–1459, 2019.
- TORRES, R.; BONOMO, M.; FERNANDES, C. A Aleatoriedade do Passeio na
  Bovespa: Testando a Eficiência do Mercado Acionário Brasileiro. *Revista
  Brasileira de Economia*, v. 56, n. 2, 2002.
- TANG, Y.; YANG, Y. FinMTEB: Finance Massive Text Embedding Benchmark. In:
  *Proceedings of the 2025 Conference on Empirical Methods in Natural
  Language Processing (EMNLP)*, 2025.

---

## Apêndice A — Reprodutibilidade

Todo o código-fonte deste trabalho — notebooks, scripts de coleta e o
módulo `utils/tcc_utils.py` (fonte única das funções de corte temporal,
geração do alvo e particionamento walk-forward) — está publicado no
repositório do projeto no GitHub, com instruções de execução em
`README.md` e ambiente fixado em `requirements.txt`. Os dados brutos,
as variáveis derivadas e os resultados reportados neste trabalho podem
ser integralmente reconstituídos a partir desse código, seguindo a ordem
de execução documentada; por seu volume, não são distribuídos no mesmo
repositório, mas podem ser disponibilizados sob solicitação. A suíte de
testes automatizados `utils/test_tcc_utils.py` (25 casos, cobrindo corte
temporal, fechamento efetivo, carimbo, horário de verão, embargo, geração
do alvo e verificações sobre os próprios dados) documenta e verifica
formalmente as garantias anti-vazamento descritas na Seção 3.3. A correção
do carimbo foi aplicada sobre os dados já extraídos, sem refazer os
embeddings (`scripts/corrige_carimbo.py`,
`scripts/regenera_features_texto.py`, `scripts/reexecuta_notebooks.py`).
