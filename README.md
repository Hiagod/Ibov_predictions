# TCC — Embeddings de notícias financeiras em português na previsão do retorno diário do Ibovespa

Avaliação com validação walk-forward de se embeddings textuais de notícias
financeiras (InfoMoney) melhoram a previsão do retorno diário (D+1) do
Ibovespa frente a baselines financeiros clássicos e a um modelo tabular forte
treinado apenas com preço.

## Estrutura do repositório

```
colabs/     notebooks (00-07), adaptados para execução local
scripts/    scripts de coleta e análises pontuais (rodam uma vez, fora dos notebooks)
data/raw/   dados brutos coletados (preços, corpus de notícias)
data/processed/  features derivadas, embeddings, sequências para o LSTM
models/     checkpoints (quando salvos)
reports/    métricas, previsões e figuras geradas pelos notebooks
utils/      tcc_utils.py — fonte única canônica das funções centrais
corpo_completo/  código da verificação com o corpo completo das matérias (ver o README da pasta)
```

## Ambiente

Windows, Python 3.11, GPU NVIDIA (usado: RTX 3070, 8 GB, driver CUDA 12.x+).

```bash
python -m venv .venv
.venv\Scripts\pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m ipykernel install --user --name tcc_ibov --display-name "tcc_ibov (.venv)"
```

Verificar: `python -c "import torch; print(torch.cuda.is_available())"` deve
imprimir `True`. Não instalar o build `+cpu` do torch — é um erro recorrente
ao migrar de ambiente.

## Ordem de execução

**Coleta (uma vez só).** Este repositório contém apenas o código; `data/raw/` e `data/processed/` não são versionados aqui por volume — reexecute os passos abaixo para reconstituí-los do zero, ou solicite os arquivos diretamente ao autor:

1. `colabs/02_coleta_noticias.ipynb` — descobre sitemaps, monta a fila de URLs (`fila_artigos.parquet`).
2. `scripts/coleta_artigos.py` — baixa e extrai headline/datePublished de cada URL da fila (shards de 1000, retomável, 5 threads).
3. `scripts/repesca.py` — refaz as URLs que falharam por erro de rede.
4. `scripts/consolida_corpus.py` — junta shards, dedup, timezone, filtro de período → `infomoney_corpus.parquet`.
5. `scripts/verifica_timezone.py` — checagem pontual (amostra) do fuso do JSON-LD. Só vale depois da migração do site do InfoMoney (24/09/2019): antes dela, o `datePublished` é hora local rotulada como UTC, e `corrige_carimbo_infomoney` (`utils/tcc_utils.py`) trata os dois casos.
6. `colabs/01_coleta_precos.ipynb` — limpeza da série de preços (Stooq).
7. `scripts/coleta_copom.py` — comunicados do Copom via API de dados abertos do BCB (49 reuniões, 2018-2024) → `copom_comunicados.parquet`.

**Pipeline de features e modelagem (nesta ordem — cada notebook depende do anterior):**

0. `colabs/00_setup.ipynb` — grava `utils/tcc_utils.py` canônico (fonte única: `build_cutoffs`, `assign_trading_day`, `make_target`, `walk_forward_splits`). **Rode sempre primeiro** — os demais notebooks só importam, nunca reescrevem esse arquivo.
1. `colabs/01_coleta_precos.ipynb` — `bvsp_clean.parquet`.
2. `colabs/02_coleta_noticias.ipynb` — `corpus_bucketed.parquet` não é gerado aqui (ver notebook 03); este notebook só monta a fila de coleta.
3. `colabs/03_features_texto.ipynb` — bucket de notícias por pregão, extração de embeddings BERTimbau e FinBERT-PT-BR (sentimento + embeddings), `features_preco.parquet`, `features_texto_bertimbau.parquet`, `features_texto_finbert.parquet`, `features_sentimento_finbert.parquet`.
4. `colabs/04_baselines.ipynb` — Random Walk, ARIMA/ARIMAX, XGBoost. Usa a mesma geometria de fold canônica (`walk_forward_splits`, `n_splits+1`, `embargo=0`) do LSTM, para permitir comparação par-a-par.
5. `scripts/features_copom.py` — classificação zero-shot hawkish/dovish dos comunicados do Copom (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, com seleção empírica de janela início/fim do texto por correlação com a decisão real de Selic) e construção de `features_copom.parquet` (forward-fill entre reuniões, mesmo corte das notícias, no fechamento efetivo da B3).
6. `colabs/05_lstm.ipynb` — LSTM só-preço (nível 4), LSTM+BERTimbau (nível 5/5b), LSTM+sentimento FinBERT (Via 2), LSTM+embeddings FinBERT (Via 1/1b), LSTM+Copom (Via 3).
7. `colabs/06_avaliacao_final.ipynb` — testes estatísticos (Diebold-Mariano pareado, Pesaran-Timmermann pooled, correção de Holm), tabela comparativa consolidada dos modelos, e métricas financeiras (Sharpe, max drawdown, com e sem custo de transação).
8. `colabs/07_ablacoes.ipynb` — robustez e ablações, em ordem: (i) texto do dia D vs. D-1; (ii) sweep de janela de lookback {1,3,5,10} vs. o padrão 21, para LSTM só-preço e LSTM+texto; (iii) MLP simples sobre a janela achatada, vs. a recorrência da LSTM; (iv) anisotropia dos embeddings (whitening) e teste de permutação; (v) qualidade textual — remoção de manchetes-template e isolamento de linguagem prospectiva/retrospectiva; (vi) interpretação qualitativa e quantitativa dos componentes de PCA do BERTimbau; (vii) agregação por recência (última manchete do dia vs. média); (viii)-(ix) representação Bag-of-Words/TF-IDF (k=8 e k=30) vs. embedding denso; (x) mesmo sinal textual acoplado ao XGBoost em vez da LSTM, para isolar se o resultado nulo decorre da arquitetura recorrente em dado pequeno.
9. `scripts/analise_sentimento_alternativo.py` — roda um segundo modelo de sentimento (`nlptown/bert-base-multilingual-uncased-sentiment`, genérico, sem ajuste financeiro) sobre o mesmo corpus de manchetes, para testar se o viés de classificação negativa do FinBERT-PT-BR é específico do modelo ou do estilo editorial do corpus.
10. `scripts/teste_reativo_vs_preditivo.py` — teste formal (bootstrap, B=20.000, correção de Holm) da diferença entre a correlação contemporânea e a preditiva do sentimento FinBERT-PT-BR com o retorno, para as 4 variáveis de sentimento agregado — sustenta o achado central da Seção 5.5/V-G (sentimento reativo, não preditivo) → `reports/teste_reativo_vs_preditivo.{parquet,json}`.

**Correção do carimbo temporal (aplicada uma vez sobre os dados já extraídos).** Sem refazer os embeddings:

1. `scripts/corrige_carimbo.py` — corrige o fuso e aplica o corte no fechamento efetivo da B3 (17h ou 18h); guarda os originais em `data/_v0_carimbo_original/`. Linhas e ordem do corpus não mudam, porque os embeddings são alinhados por posição. **Não reexecute `colabs/03` cel. 1 sobre embeddings já extraídos.**
2. `scripts/regenera_features_texto.py` — reagrega as features de texto por pregão.
3. `scripts/reexecuta_notebooks.py` — reexecuta as células de `colabs/04`, `05`, `07` e `06` que dependem do texto (logs em `reports/reexecucao/`). Com `--corpo`, usa a segunda cópia do projeto descrita em `corpo_completo/README.md`, preparada por `scripts/corrige_carimbo_corpo.py`.
4. `scripts/sensibilidade_carimbo.py` — mede o efeito da correção, retreinando as variantes de texto com os dois carimbos.

**Análises da revisão do artigo:** a Fig. 3 sai de `scripts/figura_acf.py`, com a banda de ruído branco; `scripts/coleta_janela_reacao.py` (BOVA11, Focus), `diagnostico_fechamento_b3.py`, `diagnostico_janela_abertura.py`, `modelo_abertura.py`, `diagnostico_copom_reacao.py`, `diagnostico_copom_validade.py`, `diagnostico_relato_evento.py`, `diagnostico_janela_reacao.py`, `ic_mcc_bootstrap.py`, `ic_da_bootstrap.py`, `tabela_robustez.py` e `figura_*.py`.

Cada notebook pode ser executado de ponta a ponta via:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=tcc_ibov \
  --ExecutePreprocessor.timeout=3600 \
  colabs/<notebook>.ipynb
```

## Reprodutibilidade

- Sementes fixadas (`set_seed`) e determinismo CUDA travado (`cudnn.deterministic=True`) em todo treino de LSTM.
- Modelos estocásticos (XGBoost N=10 sementes, LSTM N=5 sementes) reportam média ± desvio, nunca um único run.
- Toda transformação dependente de dados (scaler, PCA, calibração isotônica, ordem do ARIMA) é ajustada **somente no treino de cada fold** do walk-forward — nunca na série inteira.
- **Ressalva:** os resultados originais deste projeto foram gerados numa T4 do Google Colab; a migração para a RTX 3070 local reproduziu os mesmos números dentro da variação esperada entre sementes (diferenças de ponto flutuante entre GPUs/versões de CUDA não são bit-a-bit idênticas, mas não mudam nenhuma conclusão). Os números da RTX 3070 são os reportados como definitivos no texto do TCC. Na reexecução com o carimbo corrigido, os modelos que não dependem do carimbo reproduziram os resultados anteriores: XGBoost, LSTM só-preço em todos os lookbacks e LSTM+Copom idênticos; ARIMA com diferença máxima de 2×10⁻¹⁰.

## Alvo e anti-vazamento

O alvo (`y_t = ln(close_{t+1}/close_t)`) é gerado em um único ponto
(`make_target`, `utils/tcc_utils.py`). A validação é sempre walk-forward
temporal (`walk_forward_splits`), com `embargo=21` no LSTM (= lookback) para
que nenhuma sequência de teste use dias que foram de treino; baselines usam
`embargo=0`. Nenhum split é aleatório em nenhum ponto do projeto.
