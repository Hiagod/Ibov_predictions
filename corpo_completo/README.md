# Corpo completo das matérias

Código da verificação de robustez com o corpo completo das matérias, em vez da manchete (Seção V-D e Tabela III do
artigo; Seção 5.7 do `tcc.md`). O texto das matérias, os embeddings e as previsões não estão no repositório.

## Como foi executado

Em uma segunda cópia do projeto, `D:\tcc_ibov_corpo_completo`, com os mesmos `utils/`, `data/` e `reports/`. Os
scripts usam como raiz a pasta-mãe de `scripts/`, então não rodam daqui: precisam ser copiados para a segunda cópia.
`scripts/corrige_carimbo_corpo.py`, `scripts/reexecuta_notebooks.py --corpo` e `scripts/ic_mcc_bootstrap.py`, do
projeto principal, apontam para esse caminho.

| Etapa | Onde roda | Arquivo | Resultado |
|---|---|---|---|
| 1 | cópia | `scripts/coleta_corpo.py` | baixa as matérias das URLs de `infomoney_corpus.parquet`, em shards retomáveis |
| 2 | cópia | `scripts/consolida_corpo.py` | `corpus_corpo_bucketed.parquet`, com 56.819 matérias (98,6% do corpus) |
| 3 | cópia | `scripts/extrai_embeddings_corpo.py` | BERTimbau e FinBERT-PT-BR truncados em 512 tokens |
| 4 | cópia | `scripts/extrai_embeddings_corpo_chunked.py` | os mesmos modelos sobre o texto inteiro, em blocos de até 512 tokens |
| 5 | principal | `scripts/corrige_carimbo_corpo.py` | carimbo corrigido por URL, sem refazer os embeddings |
| 6 | cópia | `scripts/agrega_features_corpo.py` e `scripts/agrega_features_corpo_chunked.py` | variáveis por pregão |
| 7 | principal | `scripts/reexecuta_notebooks.py --corpo 05_lstm` | LSTMs com o corpo truncado |
| 8 | cópia | `scripts/treina_lstm_corpo_chunked.py` | LSTMs com *chunking* |
| 9 | principal | `scripts/reexecuta_notebooks.py --corpo 06_avaliacao_final` | testes e tabelas |
| 10 | principal | `scripts/ic_mcc_bootstrap.py` | linhas do corpo completo na Tabela III |

## Notebooks

`colabs/` traz os notebooks da segunda cópia, sem as saídas, que eram anteriores à correção do carimbo. Os resultados
atuais saem dos logs da reexecução. Frente aos notebooks do projeto principal:

- `05_lstm.ipynb`: muda `PROJECT` e lê `features_texto_bertimbau_corpo.parquet`, `features_texto_finbert_corpo.parquet`
  e `features_sentimento_finbert_corpo.parquet`.
- `06_avaliacao_final.ipynb`: muda só `PROJECT`.

Os demais arquivos da segunda cópia, inclusive a baseline XGBoost, vêm do projeto principal.
