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

## Para repetir em outra pasta

1. Copie o projeto e ponha `corpo_completo/scripts/` e `corpo_completo/colabs/` sobre `scripts/` e `colabs/` da cópia.
2. Troque o `PROJECT` de `colabs/05_lstm.ipynb` e `colabs/06_avaliacao_final.ipynb` pelo caminho da cópia.
3. Obtenha o que não está no repositório:
   - em `data/raw/`: `infomoney_corpus.parquet` e, para não coletar de novo, `shards_corpo/`;
   - em `data/processed/`: `corpus_bucketed.parquet`, `bvsp_clean.parquet`, `features_preco.parquet` e
     `features_copom.parquet`;
   - também em `data/processed/`, para não extrair de novo: `emb_bertimbau_corpo_raw.npy`,
     `emb_bertimbau_corpo_chunked_raw.npy`, `finbert_corpo_emb_raw.npy`, `finbert_corpo_chunked_emb_raw.npy`,
     `finbert_corpo_probs_raw.npy` e `finbert_corpo_chunked_probs_raw.npy`.
4. Rode as etapas 7 e 9 com o `reexecuta_notebooks.py` da própria cópia, sem `--corpo`: essa opção só vale para
   `D:\tcc_ibov_corpo_completo`, e sem ela o script usa a pasta em que está.
5. A etapa 5 só é necessária se o corpus da cópia for anterior à correção do carimbo.

Para recalcular a Tabela III, ajuste `CORPO` em `scripts/ic_mcc_bootstrap.py`, que lê as previsões em
`D:\tcc_ibov_corpo_completo\reports`.

Assim, em um clone novo, sem a etapa 5 e na mesma máquina, o corpus, as variáveis, as previsões das LSTMs e as tabelas
saíram idênticos aos originais. A extração foi conferida em 160 matérias, e a coleta não foi repetida.
