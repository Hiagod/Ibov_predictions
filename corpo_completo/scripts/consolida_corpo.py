"""Consolida os shards de coleta_corpo.py num unico parquet alinhado ao
corpus_bucketed.parquet ja existente (mesma ordem/URLs, so acrescenta a
coluna de corpo). Mantem so os artigos com corpo extraido com sucesso.

Uso:
    python consolida_corpo.py
"""
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT / 'data' / 'raw'
SHARD_DIR = RAW_DIR / 'shards_corpo'


def main():
    shards = sorted(SHARD_DIR.glob('shard_*.parquet'))
    if not shards:
        raise SystemExit("Nenhum shard encontrado -- rode coleta_corpo.py primeiro.")
    bruto = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    print(f"Linhas brutas: {len(bruto):,}")

    bruto = bruto.drop_duplicates(subset='url', keep='last')
    ok = bruto[(bruto['status'] == 200) & (bruto['n_paragrafos'] > 0)].copy()
    print(f"Com corpo extraido com sucesso: {len(ok):,} ({len(ok)/len(bruto):.1%})")
    print(f"Tamanho do corpo (chars) -- mediana: {ok['corpo'].str.len().median():.0f} | "
          f"min: {ok['corpo'].str.len().min()} | max: {ok['corpo'].str.len().max()}")

    corpus_bucketed = pd.read_parquet(PROJECT / 'data' / 'processed' / 'corpus_bucketed.parquet')
    print(f"\ncorpus_bucketed (base, com trading_day ja calculado): {len(corpus_bucketed):,}")

    corpo_final = corpus_bucketed.merge(
        ok[['url', 'corpo', 'n_paragrafos']], on='url', how='inner')
    print(f"Apos merge (interseccao): {len(corpo_final):,} "
          f"({len(corpo_final)/len(corpus_bucketed):.1%} do corpus original)")

    out = PROJECT / 'data' / 'processed' / 'corpus_corpo_bucketed.parquet'
    corpo_final.to_parquet(out)
    print(f"\nSalvo: {out}")


if __name__ == '__main__':
    main()
