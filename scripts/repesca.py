"""Repesca: refaz somente as URLs que falharam na coleta em massa
(status == -1, erro de rede). Menos paralelismo (3 threads) para nao
sobrecarregar o servidor numa segunda passada sobre as mesmas falhas.

Uso:
    python repesca.py
"""
import time
from pathlib import Path

import pandas as pd

from coleta_artigos import baixa_um, coleta_shard, RAW_DIR, SHARD_DIR

REPESCA_DIR = RAW_DIR / 'shards_repesca'
REPESCA_DIR.mkdir(parents=True, exist_ok=True)
SHARD_SIZE = 1000
N_THREADS = 3


def main():
    shards = sorted(SHARD_DIR.glob('shard_*.parquet'))
    if not shards:
        raise SystemExit("Nenhum shard encontrado em " + str(SHARD_DIR) +
                          " -- rode coleta_artigos.py primeiro.")

    corpus = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    falhas = corpus[corpus['status'] == -1]['url'].tolist()
    print(f"{len(falhas):,} URLs com falha de rede a repescar")
    if not falhas:
        print("Nada a fazer.")
        return

    n_shards = (len(falhas) + SHARD_SIZE - 1) // SHARD_SIZE
    done = {p.stem for p in REPESCA_DIR.glob('repesca_*.parquet')}
    t0 = time.time()
    for s in range(n_shards):
        nome = f"repesca_{s:04d}"
        if nome in done:
            continue
        lo, hi = s * SHARD_SIZE, min((s + 1) * SHARD_SIZE, len(falhas))
        df = coleta_shard(falhas[lo:hi], N_THREADS)
        df.to_parquet(REPESCA_DIR / f"{nome}.parquet")
        ok = (df['status'] == 200).sum()
        print(f"repesca {s+1}/{n_shards} | {ok}/{len(df)} recuperadas | "
              f"{time.time()-t0:.0f}s")

    print("Repesca concluida. Shards em:", REPESCA_DIR)


if __name__ == '__main__':
    main()
