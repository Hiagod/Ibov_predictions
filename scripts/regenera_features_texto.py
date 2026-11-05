"""Regenera as features de texto por pregao a partir do corpus corrigido (corrige_carimbo.py).

Mesma agregacao de colabs/03_features_texto.ipynb (cel. 10, 11, 14 e 15), sem refazer os embeddings: media
dos vetores do dia, dispersao ao centroide, n_noticias e as quatro variaveis de sentimento. Indice, colunas
e tipos seguem os arquivos originais. Antes de gravar, confere que a mesma agregacao aplicada ao carimbo
original reproduz as features originais.

Uso:
    python regenera_features_texto.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from sensibilidade_carimbo import BACKUP, EMB_COLS, PROC, SENT_COLS, agrega_sent, carrega_embeddings


def agrega_emb_livre(E, dia, index):
    """Como agrega_emb, mas aceita pregao sem manchete: emb NaN e n_noticias 0, como no notebook."""
    pos = pd.Series(np.arange(len(dia))).groupby(np.asarray(dia, dtype='datetime64[ns]')).groups
    medias = np.full((len(index), 768), np.nan, dtype=np.float32)
    n, disp = np.zeros(len(index)), np.zeros(len(index))
    for j, d in enumerate(index):
        idx = pos.get(d)
        if idx is None:
            continue
        bloco = E[np.asarray(idx)]
        centro = bloco.mean(axis=0)
        medias[j], n[j] = centro, len(idx)
        disp[j] = np.linalg.norm(bloco - centro, axis=1).mean() if len(idx) > 1 else 0.0
    return medias, n, disp


def monta_texto(E, dia, ref):
    medias, n, disp = agrega_emb_livre(E, dia, ref.index)
    out = pd.DataFrame(medias, index=ref.index, columns=EMB_COLS)
    out['n_noticias'], out['emb_dispersao'] = n, disp
    return out[ref.columns].astype(ref.dtypes.to_dict())


def monta_sent(probs, dia, ref):
    out = pd.DataFrame(agrega_sent(probs, dia, ref.index), index=ref.index, columns=SENT_COLS)
    return out[ref.columns].astype(ref.dtypes.to_dict())


def main():
    c = pd.read_parquet(PROC / 'corpus_bucketed.parquet').reset_index(drop=True)
    v0 = pd.read_parquet(BACKUP / 'processed' / 'corpus_bucketed.parquet').reset_index(drop=True)
    assert (c['url'].values == v0['url'].values).all()
    probs = np.load(PROC / 'finbert_probs_raw.npy')
    bt, fb = carrega_embeddings(c)

    trio = {'features_texto_bertimbau.parquet': (monta_texto, bt), 'features_texto_finbert.parquet': (monta_texto, fb),
            'features_sentimento_finbert.parquet': (monta_sent, probs)}
    for arq, (monta, X) in trio.items():
        ref = pd.read_parquet(BACKUP / 'processed' / arq)
        repro = monta(X, v0['trading_day'].values, ref)
        dif = float(np.abs(repro.values.astype(float) - ref.values.astype(float)).max())
        print(f'{arq}: reproducao do original, dif. max. {dif:.2e}', flush=True)
        assert dif < 1e-3, arq
        novo = monta(X, c['trading_day'].values, ref)
        assert novo.notna().all().all(), f'{arq}: pregao sem manchete'
        mudou = float((np.abs(novo.values.astype(float) - ref.values.astype(float)).max(axis=1) > 1e-6).mean())
        novo.to_parquet(PROC / arq)
        print(f'   gravado; {mudou:.1%} dos pregoes mudaram', flush=True)

    # matrizes por manchete que colabs/07 (cel. 6, 9 e 10) le; apagadas no fim da reexecucao
    np.save(PROC / 'emb_bertimbau_raw.npy', bt)
    np.save(PROC / 'finbert_emb_raw.npy', fb)
    print('matrizes por manchete gravadas', flush=True)


if __name__ == '__main__':
    main()
