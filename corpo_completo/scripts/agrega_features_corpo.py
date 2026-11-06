"""Agrega os embeddings/sentimento do CORPO COMPLETO por pregao, espelhando
exatamente a metodologia ja usada para as manchetes (notebook 03, celulas
10-11 para BERTimbau/FinBERT embeddings, celula 14 para sentimento
FinBERT): mean-pooling por trading_day + dispersao, reindexado ao periodo
de estudo (2019-01 a 2024-12).

Uso:
    python agrega_features_corpo.py
"""
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
PROC = PROJECT / 'data' / 'processed'


def agrega_embeddings(E, corpus, tag):
    dias = corpus['trading_day'].values
    idx_por_dia = defaultdict(list)
    for i, d in enumerate(dias):
        idx_por_dia[d].append(i)
    dias_unicos = np.array(sorted(pd.unique(dias)))

    emb_mean, emb_disp, n_mat, index_dias = [], [], [], []
    for d in dias_unicos:
        idx = idx_por_dia[d]
        bloco = E[idx]
        centroide = bloco.mean(axis=0)
        disp = np.linalg.norm(bloco - centroide, axis=1).mean() if len(idx) > 1 else 0.0
        emb_mean.append(centroide)
        emb_disp.append(disp)
        n_mat.append(len(idx))
        index_dias.append(d)

    emb_mean = np.vstack(emb_mean)
    feat = pd.DataFrame(emb_mean, index=pd.DatetimeIndex(index_dias),
                         columns=[f'emb_{i:03d}' for i in range(E.shape[1])])
    # nome da coluna alinhado ao usado na versao de manchetes (n_noticias),
    # ainda que aqui conte materias com corpo extraido
    feat['n_noticias'] = n_mat
    feat['emb_dispersao'] = emb_disp
    feat.index.name = 'trading_day'

    precos = pd.read_parquet(PROC / 'bvsp_clean.parquet')
    estudo = precos.loc['2019-01-01':'2024-12-31']
    feat_estudo = feat.reindex(estudo.index)
    cobertura = feat_estudo['emb_000'].notna().mean()
    print(f"  [{tag}] cobertura do periodo de estudo: {cobertura:.2%}")
    feat_estudo['n_noticias'] = feat_estudo['n_noticias'].fillna(0)

    out = PROC / f'features_texto_{tag}_corpo.parquet'
    feat_estudo.to_parquet(out)
    print(f"  Salvo: {out} | shape {feat_estudo.shape}")
    return feat_estudo


def agrega_sentimento(probs, corpus):
    polaridade = probs[:, 0] - probs[:, 1]
    pred_classe = probs.argmax(axis=1)
    df_sent = pd.DataFrame({
        'trading_day': corpus['trading_day'].values,
        'polaridade': polaridade,
        'is_pos': (pred_classe == 0).astype(int),
        'is_neg': (pred_classe == 1).astype(int),
    })
    g = df_sent.groupby('trading_day')
    feat_sent = pd.DataFrame({
        'sent_polaridade': g['polaridade'].mean(),
        'sent_dispersao': g['polaridade'].std(),
        'sent_pos_frac': g['is_pos'].mean(),
        'sent_neg_frac': g['is_neg'].mean(),
    })
    feat_sent['sent_dispersao'] = feat_sent['sent_dispersao'].fillna(0)

    precos = pd.read_parquet(PROC / 'bvsp_clean.parquet')
    estudo = precos.loc['2019-01-01':'2024-12-31']
    feat_sent = feat_sent.reindex(estudo.index)
    print(f"  [sentimento] cobertura: {feat_sent['sent_polaridade'].notna().mean():.2%}")

    out = PROC / 'features_sentimento_finbert_corpo.parquet'
    feat_sent.to_parquet(out)
    print(f"  Salvo: {out} | shape {feat_sent.shape}")

    # sanidade: correlacao contemporanea vs preditiva (deve repetir o padrao
    # ja observado com manchetes -- contemporanea forte, preditiva fraca)
    feat_preco = pd.read_parquet(PROC / 'features_preco.parquet')
    alvo = feat_preco['y']
    ret_hoje = feat_preco['ret_lag0']
    corr_pred = feat_sent['sent_polaridade'].corr(alvo)
    corr_cont = feat_sent['sent_polaridade'].corr(ret_hoje)
    print(f"  Sanidade -- sent_polaridade(corpo) -> retorno(t+1): {corr_pred:+.4f}")
    print(f"  Sanidade -- sent_polaridade(corpo) -> retorno(t)  : {corr_cont:+.4f}")
    return feat_sent


def main():
    corpus = pd.read_parquet(PROC / 'corpus_corpo_bucketed.parquet').reset_index(drop=True)
    print(f"{len(corpus):,} materias com corpo\n")

    print("=== BERTimbau (corpo) ===")
    E_bt = np.load(PROC / 'emb_bertimbau_corpo_raw.npy')
    assert len(corpus) == E_bt.shape[0]
    agrega_embeddings(E_bt, corpus, 'bertimbau')

    print("\n=== FinBERT embeddings (corpo) ===")
    E_fb = np.load(PROC / 'finbert_corpo_emb_raw.npy')
    assert len(corpus) == E_fb.shape[0]
    agrega_embeddings(E_fb, corpus, 'finbert')

    print("\n=== FinBERT sentimento (corpo) ===")
    probs_fb = np.load(PROC / 'finbert_corpo_probs_raw.npy')
    assert len(corpus) == probs_fb.shape[0]
    agrega_sentimento(probs_fb, corpus)

    print("\nTodas as features de corpo agregadas.")


if __name__ == '__main__':
    main()
