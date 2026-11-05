"""As LSTMs com embeddings recebem n_noticias em NIVEL, serie que triplica na amostra (r = 0,80 com o tempo). O artigo
exclui o nivel do volume e da Selic justamente por carregarem tendencia estrutural. Este script mede se essa
inconsistencia pesa: retreina as quatro variantes com embeddings sem o nivel de n_noticias (mantida a dispersao) e
compara com as previsoes atuais. FinBERT sent. nao usa n_noticias.

Mesma LSTM, sementes, folds, embargo, PCA e scaler do artigo. As previsoes vao para
reports/sensibilidade_carimbo/ com o sufixo sem_n_noticias; nada existente e alterado.

Uso:
    python sensibilidade_n_noticias.py   ->  reports/sensibilidade_n_noticias.json
"""
import json

import numpy as np
import pandas as pd

from sensibilidade_carimbo import EMB_COLS, LOOKBACK, MODELOS, PROC, PROJECT, concat, delta_mcc, delta_r2, mcc_vetor, roda
from tcc_utils import walk_forward_splits

OUT_JSON = PROJECT / 'reports' / 'sensibilidade_n_noticias.json'


def main():
    fp = pd.read_parquet(PROC / 'features_preco.parquet')
    X_price = fp[[k for k in fp.columns if k != 'y']].values.astype(np.float32)
    y = fp['y'].values.astype(np.float32)
    splits = walk_forward_splits(fp.index, n_splits=5, min_train=252, expanding=True, embargo=LOOKBACK)
    pos = np.concatenate([np.asarray(te, dtype=int) for _, te in splits])
    truth = y[pos] > 0
    preco = concat(np.load(PROJECT / 'reports' / 'preds_lstm_preco.npz'))

    mats = {}
    for fonte, arq in (('bertimbau', 'features_texto_bertimbau.parquet'), ('finbert', 'features_texto_finbert.parquet')):
        t = pd.read_parquet(PROC / arq).reindex(fp.index)
        mats[fonte] = (t[EMB_COLS].values.astype(np.float32), t[['emb_dispersao']].values.astype(np.float32))

    def r2(p):
        return float(1 - ((y[pos] - p) ** 2).sum() / (y[pos] ** 2).sum())

    saida = {}
    for nome, (fonte, _, arq) in MODELOS.items():
        if fonte not in mats:
            continue
        novo = concat(roda(nome, 'sem_n_noticias', X_price, y, splits, mats))
        atual = concat(np.load(PROJECT / 'reports' / arq))
        saida[nome] = {'mcc_atual': float(mcc_vetor(atual > 0, truth)), 'mcc_sem_n': float(mcc_vetor(novo > 0, truth)),
                       'r2_atual': r2(atual), 'r2_sem_n': r2(novo),
                       'efeito_dMCC': delta_mcc(novo, atual, truth), 'efeito_dR2': delta_r2(novo, atual, y[pos]),
                       'dMCC_vs_preco': delta_mcc(novo, preco, truth)}
        m = saida[nome]
        print(f"{nome:18s} MCC {m['mcc_atual']:+.3f} -> {m['mcc_sem_n']:+.3f} ({m['efeito_dMCC']['delta']:+.3f} "
              f"[{m['efeito_dMCC']['lo']:+.3f}; {m['efeito_dMCC']['hi']:+.3f}]) | R2 {m['r2_atual']:+.3f} -> {m['r2_sem_n']:+.3f} "
              f"({m['efeito_dR2']['delta']:+.3f} [{m['efeito_dR2']['lo']:+.3f}; {m['efeito_dR2']['hi']:+.3f}]) | "
              f"vs preco {m['dMCC_vs_preco']['delta']:+.3f} [{m['dMCC_vs_preco']['lo']:+.3f}; {m['dMCC_vs_preco']['hi']:+.3f}]", flush=True)
        OUT_JSON.write_text(json.dumps(saida, indent=1, ensure_ascii=False), encoding='utf-8')
    print('concluido', flush=True)


if __name__ == '__main__':
    main()
