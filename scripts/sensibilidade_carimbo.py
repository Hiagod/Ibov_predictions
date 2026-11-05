"""Sensibilidade das LSTMs com texto ao carimbo temporal (revisao do professor, C13).

A primeira versao dos resultados foi treinada com o carimbo original (V0), que tem erro de fuso ate
24/09/2019 (ver diagnostico_fechamento_b3.py). Esse erro nao toca os dias de teste, mas toca o
treino: 73% do treino do fold 1 e 17% do fold 5. Este script mede o efeito retreinando as cinco
variantes de texto com os dois carimbos, com o mesmo codigo e na mesma maquina, para que a diferenca
venha so dos dados:

    V0: trading_day original, da copia em data/_v0_carimbo_original/ (reproduz as features originais,
        conferido por assert)
    V2: fuso corrigido e corte no fechamento efetivo da B3 (17h ou 18h), o carimbo vigente

Arquitetura, sementes, folds, embargo, PCA e scaler identicos a colabs/05_lstm.ipynb. A calibracao
isotonica nao e refeita: aqui so importam as previsoes de regressao, base do MCC e da DA agrupados e
do R2. As previsoes vao para reports/sensibilidade_carimbo/; nada existente e alterado. O script
retoma de onde parou se interrompido.

Uso:
    python sensibilidade_carimbo.py   ->  reports/sensibilidade_carimbo/resumo.json
"""
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ic_mcc_bootstrap import B, BLOCO, SEED, indices_bloco, mcc_vetor

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / 'utils'))
from tcc_utils import walk_forward_splits

PROC = PROJECT / 'data' / 'processed'
BACKUP = PROJECT / 'data' / '_v0_carimbo_original'
OUT = PROJECT / 'reports' / 'sensibilidade_carimbo'
HIDDEN, DROPOUT, EPOCHS, LR, WD, BATCH, LOOKBACK = 16, 0.4, 75, 1e-3, 1e-4, 32, 21
SEEDS = [0, 1, 2, 3, 4]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMB_COLS = [f'emb_{i:03d}' for i in range(768)]
SENT_COLS = ['sent_polaridade', 'sent_dispersao', 'sent_pos_frac', 'sent_neg_frac']
# nome -> (fonte, k do PCA, arquivo salvo do artigo)
MODELOS = {'BERTimbau k=8': ('bertimbau', 8, 'preds_lstm_texto.npz'),
           'BERTimbau k=30': ('bertimbau', 30, 'preds_lstm_texto_k30.npz'),
           'FinBERT sent.': ('sentimento', None, 'preds_lstm_via2_sent.npz'),
           'FinBERT emb. k=8': ('finbert', 8, 'preds_lstm_via1_finbert_k8.npz'),
           'FinBERT emb. k=30': ('finbert', 30, 'preds_lstm_via1_finbert_k30.npz')}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden=16, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1, :])).squeeze(-1)


def treina_avalia(X_tr, y_tr, X_te, seed):
    set_seed(seed)
    model = LSTMRegressor(X_tr.shape[-1], HIDDEN, DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.MSELoss()
    Xtr, ytr = torch.tensor(X_tr, device=DEVICE), torch.tensor(y_tr, device=DEVICE)
    n = len(Xtr)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            lossf(model(Xtr[b]), ytr[b]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X_te, device=DEVICE)).cpu().numpy()


def carrega_embeddings(c):
    """Matrizes por manchete alinhadas 1:1 com corpus_bucketed. O BERTimbau foi extraido antes de
    remover uma manchete duplicada (colabs/03, cel. 9); a posicao dela esta registrada em
    indice_duplicata_bertimbau.json (corrige_carimbo.py)."""
    bt = np.vstack([np.load(f) for f in sorted((PROC / 'embeddings_bertimbau').glob('emb_*.npy'))])
    fb = np.vstack([np.load(f) for f in sorted((PROC / 'finbert_extract').glob('emb_*.npy'))])
    assert len(fb) == len(c), 'FinBERT desalinhado'
    if len(bt) != len(c):
        pos = json.loads((PROC / 'indice_duplicata_bertimbau.json').read_text(encoding='utf-8'))['posicao']
        bt = np.delete(bt, pos, axis=0)
    assert len(bt) == len(c), 'BERTimbau desalinhado'
    return bt, fb


def agrega_emb(E, dia, index):
    idx_por_dia = defaultdict(list)
    for i, d in enumerate(np.asarray(dia, dtype='datetime64[ns]')):
        idx_por_dia[d].append(i)
    medias, disp, n = np.full((len(index), 768), np.nan, dtype=np.float32), np.zeros(len(index)), np.zeros(len(index))
    for j, d in enumerate(index.values):
        idx = idx_por_dia.get(d, [])
        if not idx:
            continue
        bloco = E[idx]
        centro = bloco.mean(axis=0)
        medias[j], n[j] = centro, len(idx)
        disp[j] = np.linalg.norm(bloco - centro, axis=1).mean() if len(idx) > 1 else 0.0
    assert not np.isnan(medias).any(), 'pregao sem manchete'
    return medias, np.column_stack([n, disp]).astype(np.float32)


def agrega_sent(probs, dia, index):
    pol, classe = probs[:, 0] - probs[:, 1], probs.argmax(axis=1)
    d = pd.DataFrame({'dia': np.asarray(dia), 'pol': pol, 'pos': (classe == 0).astype(int), 'neg': (classe == 1).astype(int)})
    g = d.groupby('dia')
    s = pd.DataFrame({'sent_polaridade': g['pol'].mean(), 'sent_dispersao': g['pol'].std().fillna(0),
                      'sent_pos_frac': g['pos'].mean(), 'sent_neg_frac': g['neg'].mean()}).reindex(index)
    assert s.notna().all().all(), 'pregao sem manchete'
    return s[SENT_COLS].values.astype(np.float32)


def confere_v0(index, emb_bt, sig_bt, emb_fb, sig_fb, sent):
    """A reagregacao com o carimbo original precisa reproduzir as features usadas no artigo."""
    b = pd.read_parquet(BACKUP / 'processed' / 'features_texto_bertimbau.parquet').reindex(index)
    f = pd.read_parquet(BACKUP / 'processed' / 'features_texto_finbert.parquet').reindex(index)
    s = pd.read_parquet(BACKUP / 'processed' / 'features_sentimento_finbert.parquet').reindex(index)
    for rot, a, ref in (('BERTimbau emb', emb_bt, b[EMB_COLS]), ('BERTimbau sig', sig_bt, b[['n_noticias', 'emb_dispersao']]),
                        ('FinBERT emb', emb_fb, f[EMB_COLS]), ('FinBERT sig', sig_fb, f[['n_noticias', 'emb_dispersao']]),
                        ('sentimento', sent, s[SENT_COLS])):
        dif = float(np.abs(a - ref.values.astype(np.float32)).max())
        print(f'  V0 {rot:14s} dif. max. frente as features salvas: {dif:.2e}', flush=True)
        assert dif < 1e-3, rot


def roda(nome, versao, X_price, y, splits, mats):
    """Treina o modelo nos cinco folds e devolve {lstm_f1: previsao media entre sementes, ...}."""
    fonte, k, _ = MODELOS.get(nome, (None, None, None))
    arq = OUT / f"preds_{nome.replace(' ', '_').replace('=', '').replace('.', '')}_{versao}.npz"
    if arq.exists():
        print(f'{nome} {versao}: ja existe, pulando', flush=True)
        return dict(np.load(arq))
    t0 = time.time()
    preds = {}
    for kf, (tr_days, te_days) in enumerate(splits, 1):
        tr_set, te_set = set(tr_days), set(te_days)
        if fonte in ('bertimbau', 'finbert'):
            emb, sig = mats[fonte]
            mask = np.array([i in tr_set for i in range(len(emb))])
            pca = PCA(n_components=k, random_state=0).fit(emb[mask])
            X_day = np.concatenate([X_price, sig, pca.transform(emb).astype(np.float32)], axis=1)
        elif fonte == 'sentimento':
            X_day = np.concatenate([X_price, mats['sentimento']], axis=1)
        else:
            X_day = X_price
        sidx = np.arange(LOOKBACK - 1, len(X_day))
        seqs = np.stack([X_day[i - LOOKBACK + 1:i + 1] for i in sidx])
        targets = y[sidx]
        m_tr = np.array([i in tr_set for i in sidx])
        m_te = np.array([i in te_set for i in sidx])
        n_f = seqs.shape[-1]
        sc = StandardScaler().fit(seqs[m_tr].reshape(-1, n_f))
        X_tr = sc.transform(seqs[m_tr].reshape(-1, n_f)).reshape(seqs[m_tr].shape).astype(np.float32)
        X_te = sc.transform(seqs[m_te].reshape(-1, n_f)).reshape(seqs[m_te].shape).astype(np.float32)
        preds[f'lstm_f{kf}'] = np.mean([treina_avalia(X_tr, targets[m_tr], X_te, s) for s in SEEDS], axis=0)
    np.savez(arq, **preds)
    print(f'{nome} {versao}: {time.time() - t0:.0f}s', flush=True)
    return preds


def concat(p):
    return np.concatenate([p[f'lstm_f{k}'] for k in range(1, 6)])


def delta_mcc(pa, pb, truth):
    n = len(truth)
    idx = indices_bloco(n, BLOCO, np.random.default_rng(SEED))
    a, b, t = pa > 0, pb > 0, truth
    d = mcc_vetor(a[idx], t[idx]) - mcc_vetor(b[idx], t[idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return {'delta': float(mcc_vetor(a, t) - mcc_vetor(b, t)), 'lo': float(lo), 'hi': float(hi), 'p': p}


def delta_r2(pa, pb, yt):
    """Diferenca de R2 frente a prever zero (a menos b), com IC por block bootstrap sobre dias."""
    idx = indices_bloco(len(yt), BLOCO, np.random.default_rng(SEED))
    y2 = (yt[idx] ** 2).sum(1)
    d = ((yt[idx] - pb[idx]) ** 2).sum(1) / y2 - ((yt[idx] - pa[idx]) ** 2).sum(1) / y2
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {'delta': float((((yt - pb) ** 2).sum() - ((yt - pa) ** 2).sum()) / (yt ** 2).sum()),
            'lo': float(lo), 'hi': float(hi)}


def main():
    OUT.mkdir(exist_ok=True)
    fp = pd.read_parquet(PROC / 'features_preco.parquet')
    c = pd.read_parquet(PROC / 'corpus_bucketed.parquet').reset_index(drop=True)
    v0 = pd.read_parquet(BACKUP / 'processed' / 'corpus_bucketed.parquet').reset_index(drop=True)
    assert (c['url'].values == v0['url'].values).all(), 'corpus corrigido fora de ordem'
    probs = np.load(PROC / 'finbert_probs_raw.npy')
    bt, fb = carrega_embeddings(c)
    print(f'embeddings alinhados: BERTimbau {bt.shape}, FinBERT {fb.shape}; dispositivo {DEVICE}', flush=True)

    X_price = fp[[k for k in fp.columns if k != 'y']].values.astype(np.float32)
    y = fp['y'].values.astype(np.float32)
    splits = walk_forward_splits(fp.index, n_splits=5, min_train=252, expanding=True, embargo=LOOKBACK)
    pos = np.concatenate([np.asarray(te, dtype=int) for _, te in splits])

    dias = {'V0': v0['trading_day'].values, 'V2': c['trading_day'].values}
    mats = {}
    for v, dia in dias.items():
        eb, sb = agrega_emb(bt, dia, fp.index)
        ef, sf = agrega_emb(fb, dia, fp.index)
        se = agrega_sent(probs, dia, fp.index)
        if v == 'V0':
            confere_v0(fp.index, eb, sb, ef, sf, se)
        mats[v] = {'bertimbau': (eb, sb), 'finbert': (ef, sf), 'sentimento': se}

    preco = concat(roda('preco', 'V0', X_price, y, splits, {}))
    guardado = np.concatenate([np.load(BACKUP / 'reports' / 'preds_lstm_preco.npz')[f'lstm_f{k}'] for k in range(1, 6)])
    truth = y[pos] > 0
    rw = float((y[pos] ** 2).sum())

    def resumo(p):
        return {'da': float(((p > 0) == truth).mean()), 'mcc': float(mcc_vetor(p > 0, truth)),
                'r2': float(1 - ((y[pos] - p) ** 2).sum() / rw)}

    saida = {'preco': {'local': resumo(preco), 'artigo': resumo(guardado),
                       'corr_previsao_local_artigo': float(np.corrcoef(preco, guardado)[0, 1])},
             'modelos': {}}
    for nome, (_, _, arq) in MODELOS.items():
        v0, v2 = (concat(roda(nome, v, X_price, y, splits, mats[v])) for v in ('V0', 'V2'))
        art = np.concatenate([np.load(BACKUP / 'reports' / arq)[f'lstm_f{k}'] for k in range(1, 6)])
        saida['modelos'][nome] = {
            'artigo': resumo(art), 'V0_local': resumo(v0), 'V2': resumo(v2),
            'corr_previsao_V0_artigo': float(np.corrcoef(v0, art)[0, 1]),
            'corr_previsao_V0_V2': float(np.corrcoef(v0, v2)[0, 1]),
            'sinal_igual_V0_V2': float(((v0 > 0) == (v2 > 0)).mean()),
            'efeito_da_correcao_dMCC_V2_menos_V0': delta_mcc(v2, v0, truth),
            'efeito_da_correcao_dR2_V2_menos_V0': delta_r2(v2, v0, y[pos]),
            'dMCC_vs_preco_V0': delta_mcc(v0, preco, truth),
            'dMCC_vs_preco_V2': delta_mcc(v2, preco, truth)}
        (OUT / 'resumo.json').write_text(json.dumps(saida, indent=1, ensure_ascii=False), encoding='utf-8')
        m = saida['modelos'][nome]
        print(f"{nome:18s} MCC artigo {m['artigo']['mcc']:+.3f} | V0 {m['V0_local']['mcc']:+.3f} | V2 {m['V2']['mcc']:+.3f} | "
              f"R2 V0 {m['V0_local']['r2']:+.3f} V2 {m['V2']['r2']:+.3f} | "
              f"V2-V0 {m['efeito_da_correcao_dMCC_V2_menos_V0']['delta']:+.3f} "
              f"[{m['efeito_da_correcao_dMCC_V2_menos_V0']['lo']:+.3f}; {m['efeito_da_correcao_dMCC_V2_menos_V0']['hi']:+.3f}] | "
              f"dR2 {m['efeito_da_correcao_dR2_V2_menos_V0']['delta']:+.3f} "
              f"[{m['efeito_da_correcao_dR2_V2_menos_V0']['lo']:+.3f}; {m['efeito_da_correcao_dR2_V2_menos_V0']['hi']:+.3f}] | "
              f"vs preco V0 {m['dMCC_vs_preco_V0']['delta']:+.3f} [{m['dMCC_vs_preco_V0']['lo']:+.3f}; {m['dMCC_vs_preco_V0']['hi']:+.3f}] | "
              f"vs preco V2 {m['dMCC_vs_preco_V2']['delta']:+.3f} "
              f"[{m['dMCC_vs_preco_V2']['lo']:+.3f}; {m['dMCC_vs_preco_V2']['hi']:+.3f}]", flush=True)
    print('preco:', saida['preco'], flush=True)
    print('concluido', flush=True)


if __name__ == '__main__':
    main()
