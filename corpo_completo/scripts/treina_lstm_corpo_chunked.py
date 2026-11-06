"""Retreina as 3 variantes de LSTM+texto (BERTimbau k=8, FinBERT emb k=8,
FinBERT sentimento) sobre as features do corpo completo COM CHUNKING
(cobrindo o artigo inteiro), sob o MESMO protocolo do notebook 05_lstm.ipynb
(walk-forward, embargo=21, PCA/scaler por fold, calibracao isotonica, 5
sementes) -- para comparar contra a versao truncada em 512 tokens (Secao
V-F / robustez: corpo completo) e contra a versao com manchetes.

ANTI-VAZAMENTO: identico ao pipeline principal -- PCA e scaler ajustados
exclusivamente no treino de cada fold.

Uso:
    python treina_lstm_corpo_chunked.py
"""
import random
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, f1_score, matthews_corrcoef, brier_score_loss

PROJECT = Path(__file__).resolve().parent.parent
DIRS = {k: PROJECT / v for k, v in {
    'processed': 'data/processed', 'reports': 'reports', 'utils': 'utils',
}.items()}
sys.path.insert(0, str(DIRS['utils']))
from tcc_utils import walk_forward_splits  # noqa: E402

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HIDDEN, DROPOUT, EPOCHS, LR, WD, BATCH = 16, 0.4, 75, 1e-3, 1e-4, 32
SEEDS = [0, 1, 2, 3, 4]
LOOKBACK = 21
K_PCA = 8


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
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


def rmse(a, b): return np.sqrt(np.mean((a - b) ** 2))
def da(a, b):   return (np.sign(a) == np.sign(b)).mean()


def treina_avalia(X_tr, y_tr, X_te, seed):
    set_seed(seed)
    model = LSTMRegressor(X_tr.shape[-1], HIDDEN, DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.MSELoss()
    Xtr = torch.tensor(X_tr, device=DEVICE); ytr = torch.tensor(y_tr, device=DEVICE)
    n = len(Xtr)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(Xtr[b]), ytr[b])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_te, device=DEVICE)).cpu().numpy()
    return pred


def roda_variante(X_price, X_extra_dense, X_emb, k_pca, y_raw, index, tag):
    """X_extra_dense: features densas sempre incluidas (sig ou sentimento),
    sem PCA. X_emb: embeddings brutos (768d) para reduzir via PCA por fold,
    ou None se a variante nao usa embeddings (caso do sentimento)."""
    splits = walk_forward_splits(index, n_splits=5, min_train=252, expanding=True, embargo=LOOKBACK)
    registros, preds_media, var_exp = [], {}, []
    for kf, (tr_days, te_days) in enumerate(splits, 1):
        t0 = time.time()
        tr_set, te_set = set(tr_days), set(te_days)

        if X_emb is not None:
            tr_day_mask = np.array([i in tr_set for i in range(len(X_emb))])
            pca = PCA(n_components=k_pca, random_state=0).fit(X_emb[tr_day_mask])
            var_exp.append(pca.explained_variance_ratio_.sum())
            X_emb_pca = pca.transform(X_emb).astype(np.float32)
            X_day = np.concatenate([X_price, X_extra_dense, X_emb_pca], axis=1)
        else:
            X_day = np.concatenate([X_price, X_extra_dense], axis=1)

        seqs, targets, sidx = [], [], []
        for i in range(LOOKBACK - 1, len(X_day)):
            seqs.append(X_day[i - LOOKBACK + 1:i + 1]); targets.append(y_raw[i]); sidx.append(i)
        seqs, targets, sidx = np.array(seqs), np.array(targets), np.array(sidx)

        mask_tr = np.array([i in tr_set for i in sidx])
        mask_te = np.array([i in te_set for i in sidx])
        X_tr, y_tr = seqs[mask_tr], targets[mask_tr]
        X_te, y_te = seqs[mask_te], targets[mask_te]

        n_f = X_tr.shape[-1]
        sc = StandardScaler().fit(X_tr.reshape(-1, n_f))
        X_tr = sc.transform(X_tr.reshape(-1, n_f)).reshape(X_tr.shape).astype(np.float32)
        X_te = sc.transform(X_te.reshape(-1, n_f)).reshape(X_te.shape).astype(np.float32)

        n_cal = max(int(len(X_tr) * 0.2), 1)
        preds_seeds = []
        for s in SEEDS:
            pred = treina_avalia(X_tr, y_tr, X_te, s)
            preds_seeds.append(pred)
            pred_tr = treina_avalia(X_tr[:-n_cal], y_tr[:-n_cal], X_tr[-n_cal:], s)
            iso = IsotonicRegression(out_of_bounds='clip').fit(pred_tr, (y_tr[-n_cal:] > 0).astype(int))
            p_prob = iso.predict(pred)
            y_bin = (y_te > 0).astype(int); pred_dir = (p_prob > 0.5).astype(int)
            registros.append({
                'fold': kf, 'seed': s,
                'MAE': mean_absolute_error(y_te, pred), 'RMSE': rmse(y_te, pred),
                'DA': da(y_te, pred),
                'F1': f1_score(y_bin, pred_dir, zero_division=0),
                'MCC': matthews_corrcoef(y_bin, pred_dir),
                'Brier': brier_score_loss(y_bin, p_prob),
            })
        preds_media[kf] = np.mean(preds_seeds, axis=0)
        var_str = f" | PCA var {var_exp[-1]:.1%}" if X_emb is not None else ""
        print(f"  [{tag}] fold {kf}: {len(X_tr)} tr seqs{var_str} | {time.time()-t0:.0f}s")

    df = pd.DataFrame(registros)
    np.savez(DIRS['reports'] / f'preds_{tag}.npz', **{f'lstm_f{k}': v for k, v in preds_media.items()})
    df.to_parquet(DIRS['reports'] / f'{tag}_full.parquet')
    return df


metric_cols = ['MAE', 'RMSE', 'DA', 'F1', 'MCC', 'Brier']


def resumo_desvio(df):
    r = df.groupby('fold')[metric_cols].mean().mean()
    d = ((df.groupby('fold')[metric_cols].std() ** 2).mean()) ** 0.5
    return r, d


def main():
    print(f"Device: {DEVICE}")
    feat_preco = pd.read_parquet(DIRS['processed'] / 'features_preco.parquet')
    PRICE_COLS = [c for c in feat_preco.columns if c != 'y']
    X_price = feat_preco[PRICE_COLS].values.astype(np.float32)
    y_raw = feat_preco['y'].values.astype(np.float32)
    EMB_COLS = [f'emb_{i:03d}' for i in range(768)]
    SIG_COLS = ['n_noticias', 'emb_dispersao']
    SENT_COLS = ['sent_polaridade', 'sent_dispersao', 'sent_pos_frac', 'sent_neg_frac']

    resultados = {}

    print("\n=== LSTM + BERTimbau k=8 (corpo, chunked) ===")
    feat_bt = pd.read_parquet(DIRS['processed'] / 'features_texto_bertimbau_corpo_chunked.parquet').reindex(feat_preco.index)
    assert feat_bt['emb_000'].notna().all(), "BERTimbau chunked com NaN apos alinhar"
    reg_bt = roda_variante(X_price, feat_bt[SIG_COLS].values.astype(np.float32),
                            feat_bt[EMB_COLS].values.astype(np.float32), K_PCA, y_raw,
                            feat_preco.index, 'lstm_bertimbau_k8_corpo_chunked')
    resultados['BERTimbau k=8 (chunked)'] = resumo_desvio(reg_bt)

    print("\n=== LSTM + FinBERT emb k=8 (corpo, chunked) ===")
    feat_fb = pd.read_parquet(DIRS['processed'] / 'features_texto_finbert_corpo_chunked.parquet').reindex(feat_preco.index)
    assert feat_fb['emb_000'].notna().all(), "FinBERT emb chunked com NaN apos alinhar"
    reg_fb = roda_variante(X_price, feat_fb[SIG_COLS].values.astype(np.float32),
                            feat_fb[EMB_COLS].values.astype(np.float32), K_PCA, y_raw,
                            feat_preco.index, 'lstm_finbert_emb_k8_corpo_chunked')
    resultados['FinBERT emb k=8 (chunked)'] = resumo_desvio(reg_fb)

    print("\n=== LSTM + FinBERT sentimento (corpo, chunked) ===")
    feat_sent = pd.read_parquet(DIRS['processed'] / 'features_sentimento_finbert_corpo_chunked.parquet').reindex(feat_preco.index)
    assert feat_sent[SENT_COLS].notna().all().all(), "sentimento chunked com NaN apos alinhar"
    reg_sent = roda_variante(X_price, feat_sent[SENT_COLS].values.astype(np.float32),
                              None, None, y_raw, feat_preco.index, 'lstm_finbert_sent_corpo_chunked')
    resultados['FinBERT sentimento (chunked)'] = resumo_desvio(reg_sent)

    print("\n=== Resumo -- MCC (chunked) vs. referencias ja calculadas ===")
    n4 = pd.read_parquet(DIRS['reports'] / 'lstm_preco_full.parquet')
    r_n4, _ = resumo_desvio(n4)
    print(f"  so-preco:                        {r_n4['MCC']:+.4f}")
    for nome, (r, d) in resultados.items():
        print(f"  {nome:<32s}: {r['MCC']:+.4f} +/- {d['MCC']:.4f}")

    # referencias ja existentes (truncado em 512 tokens / manchete), se disponiveis
    for label, path in [
        ('BERTimbau k=8 (corpo truncado)', DIRS['reports'] / 'lstm_texto_full.parquet'),
        ('FinBERT emb k=8 (corpo truncado)', DIRS['reports'] / 'lstm_via1_finbert_k8_full.parquet'),
        ('FinBERT sent. (corpo truncado)', DIRS['reports'] / 'lstm_via2_sent_full.parquet'),
    ]:
        if path.exists():
            r, d = resumo_desvio(pd.read_parquet(path))
            print(f"  {label:<32s}: {r['MCC']:+.4f} +/- {d['MCC']:.4f}")

    print("\nConcluido.")


if __name__ == '__main__':
    main()
