"""IC amostral do MCC e do delta-MCC (revisao do professor, C7).

Bootstrap em blocos moveis sobre DIAS (nao sobre sementes): captura a incerteza
de amostragem, que o desvio entre sementes nao captura. Metrica: MCC agrupado dos
5 folds concatenados, com a previsao MEDIA entre sementes (a mesma base da DA
agrupada e do Pesaran-Timmermann): direcao prevista = (previsao > 0), alvo = (y > 0).
Random Walk: direcao = persistencia (retorno de hoje > 0).

Comparacoes (sempre nos dias em comum dos dois modelos):
  - variantes LSTM+texto/Copom vs. LSTM-preco   (mesmos dias)
  - ARIMA, ARIMAX, XGBoost, LSTM-preco vs. Random Walk (interseccao de dias)
Faixa de equivalencia: +-0,05 de MCC (margem sugerida no comentario).

Uso:
    python ic_mcc_bootstrap.py   ->  reports/ic_mcc_bootstrap.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / 'utils'))
from tcc_utils import walk_forward_splits

REPORTS = PROJECT / 'reports'
OUT = REPORTS / 'ic_mcc_bootstrap.json'
B = 10000
SEED = 0
BLOCO = 10          # ~ n^(1/3)
MARGEM = 0.05

LSTM_FILES = {
    'LSTM-preco': 'preds_lstm_preco.npz',
    'LSTM-BERTimbau-k8': 'preds_lstm_texto.npz',
    'LSTM-BERTimbau-k30': 'preds_lstm_texto_k30.npz',
    'LSTM-FinBERT-sent': 'preds_lstm_via2_sent.npz',
    'LSTM-FinBERT-emb-k8': 'preds_lstm_via1_finbert_k8.npz',
    'LSTM-FinBERT-emb-k30': 'preds_lstm_via1_finbert_k30.npz',
    'LSTM-Copom': 'preds_lstm_via3_copom.npz',
}


def carrega():
    fp = pd.read_parquet(PROJECT / 'data' / 'processed' / 'features_preco.parquet')
    y = fp['y'].values
    ret_hoje = fp['ret_lag0'].values
    sb = walk_forward_splits(fp.index, n_splits=5, min_train=252, expanding=True, embargo=0)
    sl = walk_forward_splits(fp.index, n_splits=5, min_train=252, expanding=True, embargo=21)
    pos_b = np.concatenate([np.asarray(te, dtype=int) for _, te in sb])
    pos_l = np.concatenate([np.asarray(te, dtype=int) for _, te in sl])

    def cat(d, prefix):
        return np.concatenate([d[f'{prefix}_f{k}'] for k in range(1, 6)])

    modelos = {}     # nome -> (posicoes, direcao prevista booleana)
    previsoes = {}   # nome -> (posicoes, previsao de retorno, escala de y)
    modelos['RandomWalk'] = (pos_b, ret_hoje[pos_b] > 0)
    previsoes['RandomWalk'] = (pos_b, np.zeros(len(pos_b)))
    d = np.load(REPORTS / 'preds_arima.npz')
    for nome, pref in (('ARIMA', 'arima'), ('ARIMAX', 'arimax')):
        p = cat(d, pref)
        modelos[nome], previsoes[nome] = (pos_b, p > 0), (pos_b, p)
    p = cat(np.load(REPORTS / 'preds_xgboost.npz'), 'reg')
    modelos['XGBoost'], previsoes['XGBoost'] = (pos_b, p > 0), (pos_b, p)
    for nome, arq in LSTM_FILES.items():
        p = cat(np.load(REPORTS / arq), 'lstm')
        modelos[nome], previsoes[nome] = (pos_l, p > 0), (pos_l, p)
    for nome, (pos, p) in modelos.items():
        assert len(pos) == len(p) == 1030, (nome, len(pos), len(p))
        assert len(previsoes[nome][1]) == 1030, nome
    return y, modelos, previsoes


def mcc_vetor(pred, truth):
    """MCC por linha: pred e truth com shape (..., n) booleanos."""
    tp = (pred & truth).sum(-1).astype(float)
    tn = (~pred & ~truth).sum(-1).astype(float)
    fp = (pred & ~truth).sum(-1).astype(float)
    fn = (~pred & truth).sum(-1).astype(float)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(invalid='ignore', divide='ignore'):
        m = (tp * tn - fp * fn) / den
    return np.where(den == 0, 0.0, m)


def indices_bloco(n, bloco, rng):
    nb = int(np.ceil(n / bloco))
    starts = rng.integers(0, n - bloco + 1, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(bloco)).reshape(B, -1)[:, :n]
    return idx


def comparacao(y, modelos, nome, ref, bloco=BLOCO):
    """delta-MCC = MCC(nome) - MCC(ref) nos dias em comum. ref=None -> MCC absoluto."""
    pos, pred = modelos[nome]
    if ref is None:
        comuns, ia = pos, np.arange(len(pos))
        truth = y[comuns] > 0
        a = pred[ia]
        pont = float(mcc_vetor(a, truth))
        rng = np.random.default_rng(SEED)
        idx = indices_bloco(len(a), bloco, rng)
        boot = mcc_vetor(a[idx], truth[idx])
    else:
        pos_r, pred_r = modelos[ref]
        comuns, ia, ir = np.intersect1d(pos, pos_r, return_indices=True)[0:3]
        truth = y[comuns] > 0
        a, r = pred[ia], pred_r[ir]
        pont = float(mcc_vetor(a, truth) - mcc_vetor(r, truth))
        rng = np.random.default_rng(SEED)
        idx = indices_bloco(len(a), bloco, rng)
        boot = mcc_vetor(a[idx], truth[idx]) - mcc_vetor(r[idx], truth[idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {'modelo': nome, 'referencia': ref, 'n_dias': int(len(comuns)),
            'mcc_ou_delta': pont, 'ep_bootstrap': float(boot.std()),
            'ic95_lo': float(lo), 'ic95_hi': float(hi),
            'ic_dentro_da_faixa': bool(lo >= -MARGEM and hi <= MARGEM),
            'ic_contem_zero': bool(lo <= 0 <= hi),
            'efeito_maximo_compativel': float(max(abs(lo), abs(hi)))}


def r2_fora_da_amostra(y, previsoes, nome, bloco=BLOCO):
    """R2 = 1 - EQM(modelo) / EQM(previsao zero = passeio aleatorio), nos dias do proprio
    modelo; IC por block bootstrap sobre dias. Negativo = pior que prever zero."""
    pos, p = previsoes[nome]
    yy = y[pos]
    num, den = (yy - p) ** 2, yy ** 2
    pont = float(1 - num.sum() / den.sum())
    rng = np.random.default_rng(SEED)
    idx = indices_bloco(len(yy), bloco, rng)
    boot = 1 - num[idx].sum(1) / den[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {'modelo': nome, 'n_dias': int(len(yy)), 'r2': pont,
            'ep_bootstrap': float(boot.std()), 'ic95_lo': float(lo), 'ic95_hi': float(hi),
            'rmse': float(np.sqrt(num.mean())), 'rmse_rw': float(np.sqrt(den.mean())),
            'dp_previsao': float(p.std())}


CORPO = Path(r'D:\tcc_ibov_corpo_completo\reports')
COLUNAS = ['BERTimbau k=8', 'BERTimbau k=30', 'FinBERT sent.', 'FinBERT emb. k=8',
           'FinBERT emb. k=30']


def delta_pareado(y, pos, a, r, bloco=BLOCO):
    """delta-MCC (a - r) nos mesmos dias, com IC por block bootstrap (mesmos indices)."""
    truth = y[pos] > 0
    pont = float(mcc_vetor(a, truth) - mcc_vetor(r, truth))
    idx = indices_bloco(len(a), bloco, np.random.default_rng(SEED))
    boot = mcc_vetor(a[idx], truth[idx]) - mcc_vetor(r[idx], truth[idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {'delta': pont, 'ep': float(boot.std()), 'lo': float(lo), 'hi': float(hi),
            'n_dias': int(len(a))}


def robustez(y):
    """Cadeia de robustez: delta-MCC de cada variante contra o baseline SEM texto da
    mesma configuracao (LSTM-preco; XGBoost-preco para as variantes com XGBoost)."""
    fp = pd.read_parquet(PROJECT / 'data' / 'processed' / 'features_preco.parquet')

    def posicoes(emb, ini=0):
        """Posicoes de teste (no indice ORIGINAL) quando os folds sao calculados sobre
        fp.index[ini:] -- a variante D-1 descarta o 1o dia e desloca os folds."""
        sp = walk_forward_splits(fp.index[ini:], n_splits=5, min_train=252, expanding=True,
                                 embargo=emb)
        return np.concatenate([np.asarray(te, dtype=int) for _, te in sp]) + ini

    def pred(caminho, pref='lstm'):
        d = np.load(caminho)
        return np.concatenate([d[f'{pref}_f{k}'] for k in range(1, 6)]) > 0

    P = REPORTS
    pos21, pos0 = posicoes(21), posicoes(0)
    preco21, xgb = pred(P / 'preds_lstm_preco.npz'), pred(P / 'preds_xgboost.npz', 'reg')
    texto = ['preds_lstm_texto.npz', 'preds_lstm_texto_k30.npz', 'preds_lstm_via2_sent.npz',
             'preds_lstm_via1_finbert_k8.npz', 'preds_lstm_via1_finbert_k30.npz']
    xgbt = ['preds_xgboost_bertimbau_k8.npz', 'preds_xgboost_bertimbau_k30.npz',
            'preds_xgboost_finbert_sent.npz', 'preds_xgboost_finbert_k8.npz',
            'preds_xgboost_finbert_k30.npz']
    linhas = []

    def add(check, coluna, a, r, pos, pos_r=None):
        if pos_r is None or np.array_equal(pos, pos_r):
            res = delta_pareado(y, pos, a, r)
        else:   # dias de teste diferentes: compara so na interseccao
            comuns, ia, ir = np.intersect1d(pos, pos_r, return_indices=True)
            res = delta_pareado(y, comuns, a[ia], r[ir])
        linhas.append({'verificacao': check, 'coluna': coluna, **res})

    for c, f in zip(COLUNAS, texto):
        add('principal', c, pred(P / f), preco21, pos21)
        add('corpo_truncado', c, pred(CORPO / f), preco21, pos21)
    for c, f in (('BERTimbau k=8', 'preds_lstm_bertimbau_k8_corpo_chunked.npz'),
                 ('FinBERT sent.', 'preds_lstm_finbert_sent_corpo_chunked.npz'),
                 ('FinBERT emb. k=8', 'preds_lstm_finbert_emb_k8_corpo_chunked.npz')):
        add('corpo_chunking', c, pred(CORPO / f), preco21, pos21)
    for c, f in zip(COLUNAS, xgbt):
        add('xgboost_texto', c, pred(P / f, 'reg'), xgb, pos0)
    add('texto_D-1', 'BERTimbau k=8', pred(P / 'preds_lstm_bertimbau_k8_lagD1.npz'), preco21,
        posicoes(21, ini=1), pos21)
    add('ultima_manchete', 'BERTimbau k=8', pred(P / 'preds_lstm_bertimbau_k8_ultima.npz'),
        preco21, pos21)
    for lb in (1, 3, 5, 10):
        add(f'lookback_{lb}', 'BERTimbau k=8', pred(P / f'preds_lstm_texto_k8_lb{lb}.npz'),
            pred(P / f'preds_lstm_preco_lb{lb}.npz'), posicoes(lb))
    add('bow', 'BERTimbau k=8', pred(P / 'preds_lstm_bow_k8.npz'), preco21, pos21)
    add('bow', 'BERTimbau k=30', pred(P / 'preds_lstm_bow_k30.npz'), preco21, pos21)
    add('mlp_sem_texto', 'MLP vs. LSTM preço', pred(P / 'preds_mlp_preco.npz'), preco21, pos21)
    return linhas


def main():
    y, modelos, previsoes = carrega()
    saida = {'B': B, 'seed': SEED, 'bloco': BLOCO, 'margem': MARGEM,
             'r2_vs_rw': [], 'absoluto': [], 'vs_lstm_preco': [], 'vs_random_walk': [], 'vs_passeio': [],
             'robustez': robustez(y), 'sensibilidade_bloco': {}}
    for nome in modelos:
        saida['r2_vs_rw'].append(r2_fora_da_amostra(y, previsoes, nome))
    for nome in modelos:
        saida['absoluto'].append(comparacao(y, modelos, nome, None))
    for nome in LSTM_FILES:
        if nome != 'LSTM-preco':
            saida['vs_lstm_preco'].append(comparacao(y, modelos, nome, 'LSTM-preco'))
    for nome in ['ARIMA', 'ARIMAX', 'XGBoost', 'LSTM-preco']:
        saida['vs_random_walk'].append(comparacao(y, modelos, nome, 'RandomWalk'))
    # persistencia (sinal do retorno de hoje) nos dias de teste de cada modelo, nao so na interseccao
    ret_hoje = pd.read_parquet(PROJECT / 'data' / 'processed' / 'features_preco.parquet')['ret_lag0'].values
    for nome, (pos, pred) in modelos.items():
        if nome != 'RandomWalk':
            par = {nome: (pos, pred), 'persistencia': (pos, ret_hoje[pos] > 0)}
            saida['vs_passeio'].append(comparacao(y, par, nome, 'persistencia'))
    for L in (5, 10, 20):
        saida['sensibilidade_bloco'][str(L)] = [
            {k: round(v, 4) if isinstance(v, float) else v for k, v in
             comparacao(y, modelos, n, 'LSTM-preco', bloco=L).items()
             if k in ('modelo', 'ep_bootstrap', 'ic95_lo', 'ic95_hi')}
            for n in LSTM_FILES if n != 'LSTM-preco']
    OUT.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding='utf-8')

    def linha(r):
        return (f"{r['modelo']:<22} ref={str(r['referencia']):<11} n={r['n_dias']:<5} "
                f"{r['mcc_ou_delta']:+.3f}  EP={r['ep_bootstrap']:.3f}  "
                f"IC95[{r['ic95_lo']:+.3f}, {r['ic95_hi']:+.3f}]  "
                f"contem0={r['ic_contem_zero']}  dentro_faixa={r['ic_dentro_da_faixa']}")
    for chave, titulo in (('absoluto', 'MCC agrupado (vs. acaso = 0)'),
                          ('vs_lstm_preco', 'delta-MCC vs. LSTM-preco'),
                          ('vs_random_walk', 'delta-MCC vs. Random Walk'),
                          ('vs_passeio', 'delta-MCC vs. persistencia, dias de cada modelo')):
        print(f'\n=== {titulo} ===')
        for r in saida[chave]:
            print(linha(r))
    print('\n=== R2 fora da amostra vs. previsao zero (passeio aleatorio) ===')
    for r in saida['r2_vs_rw']:
        print(f"{r['modelo']:<22} R2={r['r2']:+.3f}  EP={r['ep_bootstrap']:.3f}  "
              f"IC95[{r['ic95_lo']:+.3f}, {r['ic95_hi']:+.3f}]  "
              f"RMSE={r['rmse']:.5f} (RW={r['rmse_rw']:.5f})  dp_prev={100 * r['dp_previsao']:.3f}%")
    print('\n=== cadeia de robustez: delta-MCC vs. baseline sem texto (mesma configuracao) ===')
    exclui = 0
    for r in saida['robustez']:
        z = r['lo'] > 0 or r['hi'] < 0
        exclui += int(z)
        print(f"{r['verificacao']:<16} {r['coluna']:<20} n={r['n_dias']:<5} {r['delta']:+.3f} "
              f"[{r['lo']:+.3f}, {r['hi']:+.3f}] EP={r['ep']:.3f}{'  <-- exclui zero' if z else ''}")
    print(f"IC excluindo zero: {exclui} de {len(saida['robustez'])}")
    print('\n=== sensibilidade ao comprimento do bloco (EP do delta vs. LSTM-preco) ===')
    for L, rows in saida['sensibilidade_bloco'].items():
        print(f'L={L}:', [f"{r['modelo'].replace('LSTM-', '')}:{r['ep_bootstrap']:.3f}" for r in rows])


if __name__ == '__main__':
    main()
