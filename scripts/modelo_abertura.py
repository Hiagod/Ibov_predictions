"""O desenho preditivo legitimo para informacao fora do pregao (revisao do professor, C13): prever o
retorno abertura -> fechamento do BOVA11 com o que se sabe antes da abertura.

Linha t:
    alvo     : oc_t = ln(Close_t / Open_t) do BOVA11
    preco    : as 18 variaveis da linha t-1 (dados ate o fechamento de t-1) + gap_t
    texto    : manchetes publicadas depois do fechamento de t-1 e antes das 10h de t (carimbo corrigido,
               corte no fechamento efetivo), mesmas agregacoes do artigo

Mesma LSTM, sementes, folds, embargo, PCA e scaler do artigo. Compara cada variante de texto com o
modelo so com preco nesse mesmo desenho (delta-MCC com IC por block bootstrap sobre dias) e reporta
R2 frente a prever zero e Sharpe da estrategia comprada/vendida na abertura e zerada no fechamento
(sem custo e com 10 bps por dia: entrada e saida a 5 bps cada).

Uso:
    python modelo_abertura.py   ->  reports/modelo_abertura/resumo.json
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from diagnostico_janela_abertura import bandas_v2, retornos_bova
from diagnostico_relato_evento import holm
from ic_mcc_bootstrap import B, BLOCO, SEED, indices_bloco, mcc_vetor
from sensibilidade_carimbo import (LOOKBACK, SEEDS, agrega_emb, agrega_sent, carrega_embeddings, delta_mcc,
                                   treina_avalia)
from tcc_utils import walk_forward_splits

PROJECT = Path(__file__).resolve().parent.parent
PROC = PROJECT / 'data' / 'processed'
OUT = PROJECT / 'reports' / 'modelo_abertura'
MODELOS = {'so preco': (None, None), 'FinBERT sent.': ('sentimento', None), 'BERTimbau k=8': ('bertimbau', 8),
           'BERTimbau k=30': ('bertimbau', 30), 'FinBERT emb. k=8': ('finbert', 8), 'FinBERT emb. k=30': ('finbert', 30)}


def roda(nome, X_base, y, validos, splits, mats):
    fonte, k = MODELOS[nome]
    arq = OUT / f"preds_{nome.replace(' ', '_').replace('=', '').replace('.', '')}.npz"
    if arq.exists():
        print(f'{nome}: ja existe, pulando', flush=True)
        return np.load(arq)['pred']
    t0 = time.time()
    out = []
    for tr_days, te_days in splits:
        tr_set, te_set = set(tr_days), set(te_days)
        if fonte in ('bertimbau', 'finbert'):
            emb, sig = mats[fonte]
            mask = np.array([i in tr_set for i in range(len(emb))])
            pca = PCA(n_components=k, random_state=0).fit(emb[mask])
            X_day = np.concatenate([X_base, sig, pca.transform(emb).astype(np.float32)], axis=1)
        elif fonte == 'sentimento':
            X_day = np.concatenate([X_base, mats['sentimento']], axis=1)
        else:
            X_day = X_base
        # a linha 0 nao tem t-1: as sequencias comecam depois dela
        sidx = np.arange(LOOKBACK, len(X_day))
        seqs = np.stack([X_day[i - LOOKBACK + 1:i + 1] for i in sidx])
        m_tr = np.array([(i in tr_set) and validos[i] for i in sidx])
        m_te = np.array([i in te_set for i in sidx])
        n_f = seqs.shape[-1]
        sc = StandardScaler().fit(seqs[m_tr].reshape(-1, n_f))
        X_tr = sc.transform(seqs[m_tr].reshape(-1, n_f)).reshape(seqs[m_tr].shape).astype(np.float32)
        X_te = sc.transform(seqs[m_te].reshape(-1, n_f)).reshape(seqs[m_te].shape).astype(np.float32)
        out.append(np.mean([treina_avalia(X_tr, y[sidx][m_tr], X_te, s) for s in SEEDS], axis=0))
    pred = np.concatenate(out)
    np.savez(arq, pred=pred)
    print(f'{nome}: {time.time() - t0:.0f}s', flush=True)
    return pred


def main():
    OUT.mkdir(exist_ok=True)
    fp = pd.read_parquet(PROC / 'features_preco.parquet')
    precos = pd.read_parquet(PROC / 'bvsp_clean.parquet')
    c = pd.read_parquet(PROC / 'corpus_bucketed.parquet').reset_index(drop=True)
    probs = np.load(PROC / 'finbert_probs_raw.npy')
    bt, fb = carrega_embeddings(c)

    r = retornos_bova(precos.index, fp.index)
    oc, gap = r['oc'].values.astype(np.float32), r['gap'].values.astype(np.float32)
    validos = ~np.isnan(oc)
    y = np.nan_to_num(oc)                      # dias sem BOVA11 nao entram em treino nem em avaliacao
    preco = fp[[k for k in fp.columns if k != 'y']].values.astype(np.float32)
    X_base = np.column_stack([np.vstack([preco[0], preco[:-1]]), np.nan_to_num(gap)]).astype(np.float32)

    dia, banda = bandas_v2(c)
    pre = np.isin(banda, ['noite', 'manha']) & dia.isin(fp.index).values
    ep_b, sg_b = agrega_emb(bt[pre], dia.values[pre], fp.index)
    ep_f, sg_f = agrega_emb(fb[pre], dia.values[pre], fp.index)
    mats = {'bertimbau': (ep_b, sg_b), 'finbert': (ep_f, sg_f), 'sentimento': agrega_sent(probs[pre], dia.values[pre], fp.index)}
    print(f'manchetes antes da abertura: {pre.sum()} de {len(pre)}', flush=True)

    splits = walk_forward_splits(fp.index, n_splits=5, min_train=252, expanding=True, embargo=LOOKBACK)
    pos = np.concatenate([np.asarray(te, dtype=int) for _, te in splits])
    aval = validos[pos]
    yt, truth = y[pos][aval], y[pos][aval] > 0
    print(f'dias de teste: {len(pos)}; com BOVA11: {aval.sum()}', flush=True)

    def sharpe(s):
        return float(s.mean() / s.std() * np.sqrt(252))

    def ic_mcc(p):
        idx = indices_bloco(len(p), BLOCO, np.random.default_rng(SEED))
        b = mcc_vetor((p > 0)[idx], truth[idx])
        lo, hi = np.percentile(b, [2.5, 97.5])
        return float(lo), float(hi), float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())))

    preds = {nome: roda(nome, X_base, y, validos, splits, mats)[aval] for nome in MODELOS}
    saida = {'n_teste': int(aval.sum()), 'compra_na_abertura_sharpe': sharpe(yt), 'sempre_vendido_sharpe': sharpe(-yt),
             'fracao_oc_positivo': float(truth.mean()), 'modelos': {}}
    extremos = pd.DataFrame({'gap': gap, 'oc': oc}, index=fp.index)
    opostos = extremos[(extremos['gap'].abs() > 0.03) & (extremos['oc'].abs() > 0.03) & (extremos['gap'] * extremos['oc'] < 0)]
    print('dias com |gap| e |oc| > 3% de sinais opostos (erro de abertura?):\n', opostos.round(4).to_string(), flush=True)
    for nome, p in preds.items():
        pos_t = np.sign(p)
        lo, hi, p_mcc = ic_mcc(p)
        m = {'da': float(((p > 0) == truth).mean()), 'mcc': float(mcc_vetor(p > 0, truth)), 'mcc_ic95': [lo, hi], 'p_mcc': p_mcc,
             'fracao_previsao_positiva': float((p > 0).mean()),
             'r2': float(1 - ((yt - p) ** 2).sum() / (yt ** 2).sum()),
             'sharpe_0bps': sharpe(pos_t * yt), 'sharpe_10bps_dia': sharpe(pos_t * yt - 0.001)}
        if nome != 'so preco':
            m['dMCC_vs_so_preco'] = delta_mcc(p, preds['so preco'], truth)
        saida['modelos'][nome] = m
        d = m.get('dMCC_vs_so_preco')
        print(f"{nome:18s} DA {m['da']:.3f} MCC {m['mcc']:+.3f} [{lo:+.3f}; {hi:+.3f}] R2 {m['r2']:+.3f} "
              f"Sharpe {m['sharpe_0bps']:+.2f} / {m['sharpe_10bps_dia']:+.2f}"
              + (f" | dMCC {d['delta']:+.3f} [{d['lo']:+.3f}; {d['hi']:+.3f}]" if d else ''), flush=True)
    nomes = list(saida['modelos'])
    for n, x in zip(nomes, holm([saida['modelos'][n]['p_mcc'] for n in nomes])):
        saida['modelos'][n]['p_mcc_holm'] = float(x)
    com_texto = [n for n in nomes if n != 'so preco']
    for n, x in zip(com_texto, holm([saida['modelos'][n]['dMCC_vs_so_preco']['p'] for n in com_texto])):
        saida['modelos'][n]['dMCC_vs_so_preco']['p_holm'] = float(x)
    (OUT / 'resumo.json').write_text(json.dumps(saida, indent=1, ensure_ascii=False), encoding='utf-8')
    for n in nomes:
        m = saida['modelos'][n]
        print(f"{n:18s} prev. positiva {m['fracao_previsao_positiva']:.2f} | MCC p {m['p_mcc']:.3f} Holm {m['p_mcc_holm']:.3f}"
              + (f" | dMCC p {m['dMCC_vs_so_preco']['p']:.3f} Holm {m['dMCC_vs_so_preco']['p_holm']:.3f}" if n != 'so preco' else ''), flush=True)
    print('oc positivo em', round(saida['fracao_oc_positivo'], 3), 'dos dias de teste', flush=True)
    print('compra na abertura: Sharpe', round(saida['compra_na_abertura_sharpe'], 2),
          '| sempre vendido:', round(saida['sempre_vendido_sharpe'], 2), flush=True)
    print('concluido', flush=True)


if __name__ == '__main__':
    main()
