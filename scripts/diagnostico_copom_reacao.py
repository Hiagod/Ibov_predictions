"""Estudo de evento do Copom (revisao do professor, C11 e C13): o comunicado sai as 18h30 de D e
entra na linha D+1. A reacao recai no retorno D -> D+1, o contemporaneo dessa linha; o alvo da
linha e o retorno D+1 -> D+2. Mede o sinal do comunicado nas duas janelas:

    x: escore original; escore sem as frases do anuncio; variacao de cada um frente ao comunicado
       anterior (delta de linguagem); direcao da decisao; variacao da Selic; surpresa frente ao Focus
       (data/raw/focus_selic.csv, ultima expectativa antes da decisao).
    janelas: gap do BOVA11 na manha seguinte (a reacao propriamente dita); abertura -> fechamento do
       BOVA11; reacao = ret_lag0 da linha (fechamento a fechamento do indice); alvo = y da linha.

IC de 95% por bootstrap de eventos (B = 20 000, semente 0). Requer o modelo em cache:
    HF_HUB_OFFLINE=1 python diagnostico_copom_reacao.py  ->  reports/diagnostico_copom_reacao.json
"""
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from transformers import pipeline

from diagnostico_copom_validade import ANUNCIO, ARQ, LABELS, MODELO, corr_ic, sem_frases
from diagnostico_janela_abertura import retornos_bova
from diagnostico_relato_evento import holm

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / 'utils'))
from tcc_utils import assign_trading_day, build_cutoffs

PROC = PROJECT / 'data' / 'processed'
OUT = PROJECT / 'reports' / 'diagnostico_copom_reacao.json'


def p_permutacao(x, y, n=20000):
    """p bilateral por permutacao de eventos."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    obs = abs(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(0)
    xc = (x - x.mean()) / x.std()
    yc = (y - y.mean()) / y.std()
    perm = rng.permuted(np.tile(xc, (n, 1)), axis=1)
    rs = (perm * yc).mean(axis=1)
    return float(((np.abs(rs) >= obs - 1e-12).sum() + 1) / (n + 1))


def main():
    c = pd.read_parquet(ARQ).sort_values('data_reuniao').reset_index(drop=True)
    fp = pd.read_parquet(PROC / 'features_preco.parquet')
    precos = pd.read_parquet(PROC / 'bvsp_clean.parquet')

    clf = pipeline('zero-shot-classification', model=MODELO, device=0 if torch.cuda.is_available() else -1)

    def score(texto):
        out = clf(texto, candidate_labels=LABELS, multi_label=False)
        probs = dict(zip(out['labels'], out['scores']))
        return probs[LABELS[0]] - probs[LABELS[1]]

    c['sem_anuncio'] = [score(sem_frases(t, ANUNCIO)[0]) for t in c['texto']]
    c['d_score'] = c['copom_hawkish_score'].diff()
    c['d_sem_anuncio'] = c['sem_anuncio'].diff()
    c['d_selic'] = c['selic_pct'].diff()

    tz = ZoneInfo('America/Sao_Paulo')
    ts = pd.DatetimeIndex([d.replace(hour=18, minute=30, tzinfo=tz) for d in c['data_reuniao']]).tz_convert('UTC')
    c['linha'] = assign_trading_day(pd.Series(ts), build_cutoffs(pd.DatetimeIndex(precos.index))).values
    e = c[c['linha'].isin(fp.index) & c['d_score'].notna()].copy()
    e['reacao'] = fp['ret_lag0'].reindex(e['linha']).values
    e['alvo'] = fp['y'].reindex(e['linha']).values
    print(f'eventos: {len(e)} ({e["data_reuniao"].min().date()} a {e["data_reuniao"].max().date()})')

    # aberturas reais: a reacao a um comunicado das 18h30 ocorre no gap do pregao seguinte
    bova = retornos_bova(precos.index, fp.index)
    e['gap'] = bova['gap'].reindex(e['linha']).values
    e['oc'] = bova['oc'].reindex(e['linha']).values

    # surpresa: decisao realizada menos a expectativa Focus da vespera. Base de 30 dias (baseCalculo 0)
    # em todas as reunioes; a de 5 dias (baseCalculo 1), mais fresca, so existe desde 2021.
    f = pd.read_csv(PROJECT / 'data' / 'raw' / 'focus_selic.csv', parse_dates=['Data'])
    ult = f.sort_values('Data').groupby(['nro_reuniao', 'baseCalculo']).tail(1).set_index('nro_reuniao')
    f0, f1 = ult[ult['baseCalculo'] == 0], ult[ult['baseCalculo'] == 1]
    assert e['nro_reuniao'].isin(f0.index).all(), 'reuniao sem expectativa Focus'
    e['surpresa_mediana'] = e['selic_pct'].values - f0['Mediana'].reindex(e['nro_reuniao']).values
    e['surpresa_media'] = e['selic_pct'].values - f0['Media'].reindex(e['nro_reuniao']).values
    e['surpresa_mediana_5d'] = e['selic_pct'].values - f1['Mediana'].reindex(e['nro_reuniao']).values
    e['surpresa_media_5d'] = e['selic_pct'].values - f1['Media'].reindex(e['nro_reuniao']).values
    print(f"surpresa pela mediana: {(e['surpresa_mediana'].abs() > 1e-9).sum()} de {len(e)} reunioes diferentes de zero")
    print(e.loc[e['surpresa_mediana'].abs() > 1e-9, ['data_reuniao', 'selic_pct', 'surpresa_mediana', 'surpresa_media',
                                                     'gap', 'oc']].round(4).to_string())

    nomes = {'copom_hawkish_score': 'escore (nivel)', 'sem_anuncio': 'escore sem anuncio (nivel)',
             'd_score': 'delta do escore', 'd_sem_anuncio': 'delta do escore sem anuncio',
             'direcao': 'direcao da decisao', 'd_selic': 'variacao da Selic',
             'surpresa_mediana': 'surpresa (mediana Focus)', 'surpresa_media': 'surpresa (media Focus)',
             'surpresa_mediana_5d': 'surpresa (mediana, 5 dias)', 'surpresa_media_5d': 'surpresa (media, 5 dias)'}
    janelas = {'gap': 'gap', 'oc': 'oc', 'reacao': 'reacao', 'alvo': 'alvo'}
    tabela = {}
    for k, rot in nomes.items():
        tabela[rot] = {}
        for j in janelas:
            ok = e[[k, j]].notna().all(axis=1)
            tabela[rot][j] = corr_ic(e.loc[ok, k].values, e.loc[ok, j].values)
        print(f'{rot:30s} ' + ' | '.join(
            f"{j} {tabela[rot][j]['r']:+.3f} [{tabela[rot][j]['lo']:+.3f}; {tabela[rot][j]['hi']:+.3f}]" for j in janelas))

    # 10 sinais x 4 janelas, com p por permutacao e correcao de Holm. O bootstrap degenera com regressor
    # esparso: a surpresa pela mediana e zero em quase toda reuniao, e se as poucas nao nulas tem o mesmo
    # sinal, nenhuma reamostra inverte a correlacao (p = 0 espurio). A permutacao nao sofre disso.
    familia = []
    for rot, k in ((v, k) for k, v in nomes.items()):
        for j in janelas:
            ok = e[[k, j]].notna().all(axis=1)
            tabela[rot][j]['p_perm'] = p_permutacao(e.loc[ok, k].values, e.loc[ok, j].values)
            tabela[rot][j]['nao_nulos'] = int((np.abs(e.loc[ok, k].values) > 1e-9).sum())
            familia.append((f'{rot} / {j}', tabela[rot][j]))
    ph = holm([v['p_perm'] for _, v in familia])
    for (_, v), x in zip(familia, ph):
        v['p_holm'] = float(x)
    print(f'familia de {len(familia)} correlacoes (permutacao); menor p de Holm = {ph.min():.3f}; menores p:')
    for i in np.argsort([v['p_perm'] for _, v in familia])[:6]:
        v = familia[i][1]
        print(f"   {familia[i][0]:44s} r={v['r']:+.3f} n={v['n']} nao nulos={v['nao_nulos']} "
              f"p_perm={v['p_perm']:.4f} p_Holm={ph[i]:.3f}")

    # o gap se move mais depois do Copom?
    g = bova['gap'].abs()
    mg = g.index.isin(set(e['linha']))
    print(f"|gap| medio: reacao {g[mg].mean():.4f} vs. demais {g[~mg].mean():.4f}")

    # o mercado reage nesse dia? |retorno| nos dias de reacao contra os demais
    linhas = set(e['linha'])
    abs_ret = fp['ret_lag0'].abs()
    m = abs_ret.index.isin(linhas)
    dif = abs_ret[m].mean() - abs_ret[~m].mean()
    rng = np.random.default_rng(0)
    boot = [rng.choice(abs_ret[m].values, m.sum()).mean() - rng.choice(abs_ret[~m].values, (~m).sum()).mean()
            for _ in range(20000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    magnitude = {'media_abs_reacao': float(abs_ret[m].mean()), 'media_abs_demais': float(abs_ret[~m].mean()),
                 'diferenca': float(dif), 'ic95': [float(lo), float(hi)], 'n_reacao': int(m.sum())}
    print(f"|retorno| medio: reacao {abs_ret[m].mean():.4f} vs. demais {abs_ret[~m].mean():.4f}; "
          f"dif {dif:+.4f} [{lo:+.4f}; {hi:+.4f}]")

    magnitude['abs_gap_reacao'] = float(g[mg].mean())
    magnitude['abs_gap_demais'] = float(g[~mg].mean())
    magnitude['surpresas_nao_nulas_pela_mediana'] = int((e['surpresa_mediana'].abs() > 1e-9).sum())

    # o escore antecipa a surpresa? instrumento x decisao realizada menos expectativa
    instrumento = {}
    for k in ('copom_hawkish_score', 'd_score', 'sem_anuncio', 'd_sem_anuncio'):
        for sup in ('surpresa_media', 'surpresa_mediana'):
            instrumento[f'{nomes[k]} / {nomes[sup]}'] = {
                'r': float(np.corrcoef(e[k], e[sup])[0, 1]), 'p_perm': p_permutacao(e[k].values, e[sup].values), 'n': int(len(e))}
    for k, v in instrumento.items():
        print(f"{k:62s} r={v['r']:+.3f} p_perm={v['p_perm']:.3f}")
    OUT.write_text(json.dumps({'n_eventos': len(e), 'correlacoes': tabela, 'magnitude': magnitude, 'instrumento_vs_surpresa': instrumento},
                              indent=1, ensure_ascii=False), encoding='utf-8')
    print('salvo em', OUT)


if __name__ == '__main__':
    main()
