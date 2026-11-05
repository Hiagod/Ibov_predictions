"""Janela de reacao com aberturas reais (revisao do professor, C13): o texto publicado antes da
abertura se relaciona com o gap noturno ou com o retorno abertura -> fechamento?

Com o BOVA11 (data/raw/bova11_yahoo.csv), o retorno fechamento a fechamento da linha t se decompoe em
    gap_t = ln(Open_t / Close_t-1)   a reacao fora do pregao: ja ocorreu quando o pregao abre
    oc_t  = ln(Close_t / Open_t)     a janela com execucao na abertura: o desenho preditivo legitimo

Cada manchete da
linha t e PRE-ABERTURA (publicada depois do fechamento de t-1 e antes das 10h de t) ou PREGAO. So a
pre-abertura esta disponivel antes de oc_t; a do pregao entra como referencia descritiva.

IC de 95% por block bootstrap sobre dias (blocos de 10, B = 10 000).

Uso:
    python diagnostico_janela_abertura.py   ->  reports/diagnostico_janela_abertura.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from diagnostico_fechamento_b3 import hora_de_fechamento
from diagnostico_relato_evento import classe, corr_ic, holm, residuo

PROJECT = Path(__file__).resolve().parent.parent
PROC = PROJECT / 'data' / 'processed'
OUT = PROJECT / 'reports' / 'diagnostico_janela_abertura.json'


def retornos_bova(index_completo, index_estudo):
    """gap, abertura-fechamento e fechamento-fechamento do BOVA11 no calendario do indice.

    Pregao ausente ou sem negociacao (20 dias no Yahoo tem volume 0 e O=H=L=C, 18 deles seguidos em mar-abr/2021):
    a abertura e falsa, e o oc daria zero e o gap, o retorno do dia inteiro. Esses dias ficam NaN em gap e oc, e o
    gap do pregao seguinte tambem, porque dependeria de um fechamento sem negociacao. Os fechamentos acompanham
    o indice (diferenca media de 0,2%) e seguem em cc."""
    b = pd.read_csv(PROJECT / 'data' / 'raw' / 'bova11_yahoo.csv', index_col='date', parse_dates=True)
    b = b.reindex(index_completo)
    sem = b['volume'].fillna(0) == 0
    abertura, fech_ant = b['open'].where(~sem), b['close'].shift(1).where(~sem.shift(1, fill_value=False))
    r = pd.DataFrame({'gap': np.log(abertura / fech_ant), 'oc': np.log(b['close'] / abertura),
                      'cc': np.log(b['close'] / b['close'].shift(1))})
    return r.reindex(index_estudo)


def bandas_v2(c):
    """Linha (V2) e banda de cada manchete: noite = apos o fechamento de t-1 ate 24h, ou fim de semana
    e feriado; manha = da 0h as 10h do proprio dia t; pregao = 10h ao fechamento de t."""
    dia = c['trading_day']
    loc = c['ts_local'].dt.tz_localize(None)
    h = loc.dt.hour + loc.dt.minute / 60
    mesma_data = (loc.dt.normalize().values == dia.values)
    banda = np.where(~mesma_data, 'noite', np.where(h.values < 10, 'manha', 'pregao'))
    return dia, banda


def main():
    fp = pd.read_parquet(PROC / 'features_preco.parquet')
    precos = pd.read_parquet(PROC / 'bvsp_clean.parquet')
    c = pd.read_parquet(PROC / 'corpus_bucketed.parquet').reset_index(drop=True)
    probs = np.load(PROC / 'finbert_probs_raw.npy')
    assert len(c) == len(probs)
    c['pol'] = probs[:, 0] - probs[:, 1]
    c['classe'] = [classe(m) for m in c['headline']]

    r = retornos_bova(precos.index, fp.index)
    gap, oc, cc, ret0 = r['gap'].values, r['oc'].values, r['cc'].values, fp['ret_lag0'].values
    ok = ~np.isnan(cc)
    sd_ano = pd.Series(gap, index=fp.index).groupby(fp.index.year).std()
    gr = pd.Series(np.abs(gap), index=fp.index)
    fech = hora_de_fechamento(fp.index)
    validacao = {
        'pregoes_com_oc': int((~np.isnan(oc)).sum()), 'pregoes_com_gap': int((~np.isnan(gap)).sum()),
        'corr_cc_bova_indice': float(np.corrcoef(cc[ok], ret0[ok])[0, 1]),
        'desvio_gap_por_ano': {int(k): float(v) for k, v in sd_ano.items()},
        'mediana_abs_gap': float(gr.median()), 'fracao_gap_zero': float((gr == 0).mean()),
        'var_gap_sobre_var_cc': float(np.nanvar(gap) / np.nanvar(cc)),
        'corr_gap_oc': corr_ic(gap, oc),
        'erro_max_gap_mais_oc_menos_cc': float(np.nanmax(np.abs(gap + oc - cc))),
        'pregoes_18h': float((fech == 18).mean()),
    }

    dia, banda = bandas_v2(c)
    d = pd.DataFrame({'dia': dia.values, 'banda': banda, 'pol': c['pol'].values, 'classe': c['classe'].values})
    d = d[d['dia'].isin(fp.index)]
    frac = d['banda'].value_counts(normalize=True).round(4).to_dict()

    def serie(m):
        return d[m].groupby('dia')['pol'].mean().reindex(fp.index).values

    pre = d['banda'].isin(['noite', 'manha'])
    recortes = {'pre-abertura': pre, 'pre-abertura, relato': pre & (d['classe'] == 'relato'),
                'pre-abertura, evento': pre & (d['classe'] == 'evento'), 'noite': d['banda'] == 'noite',
                'manha': d['banda'] == 'manha'}
    tab, familia = {}, []
    for nome, m in recortes.items():
        s = serie(m)
        e = {'n_dias_com_manchete': int((~np.isnan(s)).sum()), 'gap': corr_ic(s, gap),
             'oc (preditivo)': corr_ic(s, oc), 'oc, parte ortogonal ao gap (preditivo)': corr_ic(residuo(s, gap), oc),
             'cc': corr_ic(s, cc)}
        tab[nome] = e
        familia += [(f'{nome} / oc', e['oc (preditivo)']), (f'{nome} / oc ortogonal', e['oc, parte ortogonal ao gap (preditivo)'])]
    ph = holm([v['p'] for _, v in familia])
    for (_, v), x in zip(familia, ph):
        v['p_holm'] = float(x)
    s_preg = serie(d['banda'] == 'pregao')
    ref = {'pregao vs oc (descritivo)': corr_ic(s_preg, oc), 'pregao vs gap': corr_ic(s_preg, gap)}

    saida = {'validacao_bova11': validacao, 'fracao_por_banda': frac, 'recortes': tab, 'referencia': ref,
             'familia_preditiva': {'n_testes': len(familia), 'menor_p_holm': float(ph.min()),
                                   'menor': familia[int(np.argmin(ph))][0]}}
    OUT.write_text(json.dumps(saida, indent=1, ensure_ascii=False), encoding='utf-8')

    print('=== validacao do BOVA11 ===')
    for k, v in validacao.items():
        print(f'{k:28s}', {a: round(b, 4) for a, b in v.items() if isinstance(b, float)} if isinstance(v, dict) and 'r' in v
              else ({a: round(b, 4) for a, b in v.items()} if isinstance(v, dict) else round(v, 5) if isinstance(v, float) else v))
    print('fracao por banda:', frac)
    fmt = lambda x: f"{x['r']:+.3f} [{x['lo']:+.3f}; {x['hi']:+.3f}]"
    print('\n=== sentimento antes da abertura ===')
    print(f"{'recorte':24s} {'gap':24s} {'oc (preditivo)':24s} {'oc ortog. ao gap':24s} p_Holm")
    for k, e in tab.items():
        p1, p2 = e['oc (preditivo)']['p_holm'], e['oc, parte ortogonal ao gap (preditivo)']['p_holm']
        print(f"{k:24s} {fmt(e['gap']):24s} {fmt(e['oc (preditivo)']):24s} "
              f"{fmt(e['oc, parte ortogonal ao gap (preditivo)']):24s} {p1:.2f}/{p2:.2f}  n={e['n_dias_com_manchete']}")
    print('referencia:', {k: fmt(v) for k, v in ref.items()})
    print('familia preditiva:', saida['familia_preditiva'])
    print('salvo em', OUT)


if __name__ == '__main__':
    main()
