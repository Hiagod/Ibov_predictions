"""Diagnostico do comentario C10 da revisao: a correlacao contemporanea do sentimento
(+0,20) mede o mercado ou a imprensa descrevendo o pregao?

(a) Relato vs. evento: classifica cada manchete por regra explicita (ativo ou ticker +
    verbo de movimento de preco = relato de mercado; o resto = evento) e mede, por classe,
    a correlacao do sentimento diario com o retorno do dia, com o de D+1 e a parte
    ortogonal ao retorno do dia (semiparcial).
(b) Antes vs. depois da publicacao (resolucao diaria; nao ha preco intradiario): para cada
    faixa de horario, correlacao do sentimento com o pregao ANTERIOR a publicacao e com o
    POSTERIOR. Pos-fechamento, madrugada e fim de semana: antes = retorno da linha anterior
    (ret_lag1), depois = retorno da propria linha (ret_lag0). Pregao: durante = ret_lag0.

Sentimento por manchete: P(positivo) - P(negativo) do FinBERT-PT-BR
(data/processed/finbert_probs_raw.npy, alinhado a corpus_bucketed.parquet).
IC de 95% por block bootstrap sobre dias (blocos de 10, B = 10 000, semente 0).

Uso:
    python diagnostico_relato_evento.py   ->  reports/diagnostico_relato_evento.json
"""
import html
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / 'utils'))
from tcc_utils import hora_fechamento_b3

PROC = PROJECT / 'data' / 'processed'
OUT = PROJECT / 'reports' / 'diagnostico_relato_evento.json'
B, SEED, BLOCO = 10000, 0, 10

# Regra v2 (apos auditoria manual da v1, que confundia resultados trimestrais e dados macro
# com relato por causa de "alta"/"queda" soltos): relato = verbo de MOVIMENTO DE PRECO e
# mencao a ativo negociado (ou ticker). "indice" e "euro" saem da lista (indice de
# confianca, Zona do Euro); "alta"/"queda" so contam dentro de expressoes de pregao.
ATIVO = re.compile(r'\b(ibovespa|ibov|bolsas?|b3|dolar|juros|dis?|petroleo|minerio|bitcoin|'
                   r'ethereum|cripto\w*|acoes|acao|papeis|papel|wall street|s&p|nasdaq|'
                   r'dow jones|treasur\w*|commodities|ouro|futuros?|bovespa|mercados?)\b')
TICKER = re.compile(r'\b[A-Z]{4}\d{1,2}\b')
PRECO = re.compile(r'\b(sobe|sobem|subiu|subir|subindo|cai|caem|caiu|cair|caindo|fecha|fecham|'
                   r'fechou|recua\w*|avanca\w*|dispara\w*|despenca\w*|salta\w*|saltou|tomba\w*|'
                   r'derrete\w*|desaba\w*|opera|operam|abre|abrem|abriu|valoriza\w*|'
                   r'desvaloriza\w*|oscila\w*|renova\w* (maxima|minima|recorde)|'
                   r'maiores (altas|quedas|baixas)|puxa\w* (alta|altas|queda|quedas)|'
                   r'(em|de) (alta|queda|baixa) (de )?\d)\b')


def normaliza(t):
    t = html.unescape(str(t))          # manchetes brutas trazem entidades ("S&amp;P", "&#8220;")
    t = unicodedata.normalize('NFKD', t).encode('ascii', 'ignore').decode()
    return t.lower()


def classe(manchete):
    n = normaliza(manchete)
    if PRECO.search(n) and (ATIVO.search(n) or TICKER.search(html.unescape(str(manchete)))):
        return 'relato'
    return 'evento'


def corr_ic(x, y):
    """Correlacao de Pearson com IC por block bootstrap sobre dias (pares completos)."""
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    n = len(x)
    r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(SEED)
    nb = int(np.ceil(n / BLOCO))
    boot = np.empty(B)
    for i0 in range(0, B, 2000):
        k = min(2000, B - i0)
        starts = rng.integers(0, n - BLOCO + 1, size=(k, nb))
        idx = (starts[:, :, None] + np.arange(BLOCO)).reshape(k, -1)[:, :n]
        xb, yb = x[idx], y[idx]
        xb = xb - xb.mean(1, keepdims=True)
        yb = yb - yb.mean(1, keepdims=True)
        boot[i0:i0 + k] = (xb * yb).sum(1) / np.sqrt((xb ** 2).sum(1) * (yb ** 2).sum(1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))
    return {'r': r, 'lo': float(lo), 'hi': float(hi), 'p': p, 'n_dias': int(n)}


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    ordem = np.argsort(ps)
    adj, acum = np.empty(len(ps)), 0.0
    for rank, i in enumerate(ordem):
        acum = max(acum, (len(ps) - rank) * ps[i])
        adj[i] = min(acum, 1.0)
    return adj


def residuo(x, z):
    """Parte de x ortogonal a z (com intercepto), nos dias com ambos definidos."""
    out = np.full_like(x, np.nan)
    ok = ~(np.isnan(x) | np.isnan(z))
    Z = np.column_stack([np.ones(ok.sum()), z[ok]])
    beta, *_ = np.linalg.lstsq(Z, x[ok], rcond=None)
    out[ok] = x[ok] - Z @ beta
    return out


def main():
    fp = pd.read_parquet(PROC / 'features_preco.parquet')
    fs = pd.read_parquet(PROC / 'features_sentimento_finbert.parquet').reindex(fp.index)
    c = pd.read_parquet(PROC / 'corpus_bucketed.parquet').reset_index(drop=True)
    probs = np.load(PROC / 'finbert_probs_raw.npy')
    assert len(c) == len(probs), 'corpus e probabilidades desalinhados'
    c['pol'] = probs[:, 0] - probs[:, 1]

    # checagem de alinhamento: a media diaria deve reproduzir sent_polaridade
    diaria = c.groupby('trading_day')['pol'].mean().reindex(fp.index)
    dif = float(np.nanmax(np.abs(diaria.values - fs['sent_polaridade'].values)))
    assert dif < 1e-6, f'media diaria nao reproduz sent_polaridade (dif max {dif})'

    t = c['ts_local'].dt.tz_localize(None)
    h = t.dt.hour + t.dt.minute / 60
    util = t.dt.dayofweek < 5
    # pos-fechamento: depois do fechamento efetivo daquele dia (17h ou 18h)
    c['faixa'] = np.select([~util, h < 10, h <= hora_fechamento_b3(t.dt.normalize())],
                           ['fim de semana', 'pre-abertura', 'pregao'], 'pos-fechamento')
    c['classe'] = [classe(m) for m in c['headline']]

    rng = np.random.default_rng(SEED)
    exemplos = {k: c.loc[c['classe'] == k, 'headline'].sample(10, random_state=1).tolist()
                for k in ('relato', 'evento')}
    frac = c['classe'].value_counts(normalize=True).round(4).to_dict()
    frac_faixa = (c.groupby('faixa')['classe'].apply(lambda s: (s == 'relato').mean())
                  .round(3).to_dict())

    ret0, ret1, y = fp['ret_lag0'].values, fp['ret_lag1'].values, fp['y'].values
    saida = {'B': B, 'bloco': BLOCO, 'seed': SEED, 'checagem_alinhamento_dif_max': dif,
             'fracao_manchetes': frac, 'fracao_relato_por_faixa': frac_faixa,
             'exemplos': exemplos, 'relato_evento': {}, 'antes_depois': {}}

    for k in ('todas', 'relato', 'evento'):
        sub = c if k == 'todas' else c[c['classe'] == k]
        pol = sub.groupby('trading_day')['pol'].mean().reindex(fp.index).values
        saida['relato_evento'][k] = {
            'contemporanea': corr_ic(pol, ret0),
            'preditiva': corr_ic(pol, y),
            'preditiva_ortogonal_ao_dia': corr_ic(residuo(pol, ret0), y),
            'dias_com_manchete': int(np.sum(~np.isnan(pol))),
        }

    for f in ('pos-fechamento', 'fim de semana', 'pre-abertura', 'pregao'):
        pol = c[c['faixa'] == f].groupby('trading_day')['pol'].mean().reindex(fp.index).values
        if f == 'pregao':
            saida['antes_depois'][f] = {'durante (ret do dia)': corr_ic(pol, ret0),
                                        'seguinte (D+1)': corr_ic(pol, y)}
        else:
            saida['antes_depois'][f] = {'antes (pregao anterior)': corr_ic(pol, ret1),
                                        'depois (pregao seguinte)': corr_ic(pol, ret0),
                                        'D+1': corr_ic(pol, y)}

    # pre-abertura por classe: os +0,20 com o pregao seguinte vem de relato (mercados externos,
    # futuros) ou de eventos? E a parte ortogonal ao retorno do dia preve D+1?
    pre = c[c['faixa'] == 'pre-abertura']
    saida['pre_abertura_por_classe'] = {}
    for k in ('relato', 'evento'):
        pol = pre[pre['classe'] == k].groupby('trading_day')['pol'].mean().reindex(fp.index).values
        saida['pre_abertura_por_classe'][k] = {
            'antes (pregao anterior)': corr_ic(pol, ret1),
            'depois (pregao seguinte)': corr_ic(pol, ret0),
            'D+1': corr_ic(pol, y),
            'D+1 ortogonal ao dia': corr_ic(residuo(pol, ret0), y)}
    pol_pre = pre.groupby('trading_day')['pol'].mean().reindex(fp.index).values
    saida['antes_depois']['pre-abertura']['D+1 ortogonal ao dia'] = corr_ic(residuo(pol_pre, ret0), y)

    # familia de testes PREDITIVOS (correlacao com D+1): p corrigido por Holm
    familia = []
    for k, v in saida['relato_evento'].items():
        familia += [(f'{k} / preditiva', v['preditiva']),
                    (f'{k} / preditiva ortogonal', v['preditiva_ortogonal_ao_dia'])]
    for f, v in saida['antes_depois'].items():
        for kk, d in v.items():
            if 'D+1' in kk:
                familia.append((f'{f} / {kk}', d))
    for k, v in saida['pre_abertura_por_classe'].items():
        familia += [(f'pre-abertura {k} / D+1', v['D+1']),
                    (f'pre-abertura {k} / D+1 ortogonal', v['D+1 ortogonal ao dia'])]
    p_holm = holm([d['p'] for _, d in familia])
    for (nome, d), ph in zip(familia, p_holm):
        d['p_holm_familia_preditiva'] = float(ph)
    saida['familia_preditiva'] = {'n_testes': len(familia),
                                  'menor_p_holm': float(p_holm.min()),
                                  'testes': {nome: float(ph) for (nome, _), ph in zip(familia, p_holm)}}

    # amostra para auditoria manual da regra de classificacao (manchetes sem entidades HTML)
    import html
    saida['auditoria'] = {k: [html.unescape(m) for m in
                              c.loc[c['classe'] == k, 'headline'].sample(40, random_state=11)]
                          for k in ('relato', 'evento')}

    OUT.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'alinhamento: dif max = {dif:.2e} | fracao: {frac} | relato por faixa: {frac_faixa}')
    for k in ('relato', 'evento'):
        print(f'\n--- exemplos {k} ---')
        for e in exemplos[k]:
            print('  ', e)
    fmt = lambda d: f"{d['r']:+.3f} [{d['lo']:+.3f}; {d['hi']:+.3f}] n={d['n_dias']}"
    print('\n=== (a) relato vs. evento ===')
    for k, v in saida['relato_evento'].items():
        print(f"{k:<7} contemp {fmt(v['contemporanea'])} | pred {fmt(v['preditiva'])} | "
              f"pred ortog {fmt(v['preditiva_ortogonal_ao_dia'])}")
    print('\n=== (b) antes vs. depois da publicacao ===')
    for f, v in saida['antes_depois'].items():
        print(f'{f}:')
        for kk, d in v.items():
            print(f'    {kk:<26} {fmt(d)}')
    print('\n=== pre-abertura por classe ===')
    for k, v in saida['pre_abertura_por_classe'].items():
        print(f'{k}:')
        for kk, d in v.items():
            print(f'    {kk:<26} {fmt(d)}')
    fp_ = saida['familia_preditiva']
    print(f"\n=== familia preditiva: {fp_['n_testes']} testes; menor p (Holm) = {fp_['menor_p_holm']:.3f} ===")
    for nome, ph in sorted(fp_['testes'].items(), key=lambda kv: kv[1])[:5]:
        print(f'    {nome:<40} p_holm={ph:.3f}')
    for k in ('relato', 'evento'):
        print(f'\n--- auditoria {k} (40) ---')
        for i, e in enumerate(saida['auditoria'][k], 1):
            print(f'  {i:2d}. {e}')


if __name__ == '__main__':
    main()
