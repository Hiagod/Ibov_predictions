"""Corte de 17h vs. fechamento efetivo da B3 (revisao do professor, C13) e erro de fuso no corpus.

Dois problemas no carimbo temporal, medidos aqui sem alterar nenhum arquivo de dados:

(1) Erro de fuso ate a migracao do site do InfoMoney (24/09/2019). Antes dela, o datePublished do
    JSON-LD traz a hora LOCAL rotulada como "+00:00"; a conversao para America/Sao_Paulo subtrai
    de novo o fuso. Efeito: manchetes publicadas ate 20h (19h no horario de verao) entram na linha
    do proprio dia, com o pregao ja encerrado. Evidencia: relatos de fechamento as ~14h15, decisao
    do Fed as 12h01 (real: 15h), coluna "Os 5 assuntos" as 04h40 antes da virada e 07h50 depois.

(2) Fechamento efetivo. A B3 acompanha o horario de Nova York: pregao ate 18h entre o fim e o
    inicio do horario de verao americano (nov-mar) e ate 17h no restante. Nesses meses, manchetes
    de 17h-18h saem com o pregao aberto e entram na linha seguinte. Nao ha vazamento nesse caso (a
    informacao entra depois, nao antes), so defasagem.

A correcao ja foi aplicada na fonte (corrige_carimbo.py, tcc_utils.py); este script compara com os dados
originais, guardados em data/_v0_carimbo_original/.

Versoes comparadas:
    V0 original              : trading_day original (corte de 17h)
    V1 fuso corrigido        : carimbo corrigido, corte de 17h
    V2 fuso + fechamento real: carimbo corrigido, corte no fechamento efetivo de cada dia

IC de 95% por block bootstrap sobre dias (funcoes de diagnostico_relato_evento.py).

Uso:
    python diagnostico_fechamento_b3.py   ->  reports/diagnostico_fechamento_b3.json
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from diagnostico_relato_evento import classe, corr_ic, holm, normaliza, residuo

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / 'utils'))
from tcc_utils import assign_trading_day, corrige_carimbo_infomoney, hora_fechamento_b3

PROC = PROJECT / 'data' / 'processed'
V0 = PROJECT / 'data' / '_v0_carimbo_original' / 'processed'
OUT = PROJECT / 'reports' / 'diagnostico_fechamento_b3.json'
TZ = 'America/Sao_Paulo'

# As datas de JANELAS_FECHAMENTO_18H (tcc_utils.py) foram conferidas nos relatos de fechamento do proprio
# corpus e em noticias (InvestNews: 03/11/2020 e 15/03/2021; 08/11/2021 e 14/03/2022; 06/11/2023).

FECHA = re.compile(r'\bibovespa\b[^.;:]{0,40}?\b(?:fecha|encerra|termina|fechou|encerrou|terminou)\b')
FAMILIA_D1 = ('preditiva', 'preditiva_ortogonal_ao_dia', 'D+1', 'D+1 ortogonal ao dia', 'seguinte (D+1)')


def carimbo_corrigido(c):
    """Instante UTC correto e mascara das manchetes anteriores a migracao do site."""
    out = corrige_carimbo_infomoney(c['datePublished'])
    assert out.notna().all(), 'carimbo ambiguo apos a correcao'
    bruto = pd.to_datetime(c['datePublished'].astype(str).str.slice(0, 19))
    return out, (bruto < pd.Timestamp('2019-09-24 18:00')).values


def hora_de_fechamento(dias):
    dias = pd.DatetimeIndex(dias)
    return pd.Series(hora_fechamento_b3(dias), index=dias, dtype=int)


def cortes(dias, efetivo):
    """Equivalente a build_cutoffs, com a hora de corte variando por dia quando efetivo=True."""
    dias = pd.DatetimeIndex(dias)
    h = hora_de_fechamento(dias) if efetivo else pd.Series(17, index=dias)
    loc = pd.DatetimeIndex([d.replace(hour=int(x)) for d, x in zip(dias, h.values)]).tz_localize(TZ)
    return pd.Series(dias, index=loc.tz_convert('UTC'))


def analisa(c, dia, hora_local, dia_semana, fech_local, fp):
    """Mesma decomposicao da Secao V-E para uma versao da atribuicao temporal."""
    ret0, ret1, y = fp['ret_lag0'].values, fp['ret_lag1'].values, fp['y'].values
    d = pd.DataFrame({'dia': dia, 'pol': c['pol'].values, 'classe': c['classe'].values})
    d['faixa'] = np.select([dia_semana >= 5, hora_local < 10, hora_local <= fech_local],
                           ['fim de semana', 'pre-abertura', 'pregao'], 'pos-fechamento')

    def serie(m=None):
        z = d if m is None else d[m]
        return z.groupby('dia')['pol'].mean().reindex(fp.index).values

    s = {'fracao_relato': float((d['classe'] == 'relato').mean()),
         'fracao_faixa': d['faixa'].value_counts(normalize=True).round(4).to_dict(),
         'relato_evento': {}, 'antes_depois': {}, 'pre_abertura_por_classe': {}}
    for k in ('todas', 'relato', 'evento'):
        p = serie(None if k == 'todas' else d['classe'] == k)
        s['relato_evento'][k] = {'contemporanea': corr_ic(p, ret0), 'preditiva': corr_ic(p, y),
                                 'preditiva_ortogonal_ao_dia': corr_ic(residuo(p, ret0), y)}
    for f in ('pos-fechamento', 'fim de semana', 'pre-abertura', 'pregao'):
        p = serie(d['faixa'] == f)
        if f == 'pregao':
            s['antes_depois'][f] = {'durante (ret do dia)': corr_ic(p, ret0), 'seguinte (D+1)': corr_ic(p, y)}
        else:
            s['antes_depois'][f] = {'antes (pregao anterior)': corr_ic(p, ret1),
                                    'depois (pregao seguinte)': corr_ic(p, ret0), 'D+1': corr_ic(p, y)}
    s['antes_depois']['pre-abertura']['D+1 ortogonal ao dia'] = corr_ic(residuo(serie(d['faixa'] == 'pre-abertura'), ret0), y)
    for k in ('relato', 'evento'):
        p = serie((d['faixa'] == 'pre-abertura') & (d['classe'] == k))
        s['pre_abertura_por_classe'][k] = {'antes (pregao anterior)': corr_ic(p, ret1),
                                           'depois (pregao seguinte)': corr_ic(p, ret0), 'D+1': corr_ic(p, y),
                                           'D+1 ortogonal ao dia': corr_ic(residuo(p, ret0), y)}
    familia = [(f'{g}/{k}/{kk}', v) for g in ('relato_evento', 'antes_depois', 'pre_abertura_por_classe')
               for k, dd in s[g].items() for kk, v in dd.items() if kk in FAMILIA_D1]
    ph = holm([v['p'] for _, v in familia])
    s['familia_preditiva'] = {'n_testes': len(familia), 'menor_p_holm': float(ph.min()),
                              'menor': familia[int(np.argmin(ph))][0],
                              'testes': {n: float(x) for (n, _), x in zip(familia, ph)}}
    return s


def main():
    fp = pd.read_parquet(PROC / 'features_preco.parquet')
    fs = pd.read_parquet(V0 / 'features_sentimento_finbert.parquet').reindex(fp.index)
    precos = pd.read_parquet(PROC / 'bvsp_clean.parquet')
    c = pd.read_parquet(V0 / 'corpus_bucketed.parquet').reset_index(drop=True)
    probs = np.load(PROC / 'finbert_probs_raw.npy')
    assert len(c) == len(probs), 'corpus e probabilidades desalinhados'
    c['pol'] = probs[:, 0] - probs[:, 1]
    c['classe'] = [classe(m) for m in c['headline']]
    dif = float(np.nanmax(np.abs(c.groupby('trading_day')['pol'].mean().reindex(fp.index).values
                                 - fs['sent_polaridade'].values)))
    assert dif < 1e-6, 'V0 nao reproduz sent_polaridade'

    ts_v1, antigo = carimbo_corrigido(c)
    loc_v0 = c['ts_local'].dt.tz_localize(None)
    loc_v1 = ts_v1.dt.tz_convert(TZ).dt.tz_localize(None)
    h0, h1 = (x.dt.hour + x.dt.minute / 60 for x in (loc_v0, loc_v1))
    dia_v0 = c['trading_day']
    dia_v1 = assign_trading_day(ts_v1, cortes(precos.index, False))
    dia_v2 = assign_trading_day(ts_v1, cortes(precos.index, True))
    fech = hora_de_fechamento(loc_v1.dt.normalize()).values
    em_estudo = dia_v0.isin(fp.index).values

    # ---- evidencias do erro e do calendario ----
    relato_f = c['headline'].map(lambda m: bool(FECHA.search(normaliza(m)))).values
    util = (loc_v1.dt.dayofweek < 5).values
    janela = util & relato_f & (h1.values >= 12)
    evid = {}
    for nome, m in (('antes da virada', antigo), ('depois da virada', ~antigo)):
        for reg in (17, 18):
            k = janela & m & (fech == reg)
            if k.sum():
                evid[f'{nome}, pregao ate {reg}h'] = {
                    'n': int(k.sum()),
                    'mediana_hora_v0': float(np.median(h0.values[k])), 'mediana_hora_v1': float(np.median(h1.values[k])),
                    'antes_do_fechamento_v0': float((h0.values[k] < reg).mean()),
                    'antes_do_fechamento_v1': float((h1.values[k] < reg).mean())}
    assuntos = c['headline'].str.contains('Os 5 assuntos').values & util
    evid_ass = {k: {'n': int((assuntos & m).sum()), 'mediana_hora_v0': float(np.median(h0.values[assuntos & m])),
                    'mediana_hora_v1': float(np.median(h1.values[assuntos & m]))}
                for k, m in (('antes da virada', antigo), ('depois da virada', ~antigo))}
    n_manc = int((c['ts_local'].dt.tz_localize(None) < '2019-09-25').sum())

    # ---- tamanho do problema ----
    muda_v1 = (dia_v0.values != dia_v1.values)
    muda_v2 = (dia_v1.values != dia_v2.values)
    linhas_ate_virada = int((fp.index <= '2019-09-24').sum())
    n_tr = [252, 458, 664, 870, 1076]
    tamanho = {
        'manchetes_com_carimbo_errado': int(antigo.sum()),
        'manchetes_com_carimbo_errado_no_estudo': int((antigo & em_estudo).sum()),
        'fracao_do_corpus': float(antigo.mean()),
        'manchetes_que_mudam_de_linha_v1': int(muda_v1.sum()),
        'manchetes_que_mudam_de_linha_v1_no_estudo': int((muda_v1 & em_estudo).sum()),
        'linhas_do_estudo_afetadas_v1': int(pd.Series(dia_v0.values[muda_v1]).isin(fp.index).pipe(
            lambda z: pd.Series(dia_v0.values[muda_v1])[z].nunique())),
        'linhas_do_estudo_ate_a_virada': linhas_ate_virada,
        'primeira_linha_de_teste': str(fp.index[252].date()),
        'fracao_do_treino_ate_a_virada_por_fold': [round(linhas_ate_virada / n, 3) for n in n_tr],
        'manchetes_17h_18h_com_pregao_aberto_v2': int(muda_v2.sum()),
        'fracao_do_corpus_v2': float(muda_v2.mean()),
        'pregoes_com_fechamento_18h_no_estudo': float((hora_de_fechamento(fp.index) == 18).mean()),
    }

    # ---- efeito direto: as manchetes que deveriam estar na linha seguinte ----
    pos_ini = c.loc[muda_v1].groupby(dia_v0[muda_v1])['pol'].mean().reindex(fp.index).values
    ate = (fp.index <= '2019-09-24')
    ret0, ret1, y = fp['ret_lag0'].values, fp['ret_lag1'].values, fp['y'].values
    serie = lambda dia: c.groupby(np.asarray(dia))['pol'].mean().reindex(fp.index).values
    s0, s1, s2 = serie(dia_v0), serie(dia_v1), serie(dia_v2)
    efeito = {
        'manchetes_deslocadas_vs_retorno_do_dia': corr_ic(pos_ini, ret0),
        'manchetes_deslocadas_vs_alvo_da_linha_errada': corr_ic(pos_ini, y),
        'janela_ate_a_virada': {
            'v0_contemporanea': corr_ic(s0[ate], ret0[ate]), 'v0_preditiva': corr_ic(s0[ate], y[ate]),
            'v1_contemporanea': corr_ic(s1[ate], ret0[ate]), 'v1_preditiva': corr_ic(s1[ate], y[ate]),
            'v0_contemporanea_lag1': corr_ic(s0[ate], ret1[ate])},
        'correlacao_da_serie_v0_v1': float(np.corrcoef(s0, s1)[0, 1]),
        'correlacao_da_serie_v1_v2': float(np.corrcoef(s1, s2)[0, 1]),
    }

    # ---- decomposicao da Secao V-E nas tres versoes ----
    dsem1 = loc_v1.dt.dayofweek.values
    versoes = {
        'V0': analisa(c, dia_v0.values, h0.values, loc_v0.dt.dayofweek.values, 17, fp),
        'V1': analisa(c, dia_v1.values, h1.values, dsem1, 17, fp),
        'V2': analisa(c, dia_v2.values, h1.values, dsem1, fech, fp),
    }

    saida = {'dif_alinhamento_v0': dif, 'evidencia_relatos_de_fechamento': evid,
             'evidencia_coluna_5_assuntos': evid_ass, 'manchetes_ate_a_virada_v0': n_manc,
             'tamanho': tamanho, 'efeito': efeito, 'versoes': versoes}
    OUT.write_text(json.dumps(saida, indent=1, ensure_ascii=False), encoding='utf-8')

    print('=== evidencia: relatos de fechamento (hora mediana; fracao antes do fechamento) ===')
    for k, v in evid.items():
        print(f"{k:34s} n={v['n']:4d} V0 {v['mediana_hora_v0']:5.2f}h ({v['antes_do_fechamento_v0']:.0%}) | "
              f"V1 {v['mediana_hora_v1']:5.2f}h ({v['antes_do_fechamento_v1']:.0%})")
    print('=== coluna "Os 5 assuntos" ===')
    for k, v in evid_ass.items():
        print(f"{k:20s} n={v['n']:4d} V0 {v['mediana_hora_v0']:.2f}h | V1 {v['mediana_hora_v1']:.2f}h")
    print('\n=== tamanho ===')
    for k, v in tamanho.items():
        print(f'{k:48s} {v}')
    fmt = lambda d: f"{d['r']:+.3f} [{d['lo']:+.3f}; {d['hi']:+.3f}] n={d['n_dias']}"
    print('\n=== efeito ===')
    print('deslocadas vs. retorno do dia     ', fmt(efeito['manchetes_deslocadas_vs_retorno_do_dia']))
    print('deslocadas vs. alvo da linha errada', fmt(efeito['manchetes_deslocadas_vs_alvo_da_linha_errada']))
    for k, v in efeito['janela_ate_a_virada'].items():
        print(f'  ate a virada {k:26s}', fmt(v))
    print('corr. entre as series: V0-V1', round(efeito['correlacao_da_serie_v0_v1'], 4),
          '| V1-V2', round(efeito['correlacao_da_serie_v1_v2'], 4))
    print('\n=== numeros da Secao V-E nas tres versoes ===')
    linhas = [('todas contemp.', 'relato_evento', 'todas', 'contemporanea'),
              ('todas preditiva', 'relato_evento', 'todas', 'preditiva'),
              ('todas ortogonal', 'relato_evento', 'todas', 'preditiva_ortogonal_ao_dia'),
              ('relato contemp.', 'relato_evento', 'relato', 'contemporanea'),
              ('evento contemp.', 'relato_evento', 'evento', 'contemporanea'),
              ('pos-fech. antes', 'antes_depois', 'pos-fechamento', 'antes (pregao anterior)'),
              ('pos-fech. depois', 'antes_depois', 'pos-fechamento', 'depois (pregao seguinte)'),
              ('pre-ab. antes', 'antes_depois', 'pre-abertura', 'antes (pregao anterior)'),
              ('pre-ab. depois', 'antes_depois', 'pre-abertura', 'depois (pregao seguinte)'),
              ('pre-ab. relato', 'pre_abertura_por_classe', 'relato', 'depois (pregao seguinte)'),
              ('pre-ab. evento', 'pre_abertura_por_classe', 'evento', 'depois (pregao seguinte)'),
              ('pregao (dia)', 'antes_depois', 'pregao', 'durante (ret do dia)')]
    for rot, g, k, kk in linhas:
        print(f'{rot:18s}', ' | '.join(f"{v}: {fmt(versoes[v][g][k][kk])}" for v in versoes))
    for v in versoes:
        f = versoes[v]['familia_preditiva']
        print(f"{v}: fracao de relatos {versoes[v]['fracao_relato']:.3f}; {f['n_testes']} testes, "
              f"menor p Holm {f['menor_p_holm']:.3f} ({f['menor']})")
    print('salvo em', OUT)


if __name__ == '__main__':
    main()
