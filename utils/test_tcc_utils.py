"""Testes de tcc_utils.py -- casos-limite do corte diario (fechamento efetivo da B3,
17h ou 18h America/Sao_Paulo), do carimbo do InfoMoney e da validacao walk-forward.
Cobrem as categorias descritas na metodologia do TCC: corte inclusivo, virada de fim
de semana, feriado, horario de verao historico, o timestamp UTC vindo do JSON-LD e,
na ultima secao, os proprios dados: relatos de fechamento precisam sair depois do
fechamento e a rotina editorial nao pode saltar na migracao do site. Sem data/, que nao
esta no repositorio, os testes sobre os dados sao pulados, e o resumo informa quantos.

Uso:
    .venv\\Scripts\\python -m pytest utils/test_tcc_utils.py -v
ou, sem pytest instalado:
    .venv\\Scripts\\python utils/test_tcc_utils.py
"""
import sys
import unittest
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tcc_utils import (build_cutoffs, assign_trading_day, corrige_carimbo_infomoney,
                       hora_fechamento_b3, make_target, walk_forward_splits)

TZ_BR = ZoneInfo('America/Sao_Paulo')


def _cutoffs_para(dias_str):
    dias = pd.DatetimeIndex(pd.to_datetime(dias_str))
    return build_cutoffs(dias)


def _local(dia_str, hora_str):
    """Timestamp aware construido via zoneinfo (respeita DST historico),
    nunca por offset fixo escrito a mao -- e exatamente o bug que este
    arquivo pegou na primeira versao (ver git history / handoff)."""
    return pd.Timestamp(f'{dia_str} {hora_str}').tz_localize(TZ_BR)


# ---------- assign_trading_day ----------

def test_antes_do_corte_fica_no_mesmo_dia():
    cutoffs = _cutoffs_para(['2019-01-02', '2019-01-03'])
    ts = pd.Series([_local('2019-01-02', '16:00:00')])
    out = assign_trading_day(ts, cutoffs)
    assert out.iloc[0] == pd.Timestamp('2019-01-02'), out.iloc[0]


def test_corte_e_inclusivo_17h_exato_fica_no_mesmo_dia():
    # noticia EXATAMENTE as 17:00 local ainda conta para o pregao D
    # (searchsorted side='left': instante igual ao cutoff fica ANTES dele
    # na ordenacao, ou seja e classificado como "ainda dentro" de D)
    cutoffs = _cutoffs_para(['2019-06-03', '2019-06-04'])
    ts = pd.Series([_local('2019-06-03', '17:00:00')])
    out = assign_trading_day(ts, cutoffs)
    assert out.iloc[0] == pd.Timestamp('2019-06-03'), out.iloc[0]


def test_um_segundo_apos_o_corte_vai_para_o_proximo_pregao():
    cutoffs = _cutoffs_para(['2019-06-03', '2019-06-04'])
    ts = pd.Series([_local('2019-06-03', '17:00:01')])
    out = assign_trading_day(ts, cutoffs)
    assert out.iloc[0] == pd.Timestamp('2019-06-04'), out.iloc[0]


def test_depois_do_corte_vai_para_o_proximo_pregao():
    cutoffs = _cutoffs_para(['2019-01-02', '2019-01-03'])
    ts = pd.Series([_local('2019-01-02', '18:30:00')])  # ex.: COPOM 18h30
    out = assign_trading_day(ts, cutoffs)
    assert out.iloc[0] == pd.Timestamp('2019-01-03'), out.iloc[0]


def test_sexta_a_noite_pula_para_segunda():
    # sexta 2019-01-04, proximo pregao segunda 2019-01-07 (sem pregao no fds)
    cutoffs = _cutoffs_para(['2019-01-04', '2019-01-07'])
    ts = pd.Series([_local('2019-01-04', '19:00:00')])
    out = assign_trading_day(ts, cutoffs)
    assert out.iloc[0] == pd.Timestamp('2019-01-07'), out.iloc[0]


def test_feriado_no_calendario_de_pregoes_e_pulado():
    # calendario de pregoes ja vem sem feriados (vindo do indice de precos);
    # noticia apos o corte de quinta (Tiradentes na sexta seguinte) cai no
    # primeiro pregao seguinte do calendario, qualquer que seja a distancia
    cutoffs = _cutoffs_para(['2019-04-18', '2019-04-22'])
    ts = pd.Series([_local('2019-04-18', '18:00:00')])
    out = assign_trading_day(ts, cutoffs)
    assert out.iloc[0] == pd.Timestamp('2019-04-22'), out.iloc[0]


def test_horario_de_verao_historico_nao_e_offset_fixo():
    # 15/jan/2019: Brasil em horario de verao (BRST, UTC-2) ate 17/fev/2019, com
    # pregao ate 18h. Cutoff local 18:00 BRST == 20:00 UTC. Um instante 20:30 UTC
    # (18:30 BRST, apos o corte) deve cair no pregao seguinte. Se o codigo tratasse
    # o fuso como offset FIXO de -3h (erro classico), 20:30 UTC vira 17:30
    # "local" -- antes do corte, mesmo dia -- resultado ERRADO. Este teste
    # falharia sob essa implementacao incorreta.
    cutoffs = _cutoffs_para(['2019-01-15', '2019-01-16'])
    ts_utc = pd.Series([pd.Timestamp('2019-01-15T20:30:00', tz='UTC')])
    out = assign_trading_day(ts_utc, cutoffs)
    assert out.iloc[0] == pd.Timestamp('2019-01-16'), out.iloc[0]


def test_utc_real_do_json_ld_e_convertido_corretamente():
    # instante fixo em UTC (formato do JSON-LD do InfoMoney, ex. datePublished
    # terminando em "Z" ou "+00:00") deve ser convertido considerando o
    # deslocamento vigente na data -- aqui, junho, sem horario de verao
    # (BRT, UTC-3): cutoff 17h local == 20h UTC.
    cutoffs = _cutoffs_para(['2019-06-04', '2019-06-05'])
    ts = pd.Series([pd.Timestamp('2019-06-04T20:30:00', tz='UTC')])  # 17h30 BRT
    out = assign_trading_day(ts, cutoffs)
    assert out.iloc[0] == pd.Timestamp('2019-06-05'), out.iloc[0]


def test_noticia_apos_ultimo_corte_vira_nat():
    cutoffs = _cutoffs_para(['2019-01-02'])
    ts = pd.Series([_local('2019-01-03', '12:00:00')])
    out = assign_trading_day(ts, cutoffs)
    assert pd.isna(out.iloc[0])


def test_madrugada_fica_no_pregao_do_dia_corrente():
    cutoffs = _cutoffs_para(['2019-01-02', '2019-01-03'])
    ts = pd.Series([_local('2019-01-02', '06:00:00')])
    out = assign_trading_day(ts, cutoffs)
    assert out.iloc[0] == pd.Timestamp('2019-01-02'), out.iloc[0]


# ---------- fechamento efetivo da B3 ----------

def test_pregao_ate_18h_no_inverno_americano():
    cutoffs = _cutoffs_para(['2023-01-10', '2023-01-11'])
    dentro = pd.Series([_local('2023-01-10', '17:30:00')])
    fora = pd.Series([_local('2023-01-10', '18:00:01')])
    assert assign_trading_day(dentro, cutoffs).iloc[0] == pd.Timestamp('2023-01-10')
    assert assign_trading_day(fora, cutoffs).iloc[0] == pd.Timestamp('2023-01-11')


def test_pregao_ate_17h_no_verao_americano():
    cutoffs = _cutoffs_para(['2023-06-14', '2023-06-15'])
    fora = pd.Series([_local('2023-06-14', '17:30:00')])
    assert assign_trading_day(fora, cutoffs).iloc[0] == pd.Timestamp('2023-06-15')


def test_datas_de_transicao_do_fechamento():
    casos = {'2019-11-01': 17, '2019-11-04': 18, '2020-03-06': 18, '2020-03-09': 17,
             '2021-03-12': 18, '2021-03-15': 17, '2021-11-05': 17, '2021-11-08': 18,
             '2023-11-01': 17, '2023-11-06': 18, '2024-03-08': 18, '2024-03-11': 17}
    for dia, hora in casos.items():
        assert hora_fechamento_b3([pd.Timestamp(dia)])[0] == hora, dia


def test_copom_18h30_fica_sempre_depois_do_fechamento():
    for d0, d1 in (('2023-02-01', '2023-02-02'), ('2023-06-21', '2023-06-22')):
        ts = pd.Series([_local(d0, '18:30:00')])
        assert assign_trading_day(ts, _cutoffs_para([d0, d1])).iloc[0] == pd.Timestamp(d1)


# ---------- carimbo do InfoMoney ----------

def _carimbo_local(valor):
    return corrige_carimbo_infomoney(pd.Series([valor])).dt.tz_convert(TZ_BR).dt.strftime('%Y-%m-%d %H:%M').iloc[0]


def test_carimbo_antes_da_migracao_e_hora_local():
    # decisao do Fed de 31/07/2019: 15h em Brasilia, gravada como 15:01+00:00
    assert _carimbo_local('2019-07-31T15:01:00+00:00') == '2019-07-31 15:01'


def test_carimbo_antes_da_migracao_em_horario_de_verao():
    # Copom de 12/12/2018 (BRST): gravado 18:21+00:00, publicado as 18h21 locais
    assert _carimbo_local('2018-12-12T18:21:00+00:00') == '2018-12-12 18:21'


def test_carimbo_depois_da_migracao_e_utc():
    # Copom de 30/10/2019: gravado 21:07+00:00, publicado as 18h07 locais
    assert _carimbo_local('2019-10-30T21:07:00+00:00') == '2019-10-30 18:07'


def test_virada_do_carimbo_no_dia_da_migracao():
    assert _carimbo_local('2019-09-24T17:00:00+00:00') == '2019-09-24 17:00'
    assert _carimbo_local('2019-09-24T20:20:00+00:00') == '2019-09-24 17:20'


# ---------- os proprios dados ----------

def _corpus():
    arq = Path(__file__).parent.parent / 'data' / 'processed' / 'corpus_bucketed.parquet'
    if not arq.exists():
        raise unittest.SkipTest('sem data/processed/corpus_bucketed.parquet')
    c = pd.read_parquet(arq)
    t = c['ts_local'].dt.tz_localize(None)
    return c.assign(_t=t, _hora=t.dt.hour + t.dt.minute / 60, _util=t.dt.dayofweek < 5,
                    _antes=t < '2019-09-24 18:00', _fech=hora_fechamento_b3(t.dt.normalize()))


def test_dados_relatos_de_fechamento_saem_depois_do_fechamento():
    c = _corpus()
    rel = c[c['headline'].str.contains(r'Ibovespa[^.;:]{0,40}?\b(?:fecha|encerra)\b', regex=True)
            & c['_util'] & (c['_hora'] >= 12)]
    for rotulo, g in rel.groupby('_antes'):
        cedo = (g['_hora'] < g['_fech']).mean()
        assert cedo < 0.15, f'{rotulo}: {cedo:.1%} dos relatos antes do fechamento'
    # a hora mediana do relato acompanha o calendario de 17h e 18h
    for fech, g in rel.groupby('_fech'):
        assert fech <= g['_hora'].median() <= fech + 1, f'{fech}h: {g["_hora"].median():.2f}'


def test_dados_rotina_editorial_nao_salta_na_migracao():
    c = _corpus()
    col = c[c['headline'].str.contains('Os 5 assuntos') & c['_util']]
    antes, depois = col[col['_antes']]['_hora'].median(), col[~col['_antes']]['_hora'].median()
    assert abs(antes - depois) < 1, f'coluna diaria: {antes:.2f}h antes e {depois:.2f}h depois'


def test_dados_linha_do_pregao_coincide_com_o_corte():
    c = _corpus()
    # nenhuma manchete entra na linha de um pregao ja encerrado quando foi publicada
    cortes = c['trading_day'] + pd.to_timedelta(hora_fechamento_b3(c['trading_day']), unit='h')
    assert (c['_t'] <= cortes).all(), 'manchete publicada depois do fechamento da propria linha'


# ---------- make_target ----------

def test_make_target_shift_menos_um():
    precos = pd.DataFrame(
        {'log_ret': [0.01, -0.02, 0.03]},
        index=pd.date_range('2019-01-02', periods=3, freq='B'))
    y = make_target(precos)
    assert y.iloc[0] == -0.02
    assert y.iloc[1] == 0.03
    assert pd.isna(y.iloc[2])  # ultimo dia sem alvo


# ---------- walk_forward_splits ----------

def test_walk_forward_respeita_ordem_temporal():
    idx = pd.date_range('2019-01-01', periods=400, freq='B')
    splits = walk_forward_splits(idx, n_splits=3, min_train=100, embargo=0)
    for tr, te in splits:
        assert tr.max() < te.min(), "treino deve terminar antes do teste comecar"


def test_walk_forward_embargo_cria_gap_exato():
    idx = pd.date_range('2019-01-01', periods=400, freq='B')
    splits = walk_forward_splits(idx, n_splits=3, min_train=100, embargo=21)
    for tr, te in splits:
        gap = te[0] - tr[-1] - 1
        assert gap == 21, f"gap esperado 21, obtido {gap}"


def test_walk_forward_expanding_treino_cresce():
    idx = pd.date_range('2019-01-01', periods=400, freq='B')
    splits = walk_forward_splits(idx, n_splits=3, min_train=100, expanding=True)
    tamanhos = [len(tr) for tr, _ in splits]
    assert tamanhos == sorted(tamanhos), "treino expansivel deve so crescer"


if __name__ == '__main__':
    testes = [v for k, v in list(globals().items()) if k.startswith('test_')]
    falhas = pulados = 0
    for t in testes:
        try:
            t()
            print(f"OK   {t.__name__}")
        except unittest.SkipTest as e:
            pulados += 1
            print(f"PULO {t.__name__}: {e}")
        except AssertionError as e:
            falhas += 1
            print(f"FAIL {t.__name__}: {e}")
    resumo = f"{len(testes) - falhas - pulados}/{len(testes)} passaram"
    print(f"\n{resumo}, {pulados} pulados" if pulados else f"\n{resumo}")
    if falhas:
        sys.exit(1)
