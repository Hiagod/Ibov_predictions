"""Testes de tcc_utils.py -- casos-limite do corte diario (17h America/Sao_Paulo)
e da validacao walk-forward. Cobrem as categorias descritas na metodologia do
TCC: corte inclusivo, virada de fim de semana, feriado, horario de verao
historico, e o timestamp UTC real vindo do JSON-LD do InfoMoney.

Uso:
    .venv\\Scripts\\python -m pytest utils/test_tcc_utils.py -v
ou, sem pytest instalado:
    .venv\\Scripts\\python utils/test_tcc_utils.py
"""
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tcc_utils import build_cutoffs, assign_trading_day, make_target, walk_forward_splits

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
    cutoffs = _cutoffs_para(['2019-01-02', '2019-01-03'])
    ts = pd.Series([_local('2019-01-02', '17:00:00')])
    out = assign_trading_day(ts, cutoffs)
    assert out.iloc[0] == pd.Timestamp('2019-01-02'), out.iloc[0]


def test_um_segundo_apos_o_corte_vai_para_o_proximo_pregao():
    cutoffs = _cutoffs_para(['2019-01-02', '2019-01-03'])
    ts = pd.Series([_local('2019-01-02', '17:00:01')])
    out = assign_trading_day(ts, cutoffs)
    assert out.iloc[0] == pd.Timestamp('2019-01-03'), out.iloc[0]


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
    # 15/jan/2019: Brasil em horario de verao (BRST, UTC-2) ate 17/fev/2019.
    # Cutoff local 17:00 BRST == 19:00 UTC. Um instante 19:30 UTC (17:30
    # BRST, apos o corte) deve cair no pregao seguinte. Se o codigo tratasse
    # o fuso como offset FIXO de -3h (erro classico), 19:30 UTC vira 16:30
    # "local" -- antes do corte, mesmo dia -- resultado ERRADO. Este teste
    # falharia sob essa implementacao incorreta.
    cutoffs = _cutoffs_para(['2019-01-15', '2019-01-16'])
    ts_utc = pd.Series([pd.Timestamp('2019-01-15T19:30:00', tz='UTC')])
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
    falhas = 0
    for t in testes:
        try:
            t()
            print(f"OK   {t.__name__}")
        except AssertionError as e:
            falhas += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(testes) - falhas}/{len(testes)} passaram")
    if falhas:
        sys.exit(1)
