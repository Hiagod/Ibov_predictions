"""tcc_utils -- funcoes centrais do TCC (fonte unica, canonica).

Gravado uma unica vez pelo notebook 00_setup. Os demais notebooks apenas
importam daqui -- nenhuma celula deve reescrever este arquivo em runtime.
"""
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

TZ_BR = ZoneInfo('America/Sao_Paulo')

# A B3 acompanha o relogio de Nova York: pregao ate 18h do 1o pregao apos o fim do horario de verao
# americano ate a vespera do 1o pregao apos o seu inicio; ate 17h no restante. Intervalos [inicio, fim).
JANELAS_FECHAMENTO_18H = [
    ('2018-11-05', '2019-03-11'), ('2019-11-04', '2020-03-09'), ('2020-11-03', '2021-03-15'),
    ('2021-11-08', '2022-03-14'), ('2022-11-07', '2023-03-13'), ('2023-11-06', '2024-03-11'),
    ('2024-11-04', '2025-03-10'),
]

# Ate a migracao do site do InfoMoney (24/09/2019), o datePublished do JSON-LD traz hora LOCAL rotulada
# como "+00:00"; depois, UTC verdadeiro. No dia da virada nao ha manchetes entre 17:00 e 20:20 no valor
# bruto, entao qualquer instante nesse vao separa os dois regimes.
FIM_CARIMBO_LOCAL_INFOMONEY = pd.Timestamp('2019-09-24 18:00')


def hora_fechamento_b3(trading_days) -> np.ndarray:
    """Hora local de fechamento do pregao regular em cada dia (17 ou 18)."""
    dias = pd.DatetimeIndex(trading_days)
    horas = np.full(len(dias), 17)
    for inicio, fim in JANELAS_FECHAMENTO_18H:
        horas[(dias >= inicio) & (dias < fim)] = 18
    return horas


def corrige_carimbo_infomoney(date_published: pd.Series) -> pd.Series:
    """Instante UTC da publicacao a partir do datePublished bruto do JSON-LD."""
    bruto = pd.to_datetime(date_published.astype(str).str.slice(0, 19))
    antigo = (bruto < FIM_CARIMBO_LOCAL_INFOMONEY).values
    out = bruto.dt.tz_localize('UTC')
    out[antigo] = (bruto[antigo].dt.tz_localize(TZ_BR, ambiguous='NaT', nonexistent='shift_forward')
                   .dt.tz_convert('UTC'))
    return out


def build_cutoffs(trading_days: pd.DatetimeIndex) -> pd.Series:
    dias = pd.DatetimeIndex(trading_days)
    cutoffs_local = pd.DatetimeIndex(
        [d.replace(hour=int(h), tzinfo=TZ_BR) for d, h in zip(dias, hora_fechamento_b3(dias))]
    )
    return pd.Series(dias, index=cutoffs_local.tz_convert('UTC'))


def assign_trading_day(ts_utc: pd.Series, cutoffs: pd.Series) -> pd.Series:
    cut_values = cutoffs.index.values
    day_values = cutoffs.values
    pos = np.searchsorted(cut_values, ts_utc.values, side='left')
    ok = pos < len(day_values)
    out = np.full(len(ts_utc), np.datetime64('NaT'), dtype='datetime64[ns]')
    out[ok] = day_values[pos[ok]]
    return pd.Series(out, index=ts_utc.index, name='trading_day')


def make_target(precos: pd.DataFrame) -> pd.Series:
    """Alvo UNICO do projeto: log-retorno de D+1.
    y_t = ln(close_{t+1}/close_t) = log_ret deslocado -1.
    O shift(-1) acontece SO AQUI. O ultimo dia fica sem alvo (NaN) e
    e descartado pelo chamador. Todos os modelos usam este alvo."""
    y = precos['log_ret'].shift(-1)
    y.name = 'target_logret_d1'
    return y


def walk_forward_splits(index, n_splits=5, min_train=252, expanding=True, embargo=0):
    """Gera (train_idx, test_idx) respeitando ordem temporal.
    - expanding=True: treino cresce a cada fold.
    - embargo>0: descarta os primeiros \`embargo\` dias de cada bloco de teste,
      criando um gap treino->teste. Necessario quando o modelo usa janelas de
      lookback (LSTM): sem isso, as 1as sequencias de teste conteriam dias do
      treino. Para baselines sem lookback, embargo=0 (comportamento original).
    Retorna posicoes inteiras (iloc)."""
    n = len(index)
    fold_size = (n - min_train) // (n_splits + 1)
    if fold_size <= 0:
        raise ValueError("Serie curta demais p/ esse n_splits/min_train")
    splits = []
    for k in range(1, n_splits + 1):
        train_end = min_train + (k - 1) * fold_size
        test_start = train_end + embargo
        test_end = test_start + fold_size
        if test_start >= n:
            break
        train_start = 0 if expanding else max(0, train_end - min_train)
        splits.append((
            np.arange(train_start, train_end),
            np.arange(test_start, min(test_end, n)),
        ))
    return splits
