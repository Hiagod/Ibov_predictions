"""Coleta os dados que permitem testar a janela de reacao (revisao do professor, C11 e C13).

data/raw/bova11_yahoo.csv : OHLC diario do BOVA11 (Yahoo Finance, BOVA11.SA). As fontes de indice
                            testadas nao tem abertura real (Stooq ^BVP e Yahoo ^BVSP: abertura =
                            fechamento anterior); o ETF tem, e e onde a execucao na abertura ocorreria.
data/raw/focus_selic.csv  : expectativas de Selic por reuniao do Copom (Focus, API Olinda do BCB),
                            mantidos so os 10 dias anteriores a cada decisao. baseCalculo = 0:
                            respostas de 30 dias; 1: de 5 dias.

Uso:
    python coleta_janela_reacao.py
"""
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

PROJECT = Path(__file__).resolve().parent.parent
RAW = PROJECT / 'data' / 'raw'
UA = {'User-Agent': 'Mozilla/5.0'}
INICIO, FIM = '2018-07-01', '2024-12-31'
OLINDA = 'https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoSelic'


def coleta_bova11():
    p1, p2 = (int(pd.Timestamp(x, tz='UTC').timestamp()) for x in (INICIO, '2025-01-01'))
    url = (f'https://query1.finance.yahoo.com/v8/finance/chart/BOVA11.SA'
           f'?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit')
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    res = r.json()['chart']['result'][0]
    q = res['indicators']['quote'][0]
    tz = res['meta']['exchangeTimezoneName']
    dias = pd.to_datetime(res['timestamp'], unit='s', utc=True).tz_convert(tz).tz_localize(None).normalize()
    df = pd.DataFrame({k: q[k] for k in ('open', 'high', 'low', 'close', 'volume')}, index=dias)
    df.index.name = 'date'
    df = df.dropna(subset=['open', 'close']).sort_index()
    assert df.index.is_unique, 'datas repetidas no BOVA11'
    arq = RAW / 'bova11_yahoo.csv'
    df.to_csv(arq)
    print(f"BOVA11: {len(df)} pregoes, {df.index[0].date()} a {df.index[-1].date()}, "
          f"eventos={list(res.get('events', {}).keys())}, {arq.stat().st_size / 1024:.0f} KB")


def coleta_focus():
    filtro = quote(f"Data ge '{INICIO}' and Data le '{FIM}'")
    url = f'{OLINDA}?$format=json&$top=200000&$filter={filtro}'
    r = requests.get(url, headers=UA, timeout=180)
    r.raise_for_status()
    f = pd.DataFrame(r.json()['value'])
    f['Data'] = pd.to_datetime(f['Data'])
    print(f'Focus bruto: {len(f)} linhas; colunas {f.columns.tolist()}')

    # Reuniao do Focus = "R<ordem>/<ano>"; 8 reunioes por ano, e a de dez/2018 e a R8
    c = pd.read_parquet(RAW / 'copom_comunicados.parquet').sort_values('data_reuniao')
    ordem = c.groupby(c['data_reuniao'].dt.year).cumcount() + 1
    ordem[c['data_reuniao'].dt.year == 2018] = 8
    assert (c['data_reuniao'].dt.year[c['data_reuniao'].dt.year > 2018].value_counts() == 8).all()
    c['Reuniao'] = 'R' + ordem.astype(str) + '/' + c['data_reuniao'].dt.year.astype(str)

    partes = []
    for _, m in c.iterrows():
        z = f[(f['Reuniao'] == m['Reuniao']) & (f['Data'] < m['data_reuniao'])
              & (f['Data'] >= m['data_reuniao'] - pd.Timedelta(days=10))]
        partes.append(z.assign(nro_reuniao=m['nro_reuniao'], data_reuniao=m['data_reuniao']))
    s = pd.concat(partes).sort_values(['nro_reuniao', 'baseCalculo', 'Data'])
    faltam = sorted(set(c['nro_reuniao']) - set(s['nro_reuniao']))
    arq = RAW / 'focus_selic.csv'
    s.to_csv(arq, index=False)
    print(f'Focus: {len(s)} linhas, {s["nro_reuniao"].nunique()} reunioes, sem dados: {faltam}, '
          f'{arq.stat().st_size / 1024:.0f} KB')


if __name__ == '__main__':
    coleta_bova11()
    coleta_focus()
