"""Consolida os shards de coleta_artigos.py + repesca.py num corpus unico,
limpo e com timestamps corretos.

Passos (nesta ordem):
  1. Concatena shards da coleta original e da repesca (repesca por ULTIMO,
     assim o dedup por URL com keep='last' prefere o resultado da repesca
     quando ela teve sucesso onde a coleta original falhou).
  2. Mantem so status == 200 com datePublished nao-nulo.
  3. Parseia datePublished como UTC verdadeiro (confirmado por
     verifica_timezone.py -- o JSON-LD do InfoMoney usa UTC real, nao um
     offset fixo de -3h) e converte para America/Sao_Paulo via zoneinfo,
     que trata o horario de verao historico corretamente.
  4. Filtra ao periodo de coleta (2018-06 a 2025-02, com margem -- o corte
     fino por periodo de estudo acontece no notebook 03).
  5. Remove duplicatas de manchete no mesmo dia local (republicacoes).
  6. Salva infomoney_corpus.parquet com o schema canonico.

Uso:
    python consolida_corpus.py
"""
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

PROJECT = Path(r'D:\tcc_ibov')
RAW_DIR = PROJECT / 'data' / 'raw'
SHARD_DIR = RAW_DIR / 'shards_artigos'
REPESCA_DIR = RAW_DIR / 'shards_repesca'

TZ_BR = ZoneInfo('America/Sao_Paulo')
PERIODO_LO = pd.Timestamp('2018-06-01', tz='UTC')
PERIODO_HI = pd.Timestamp('2025-02-01', tz='UTC')

SCHEMA_COLS = ['url', 'status', 'error', 'headline', 'datePublished',
               'dateModified', 'via', 'ts_utc', 'ts_local']


def carrega_shards() -> pd.DataFrame:
    partes = []
    for pasta in (SHARD_DIR, REPESCA_DIR):
        arquivos = sorted(pasta.glob('*.parquet'))
        if arquivos:
            partes.append(pd.concat([pd.read_parquet(p) for p in arquivos],
                                     ignore_index=True))
    if not partes:
        raise SystemExit("Nenhum shard encontrado -- rode coleta_artigos.py "
                          "(e repesca.py) primeiro.")
    # repesca concatenada por ultimo -> keep='last' no dedup prefere repesca
    bruto = pd.concat(partes, ignore_index=True)
    print(f"Linhas brutas (coleta + repesca): {len(bruto):,}")
    return bruto


def main():
    bruto = carrega_shards()

    bruto = bruto.drop_duplicates(subset='url', keep='last')
    print(f"Apos dedup por URL: {len(bruto):,}")

    corpus = bruto[(bruto['status'] == 200) & bruto['datePublished'].notna()].copy()
    print(f"Com status 200 e datePublished: {len(corpus):,}")

    # datePublished do JSON-LD e UTC verdadeiro (confirmado empiricamente --
    # ver verifica_timezone.py). pd.to_datetime com utc=True normaliza
    # qualquer offset presente na string ISO para UTC.
    corpus['ts_utc'] = pd.to_datetime(corpus['datePublished'], utc=True,
                                       errors='coerce')
    antes = len(corpus)
    corpus = corpus.dropna(subset=['ts_utc'])
    if antes != len(corpus):
        print(f"Descartadas {antes - len(corpus)} linhas com datePublished "
              f"nao parseavel")

    corpus = corpus[corpus['ts_utc'].between(PERIODO_LO, PERIODO_HI)]
    print(f"Apos filtro de periodo ({PERIODO_LO.date()} a {PERIODO_HI.date()}): "
          f"{len(corpus):,}")

    corpus['ts_local'] = corpus['ts_utc'].dt.tz_convert(TZ_BR)

    # Duplicatas de manchete no mesmo dia local (republicacoes com sufixo
    # "-2" etc.) -- mantem a primeira publicacao.
    dia_local = corpus['ts_local'].dt.date
    antes = len(corpus)
    corpus = corpus.loc[~corpus.assign(_dia=dia_local)
                         .duplicated(subset=['headline', '_dia'], keep='first')]
    print(f"Apos remover duplicatas headline+dia: {len(corpus):,} "
          f"(removidas {antes - len(corpus)})")

    corpus = corpus[SCHEMA_COLS].reset_index(drop=True)
    corpus.to_parquet(RAW_DIR / 'infomoney_corpus.parquet')
    print(f"\nSalvo: {RAW_DIR / 'infomoney_corpus.parquet'} | shape {corpus.shape}")


if __name__ == '__main__':
    main()
