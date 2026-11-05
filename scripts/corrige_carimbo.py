"""Corrige o carimbo temporal do corpus na fonte (revisao do professor, C13). Roda uma unica vez.

Dois erros, descritos em diagnostico_fechamento_b3.py:
  (1) ate a migracao do site do InfoMoney (24/09/2019), datePublished traz hora local rotulada como UTC;
  (2) o corte fixo de 17h nao coincide com o fechamento da B3 nos meses em que o pregao vai ate 18h.

O script guarda os arquivos originais em data/_v0_carimbo_original/ e atualiza ts_utc, ts_local e
trading_day em infomoney_corpus.parquet e corpus_bucketed.parquet. Linhas e ordem nao mudam: os embeddings
e as probabilidades do FinBERT sao alinhados por posicao a corpus_bucketed.parquet.

Tambem grava data/processed/indice_duplicata_bertimbau.json: a extracao do BERTimbau tem uma linha a mais,
removida depois por duplicidade de manchete + pregao (colabs/03, cel. 9). Essa posicao era deduzida do
carimbo antigo e agora fica registrada.

Uso:
    python corrige_carimbo.py
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / 'utils'))
from tcc_utils import TZ_BR, assign_trading_day, build_cutoffs, corrige_carimbo_infomoney

RAW = PROJECT / 'data' / 'raw'
PROC = PROJECT / 'data' / 'processed'
REP = PROJECT / 'reports'
BACKUP = PROJECT / 'data' / '_v0_carimbo_original'


def guarda_originais():
    if (BACKUP / 'CONCLUIDO').exists():
        print('originais ja guardados em', BACKUP)
        return
    (BACKUP / 'raw').mkdir(parents=True, exist_ok=True)
    (BACKUP / 'processed').mkdir(exist_ok=True)
    (BACKUP / 'reports').mkdir(exist_ok=True)
    shutil.copy2(RAW / 'infomoney_corpus.parquet', BACKUP / 'raw')
    for arq in ('corpus_bucketed.parquet', 'features_texto_bertimbau.parquet', 'features_texto_finbert.parquet',
                'features_sentimento_finbert.parquet', 'features_sentimento_alt.parquet', 'features_copom.parquet'):
        shutil.copy2(PROC / arq, BACKUP / 'processed')
    for p in REP.iterdir():
        if p.is_file():
            shutil.copy2(p, BACKUP / 'reports')
    (BACKUP / 'CONCLUIDO').write_text('copia feita antes da correcao do carimbo\n', encoding='utf-8')
    print('originais guardados em', BACKUP)


def registra_duplicata_bertimbau():
    """Posicao, nas shards do BERTimbau, da linha removida em colabs/03 cel. 9 (usa o carimbo original)."""
    arq = PROC / 'indice_duplicata_bertimbau.json'
    if arq.exists():
        return
    bruto = pd.read_parquet(BACKUP / 'raw' / 'infomoney_corpus.parquet').reset_index(drop=True)
    precos = pd.read_parquet(PROC / 'bvsp_clean.parquet')
    # corte fixo de 17h, como na extracao original
    dias = pd.DatetimeIndex(precos.index)
    cortes = pd.Series(dias, index=pd.DatetimeIndex([d.replace(hour=17, tzinfo=TZ_BR) for d in dias]).tz_convert('UTC'))
    bruto['trading_day'] = assign_trading_day(bruto['ts_utc'], cortes)
    bruto = bruto.dropna(subset=['trading_day']).reset_index(drop=True)
    dup = bruto.duplicated(subset=['headline', 'trading_day'], keep='first')
    assert dup.sum() == 1 and len(bruto) == 57612, (dup.sum(), len(bruto))
    fica = bruto.loc[~dup, 'url'].values
    assert (fica == pd.read_parquet(BACKUP / 'processed' / 'corpus_bucketed.parquet')['url'].values).all()
    arq.write_text(json.dumps({'posicao': int(np.flatnonzero(dup.values)[0]), 'url': bruto.loc[dup, 'url'].iloc[0]},
                              ensure_ascii=False, indent=1), encoding='utf-8')
    print('duplicata do BERTimbau registrada:', arq.read_text(encoding='utf-8').replace('\n', ' '))


def main():
    guarda_originais()
    registra_duplicata_bertimbau()
    precos = pd.read_parquet(PROC / 'bvsp_clean.parquet')
    cortes = build_cutoffs(pd.DatetimeIndex(precos.index))

    bruto = pd.read_parquet(RAW / 'infomoney_corpus.parquet')
    ts = corrige_carimbo_infomoney(bruto['datePublished'])
    print(f'corpus bruto: {(ts != bruto["ts_utc"]).sum()} de {len(bruto)} carimbos alterados')
    bruto['ts_utc'], bruto['ts_local'] = ts, ts.dt.tz_convert(TZ_BR)
    bruto.to_parquet(RAW / 'infomoney_corpus.parquet')

    c = pd.read_parquet(PROC / 'corpus_bucketed.parquet')
    urls, dia_antigo = c['url'].values.copy(), c['trading_day'].values.copy()
    ts = corrige_carimbo_infomoney(c['datePublished'])
    print(f'corpus_bucketed: {(ts != c["ts_utc"]).sum()} de {len(c)} carimbos alterados')
    c['ts_utc'], c['ts_local'] = ts, ts.dt.tz_convert(TZ_BR)
    c['trading_day'] = assign_trading_day(c['ts_utc'], cortes).values
    assert c['trading_day'].notna().all(), 'manchete sem pregao apos a correcao'
    assert (c['url'].values == urls).all()
    print(f'manchetes que mudam de linha: {(c["trading_day"].values != dia_antigo).sum()}; '
          f'duplicatas manchete + pregao: {c.duplicated(subset=["headline", "trading_day"]).sum()}')
    c.to_parquet(PROC / 'corpus_bucketed.parquet')
    print('corpus corrigido')


if __name__ == '__main__':
    main()
