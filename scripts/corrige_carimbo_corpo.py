"""Leva a correcao do carimbo ao repositorio do corpo completo (D:\\tcc_ibov_corpo_completo), que nao esta no
git: guarda os originais em data/_v0_carimbo_original/, atualiza ts_utc, ts_local e trading_day por URL (linhas
e ordem nao mudam, os embeddings sao alinhados por posicao) e copia tcc_utils.py e as features de manchete
corrigidas do projeto principal.

A baseline XGBoost so com preco desse repositorio estava defasada frente a do principal (commit "Corrige baseline XGBoost defasado"); as
previsoes do principal sao copiadas para que os testes pareados usem a mesma baseline do artigo.

Uso:
    python corrige_carimbo_corpo.py
"""
import shutil
from pathlib import Path

import pandas as pd

PRINCIPAL = Path(__file__).resolve().parent.parent
CORPO = Path(r'D:\tcc_ibov_corpo_completo')
PROC = CORPO / 'data' / 'processed'
BACKUP = CORPO / 'data' / '_v0_carimbo_original'


def guarda_originais():
    if (BACKUP / 'CONCLUIDO').exists():
        print('originais ja guardados em', BACKUP)
        return
    (BACKUP / 'processed').mkdir(parents=True, exist_ok=True)
    (BACKUP / 'reports').mkdir(exist_ok=True)
    (BACKUP / 'utils').mkdir(exist_ok=True)
    for p in PROC.glob('*.parquet'):
        shutil.copy2(p, BACKUP / 'processed')
    for p in (CORPO / 'reports').iterdir():
        if p.is_file():
            shutil.copy2(p, BACKUP / 'reports')
    shutil.copy2(CORPO / 'utils' / 'tcc_utils.py', BACKUP / 'utils')
    (BACKUP / 'CONCLUIDO').write_text('copia feita antes da correcao do carimbo\n', encoding='utf-8')
    print('originais guardados em', BACKUP)


def main():
    guarda_originais()
    ref = pd.read_parquet(PRINCIPAL / 'data' / 'processed' / 'corpus_bucketed.parquet')
    ref = ref.drop_duplicates('url').set_index('url')
    for arq in ('corpus_bucketed.parquet', 'corpus_corpo_bucketed.parquet'):
        c = pd.read_parquet(PROC / arq)
        urls = c['url'].values.copy()
        antes = c['trading_day'].values.copy()
        assert c['url'].isin(ref.index).all(), f'{arq}: URL fora do corpus principal'
        for col in ('ts_utc', 'ts_local', 'trading_day'):
            c[col] = ref.loc[c['url'], col].values if col == 'trading_day' else ref.loc[c['url'], col].array
        assert (c['url'].values == urls).all() and c['trading_day'].notna().all()
        c.to_parquet(PROC / arq)
        print(f'{arq}: {len(c)} linhas; {(c["trading_day"].values != antes).sum()} mudam de pregao')

    shutil.copy2(PRINCIPAL / 'utils' / 'tcc_utils.py', CORPO / 'utils' / 'tcc_utils.py')
    for arq in ('features_texto_bertimbau.parquet', 'features_texto_finbert.parquet', 'features_sentimento_finbert.parquet'):
        shutil.copy2(PRINCIPAL / 'data' / 'processed' / arq, PROC / arq)
    for arq in ('preds_xgboost.npz', 'baseline_xgboost.parquet', 'baseline_xgboost_full.parquet'):
        shutil.copy2(PRINCIPAL / 'reports' / arq, CORPO / 'reports' / arq)
    print('tcc_utils, features de manchete e baseline XGBoost copiados do principal')


if __name__ == '__main__':
    main()
