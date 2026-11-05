"""Gera as linhas LaTeX da tabela unica da cadeia de robustez (revisao do professor, C12)
a partir de reports/ic_mcc_bootstrap.json (secao 'robustez', gerada por ic_mcc_bootstrap.py).

Celula: delta-MCC agrupado frente ao modelo sem texto na mesma configuracao, com IC de 95%
por block bootstrap sobre dias. '--' = combinacao nao testada.

Uso:
    python tabela_robustez.py   ->  imprime as linhas e um resumo para o texto
"""
import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
dados = json.loads((PROJECT / 'reports' / 'ic_mcc_bootstrap.json').read_text(encoding='utf-8'))
rob = dados['robustez']

COLUNAS = ['BERTimbau k=8', 'BERTimbau k=30', 'FinBERT sent.', 'FinBERT emb. k=8',
           'FinBERT emb. k=30']
LINHAS = [
    ('principal', r'Principal (manchete, LSTM, lookback 21)'),
    ('texto_D-1', r'Texto do dia anterior ($D-1$)'),
    ('ultima_manchete', r'\'Ultima manchete antes do corte'),
    ('lookback_1', r'Lookback de 1 dia'),
    ('lookback_3', r'Lookback de 3 dias'),
    ('lookback_5', r'Lookback de 5 dias'),
    ('lookback_10', r'Lookback de 10 dias'),
    ('bow', r'TF-IDF+SVD no lugar do embedding'),
    ('corpo_truncado', r'Corpo completo (512 tokens)'),
    ('corpo_chunking', r'Corpo completo (\emph{chunking})'),
    ('xgboost_texto', r'XGBoost no lugar da LSTM'),
]


def num(x):
    r = round(x, 2)
    if r == 0:
        return '0{,}00'
    return f'{r:+.2f}'.replace('.', '{,}')


def celula(r):
    return f"${num(r['delta'])}\\;[{num(r['lo'])};\\,{num(r['hi'])}]$"


def linhas_latex():
    idx = {(r['verificacao'], r['coluna']): r for r in rob}
    saida = []
    for chave, rotulo in LINHAS:
        cels = [celula(idx[(chave, c)]) if (chave, c) in idx else '--' for c in COLUNAS]
        saida.append(rotulo + ' & ' + ' & '.join(cels) + r' \\')
    return saida


def main():
    print('\n'.join(linhas_latex()))
    texto = [r for r in rob if r['verificacao'] != 'mlp_sem_texto']
    his = np.array([r['hi'] for r in texto])
    los = np.array([r['lo'] for r in texto])
    exclui = sum(1 for r in texto if r['lo'] > 0 or r['hi'] < 0)
    mlp = next(r for r in rob if r['verificacao'] == 'mlp_sem_texto')
    print(f'\n% resumo: {len(texto)} intervalos com texto; excluem zero: {exclui}; '
          f'limites superiores de {his.min():+.3f} a {his.max():+.3f} (mediana {np.median(his):+.3f}); '
          f'limites inferiores de {los.min():+.3f} a {los.max():+.3f}')
    print(f"% MLP sem texto vs. LSTM preco: {mlp['delta']:+.3f} [{mlp['lo']:+.3f}; {mlp['hi']:+.3f}]")


if __name__ == '__main__':
    main()
