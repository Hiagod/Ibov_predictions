"""Figura da Secao V-E: correlacao de cada variavel de sentimento com o retorno do mesmo dia e com o de D+1.

Valores de reports/teste_reativo_vs_preditivo.json (teste_reativo_vs_preditivo.py).

Uso:
    python figura_sentimento_reativo.py   ->  entrega_final/fig_sentimento_reativo.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / 'entrega_final' / 'fig_sentimento_reativo.png'
AZUL, LARANJA = '#4C72B0', '#DD8452'
NOMES = {'sent_polaridade': 'Polaridade', 'sent_dispersao': 'Dispersão',
         'sent_pos_frac': 'Frac. positiva', 'sent_neg_frac': 'Frac. negativa'}


def main():
    res = json.loads((PROJECT / 'reports' / 'teste_reativo_vs_preditivo.json').read_text(encoding='utf-8'))['resultados']
    x = np.arange(len(res))
    fig, ax = plt.subplots(figsize=(6.2, 2.45), dpi=300)
    ax.bar(x - 0.19, [r['r_contemporanea'] for r in res], 0.38, color=AZUL, label='Retorno contemporâneo (D)')
    ax.bar(x + 0.19, [r['r_preditiva'] for r in res], 0.38, color=LARANJA, label='Retorno preditivo (D+1)')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x, [NOMES[r['variavel']] for r in res])
    ax.set_ylabel('Correlação de Pearson')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.1f}'.replace('.', ',').replace('-', '\u2212')))
    ax.set_title('Sentimento FinBERT: reativo vs. preditivo')
    ax.legend(fontsize=7.5, loc='upper right')
    fig.tight_layout()
    fig.savefig(OUT)
    print('salvo:', OUT)


if __name__ == '__main__':
    main()
