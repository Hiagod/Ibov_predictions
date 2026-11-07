"""Figura de autocorrelacao do retorno e do retorno ao quadrado (Secao V-A do artigo).

Mesmas series e defasagens da celula do notebook 04. Muda a banda: 1,96/raiz(n), a de ruido branco, que e a
que a legenda descreve; o padrao de plot_acf e a de Bartlett, que se alarga com a defasagem.

Uso:
    python figura_acf.py   ->  entrega_final/acf_pacf_retorno.png
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / 'entrega_final' / 'acf_pacf_retorno.png'
VIRGULA = FuncFormatter(lambda v, _: f'{v:.1f}'.replace('.', ',').replace('-', '−'))


def main():
    fp = pd.read_parquet(PROJECT / 'data' / 'processed' / 'features_preco.parquet')
    px = pd.read_parquet(PROJECT / 'data' / 'processed' / 'bvsp_clean.parquet')
    r = px.loc[fp.index, 'log_ret']

    plt.rcParams['font.size'] = 7
    fig, ax = plt.subplots(2, 2, figsize=(6.5, 3.5), dpi=300)
    plot_acf(r, lags=30, ax=ax[0, 0], bartlett_confint=False, title='ACF, retorno', markersize=2.5)
    plot_pacf(r, lags=30, ax=ax[0, 1], method='ywm', title='PACF, retorno', markersize=2.5)
    plot_acf(r ** 2, lags=30, ax=ax[1, 0], bartlett_confint=False, title='ACF, retorno²', markersize=2.5)
    plot_pacf(r ** 2, lags=30, ax=ax[1, 1], method='ywm', title='PACF, retorno²', markersize=2.5)
    for a in ax.flat:
        a.set_ylim(-0.3, 1.05)
        a.yaxis.set_major_formatter(VIRGULA)
        a.tick_params(labelsize=6.5, length=2)
        a.title.set_fontsize(7.5)
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT, dpi=300)

    banda = 1.96 / np.sqrt(len(r))
    a1, a2 = acf(r, nlags=30, fft=False)[1:], acf(r ** 2, nlags=30, fft=False)[1:]
    print(f'n={len(r)} banda=+-{banda:.3f}')
    print(f'retorno: maior |r| {np.abs(a1).max():.3f}, lags fora da banda {int((np.abs(a1) > banda).sum())} de 30')
    print(f'retorno2: lag 1 {a2[0]:.3f}, lags fora da banda {int((np.abs(a2) > banda).sum())} de 30')
    print('salvo:', OUT)


if __name__ == '__main__':
    main()
