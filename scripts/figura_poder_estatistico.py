"""Curva de poder do teste de acuracia direcional (revisao do professor, C4).

Teste binomial bilateral (alfa = 5%), H0: DA = 50%, n = 1030 dias agrupados (os 5
folds de teste concatenados, mesma base do Pesaran-Timmermann do artigo).
Eixo superior: Sharpe equivalente de uma estrategia long/short com acerto
direcional DA, sem custos:  Sharpe = (2*DA - 1) * E|r| / sqrt(E[r^2]) * sqrt(252),
com E|r| e E[r^2] medidos nos mesmos dias de teste. Aproximacao: supoe acerto de
sinal independente da magnitude do retorno.

Uso:
    python figura_poder_estatistico.py   ->  entrega_final/fig_poder.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import binom

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / 'utils'))
from tcc_utils import walk_forward_splits

OUT = PROJECT / 'entrega_final' / 'fig_poder.png'
ALFA = 0.05
AZUL, LARANJA = '#4C72B0', '#DD8452'


def pct(x, nd=1):
    return f'{x:.{nd}f}'.replace('.', ',')


def main():
    pt = pd.read_parquet(PROJECT / 'reports' / 'testes_pt_pooled.parquet')
    n = int(pt['n_dias'].iloc[0])
    da = pt['DA_pooled'].values

    # regiao de rejeicao exata do binomial bilateral
    k_lo = int(binom.ppf(ALFA / 2, n, 0.5))
    while binom.cdf(k_lo, n, 0.5) > ALFA / 2:
        k_lo -= 1
    k_hi = int(binom.isf(ALFA / 2, n, 0.5))
    while binom.sf(k_hi - 1, n, 0.5) > ALFA / 2:
        k_hi += 1

    def poder(p):
        return binom.cdf(k_lo, n, p) + binom.sf(k_hi - 1, n, p)

    p50 = brentq(lambda p: poder(p) - 0.5, 0.505, 0.6)
    p80 = brentq(lambda p: poder(p) - 0.8, 0.505, 0.6)

    # razao E|r| / sqrt(E[r^2]) nos mesmos dias de teste (baselines: embargo 0)
    fp = pd.read_parquet(PROJECT / 'data' / 'processed' / 'features_preco.parquet')
    splits = walk_forward_splits(fp.index, n_splits=5, min_train=252, expanding=True, embargo=0)
    pos_teste = np.concatenate([np.asarray(te, dtype=int) for _, te in splits])
    y = fp['y'].values[pos_teste]
    assert len(y) == n and not np.isnan(y).any(), 'dias de teste nao batem com o n do teste PT'
    razao = np.mean(np.abs(y)) / np.sqrt(np.mean(y ** 2))

    def sharpe(p):
        return (2 * np.asarray(p) - 1) * razao * np.sqrt(252)

    def da_de_sharpe(s):
        return (np.asarray(s) / (razao * np.sqrt(252)) + 1) / 2

    print(f'n={n} | regiao de rejeicao: X<={k_lo} ou X>={k_hi} '
          f'(DA <= {k_lo/n:.4f} ou >= {k_hi/n:.4f})')
    print(f'DA com 50% de poder: {p50:.4f} | com 80% de poder: {p80:.4f}')
    print(f'dias de teste usados na razao: {len(y)} | E|r|/sqrt(E[r^2]) = {razao:.4f}')
    print(f'Sharpe equivalente: 50% de poder -> {sharpe(p50):.2f} | 80% de poder -> {sharpe(p80):.2f}'
          f' | DA 53,0% -> {sharpe(0.53):.2f}')
    print('DA agrupada observada:', {m: round(float(d), 4) for m, d in zip(pt.index, da)})
    print(f'faixa das 11 DAs: {da.min():.4f} a {da.max():.4f}; todas com poder<80%:',
          bool(np.all((da > 1 - p80) & (da < p80))))
    print(f'Poder em 52,3% (maior DA observada): {poder(da.max()):.3f}')

    plt.rcParams['font.size'] = 6.5
    fig, ax = plt.subplots(figsize=(3.45, 2.5), dpi=300)
    xs = np.linspace(0.46, 0.58, 480)
    ax.axvspan(0.46, p80, color='#E4E4E4', zorder=0)
    ax.plot(xs, poder(xs), color=AZUL, lw=1.4, zorder=3)
    for nivel, px in ((0.5, p50), (0.8, p80)):
        ax.plot([0.46, px], [nivel, nivel], color='black', lw=0.5, ls='--', zorder=1)
        ax.plot([px, px], [0, nivel], color='black', lw=0.5, ls='--', zorder=1)
    # 50%: acima da linha tracejada, para nao cruzar a vertical dos 80%
    ax.text(p50 - 0.002, 0.515, f'50% em {pct(100*p50)}%', fontsize=6.2, va='bottom', ha='right')
    ax.text(p80 + 0.0015, 0.725, f'80% em {pct(100*p80)}%', fontsize=6.2, va='top', ha='left')
    ax.text(0.4705, 0.6, 'efeitos que o desenho\nnão distingue de zero\n(poder < 80%)',
            fontsize=6.2, va='bottom', ha='left')
    # DAs observadas (marcas no eixo)
    for i, d in enumerate(da):
        cor = LARANJA if pt.index[i] == 'XGBoost' else '#555555'
        ax.plot([d, d], [0, 0.07], color=cor, lw=1.1, zorder=4)
    ax.plot([], [], color='#555555', lw=1.1, label='DA observada')
    ax.plot([], [], color=LARANJA, lw=1.1, label=f'XGBoost ({pct(100*da.max())}%)')
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.07), fontsize=6.0, frameon=False,
              handlelength=1.2)
    ax.set_xlim(0.46, 0.58)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58])
    ax.set_xticklabels(['46', '48', '50', '52', '54', '56', '58'])
    ax.set_xlabel('DA verdadeira (%)')
    ax.set_ylabel('Probabilidade de rejeitar $H_0$')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.1f}'.replace('.', ',')))
    sec = ax.secondary_xaxis('top', functions=(sharpe, da_de_sharpe))
    sec.set_xlabel('Sharpe equivalente (≈, sem custos)')
    sec.set_xticks([-1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0])
    sec.set_xticklabels(['−1,0', '−0,5', '0', '0,5', '1,0', '1,5', '2,0'])
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT, dpi=300)
    print('salvo:', OUT)


if __name__ == '__main__':
    main()
