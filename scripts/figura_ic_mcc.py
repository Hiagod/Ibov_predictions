"""Figura de resultados com incerteza amostral (revisao do professor, C7 e C8):
substitui as barras de desvio entre sementes por intervalos de confianca de 95% por
block bootstrap sobre dias.

(a) R2 fora da amostra contra a previsao zero (passeio aleatorio);
(b) MCC agrupado de cada modelo, em relacao ao acaso (0);
(c) delta-MCC frente ao LSTM so com preco (circulo cheio) e frente ao passeio aleatorio (circulo vazio).
Faixa cinza em (b) e (c): margem de equivalencia +-0,05.
Le reports/ic_mcc_bootstrap.json (gerado por ic_mcc_bootstrap.py).

Uso:
    python figura_ic_mcc.py   ->  entrega_final/fig_ic_mcc.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / 'entrega_final' / 'fig_ic_mcc.png'
dados = json.loads((PROJECT / 'reports' / 'ic_mcc_bootstrap.json').read_text(encoding='utf-8'))
MARGEM = dados['margem']

ROTULO = {
    'RandomWalk': 'Passeio aleatório', 'ARIMA': 'ARIMA', 'ARIMAX': 'ARIMAX',
    'XGBoost': 'XGBoost', 'LSTM-preco': 'LSTM preço',
    'LSTM-BERTimbau-k8': 'LSTM+BERTimbau $k$=8', 'LSTM-BERTimbau-k30': 'LSTM+BERTimbau $k$=30',
    'LSTM-FinBERT-sent': 'LSTM+FinBERT sent.', 'LSTM-FinBERT-emb-k8': 'LSTM+FinBERT emb. $k$=8',
    'LSTM-FinBERT-emb-k30': 'LSTM+FinBERT emb. $k$=30', 'LSTM-Copom': 'LSTM+Copom',
}
ROTULO_CURTO = {k: v.replace('LSTM+', '') for k, v in ROTULO.items()}
CINZA, AZUL, LARANJA = '#7F7F7F', '#4C72B0', '#DD8452'
VIRGULA = FuncFormatter(lambda v, _: f'{v:.1f}'.replace('.', ',').replace('-', '\u2212'))


def cor(nome):
    if nome in ('RandomWalk', 'ARIMA', 'ARIMAX', 'XGBoost'):
        return CINZA
    return AZUL if nome == 'LSTM-preco' else LARANJA


def painel(ax, itens, titulo, xlabel, xlim, faixa=True, rotulos=None):
    """itens: lista de (modelo, ponto, lo, hi); lo/hi None = valor de referencia sem IC."""
    if faixa:
        ax.axvspan(-MARGEM, MARGEM, color='#E4E4E4', zorder=0)
    ax.axvline(0, color='black', lw=0.6, zorder=1)
    for i, (nome, x, lo, hi) in enumerate(itens):
        if lo is None:
            ax.plot(x, i, 'o', ms=3.4, mfc='white', mec=cor(nome), zorder=3)
            continue
        lo_vis = max(lo, xlim[0])
        ax.errorbar(x, i, xerr=[[x - lo_vis], [hi - x]], fmt='o', ms=3.4, color=cor(nome),
                    ecolor=cor(nome), elinewidth=1.0, capsize=2.2, zorder=3)
        if lo < xlim[0]:      # IC cortado pelo eixo: marca com seta
            ax.plot(xlim[0], i, marker='<', ms=3.2, color=cor(nome), clip_on=False, zorder=4)
    ax.set_yticks(range(len(itens)))
    if rotulos is None:
        ax.set_yticklabels([])
    else:
        ax.set_yticklabels([rotulos[n] for n, *_ in itens], fontsize=6.4)
    ax.set_ylim(len(itens) - 0.4, -0.7)
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_formatter(VIRGULA)
    ax.set_xlabel(xlabel, fontsize=6.8)
    ax.set_title(titulo, fontsize=7, loc='left')
    ax.tick_params(axis='x', labelsize=6.4)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='x', lw=0.3, alpha=0.5, zorder=0)


def painel_delta(ax, ordem, vs_lstm, vs_passeio, xlim):
    """Um modelo por linha, nas mesmas linhas de (a) e (b), com os dois deltas lado a lado."""
    ax.axvspan(-MARGEM, MARGEM, color='#E4E4E4', zorder=0)
    ax.axvline(0, color='black', lw=0.6, zorder=1)
    for i, nome in enumerate(ordem):
        if nome in vs_lstm:
            x, lo, hi = vs_lstm[nome]
            ax.errorbar(x, i - 0.17, xerr=[[x - lo], [hi - x]], fmt='o', ms=3.0, color=LARANJA, ecolor=LARANJA,
                        elinewidth=0.9, capsize=1.8, zorder=3)
        if nome in vs_passeio:
            x, lo, hi = vs_passeio[nome]
            ax.errorbar(x, i + 0.17, xerr=[[x - lo], [hi - x]], fmt='o', ms=3.0, mfc='white', mec=CINZA, ecolor=CINZA,
                        elinewidth=0.9, capsize=1.8, zorder=3)
    ax.set_yticks(range(len(ordem)))
    ax.set_yticklabels([])
    ax.set_ylim(len(ordem) - 0.4, -0.7)
    ax.set_xlim(*xlim)
    ax.set_xticks([-0.1, 0, 0.1])
    ax.xaxis.set_major_formatter(VIRGULA)
    ax.set_xlabel('$\\Delta$MCC (IC de 95%)', fontsize=6.8)
    ax.set_title('(c) $\\Delta$MCC', fontsize=7, loc='left')
    ax.tick_params(axis='x', labelsize=6.4)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='x', lw=0.3, alpha=0.5, zorder=0)
    ax.plot([], [], 'o', ms=3.0, color=LARANJA, label='vs. LSTM preço')
    ax.plot([], [], 'o', ms=3.0, mfc='white', mec=CINZA, label='vs. passeio aleatório')
    ax.legend(loc='upper right', fontsize=5.6, frameon=False, handletextpad=0.1, borderaxespad=0.1, labelspacing=0.2,
              bbox_to_anchor=(1.0, 1.13), ncol=2, columnspacing=0.6)


def main():
    plt.rcParams['font.size'] = 6.5
    r2 = [(r['modelo'], r['r2'], None if r['modelo'] == 'RandomWalk' else r['ic95_lo'],
           r['ic95_hi']) for r in dados['r2_vs_rw']]
    mcc = [(r['modelo'], r['mcc_ou_delta'], r['ic95_lo'], r['ic95_hi']) for r in dados['absoluto']]
    ordem = [r['modelo'] for r in dados['absoluto']]
    vs_lstm = {r['modelo']: (r['mcc_ou_delta'], r['ic95_lo'], r['ic95_hi']) for r in dados['vs_lstm_preco']}
    vs_passeio = {r['modelo']: (r['mcc_ou_delta'], r['ic95_lo'], r['ic95_hi']) for r in dados['vs_passeio']}

    fig = plt.figure(figsize=(7.2, 2.1), dpi=300)
    a = fig.add_axes([0.185, 0.185, 0.225, 0.715])
    b = fig.add_axes([0.43, 0.185, 0.225, 0.715])
    c = fig.add_axes([0.675, 0.185, 0.315, 0.715])
    painel(a, r2, '(a) R² fora da amostra', 'R² (IC de 95%)', (-0.5, 0.06),
           faixa=False, rotulos=ROTULO)
    painel(b, mcc, '(b) MCC agrupado', 'MCC agrupado (IC de 95%)', (-0.15, 0.16))
    painel_delta(c, ordem, vs_lstm, vs_passeio, (-0.13, 0.19))
    fig.savefig(OUT, dpi=300)
    print('salvo:', OUT)


if __name__ == '__main__':
    main()
