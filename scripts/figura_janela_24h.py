"""Figura da atribuicao temporal (Secao III-C do artigo): faixa de 24h entre dois
fechamentos da B3 (17h ou 18h, conforme o periodo), com as janelas de retorno (alvo
vs. contemporaneo) de cada linha. O eixo usa as horas do pregao ate 17h.

Os percentuais de manchetes por intervalo do relogio sao calculados do corpus
(data/processed/corpus_bucketed.parquet), nao digitados.

Uso:
    python figura_janela_24h.py   ->  entrega_final/fig_janela_24h.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / 'entrega_final' / 'fig_janela_24h.png'
sys.path.insert(0, str(PROJECT / 'utils'))
from tcc_utils import hora_fechamento_b3

AZUL, LARANJA = '#4C72B0', '#DD8452'   # mesma paleta da Fig. de sentimento
CINZA_ESC, CINZA_CLA = '#C9C9C9', '#E4E4E4'


def fracoes_por_intervalo():
    c = pd.read_parquet(PROJECT / 'data' / 'processed' / 'corpus_bucketed.parquet')
    t = c['ts_local'].dt.tz_localize(None)
    h = t.dt.hour + t.dt.minute / 60
    util = t.dt.dayofweek < 5
    fech = hora_fechamento_b3(t.dt.normalize())
    return {
        'noite': ((h > fech) & util).mean(),
        'madrugada': ((h < 10) & util).mean(),
        'pregao': ((h >= 10) & (h <= fech) & util).mean(),
        'fds': (~util).mean(),
    }


def pct(x):
    return f'{100 * x:.1f}'.replace('.', ',') + '%'


def main():
    f = fracoes_por_intervalo()
    # eixo x: horas desde o fechamento do dia D (desenhado com pregao ate 17h)
    fig, ax = plt.subplots(figsize=(7.2, 1.95), dpi=300)
    ax.set_xlim(-8, 38)
    ax.set_ylim(1.9, 10)
    ax.axis('off')
    plt.rcParams['font.size'] = 6.5

    AX = 2.9   # altura do eixo do relogio
    # --- linha de manchetes ---
    y0, h0 = 6.9, 1.25
    blocos = [
        (-8, 0, 'linha D', CINZA_CLA),
        (0, 7, f'fechamento–24h\n{pct(f["noite"])}', CINZA_ESC),
        (7, 17, f'0h–10h (pré-abertura)\n{pct(f["madrugada"])}', CINZA_ESC),
        (17, 24, f'10h–fech. (pregão)\n{pct(f["pregao"])}', CINZA_ESC),
        (24, 38, 'linha D+2', CINZA_CLA),
    ]
    for x0, x1, txt, cor in blocos:
        ax.add_patch(Rectangle((x0 + 0.08, y0), x1 - x0 - 0.16, h0, fc=cor, ec='none'))
        ax.text((x0 + x1) / 2, y0 + h0 / 2, txt, ha='center', va='center', fontsize=6.3)
    ax.text(12, y0 + h0 + 1.15, 'Manchetes atribuídas à linha D+1 (publicadas entre os dois fechamentos)',
            ha='center', va='bottom', fontsize=6.8, fontweight='bold')

    # --- marcadores de eventos ---
    for x, ls in [(0, '-'), (24, '-'), (17, ':')]:
        ax.plot([x, x], [AX, y0 - 0.05], color='black', lw=0.7, ls=ls, zorder=0)
    ax.plot([1.5, 1.5], [AX, y0 + h0 + 0.1], color='#B03A2E', lw=0.9, ls='--', zorder=0)
    ax.text(-0.35, y0 + h0 + 0.35, 'fechamento', ha='right', va='bottom', fontsize=6.3)
    ax.text(1.85, y0 + h0 + 0.35, 'Copom 18h30', ha='left', va='bottom', fontsize=6.3,
            color='#B03A2E')
    ax.text(24.35, y0 + h0 + 0.35, 'fechamento', ha='left', va='bottom', fontsize=6.3)

    # --- janelas de retorno ---
    def barra(x0, x1, y, h, cor, txt, alpha=1.0, seta=False):
        ax.add_patch(Rectangle((x0 + 0.08, y), x1 - x0 - 0.16, h, fc=cor, ec='none',
                               alpha=alpha))
        ax.text((x0 + x1) / 2, y + h / 2, txt, ha='center', va='center', fontsize=6.3,
                color='white' if alpha == 1.0 else 'black')
        if seta:
            ax.add_patch(FancyArrowPatch((x1 - 0.1, y + h / 2), (x1 + 0.9, y + h / 2),
                                         arrowstyle='-|>', mutation_scale=6, color=cor,
                                         lw=0.8))

    yo, yb, hh = 5.2, 3.7, 1.2
    barra(0, 24, yo, hh, LARANJA,
          'alvo da linha D: retorno D→D+1 (fechamento de D ao fechamento de D+1)')
    barra(0, 17, yb, hh, AZUL, 'contemporâneo da linha D+1: gap noturno')
    barra(17, 24, yb, hh, '#34527F', 'abertura→fechamento')
    barra(24, 38, yo, hh, LARANJA, 'alvo da linha D+1:\nD+1→D+2', alpha=0.45)
    barra(-8, 0, yb, hh, AZUL, 'contemporâneo da\nlinha D: D−1→D', alpha=0.45)

    # --- eixo do relogio ---
    ax.plot([-8, 38], [AX, AX], color='black', lw=0.7)
    for x, lab in [(0, '17h ou 18h (D)'), (7, '0h (D+1)'), (17, '10h (abertura)'), (24, '17h ou 18h (D+1)')]:
        ax.plot([x, x], [AX, AX - 0.25], color='black', lw=0.7)
        ax.text(x, AX - 0.4, lab, ha='center', va='top', fontsize=6.3)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    fig.savefig(OUT, dpi=300)
    print('salvo:', OUT)
    print({k: pct(v) for k, v in f.items()})


if __name__ == '__main__':
    main()
