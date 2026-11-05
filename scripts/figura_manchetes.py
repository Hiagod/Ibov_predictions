"""Figura da Secao III-B: manchetes por pregao (mesmo grafico de colabs/03, cel. 2), sobre o corpus corrigido.

colabs/03 cel. 1 nao pode ser reexecutada depois da correcao: regravaria o corpus e quebraria o alinhamento
por posicao com os embeddings. Daqui saem tambem os numeros da legenda.

Uso:
    python figura_manchetes.py   ->  entrega_final/manchetes_por_pregao.png
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / 'entrega_final' / 'manchetes_por_pregao.png'


def main():
    c = pd.read_parquet(PROJECT / 'data' / 'processed' / 'corpus_bucketed.parquet')
    precos = pd.read_parquet(PROJECT / 'data' / 'processed' / 'bvsp_clean.parquet')
    dias = precos.loc['2019-01-01':'2024-12-31'].index
    por_dia = c.groupby('trading_day').size().reindex(dias, fill_value=0)
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(por_dia.index, por_dia.values, lw=0.4, alpha=0.4, color='steelblue')
    ax.plot(por_dia.index, por_dia.rolling(21).mean(), lw=1.8, color='darkblue', label='média móvel 21 pregões')
    ax.set_title('Manchetes InfoMoney (mercados + economia) por pregão — 2019–2024')
    ax.set_ylabel('manchetes/pregão')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT, dpi=130)
    print('salvo:', OUT)
    print('minimo por pregao:', por_dia.min(), '| sem manchete:', (por_dia == 0).sum(), '| pregoes:', len(por_dia))
    print('media anual:', por_dia.groupby(por_dia.index.year).mean().round(1).to_dict())


if __name__ == '__main__':
    main()
