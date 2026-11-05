"""Diagnostico do tema 'janela de reacao' (revisao do professor, C10 e C13).

B1  Ortogonalizacao: a parte do sentimento NAO explicada pelo retorno do mesmo
    dia (e, em variante, por todas as variaveis de preco) prediz o retorno de D+1?
B0  Viabilidade do 'Open' do Stooq: ele contem informacao propria (gap
    noturno) ou e quase o fechamento anterior? Qualquer teste abertura->fechamento
    depende disso.

Saida: reports/diagnostico_janela_reacao.json (cada bloco e acrescentado por
uma funcao; rodar o script regenera todos).

Uso:
    python diagnostico_janela_reacao.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
PROC = PROJECT / 'data' / 'processed'
RAW = PROJECT / 'data' / 'raw'
OUT = PROJECT / 'reports' / 'diagnostico_janela_reacao.json'
SENT_COLS = ['sent_polaridade', 'sent_dispersao', 'sent_pos_frac', 'sent_neg_frac']
B = 20000
SEED = 0


def semipartial(x, y, z):
    """corr(x residualizado em z, y) a partir das tres correlacoes simples."""
    r_xy = np.corrcoef(x, y)[0, 1]
    r_xz = np.corrcoef(x, z)[0, 1]
    r_zy = np.corrcoef(z, y)[0, 1]
    return (r_xy - r_xz * r_zy) / np.sqrt(1 - r_xz ** 2), r_xy, r_xz, r_zy


def b1_ortogonalizacao():
    fp = pd.read_parquet(PROC / 'features_preco.parquet')
    fs = pd.read_parquet(PROC / 'features_sentimento_finbert.parquet').reindex(fp.index)
    z = fp['ret_lag0'].values
    y = fp['y'].values
    price_cols = [c for c in fp.columns if c != 'y']
    Z = np.column_stack([np.ones(len(fp)), fp[price_cols].values])
    n = len(fp)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(B, n))
    res = []
    for col in SENT_COLS:
        x = fs[col].values
        sp, r_xy, r_xz, r_zy = semipartial(x, y, z)
        # bootstrap de dias com reposicao (a regressao e re-estimada em cada reamostra)
        boot = np.empty(B)
        for b in range(B):
            ib = idx[b]
            boot[b] = semipartial(x[ib], y[ib], z[ib])[0]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
        # variante: residualiza em todas as 18 variaveis de preco
        beta, *_ = np.linalg.lstsq(Z, x, rcond=None)
        resid_all = x - Z @ beta
        sp_all = np.corrcoef(resid_all, y)[0, 1]
        res.append({
            'variavel': col,
            'r_preditiva_bruta': r_xy,
            'r_contemporanea': r_xz,
            'r_retorno_hoje_vs_amanha': r_zy,
            'cadeia_r_xz_x_r_zy': r_xz * r_zy,
            'semiparcial_ret_lag0': sp,
            'ic95_lo': lo, 'ic95_hi': hi, 'p_bootstrap': p,
            'semiparcial_todas_vars_preco': sp_all,
        })
    return {'n': n, 'B': B, 'seed': SEED, 'resultados': res}


def b0_viabilidade_open():
    df = pd.read_csv(RAW / 'bvsp_stooq.csv', parse_dates=['Date']).set_index('Date')
    df = df.loc['2019-01-01':'2024-12-31']
    gap = np.log(df['Open'] / df['Close'].shift(1)).dropna()
    r_cc = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    r_oc = np.log(df['Close'] / df['Open']).reindex(gap.index)
    return {
        'n': int(len(gap)),
        'gap_std': float(gap.std()),
        'gap_abs_mediano': float(gap.abs().median()),
        'frac_gap_exatamente_zero': float((gap == 0).mean()),
        'frac_abs_gap_lt_5bp': float((gap.abs() < 0.0005).mean()),
        'ret_cc_std': float(r_cc.std()),
        'ret_oc_std': float(r_oc.std()),
        'corr_ret_cc_ret_oc': float(np.corrcoef(r_cc.reindex(gap.index), r_oc)[0, 1]),
        'corr_gap_ret_cc': float(np.corrcoef(gap, r_cc.reindex(gap.index))[0, 1]),
    }


def main():
    out = {'B1_ortogonalizacao': b1_ortogonalizacao(),
           'B0_viabilidade_open': b0_viabilidade_open()}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')

    print('=== B1: sentimento ortogonalizado ao retorno do dia -> retorno de D+1 ===')
    print(f"n={out['B1_ortogonalizacao']['n']}  B={B}")
    for r in out['B1_ortogonalizacao']['resultados']:
        print(f"{r['variavel']:<16} r_pred bruta={r['r_preditiva_bruta']:+.4f} | "
              f"cadeia={r['cadeia_r_xz_x_r_zy']:+.4f} | semiparcial={r['semiparcial_ret_lag0']:+.4f} "
              f"IC95[{r['ic95_lo']:+.3f},{r['ic95_hi']:+.3f}] p={r['p_bootstrap']:.3f} | "
              f"resid(18 vars preco)={r['semiparcial_todas_vars_preco']:+.4f}")
    print('\n=== B0: o Open do Stooq tem informacao propria? ===')
    for k, v in out['B0_viabilidade_open'].items():
        print(f'{k}: {v:.5f}' if isinstance(v, float) else f'{k}: {v}')


if __name__ == '__main__':
    main()
