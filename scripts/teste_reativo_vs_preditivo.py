"""Teste formal para a Secao 5.5 do TCC ('Sentimento reativo versus
sentimento preditivo'): a correlacao contemporanea entre sentimento e
retorno (mesmo dia) difere estatisticamente da correlacao preditiva
(sentimento do dia t vs. retorno de D+1, o alvo)?

Sao duas correlacoes DEPENDENTES: compartilham a variavel de sentimento, e
os dois retornos comparados (t e t+1) sao adjacentes na mesma serie, logo
correlacionados entre si -- um teste de diferenca entre correlacoes
INDEPENDENTES (ex.: Fisher z simples) nao se aplica aqui.

Metodo: block bootstrap sobre dias (blocos de 10 dias, os mesmos do resto do
artigo, B=20000), porque o sentimento diario e autocorrelacionado; recomputa as
duas correlacoes e a diferenca a cada reamostra. Reporta IC 95% percentil da
diferenca e p-valor bicaudal via proporcao de reamostras que cruzam zero (piso
2/B), com correcao de Holm sobre a familia das 4 variaveis de sentimento
agregado testadas, e a sensibilidade a blocos de 5 e 21 dias.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(r'D:\tcc_ibov')
DIRS = {k: PROJECT / v for k, v in {
    'processed': 'data/processed', 'reports': 'reports',
}.items()}

B = 20000
BLOCO = 10
SEED = 0
SENT_COLS = ['sent_polaridade', 'sent_dispersao', 'sent_pos_frac', 'sent_neg_frac']
NAMES = {'sent_polaridade': 'Polaridade', 'sent_dispersao': 'Dispersão',
         'sent_pos_frac': 'Frac. positiva', 'sent_neg_frac': 'Frac. negativa'}


def holm_correction(pvals):
    order = np.argsort(pvals)
    n = len(pvals)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def indices_bloco(n, bloco):
    rng = np.random.default_rng(SEED)
    nb = int(np.ceil(n / bloco))
    inicios = rng.integers(0, n - bloco + 1, size=(B, nb))
    return (inicios[:, :, None] + np.arange(bloco)).reshape(B, -1)[:, :n]


def corr_linhas(a, b):
    am = a - a.mean(1, keepdims=True)
    bm = b - b.mean(1, keepdims=True)
    return (am * bm).sum(1) / np.sqrt((am ** 2).sum(1) * (bm ** 2).sum(1))


def p_bicaudal(d):
    return max(min(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0), 2 / B)


def main():
    feat_preco = pd.read_parquet(DIRS['processed'] / 'features_preco.parquet')
    feat_sent = pd.read_parquet(DIRS['processed'] / 'features_sentimento_finbert.parquet')
    feat_sent = feat_sent.reindex(feat_preco.index)
    assert feat_sent.notna().all().all(), "sentimento com NaN apos alinhar"

    ret_t = feat_preco['ret_lag0'].values   # retorno contemporaneo (dia t)
    y_t1 = feat_preco['y'].values           # retorno de D+1 (o alvo)
    n = len(feat_preco)
    r_ret_y = np.corrcoef(ret_t, y_t1)[0, 1]
    print(f"n = {n} dias | autocorrelacao lag-1 do retorno = {r_ret_y:+.4f}")

    idx = indices_bloco(n, BLOCO)
    registros = []
    for col in SENT_COLS:
        x = feat_sent[col].values
        r_contemp = np.corrcoef(x, ret_t)[0, 1]
        r_predit = np.corrcoef(x, y_t1)[0, 1]
        delta_obs = r_contemp - r_predit

        deltas_boot = corr_linhas(x[idx], ret_t[idx]) - corr_linhas(x[idx], y_t1[idx])

        ci_lo, ci_hi = np.percentile(deltas_boot, [2.5, 97.5])
        p_two_sided = p_bicaudal(deltas_boot)

        registros.append(dict(
            variavel=col, r_contemporanea=r_contemp, r_preditiva=r_predit,
            delta=delta_obs, ci_lo=ci_lo, ci_hi=ci_hi, p_bootstrap=p_two_sided,
        ))
        print(f"  {NAMES[col]:<16s}: contemp={r_contemp:+.4f} | predit={r_predit:+.4f} | "
              f"delta={delta_obs:+.4f} | IC95%=[{ci_lo:+.4f},{ci_hi:+.4f}] | p={p_two_sided:.4f}")

    df = pd.DataFrame(registros)
    df['p_holm'] = holm_correction(df['p_bootstrap'].values)
    print("\n=== Apos correcao de Holm (familia de 4 variaveis de sentimento) ===")
    print(df[['variavel', 'delta', 'p_bootstrap', 'p_holm']].round(4).to_string(index=False))

    sensibilidade = {}
    for bloco in (5, 21):
        ib = indices_bloco(n, bloco)
        ps = [p_bicaudal(corr_linhas(feat_sent[c].values[ib], ret_t[ib]) - corr_linhas(feat_sent[c].values[ib], y_t1[ib]))
              for c in SENT_COLS]
        sensibilidade[str(bloco)] = {c: {'p': float(p), 'p_holm': float(h)} for c, p, h in zip(SENT_COLS, ps, holm_correction(np.array(ps)))}
        print(f"blocos de {bloco}:", {c: round(v['p_holm'], 4) for c, v in sensibilidade[str(bloco)].items()})

    df.to_parquet(DIRS['reports'] / 'teste_reativo_vs_preditivo.parquet')
    with open(DIRS['reports'] / 'teste_reativo_vs_preditivo.json', 'w', encoding='utf-8') as f:
        json.dump(dict(n=n, r_autocorr_lag1=r_ret_y, B=B, bloco=BLOCO, seed=SEED,
                        resultados=df.to_dict(orient='records'), sensibilidade_bloco=sensibilidade),
                  f, indent=2, ensure_ascii=False)
    print("\nSalvos: reports/teste_reativo_vs_preditivo.{parquet,json}")


if __name__ == '__main__':
    main()
