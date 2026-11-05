"""Confere os ingredientes que o comentario C5 da revisao aponta para um teste de volatilidade, antes de
registra-lo como trabalho futuro. Nao ajusta modelo de volatilidade: so verifica se as afirmacoes se sustentam.

  1. A maxima e a minima do Stooq medem o pregao? O Open do indice e o fechamento anterior; se entra no High/Low,
     a amplitude incorpora o gap noturno. Referencia: BOVA11, com abertura real.
  2. Parkinson e Garman-Klass sao calculaveis? Garman-Klass usa a abertura; Parkinson, so maxima e minima.
  3. r(componente do PCA, n_noticias) = +0,43 sobrevive a retirar a tendencia? n_noticias triplica na amostra. Mesmo
     PCA de colabs/07 cel. 9 (ajustado por manchete na serie inteira; diagnostico, nao modelo).
  4. Volume de noticias se relaciona com a volatilidade do dia seguinte (Parkinson do BOVA11), descontada a
     volatilidade do dia e a tendencia do volume?

Tendencia retirada pela diferenca frente a media movel dos 63 pregoes anteriores (so dados passados).
IC de 95% por block bootstrap sobre dias (funcoes de diagnostico_relato_evento.py).

Uso:
    python diagnostico_ingredientes_volatilidade.py   ->  reports/diagnostico_ingredientes_volatilidade.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from diagnostico_relato_evento import corr_ic, residuo
from sensibilidade_carimbo import carrega_embeddings

PROJECT = Path(__file__).resolve().parent.parent
PROC = PROJECT / 'data' / 'processed'
OUT = PROJECT / 'reports' / 'diagnostico_ingredientes_volatilidade.json'


def parkinson(df):
    return np.log(df['high'] / df['low']) ** 2 / (4 * np.log(2))


def garman_klass(df):
    return 0.5 * np.log(df['high'] / df['low']) ** 2 - (2 * np.log(2) - 1) * np.log(df['close'] / df['open']) ** 2


def sem_tendencia(s, janela=63):
    return s - s.shift(1).rolling(janela).mean()


def fmt(d):
    return f"{d['r']:+.3f} [{d['lo']:+.3f}; {d['hi']:+.3f}] n={d['n_dias']}"


def main():
    fp = pd.read_parquet(PROC / 'features_preco.parquet')
    p = pd.read_csv(PROJECT / 'data' / 'raw' / 'bvsp_stooq.csv', parse_dates=['Date'], index_col='Date').sort_index()
    p.columns = [c.lower() for c in p.columns]
    b = pd.read_csv(PROJECT / 'data' / 'raw' / 'bova11_yahoo.csv', parse_dates=['date'], index_col='date')
    txt = pd.read_parquet(PROC / 'features_texto_bertimbau.parquet').reindex(fp.index)

    p = p.reindex(fp.index)
    fech_ant = b['close'].shift(1).reindex(fp.index)
    b = b.reindex(fp.index)
    # sem negociacao (volume 0, O=H=L=C): amplitude e abertura falsas; ver retornos_bova
    b.loc[b['volume'].fillna(0) == 0, ['open', 'high', 'low']] = np.nan
    amp_i, amp_b = np.log(p['high'] / p['low']), np.log(b['high'] / b['low'])
    amp_e = np.log(pd.concat([b['high'], fech_ant], axis=1).max(axis=1) / pd.concat([b['low'], fech_ant], axis=1).min(axis=1))
    gap = np.log(b['open'] / fech_ant)
    ok = amp_i.notna() & amp_b.notna() & amp_e.notna() & gap.notna()
    tol = 1e-6
    gr, peq = ok & (gap.abs() > 0.01), ok & (gap.abs() < 0.002)
    pk, gk = parkinson(p), garman_klass(p)
    pk_b, gk_b = parkinson(b), garman_klass(b)
    lv = np.log(pk_b.where(pk_b > 0))
    estimadores = {
        'fracao_open_igual_high_ou_low_indice': float(((np.abs(p['high'] / p['open'] - 1) < tol) | (np.abs(p['low'] / p['open'] - 1) < tol)).mean()),
        'fracao_open_igual_high_ou_low_bova11': float(((np.abs(b['high'] / b['open'] - 1) < tol) | (np.abs(b['low'] / b['open'] - 1) < tol)).mean()),
        'corr_amplitude_indice_vs_pregao_bova11': float(np.corrcoef(amp_i[ok], amp_b[ok])[0, 1]),
        'corr_amplitude_indice_vs_bova11_estendida_ao_fechamento_anterior': float(np.corrcoef(amp_i[ok], amp_e[ok])[0, 1]),
        'dias_gap_maior_1pct': int(gr.sum()),
        'razao_amplitude_indice_sobre_pregao_bova11_gap_grande': float((amp_i / amp_b)[gr].median()),
        'razao_amplitude_indice_sobre_estendida_gap_grande': float((amp_i / amp_e)[gr].median()),
        'razao_amplitude_indice_sobre_pregao_bova11_gap_pequeno': float((amp_i / amp_b)[peq].median()),
        'corr_abs_gap_vs_excesso_de_amplitude': float(np.corrcoef(gap[ok].abs(), (amp_i - amp_b)[ok])[0, 1]),
        'razao_var_parkinson_indice_sobre_bova11': float(pk[ok].mean() / pk_b[ok].mean()),
        'razao_var_gk_sobre_parkinson_indice': float(gk.mean() / pk.mean()),
        'razao_var_gk_sobre_parkinson_bova11': float(gk_b.mean() / pk_b.mean()),
        'acf1_log_parkinson_bova11': float(lv.autocorr(1)),
        'acf5_log_parkinson_bova11': float(lv.autocorr(5)),
        'acf21_log_parkinson_bova11': float(lv.autocorr(21)),
    }

    c = pd.read_parquet(PROC / 'corpus_bucketed.parquet').reset_index(drop=True)
    bt, _ = carrega_embeddings(c)
    pca = PCA(n_components=8, random_state=0).fit(bt)
    pc = pd.DataFrame(pca.transform(bt)).groupby(c['trading_day'].values).mean().reindex(fp.index)
    nn = txt['n_noticias'].astype(float)
    tempo = pd.Series(np.arange(len(nn), dtype=float), index=nn.index)
    nn_st = sem_tendencia(nn)
    r_pc = {i: float(np.corrcoef(pc[i], nn)[0, 1]) for i in range(8)}
    r_st = {i: corr_ic(sem_tendencia(pc[i]).values, nn_st.values) for i in range(8)}
    k = max(r_pc, key=lambda x: abs(r_pc[x]))
    componente = {
        'variancia_explicada': [float(x) for x in pca.explained_variance_ratio_],
        'r_por_componente': {f'comp_{i}': v for i, v in r_pc.items()},
        'r_sem_tendencia_por_componente': {f'comp_{i}': v['r'] for i, v in r_st.items()},
        'componente_mais_correlacionado': f'comp_{k}',
        'comp_vs_n_noticias': corr_ic(pc[k].values, nn.values),
        'comp_vs_tempo': corr_ic(pc[k].values, tempo.values),
        'n_noticias_vs_tempo': corr_ic(nn.values, tempo.values),
        'comp_vs_n_noticias_sem_tendencia': r_st[k],
    }

    # volume de noticias da linha t x volatilidade do pregao (Parkinson do BOVA11): mesmo dia e dia seguinte
    prox = lv.shift(-1)
    lv_st = sem_tendencia(lv)
    prox_st = lv_st.shift(-1)
    volume = {
        'n_noticias_vs_mesmo_dia': corr_ic(nn.values, lv.values),
        'n_noticias_vs_dia_seguinte': corr_ic(nn.values, prox.values),
        'n_noticias_sem_tendencia_vs_mesmo_dia': corr_ic(nn_st.values, lv_st.values),
        'n_noticias_sem_tendencia_vs_dia_seguinte': corr_ic(nn_st.values, prox_st.values),
        'parte_nao_explicada_pela_vol_do_dia_vs_dia_seguinte': corr_ic(residuo(nn_st.values, lv_st.values), prox_st.values),
        'persistencia_log_parkinson_sem_tendencia': corr_ic(lv_st.values, prox_st.values),
    }

    saida = {'estimadores': estimadores, 'componente': componente, 'volume_de_noticias': volume}
    OUT.write_text(json.dumps(saida, indent=1, ensure_ascii=False), encoding='utf-8')
    print('=== estimadores ===')
    for a, v in estimadores.items():
        print(f'{a:66s} {v:.4f}')
    print('\n=== componente do PCA x n_noticias ===')
    print('variancia:', [round(x, 3) for x in componente['variancia_explicada']])
    print('r:', {a: round(v, 3) for a, v in componente['r_por_componente'].items()})
    print('r sem tendencia:', {a: round(v, 3) for a, v in componente['r_sem_tendencia_por_componente'].items()})
    for a, v in componente.items():
        if isinstance(v, dict) and 'r' in v:
            print(f'{a:36s} {fmt(v)}')
    print('\n=== volume de noticias x volatilidade (log Parkinson, BOVA11) ===')
    for a, v in volume.items():
        print(f'{a:52s} {fmt(v)}')
    print('salvo em', OUT)


if __name__ == '__main__':
    main()
