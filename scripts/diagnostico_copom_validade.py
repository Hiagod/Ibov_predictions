"""Validade do classificador do Copom sem o rotulo escrito no texto (revisao do professor, C11).

O r = +0,615 entre o escore e a direcao da decisao e o argumento de validade do instrumento. Mas
a decisao e enunciada no proprio comunicado, e em parte deles dentro da janela classificada
(primeiros 512 tokens). Este diagnostico mede o quanto do r e a leitura do anuncio:

  1. confere a reproducao do escore original (mesmo modelo, rotulos e truncamento);
  2. mede em quantos comunicados o anuncio cai na janela e o r em cada grupo;
  3. retira do texto as frases do anuncio e reclassifica:
       anuncio  -> frases com "decidiu/deliberou ..." sobre a taxa
       estrito  -> tambem frases com verbo de decisao + taxa (ex.: "elevando a taxa basica"),
                   o que pode remover parte do tom legitimo (limite conservador).

Saida: reports/diagnostico_copom_validade.json. Nao altera dados nem features.

Uso:
    python diagnostico_copom_validade.py
"""
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import pipeline

PROJECT = Path(__file__).resolve().parent.parent
ARQ = PROJECT / 'data' / 'raw' / 'copom_comunicados.parquet'
OUT = PROJECT / 'reports' / 'diagnostico_copom_validade.json'
MODELO = 'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli'
LABELS = [
    "comunicado com viés de aperto monetário, sinalizando alta de juros",
    "comunicado com viés de afrouxamento monetário, sinalizando queda de juros",
]
B = 20000
SEED = 0
MAX_TOKENS = 512

SEPARA = re.compile(r'(?<=[.;:!?])\s+|\n+')
ANUNCIO = re.compile(r'\b(?:decid|deliber)\w*\b[^.]{0,200}?(?:selic|juros|taxa)')
ESTRITO = re.compile(r'\b(?:elev|reduz|mant|aument|cort|redu)\w*\b[^.]{0,80}?(?:selic|taxa basica|taxa de juros)'
                     r'|\b(?:selic|taxa basica)\b[^.]{0,60}?\d+,\d+\s*(?:%|ponto)')


def norm(t):
    return ''.join(k for k in unicodedata.normalize('NFD', t) if unicodedata.category(k) != 'Mn').lower()


def frases(texto):
    return [f for f in SEPARA.split(texto.replace('​', '').strip()) if f.strip()]


def sem_frases(texto, *padroes):
    """Texto sem as frases que casam com algum padrao, e a lista das frases retiradas."""
    fica, sai = [], []
    for f in frases(texto):
        (sai if any(p.search(norm(f)) for p in padroes) else fica).append(f)
    return ' '.join(fica), sai


def corr_ic(x, d):
    x, d = np.asarray(x, float), np.asarray(d, float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    xs, ds = x[idx], d[idx]
    xc, dc = xs - xs.mean(1, keepdims=True), ds - ds.mean(1, keepdims=True)
    den = np.sqrt((xc ** 2).sum(1) * (dc ** 2).sum(1))
    with np.errstate(invalid='ignore', divide='ignore'):
        rs = (xc * dc).sum(1) / den
    lo, hi = np.nanpercentile(rs, [2.5, 97.5])
    p = float(min(1.0, 2 * min(np.nanmean(rs <= 0), np.nanmean(rs >= 0))))
    return {'r': float(np.corrcoef(x, d)[0, 1]), 'lo': float(lo), 'hi': float(hi), 'p': p, 'n': int(len(x))}


def main():
    c = pd.read_parquet(ARQ)
    d = c['direcao'].values.astype(float)
    clf = pipeline('zero-shot-classification', model=MODELO, device=0 if torch.cuda.is_available() else -1)
    tok = clf.tokenizer

    def score(texto):
        out = clf(texto, candidate_labels=LABELS, multi_label=False)
        probs = dict(zip(out['labels'], out['scores']))
        return probs[LABELS[0]] - probs[LABELS[1]]

    # 1. reproducao do escore original
    original = np.array([score(t) for t in c['texto']])
    dif = float(np.abs(original - c['copom_hawkish_score'].values).max())
    print(f'reproducao: dif. maxima frente ao escore salvo = {dif:.2e}')
    assert dif < 1e-2, 'classificador nao reproduz o escore salvo'

    # 2. anuncio dentro da janela de 512 tokens
    janela = []
    for t in c['texto']:
        ids = tok(t, truncation=True, max_length=MAX_TOKENS)['input_ids']
        janela.append(bool(ANUNCIO.search(norm(tok.decode(ids, skip_special_tokens=True)))))
    janela = np.array(janela)
    print(f'anuncio na janela: {janela.sum()} de {len(janela)}')

    # 3. reclassificacao sem o anuncio
    sem_a = [sem_frases(t, ANUNCIO) for t in c['texto']]
    sem_e = [sem_frases(t, ANUNCIO, ESTRITO) for t in c['texto']]
    assert all(s for _, s in sem_a), 'comunicado sem frase de anuncio identificada'
    score_a = np.array([score(t) for t, _ in sem_a])
    score_e = np.array([score(t) for t, _ in sem_e])

    saida = {
        'B': B, 'seed': SEED, 'n': len(c),
        'reproducao_dif_max': dif,
        'anuncio_na_janela': int(janela.sum()),
        'r_original': corr_ic(original, d),
        'r_original_com_anuncio_na_janela': corr_ic(original[janela], d[janela]),
        'r_original_sem_anuncio_na_janela': corr_ic(original[~janela], d[~janela]),
        'r_sem_anuncio': corr_ic(score_a, d),
        'r_estrito': corr_ic(score_e, d),
        'r_sem_anuncio_nos_que_tinham_na_janela': corr_ic(score_a[janela], d[janela]),
        'r_estrito_nos_que_tinham_na_janela': corr_ic(score_e[janela], d[janela]),
        'grupo_sem_anuncio_na_janela': {
            'inicio': str(c.loc[~janela, 'data_reuniao'].min().date()),
            'fim': str(c.loc[~janela, 'data_reuniao'].max().date()),
            'direcao': {str(k): int(v) for k, v in c.loc[~janela, 'direcao'].value_counts().items()},
        },
        'frases_retiradas_por_comunicado': {
            'anuncio_media': float(np.mean([len(s) for _, s in sem_a])),
            'estrito_media': float(np.mean([len(s) for _, s in sem_e])),
        },
        'auditoria_anuncio': [{'reuniao': int(n), 'frases': s}
                              for n, (_, s) in zip(c['nro_reuniao'], sem_a)],
    }
    OUT.write_text(json.dumps(saida, ensure_ascii=False, indent=1), encoding='utf-8')
    for k in ('r_original', 'r_original_com_anuncio_na_janela', 'r_original_sem_anuncio_na_janela',
              'r_sem_anuncio', 'r_estrito', 'r_sem_anuncio_nos_que_tinham_na_janela',
              'r_estrito_nos_que_tinham_na_janela'):
        v = saida[k]
        print(f"{k:42s} r={v['r']:+.3f} IC=[{v['lo']:+.3f}; {v['hi']:+.3f}] n={v['n']}")
    print('grupo sem anuncio na janela:', saida['grupo_sem_anuncio_na_janela'])
    print('frases retiradas (media):', saida['frases_retiradas_por_comunicado'])
    print('salvo em', OUT)


if __name__ == '__main__':
    main()
