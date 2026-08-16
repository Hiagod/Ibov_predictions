"""Classificacao zero-shot dos comunicados do Copom + features diarias.

Consome data/raw/copom_comunicados.parquet (gerado por coleta_copom.py) e
produz data/processed/features_copom.parquet, com uma linha por pregao do
periodo de estudo (2019-01 a 2024-12).

Classificacao hawkish/dovish (zero-shot, MoritzLaurer/mDeBERTa-v3-base-mnli-xnli):
os comunicados tem 1.219-1.876 tokens (mediana 1.482) -- TODOS excedem o
limite de 512 tokens do modelo, e apenas 2 de 49 abrem com a frase da
decisao ("O Copom decidiu..."), que costuma vir no fim. Por isso o texto e
classificado em DUAS janelas: o inicio (truncamento natural do pipeline) e o
trecho FINAL (ultimos ~1.500 chars, onde a decisao e enunciada). A janela
usada nas features e a de maior correlacao com a direcao REAL da decisao de
Selic (extraida do titulo oficial por regex, independente do modelo) --
criterio de selecao supervisionado por um rotulo auxiliar, sem contato com o
alvo de previsao (retornos), portanto sem vazamento.

Features diarias (forward-fill entre reunioes -- o regime de politica
monetaria persiste ate a proxima decisao):
  copom_hawkish_score  P(hawkish) - P(dovish) da janela selecionada
  direcao_ultima       +1 elevou / -1 reduziu / 0 manteve (ultima decisao)
  dias_desde_copom     pregoes desde o pregao em que a decisao foi absorvida
                       (calculado ANTES do ffill -- a versao original deste
                       script calculava depois, produzindo constante 0)
O nivel absoluto da Selic e salvo mas NAO entra como feature (tendencia
estrutural, mesma logica da exclusao do nivel de volume).

Uso:
    python features_copom.py
"""
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PROJECT = Path(r'D:\tcc_ibov')
sys.path.insert(0, str(PROJECT / 'utils'))
from tcc_utils import build_cutoffs, assign_trading_day

RAW = PROJECT / 'data' / 'raw'
PROC = PROJECT / 'data' / 'processed'
TZ_BR = ZoneInfo('America/Sao_Paulo')
TAIL_CHARS = 1500   # ~430 tokens -- cabe no limite de 512 do mDeBERTa

LABELS = [
    "comunicado com viés de aperto monetário, sinalizando alta de juros",
    "comunicado com viés de afrouxamento monetário, sinalizando queda de juros",
]


def classifica(df: pd.DataFrame) -> pd.DataFrame:
    """Gera copom_hawkish_head (inicio, truncado pelo pipeline) e
    copom_hawkish_tail (ultimos TAIL_CHARS). Idempotente."""
    precisa_head = 'copom_hawkish_head' not in df.columns
    precisa_tail = 'copom_hawkish_tail' not in df.columns
    if not precisa_head and not precisa_tail:
        print("scores head e tail ja presentes -- pulando classificacao")
        return df

    import torch
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline("zero-shot-classification",
                   model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", device=device)

    def score(texto):
        out = clf(texto, candidate_labels=LABELS, multi_label=False)
        probs = dict(zip(out["labels"], out["scores"]))
        return probs[LABELS[0]] - probs[LABELS[1]]

    if precisa_head:
        if 'copom_hawkish_score' in df.columns:
            # migracao: score antigo (unico) era o head
            df['copom_hawkish_head'] = df['copom_hawkish_score']
            print("head migrado do score antigo")
        else:
            df['copom_hawkish_head'] = [score(t) for t in df['texto']]
            print("head classificado")
    if precisa_tail:
        df['copom_hawkish_tail'] = [score(t[-TAIL_CHARS:]) for t in df['texto']]
        print("tail classificado")
    return df


def main():
    df = pd.read_parquet(RAW / 'copom_comunicados.parquet')
    print(f"{len(df)} comunicados")

    df = classifica(df)

    corr_head = np.corrcoef(df['direcao'], df['copom_hawkish_head'])[0, 1]
    corr_tail = np.corrcoef(df['direcao'], df['copom_hawkish_tail'])[0, 1]
    print(f"\nSanidade vs direcao real da Selic: head={corr_head:+.3f} | tail={corr_tail:+.3f}")

    usar = 'copom_hawkish_tail' if corr_tail > corr_head else 'copom_hawkish_head'
    print(f"Janela selecionada: {usar}")
    df['copom_hawkish_score'] = df[usar]
    df.to_parquet(RAW / 'copom_comunicados.parquet')

    # ---------------- bucket para pregao ----------------
    # Comunicado liberado ~18h30 do dia da reuniao (apos o fechamento) -> cai
    # no proximo pregao pela mesma regra de corte (17h) das noticias.
    ts_evento = pd.DatetimeIndex(
        [d.replace(hour=18, minute=30, tzinfo=TZ_BR) for d in df['data_reuniao']]
    ).tz_convert('UTC')
    ts_evento = pd.Series(ts_evento, name='ts_utc')

    precos = pd.read_parquet(PROC / 'bvsp_clean.parquet')
    cutoffs = build_cutoffs(pd.DatetimeIndex(precos.index))
    df['trading_day'] = assign_trading_day(ts_evento, cutoffs).values
    df_ok = df.dropna(subset=['trading_day'])
    print(f"comunicados sem pregao seguinte (fim da serie): {len(df) - len(df_ok)}")

    # ---------------- serie diaria (indice completo com buffer) ----------------
    serie = pd.DataFrame(index=precos.index)
    evento = df_ok.set_index('trading_day')[['copom_hawkish_score', 'direcao', 'selic_pct']]
    serie = serie.join(evento)

    # dias_desde_copom: ANTES do ffill, sobre a coluna de eventos crus
    tem_evento = serie['copom_hawkish_score'].notna()
    grupo = tem_evento.cumsum()
    serie['dias_desde_copom'] = serie.groupby(grupo).cumcount()

    serie['copom_hawkish_score'] = serie['copom_hawkish_score'].ffill()
    serie['selic_pct'] = serie['selic_pct'].ffill()
    serie['direcao_ultima'] = serie['direcao'].ffill().fillna(0)
    serie = serie.drop(columns=['direcao'])

    # recorte do periodo de estudo (buffer descartado, semente 219 ja aplicada)
    estudo = precos.loc['2019-01-01':'2024-12-31']
    serie_estudo = serie.reindex(estudo.index)
    assert serie_estudo['copom_hawkish_score'].notna().all(), \
        "semente (reuniao 219) nao cobriu o inicio do estudo"
    assert serie_estudo['dias_desde_copom'].nunique() > 1, \
        "dias_desde_copom constante -- bug do cumcount pos-ffill regrediu"

    serie_estudo.index.name = 'trading_day'
    out = PROC / 'features_copom.parquet'
    serie_estudo.to_parquet(out)
    print(f"\nSalvo: {out} | shape {serie_estudo.shape}")
    print(f"dias_desde_copom: min {serie_estudo['dias_desde_copom'].min()} | "
          f"max {serie_estudo['dias_desde_copom'].max()} | "
          f"nunique {serie_estudo['dias_desde_copom'].nunique()}")


if __name__ == '__main__':
    main()
