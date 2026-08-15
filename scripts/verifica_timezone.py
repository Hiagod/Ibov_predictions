"""Verificacao empirica do timezone do JSON-LD do InfoMoney: compara o
datePublished do JSON-LD com o horario exibido na propria pagina (tag
<time datetime=...> perto da manchete) para confirmar se o offset do
JSON-LD e UTC verdadeiro ou um deslocamento fixo (ex.: -03:00 já aplicado).

Achado (documentado no TCC): os dois horarios representam o MESMO instante
-- ex. "18:23+00:00" no JSON-LD == "15:23-03:00" exibido na pagina. Ou seja,
o JSON-LD usa UTC real, e a conversao para America/Sao_Paulo (via zoneinfo,
que trata horario de verao historico) e a forma correta de obter o horario
local, nao uma subtracao fixa de 3h.

Uso:
    python verifica_timezone.py [--n 12]
"""
import argparse
import json
import re
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

PROJECT = Path(r'D:\tcc_ibov')
RAW_DIR = PROJECT / 'data' / 'raw'
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TCC-research"}
TZ_BR = ZoneInfo('America/Sao_Paulo')


def extrai_json_ld_data(html: str) -> str | None:
    for m in re.finditer(
            r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
            html, re.DOTALL):
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        candidatos = data if isinstance(data, list) else \
                     data.get('@graph', [data]) if isinstance(data, dict) else []
        for c in candidatos:
            if isinstance(c, dict) and c.get('@type') in (
                    'NewsArticle', 'Article', 'ReportageNewsArticle'):
                return c.get('datePublished')
    return None


def extrai_time_tag(html: str) -> str | None:
    """Horario exibido na pagina: tag <time datetime="..."> proxima da
    manchete (renderizado pelo InfoMoney no fuso local do leitor)."""
    m = re.search(r'<time[^>]+datetime=["\']([^"\']+)["\']', html)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=12)
    args = ap.parse_args()

    corpus_path = RAW_DIR / 'infomoney_corpus.parquet'
    if not corpus_path.exists():
        raise SystemExit(f"{corpus_path} nao encontrado -- rode "
                          f"consolida_corpus.py primeiro.")
    corpus = pd.read_parquet(corpus_path)
    amostra = corpus.sample(n=min(args.n, len(corpus)), random_state=42)

    linhas = []
    for _, row in amostra.iterrows():
        try:
            r = requests.get(row['url'], headers=HEADERS, timeout=30)
            json_ld_raw = extrai_json_ld_data(r.text)
            time_tag_raw = extrai_time_tag(r.text)
        except requests.exceptions.RequestException as e:
            json_ld_raw = time_tag_raw = None
            print(f"erro em {row['url']}: {type(e).__name__}")

        ts_json_ld = pd.to_datetime(json_ld_raw, utc=True, errors='coerce') \
            if json_ld_raw else pd.NaT
        ts_time_tag = pd.to_datetime(time_tag_raw, utc=True, errors='coerce') \
            if time_tag_raw else pd.NaT

        mesmo_instante = (pd.notna(ts_json_ld) and pd.notna(ts_time_tag)
                           and ts_json_ld == ts_time_tag)

        linhas.append({
            'url': row['url'],
            'json_ld_raw': json_ld_raw,
            'time_tag_raw': time_tag_raw,
            'json_ld_utc': ts_json_ld,
            'time_tag_utc': ts_time_tag,
            'mesmo_instante': mesmo_instante,
        })
        print(f"[{'OK' if mesmo_instante else 'DIVERGE'}] "
              f"json-ld={json_ld_raw} | pagina={time_tag_raw}")
        time.sleep(2)

    df = pd.DataFrame(linhas)
    n_ok = df['mesmo_instante'].sum()
    print(f"\n{n_ok}/{len(df)} confirmam mesmo instante "
          f"(JSON-LD em UTC verdadeiro, nao offset fixo)")
    df.to_parquet(RAW_DIR / 'verificacao_timezone.parquet')
    print(f"Salvo: {RAW_DIR / 'verificacao_timezone.parquet'}")


if __name__ == '__main__':
    main()
