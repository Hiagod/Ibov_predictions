"""Coleta em massa: baixa cada URL da fila e extrai headline/datePublished do
JSON-LD. Roda em shards de 1000 URLs (1 parquet cada, retomavel) com pool de
threads, pois o servidor e lento (~3-4 req/s sustentaveis nesta config).

Uso:
    python coleta_artigos.py [--fila fila_artigos.parquet] [--threads 5]

Reconstruido a partir do notebook 02_coleta_noticias.ipynb (celula 5, funcao
extrai_meta testada em amostra de 12 URLs) e da descricao do pipeline no
handoff do TCC -- o script original rodou em outra maquina e nao foi versionado.
"""
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

PROJECT = Path(r'D:\tcc_ibov')
RAW_DIR = PROJECT / 'data' / 'raw'
SHARD_DIR = RAW_DIR / 'shards_artigos'
SHARD_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TCC-research"}
SHARD_SIZE = 1000
STREAM_CAP_BYTES = 200_000   # JSON-LD esta no <head>; nao precisa do body inteiro


def extrai_meta(html: str) -> dict:
    """title + datas via JSON-LD (fonte de verdade); fallback p/ meta OpenGraph.
    Identico ao testado em notebook 02 (amostra 12/12 com timezone extraido)."""
    out = {'headline': None, 'datePublished': None, 'dateModified': None, 'via': None}

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
                out.update({
                    'headline': c.get('headline'),
                    'datePublished': c.get('datePublished'),
                    'dateModified': c.get('dateModified'),
                    'via': 'json-ld',
                })
                return out

    mt = re.search(r'<meta[^>]+property=["\']article:published_time["\'][^>]+'
                   r'content=["\']([^"\']+)', html)
    tt = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)', html)
    if mt:
        out.update({'datePublished': mt.group(1),
                    'headline': tt.group(1) if tt else None,
                    'via': 'meta-og'})
    return out


def baixa_um(url: str) -> dict:
    """GET em streaming, corta em STREAM_CAP_BYTES (o JSON-LD fica no <head>).
    status numerico: 200 normal, -1 = erro de rede (timeout, conexao etc.).
    error fica em coluna separada -- misturar status numerico com string de
    erro na mesma coluna quebra a serializacao parquet (coluna de tipo misto)."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=30, stream=True)
        chunks, total = [], 0
        for chunk in r.iter_content(chunk_size=8192):
            chunks.append(chunk)
            total += len(chunk)
            if total >= STREAM_CAP_BYTES:
                break
        r.close()
        html = b''.join(chunks).decode('utf-8', errors='replace')
        meta = extrai_meta(html)
        return {'url': url, 'status': r.status_code, 'error': None, **meta}
    except requests.exceptions.RequestException as e:
        return {'url': url, 'status': -1, 'error': type(e).__name__,
                'headline': None, 'datePublished': None, 'dateModified': None, 'via': None}


def coleta_shard(urls: list[str], n_threads: int) -> pd.DataFrame:
    resultados = []
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futuros = {ex.submit(baixa_um, u): u for u in urls}
        for fut in as_completed(futuros):
            resultados.append(fut.result())
    return pd.DataFrame(resultados)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fila', default='fila_artigos.parquet')
    ap.add_argument('--threads', type=int, default=5)
    args = ap.parse_args()

    fila = pd.read_parquet(RAW_DIR / args.fila)
    urls = fila['url'].tolist()
    n = len(urls)
    n_shards = (n + SHARD_SIZE - 1) // SHARD_SIZE
    print(f"{n:,} URLs em {n_shards} shards de ate {SHARD_SIZE}")

    done = {p.stem for p in SHARD_DIR.glob('shard_*.parquet')}
    t0 = time.time()
    for s in range(n_shards):
        nome = f"shard_{s:04d}"
        if nome in done:
            continue
        lo, hi = s * SHARD_SIZE, min((s + 1) * SHARD_SIZE, n)
        df = coleta_shard(urls[lo:hi], args.threads)
        df.to_parquet(SHARD_DIR / f"{nome}.parquet")
        ok = (df['status'] == 200).sum()
        elapsed = time.time() - t0
        print(f"shard {s+1}/{n_shards} | {ok}/{len(df)} OK | "
              f"{elapsed:.0f}s total | {(hi)/max(elapsed,1e-9):.1f} URLs/s")

    print("Coleta concluida. Shards em:", SHARD_DIR)


if __name__ == '__main__':
    main()
