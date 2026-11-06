"""Coleta o CORPO COMPLETO de cada materia (nao so a manchete), a partir das
mesmas 57.611 URLs ja conhecidas em data/raw/infomoney_corpus.parquet.

Diferenca em relacao a coleta_artigos.py (versao manchete): aqui a pagina
e baixada INTEIRA (sem corte em ~200KB), pois o corpo do texto fica bem
depois do <head> no HTML. O corpo e extraido via os paragrafos filhos
diretos de <article class="im-article">, que exclui de forma limpa
propagandas, scripts e iframes (estrutura verificada manualmente em
artigos de 2019 e da versao atual do site -- estavel ao longo dos anos,
pois toda pagina e servida pelo template atual do CMS independente da
data de publicacao original).

Uso:
    python coleta_corpo.py
"""
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT / 'data' / 'raw'
SHARD_DIR = RAW_DIR / 'shards_corpo'
SHARD_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TCC-research"}
SHARD_SIZE = 1000
N_THREADS = 6


def extrai_corpo(html: str) -> tuple[str, int]:
    """Extrai o texto dos paragrafos filhos diretos de article.im-article.
    Retorna (corpo, n_paragrafos). Corpo vazio se a estrutura nao bater
    (ex.: pagina removida/redirecionada) -- fica marcado para revisao."""
    soup = BeautifulSoup(html, 'html.parser')
    art = soup.select_one('article.im-article')
    if art is None:
        return '', 0
    paras = [p.get_text(strip=True) for p in art.find_all('p', recursive=False)]
    paras = [p for p in paras if p]
    return '\n\n'.join(paras), len(paras)


def baixa_um(url: str) -> dict:
    try:
        r = requests.get(url, headers=HEADERS, timeout=45)
        corpo, n_par = extrai_corpo(r.text)
        return {'url': url, 'status': r.status_code, 'error': None,
                'corpo': corpo, 'n_paragrafos': n_par}
    except requests.exceptions.RequestException as e:
        return {'url': url, 'status': -1, 'error': type(e).__name__,
                'corpo': '', 'n_paragrafos': 0}


def coleta_shard(urls: list[str]) -> pd.DataFrame:
    resultados = []
    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        futuros = {ex.submit(baixa_um, u): u for u in urls}
        for fut in as_completed(futuros):
            resultados.append(fut.result())
    return pd.DataFrame(resultados)


def main():
    corpus = pd.read_parquet(RAW_DIR / 'infomoney_corpus.parquet')
    urls = corpus['url'].tolist()
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
        df = coleta_shard(urls[lo:hi])
        df.to_parquet(SHARD_DIR / f"{nome}.parquet")
        ok = ((df['status'] == 200) & (df['n_paragrafos'] > 0)).sum()
        elapsed = time.time() - t0
        print(f"shard {s+1}/{n_shards} | {ok}/{len(df)} OK com corpo | "
              f"{elapsed:.0f}s total | {hi/max(elapsed,1e-9):.1f} URLs/s")

    print("Coleta concluida. Shards em:", SHARD_DIR)


if __name__ == '__main__':
    main()
