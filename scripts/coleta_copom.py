"""Coleta os comunicados do Copom (decisao de politica monetaria) via API
publica de dados abertos do Banco Central (dadosabertos.bcb.gov.br, dataset
"atas-comunicados-copom"). Cobre o periodo de estudo (2019-2024, reunioes
220 a 267 -- 48 reunioes) com o texto integral do comunicado de cada uma.

O comunicado e liberado no MESMO DIA da decisao (~18h30, apos o fechamento
do pregao), mas a API so retorna a data (nao o horario). Tratamos cada
comunicado como publicado as 18h30 America/Sao_Paulo no dia da reuniao --
consistente com a pratica institucional conhecida do Copom e com o mesmo
corte de 17h ja usado para noticias (assign_trading_day empurra para o
proximo pregao). A ata detalhada (publicada ~6 dias uteis depois, com data
de publicacao propria) NAO e usada aqui -- so o comunicado imediato, que e
o que estava disponivel em tempo real no fechamento do dia da decisao.

Uso:
    python coleta_copom.py
"""
import re
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT = Path(r'D:\tcc_ibov')
RAW_DIR = PROJECT / 'data' / 'raw'
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE = 'https://www.bcb.gov.br/api/servico/sitebcb/copom'
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TCC-research"}

# Reunioes 219 (2018-12-12) a 267 (2024-12-11). A reuniao 219 e anterior ao
# periodo de estudo (2019-01 a 2024-12) e serve de SEMENTE: sem ela, os
# primeiros ~25 pregoes de 2019 (antes da 1a reuniao de 2019, em 06/02)
# ficariam sem nenhum estado de Copom conhecido para o forward-fill.
PRIMEIRA_REUNIAO = 219
ULTIMA_REUNIAO = 267


def limpa_html(texto_html: str) -> str:
    """Remove tags HTML simples do texto do comunicado, mantendo paragrafos."""
    texto = re.sub(r'<p[^>]*>', '\n', texto_html)
    texto = re.sub(r'<[^>]+>', '', texto)
    texto = re.sub(r'&#58;', ':', texto)
    texto = re.sub(r'&[a-z]+;', ' ', texto)
    texto = re.sub(r'\n{2,}', '\n', texto).strip()
    return texto


def busca_reuniao(nro: int, tries: int = 3) -> dict | None:
    url = f'{BASE}/comunicados_detalhes?nro_reuniao={nro}'
    for i in range(1, tries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.encoding = 'utf-8'
            data = r.json()
            item = data['conteudo'][0]
            return {
                'nro_reuniao': item['nro_reuniao'],
                'data_reuniao': item['dataReferencia'],
                'titulo': item['titulo'],
                'texto': limpa_html(item['textoComunicado']),
            }
        except Exception as e:
            print(f"  reuniao {nro} tentativa {i}: erro {type(e).__name__}: {e}")
            time.sleep(3)
    return None


def extrai_selic(titulo: str) -> float | None:
    """Extrai a taxa Selic alvo (%) do titulo, ex.: 'para 14,25%' -> 14.25."""
    m = re.search(r'(\d+,\d+)\s*%', titulo)
    if m:
        return float(m.group(1).replace(',', '.'))
    return None


def extrai_direcao(titulo: str) -> int:
    """+1 eleva (hawkish/aperto), -1 reduz (dovish/afrouxa), 0 mantem."""
    t = titulo.lower()
    if 'eleva' in t:
        return 1
    if 'reduz' in t:
        return -1
    return 0


def main():
    registros = []
    for nro in range(PRIMEIRA_REUNIAO, ULTIMA_REUNIAO + 1):
        item = busca_reuniao(nro)
        if item is None:
            print(f"FALHOU: reuniao {nro}")
            continue
        item['selic_pct'] = extrai_selic(item['titulo'])
        item['direcao'] = extrai_direcao(item['titulo'])
        registros.append(item)
        print(f"reuniao {nro} | {item['data_reuniao']} | dir={item['direcao']:+d} | "
              f"selic={item['selic_pct']} | {len(item['texto'])} chars")
        time.sleep(1)  # gentil com a API publica

    df = pd.DataFrame(registros).sort_values('nro_reuniao').reset_index(drop=True)
    df['data_reuniao'] = pd.to_datetime(df['data_reuniao'])
    assert df['texto'].str.len().gt(100).all(), "algum comunicado veio vazio/curto -- checar"

    out = RAW_DIR / 'copom_comunicados.parquet'
    df.to_parquet(out)
    print(f"\n{len(df)} reunioes salvas em {out}")
    print(f"Periodo: {df['data_reuniao'].min().date()} a {df['data_reuniao'].max().date()}")


if __name__ == '__main__':
    main()
