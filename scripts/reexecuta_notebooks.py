"""Reexecuta as celulas dos notebooks que dependem do carimbo corrigido, com o mesmo codigo do artigo.

Fica de fora o que nao depende do texto (XGBoost so com preco, MLP, ACF) e o que nao deve rodar de novo:
extracao de embeddings e colabs/03 cel. 1, que regravaria o corpus e quebraria o alinhamento por posicao.
A saida de cada notebook vai para reports/reexecucao/<notebook>.log. Com --corpo, usa os notebooks de
D:\\tcc_ibov_corpo_completo, cujo PROJECT aponta para aquela pasta.

Uso:
    python reexecuta_notebooks.py 04_baselines
    python reexecuta_notebooks.py 05_lstm 07_ablacoes 06_avaliacao_final
    python reexecuta_notebooks.py --corpo 05_lstm
"""
import json
import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

PROJECT = Path(r'D:\tcc_ibov_corpo_completo') if '--corpo' in sys.argv else Path(__file__).resolve().parent.parent
LOGS = PROJECT / 'reports' / 'reexecucao'
CELULAS = {'04_baselines': [0, 1, 5],
           '05_lstm': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
           '07_ablacoes': [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13],
           '06_avaliacao_final': list(range(10))}


class Tee:
    def __init__(self, *destinos):
        self.destinos = destinos

    def write(self, s):
        for d in self.destinos:
            d.write(s)
            d.flush()

    def flush(self):
        for d in self.destinos:
            d.flush()


def roda(nome):
    d = json.loads((PROJECT / 'colabs' / f'{nome}.ipynb').read_text(encoding='utf-8'))
    ns = {'__name__': '__main__', 'display': print}
    original = sys.stdout
    with open(LOGS / f'{nome}.log', 'w', encoding='utf-8') as log:
        sys.stdout = Tee(original, log)
        try:
            for i in CELULAS[nome]:
                c = d['cells'][i]
                src = '\n'.join(l for l in ''.join(c['source']).splitlines() if not l.lstrip().startswith(('!', '%')))
                print(f'\n########## {nome}[{i}] {src.splitlines()[0][:70]}', flush=True)
                t0 = time.time()
                exec(compile(src, f'{nome}[{i}]', 'exec'), ns)
                print(f'########## {nome}[{i}] concluida em {time.time() - t0:.0f}s', flush=True)
        except Exception:
            traceback.print_exc(file=sys.stdout)
            raise
        finally:
            sys.stdout = original


if __name__ == '__main__':
    LOGS.mkdir(exist_ok=True)
    nomes = [a for a in sys.argv[1:] if not a.startswith('--')]
    for nome in nomes:
        roda(nome)
    print('concluido:', PROJECT, nomes, flush=True)
