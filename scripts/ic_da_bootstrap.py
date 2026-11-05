"""IC de 95% da DA agrupada por block bootstrap sobre dias e Sharpe equivalente (revisao do
professor, C1): base da frase "os intervalos excluem acuracia direcional acima de ~55%".

Mesmos blocos, B e semente de ic_mcc_bootstrap.py, previsao media entre sementes e folds
concatenados (n = 1030). Sharpe equivalente sem custos, como em figura_poder_estatistico.py:
    Sharpe = (2*DA - 1) * E|r|/sqrt(E[r^2]) * sqrt(252), com a razao medida nos dias de teste.

Saida: reports/ic_da_bootstrap.json.

Uso:
    python ic_da_bootstrap.py
"""
import json

import numpy as np

from ic_mcc_bootstrap import B, BLOCO, REPORTS, SEED, carrega, indices_bloco

OUT = REPORTS / 'ic_da_bootstrap.json'


def main():
    y, modelos, _ = carrega()
    pos = modelos['RandomWalk'][0]
    razao = float(np.abs(y[pos]).mean() / np.sqrt((y[pos] ** 2).mean()))

    def sharpe(da):
        return float((2 * da - 1) * razao * np.sqrt(252))

    linhas = []
    for nome, (pos, pred) in modelos.items():
        acerto = pred == (y[pos] > 0)
        idx = indices_bloco(len(pos), BLOCO, np.random.default_rng(SEED))
        lo, hi = np.percentile(acerto[idx].mean(1), [2.5, 97.5])
        linhas.append({'modelo': nome, 'n_dias': int(len(pos)), 'da': float(acerto.mean()),
                       'ic95_lo': float(lo), 'ic95_hi': float(hi), 'sharpe_no_limite_superior': sharpe(hi)})
        print(f'{nome:22s} DA={acerto.mean():.4f} IC=[{lo:.4f}; {hi:.4f}] Sharpe(lim.sup.)={sharpe(hi):+.2f}')

    maior = max(linhas, key=lambda r: r['ic95_hi'])
    saida = {'B': B, 'bloco': BLOCO, 'seed': SEED, 'razao_abs_rms': razao,
             'maior_limite_superior': maior, 'modelos': linhas}
    OUT.write_text(json.dumps(saida, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"\nrazao E|r|/rms = {razao:.3f}; maior limite superior: {maior['modelo']} "
          f"{maior['ic95_hi']:.4f} (Sharpe {maior['sharpe_no_limite_superior']:.2f})")
    print('salvo em', OUT)


if __name__ == '__main__':
    main()
