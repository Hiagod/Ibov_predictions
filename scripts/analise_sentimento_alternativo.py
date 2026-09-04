"""Mitigacao do vies de classificacao do FinBERT-PT-BR (Limitacoes): roda um
segundo modelo de sentimento, de linhagem INDEPENDENTE (nao finance-tuned,
multilingue) sobre o mesmo corpus de manchetes, para verificar se a
sobrerrepresentacao de sentimento negativo (53% neg. vs 21% pos.) e um
artefato do FinBERT-PT-BR especificamente (por exemplo, por confundir
vocabulario tecnico financeiro neutro com tom negativo) ou um padrao mais
amplo do proprio corpus de manchetes.

Modelo escolhido: nlptown/bert-base-multilingual-uncased-sentiment (escala
de 1 a 5 estrelas, sem ajuste financeiro, linhagem/equipe diferente do
FinBERT-PT-BR -- ao contrario de turing-usp/FinBertPTBR, que e uma versao
anterior descontinuada do proprio FinBERT-PT-BR e por isso NAO serviria como
comparacao independente).

ANTI-VAZAMENTO: extracao de sentimento e feature-level, sem uso do alvo --
mesma logica ja usada para a extracao original do FinBERT-PT-BR.

Uso:
    python analise_sentimento_alternativo.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT = Path(__file__).resolve().parent.parent
PROC = PROJECT / 'data' / 'processed'
REPORTS = PROJECT / 'reports'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = 'nlptown/bert-base-multilingual-uncased-sentiment'
MAX_LEN = 64
BATCH = 128


@torch.no_grad()
def classifica_lote(model, tokenizer, textos):
    enc = tokenizer(textos, padding=True, truncation=True,
                     max_length=MAX_LEN, return_tensors='pt').to(DEVICE)
    logits = model(**enc).logits
    probs = F.softmax(logits, dim=-1).cpu().numpy()  # (B, 5) -- 1..5 estrelas
    return probs


def main():
    print(f"Device: {DEVICE}")
    corpus = pd.read_parquet(PROC / 'corpus_bucketed.parquet').reset_index(drop=True)
    textos = corpus['headline'].astype(str).tolist()
    print(f"{len(textos):,} manchetes a classificar com {MODEL_NAME}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    n = len(textos)
    n_batches = (n + BATCH - 1) // BATCH
    probs_all = []
    import time
    t0 = time.time()
    for b in range(n_batches):
        lo, hi = b * BATCH, min((b + 1) * BATCH, n)
        probs_all.append(classifica_lote(model, tok, textos[lo:hi]))
        if (b + 1) % 50 == 0 or b == n_batches - 1:
            print(f"  lote {b+1}/{n_batches} | {time.time()-t0:.0f}s")
    probs = np.vstack(probs_all)  # (n, 5), colunas = P(1 estrela)..P(5 estrelas)
    np.save(PROC / 'sentimento_alt_probs_raw.npy', probs)

    # estrelas esperadas (1..5) e polaridade continua analoga a sent_polaridade:
    # media ponderada das estrelas, reescalada para [-1, +1] (3 estrelas = 0)
    estrelas = np.arange(1, 6)
    estrela_esperada = probs @ estrelas
    polaridade_alt = (estrela_esperada - 3) / 2

    # classe discreta: 1-2 estrelas = negativo, 3 = neutro, 4-5 = positivo
    classe_pred = np.argmax(probs, axis=1) + 1  # 1..5
    is_neg = classe_pred <= 2
    is_pos = classe_pred >= 4
    print(f"\nDistribuicao (modelo alternativo, generico, sem ajuste financeiro):")
    print(f"  negativo (1-2 estrelas): {is_neg.mean():.1%}")
    print(f"  neutro   (3 estrelas):   {(~is_neg & ~is_pos).mean():.1%}")
    print(f"  positivo (4-5 estrelas): {is_pos.mean():.1%}")
    print(f"  (referencia FinBERT-PT-BR: 53% negativo, 21% positivo)")

    df_alt = pd.DataFrame({
        'trading_day': corpus['trading_day'].values,
        'polaridade_alt': polaridade_alt,
        'is_pos_alt': is_pos.astype(int),
        'is_neg_alt': is_neg.astype(int),
    })
    g = df_alt.groupby('trading_day')
    feat_alt = pd.DataFrame({
        'sent_polaridade_alt': g['polaridade_alt'].mean(),
        'sent_pos_frac_alt': g['is_pos_alt'].mean(),
        'sent_neg_frac_alt': g['is_neg_alt'].mean(),
    })

    precos = pd.read_parquet(PROC / 'bvsp_clean.parquet')
    estudo = precos.loc['2019-01-01':'2024-12-31']
    feat_alt = feat_alt.reindex(estudo.index)
    print(f"\nCobertura do periodo de estudo: {feat_alt['sent_polaridade_alt'].notna().mean():.2%}")

    out = PROC / 'features_sentimento_alt.parquet'
    feat_alt.to_parquet(out)
    print(f"Salvo: {out}")

    # sanidade -- mesma comparacao contemporaneo vs preditivo ja feita p/ FinBERT-PT-BR
    feat_preco = pd.read_parquet(PROC / 'features_preco.parquet')
    alvo = feat_preco['y']
    ret_hoje = feat_preco['ret_lag0']
    corr_pred = feat_alt['sent_polaridade_alt'].corr(alvo)
    corr_cont = feat_alt['sent_polaridade_alt'].corr(ret_hoje)
    print(f"\nSanidade -- sent_polaridade_alt -> retorno(t+1) [preditivo]: {corr_pred:+.4f}")
    print(f"Sanidade -- sent_polaridade_alt -> retorno(t)   [contemporaneo]: {corr_cont:+.4f}")

    # comparacao direta com FinBERT-PT-BR original
    feat_fb = pd.read_parquet(PROC / 'features_sentimento_finbert.parquet').reindex(estudo.index)
    corr_entre_modelos = feat_alt['sent_polaridade_alt'].corr(feat_fb['sent_polaridade'])
    print(f"\nCorrelacao entre polaridade do modelo alternativo e do FinBERT-PT-BR: "
          f"{corr_entre_modelos:+.4f} (mede se os dois modelos concordam sobre QUAIS dias sao mais "
          f"positivos/negativos, independente do nivel absoluto de cada um)")

    resumo = pd.DataFrame([{
        'modelo': 'nlptown/bert-base-multilingual-uncased-sentiment',
        'frac_negativo': is_neg.mean(), 'frac_positivo': is_pos.mean(),
        'corr_preditiva': corr_pred, 'corr_contemporanea': corr_cont,
        'corr_com_finbert_ptbr': corr_entre_modelos,
    }])
    resumo.to_parquet(REPORTS / 'sentimento_alt_resumo.parquet')
    print(f"\nSalvo: {REPORTS / 'sentimento_alt_resumo.parquet'}")


if __name__ == '__main__':
    main()
