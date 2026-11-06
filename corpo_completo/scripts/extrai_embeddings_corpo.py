"""Extrai embeddings BERTimbau e FinBERT-PT-BR sobre o CORPO COMPLETO das
materias (nao mais so a manchete). Mesma logica de mean-pooling mascarado
das celulas do notebook 03, mas com max_length=512 (limite do modelo, vs.
64 usado para manchetes) e batch menor (textos mais longos consomem mais
memoria de GPU por causa da atencao O(n^2)).

ANTI-VAZAMENTO: extracao de embeddings e feature-level, nao depende de
rotulo/alvo -- sem risco de vazamento aqui (mesma logica ja usada p/
manchetes). O vazamento so poderia ocorrer na normalizacao/PCA feitos
depois, por fold -- inalterado em relacao ao pipeline original.

Uso:
    python extrai_embeddings_corpo.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification

PROJECT = Path(__file__).resolve().parent.parent
PROC = PROJECT / 'data' / 'processed'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LEN = 512
BATCH = 16   # textos longos -- batch menor que o usado p/ manchetes (128)


@torch.no_grad()
def embed_batch(model, tokenizer, texts, max_len=MAX_LEN):
    enc = tokenizer(texts, padding=True, truncation=True,
                     max_length=max_len, return_tensors='pt').to(DEVICE)
    out = model(**enc).last_hidden_state
    mask = enc['attention_mask'].unsqueeze(-1).float()
    summed = (out * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return (summed / counts).cpu().numpy()


def extrai_em_lotes(model, tokenizer, textos, nome, checkpoint_every=20):
    ckpt_dir = PROC / f'embeddings_corpo_{nome}'
    ckpt_dir.mkdir(exist_ok=True)
    n = len(textos)
    n_batches = (n + BATCH - 1) // BATCH
    done = sorted(ckpt_dir.glob('emb_*.npy'))
    start_batch = len(done) * checkpoint_every
    if start_batch:
        print(f"  [{nome}] retomando do lote {start_batch}")

    buf = []
    import time
    t0 = time.time()
    for b in range(start_batch, n_batches):
        lo, hi = b * BATCH, min((b + 1) * BATCH, n)
        buf.append(embed_batch(model, tokenizer, textos[lo:hi]))
        if (b + 1) % checkpoint_every == 0 or b == n_batches - 1:
            shard = np.vstack(buf)
            idx = b // checkpoint_every
            np.save(ckpt_dir / f'emb_{idx:04d}.npy', shard)
            buf = []
            print(f"  [{nome}] lote {b+1}/{n_batches} | ckpt {idx:04d} | "
                  f"{time.time()-t0:.0f}s")

    shards = sorted(ckpt_dir.glob('emb_*.npy'))
    E = np.vstack([np.load(s) for s in shards])
    assert E.shape[0] == n, f"desalinhado: {E.shape[0]} != {n}"
    return E


@torch.no_grad()
def finbert_batch(model, tokenizer, texts, max_len=MAX_LEN):
    """probs (B,3) via forward oficial; emb (B,768) via encoder .bert --
    mesmo caminho oficial usado na extracao de manchetes (celula 12/13 do
    notebook 03), sem reconstrucao manual da cabeca de classificacao."""
    enc = tokenizer(texts, padding=True, truncation=True,
                     max_length=max_len, return_tensors='pt').to(DEVICE)
    logits = model(**enc).logits
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    hidden = model.bert(**enc).last_hidden_state
    mask = enc['attention_mask'].unsqueeze(-1).float()
    emb = ((hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)).cpu().numpy()
    return probs, emb


def extrai_finbert_em_lotes(model, tokenizer, textos, checkpoint_every=20):
    ckpt_dir = PROC / 'finbert_corpo_extract'
    ckpt_dir.mkdir(exist_ok=True)
    n = len(textos)
    n_batches = (n + BATCH - 1) // BATCH
    done = sorted(ckpt_dir.glob('probs_*.npy'))
    start_batch = len(done) * checkpoint_every
    if start_batch:
        print(f"  [finbert] retomando do lote {start_batch}")

    buf_p, buf_e = [], []
    import time
    t0 = time.time()
    for b in range(start_batch, n_batches):
        lo, hi = b * BATCH, min((b + 1) * BATCH, n)
        p, e = finbert_batch(model, tokenizer, textos[lo:hi])
        buf_p.append(p); buf_e.append(e)
        if (b + 1) % checkpoint_every == 0 or b == n_batches - 1:
            idx = b // checkpoint_every
            np.save(ckpt_dir / f'probs_{idx:04d}.npy', np.vstack(buf_p))
            np.save(ckpt_dir / f'emb_{idx:04d}.npy', np.vstack(buf_e))
            buf_p, buf_e = [], []
            print(f"  [finbert] lote {b+1}/{n_batches} | ckpt {idx:04d} | "
                  f"{time.time()-t0:.0f}s")

    probs = np.vstack([np.load(s) for s in sorted(ckpt_dir.glob('probs_*.npy'))])
    emb = np.vstack([np.load(s) for s in sorted(ckpt_dir.glob('emb_*.npy'))])
    assert probs.shape[0] == n and emb.shape[0] == n
    return probs, emb


def main():
    print(f"Device: {DEVICE}")
    corpus = pd.read_parquet(PROC / 'corpus_corpo_bucketed.parquet').reset_index(drop=True)
    textos = corpus['corpo'].tolist()
    print(f"{len(textos):,} corpos a processar")

    lens_chars = [len(t) for t in textos]
    print(f"Tamanho do corpo (chars) -- mediana: {np.median(lens_chars):.0f} | "
          f"p90: {np.percentile(lens_chars, 90):.0f} | max: {max(lens_chars)}")

    # ---------------- BERTimbau ----------------
    print("\n=== BERTimbau ===")
    tok_bt = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
    model_bt = AutoModel.from_pretrained('neuralmind/bert-base-portuguese-cased').to(DEVICE).eval()
    E_bt = extrai_em_lotes(model_bt, tok_bt, textos, 'bertimbau')
    np.save(PROC / 'emb_bertimbau_corpo_raw.npy', E_bt)
    print(f"Salvo: emb_bertimbau_corpo_raw.npy {E_bt.shape}")
    del model_bt
    torch.cuda.empty_cache()

    # ---------------- FinBERT-PT-BR ----------------
    print("\n=== FinBERT-PT-BR ===")
    tok_fb = AutoTokenizer.from_pretrained('lucas-leme/FinBERT-PT-BR')
    model_fb = BertForSequenceClassification.from_pretrained('lucas-leme/FinBERT-PT-BR').to(DEVICE).eval()
    probs_fb, E_fb = extrai_finbert_em_lotes(model_fb, tok_fb, textos)
    np.save(PROC / 'finbert_corpo_probs_raw.npy', probs_fb)
    np.save(PROC / 'finbert_corpo_emb_raw.npy', E_fb)
    print(f"Salvo: finbert_corpo_probs_raw.npy {probs_fb.shape}, "
          f"finbert_corpo_emb_raw.npy {E_fb.shape}")

    print("\nExtracao concluida.")


if __name__ == '__main__':
    main()
