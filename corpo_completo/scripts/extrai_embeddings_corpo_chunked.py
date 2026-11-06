"""Extrai embeddings BERTimbau e FinBERT-PT-BR sobre o CORPO COMPLETO das
materias usando CHUNKING em vez de truncamento simples em 512 tokens
(scripts/extrai_embeddings_corpo.py). Mitiga a limitacao "escopo textual
truncado": artigos longos (mediana 2.584 caracteres, p90 7.540) tinham
apenas sua porcao INICIAL representada no embedding; aqui o texto completo
e dividido em janelas de ate 512 tokens, cada uma embutida separadamente, e
as janelas de uma mesma materia sao agregadas por media -- o embedding final
de cada materia passa a refletir o artigo inteiro, nao so o inicio.

ANTI-VAZAMENTO: mesma logica de extracao ja usada nas duas versoes
anteriores (manchete e corpo truncado) -- feature-level, sem uso do alvo.

Uso:
    python extrai_embeddings_corpo_chunked.py
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
BATCH = 16
CHECKPOINT_EVERY = 200  # em numero de LOTES de chunks (nao de artigos)


def chunca_texto(tokenizer, texto, max_len=MAX_LEN):
    """Tokeniza sem truncar e divide em janelas de ate max_len-2 tokens de
    conteudo (reserva 2 posicoes para [CLS]/[SEP]). Retorna lista de listas
    de input_ids, cada uma ja com os tokens especiais."""
    ids = tokenizer.encode(texto, add_special_tokens=False, truncation=False)
    if not ids:
        return [[tokenizer.cls_token_id, tokenizer.sep_token_id]]
    content_len = max_len - 2
    return [[tokenizer.cls_token_id] + ids[i:i + content_len] + [tokenizer.sep_token_id]
            for i in range(0, len(ids), content_len)]


def prepara_chunks_flat(tokenizer, textos):
    """Achata (artigo, texto) em uma lista plana de chunks + indice de qual
    artigo cada chunk pertence, para poder embutir em lotes eficientes e
    depois reagregar por artigo."""
    flat_chunks, article_of_chunk = [], []
    for art_idx, texto in enumerate(textos):
        for c in chunca_texto(tokenizer, texto):
            flat_chunks.append(c)
            article_of_chunk.append(art_idx)
    return flat_chunks, np.array(article_of_chunk, dtype=np.int64)


@torch.no_grad()
def embed_chunk_batch(model, tokenizer, chunk_id_lists):
    enc = tokenizer.pad({'input_ids': chunk_id_lists}, padding=True, return_tensors='pt').to(DEVICE)
    out = model(**enc).last_hidden_state
    mask = enc['attention_mask'].unsqueeze(-1).float()
    summed = (out * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return (summed / counts).cpu().numpy()


@torch.no_grad()
def finbert_chunk_batch(model, tokenizer, chunk_id_lists):
    enc = tokenizer.pad({'input_ids': chunk_id_lists}, padding=True, return_tensors='pt').to(DEVICE)
    logits = model(**enc).logits
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    hidden = model.bert(**enc).last_hidden_state
    mask = enc['attention_mask'].unsqueeze(-1).float()
    emb = ((hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)).cpu().numpy()
    return probs, emb


def agrega_chunks_por_artigo(valores_chunk, article_of_chunk, n_artigos):
    """Media dos chunks de cada artigo -> um vetor por artigo."""
    dim = valores_chunk.shape[1]
    soma = np.zeros((n_artigos, dim), dtype=np.float64)
    conta = np.zeros(n_artigos, dtype=np.int64)
    np.add.at(soma, article_of_chunk, valores_chunk)
    np.add.at(conta, article_of_chunk, 1)
    conta = np.clip(conta, 1, None)
    return (soma / conta[:, None]).astype(np.float32)


def extrai_bertimbau_chunked(textos):
    ckpt_dir = PROC / 'embeddings_corpo_bertimbau_chunked'
    ckpt_dir.mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
    model = AutoModel.from_pretrained('neuralmind/bert-base-portuguese-cased').to(DEVICE).eval()

    print("  Tokenizando e dividindo em chunks (BERTimbau)...")
    flat_chunks, article_of_chunk = prepara_chunks_flat(tok, textos)
    n_chunks = len(flat_chunks)
    n_artigos = len(textos)
    print(f"  {n_artigos:,} artigos -> {n_chunks:,} chunks "
          f"(media {n_chunks/n_artigos:.2f} chunks/artigo)")

    n_batches = (n_chunks + BATCH - 1) // BATCH
    done = sorted(ckpt_dir.glob('emb_*.npy'))
    start_batch = len(done) * CHECKPOINT_EVERY
    if start_batch:
        print(f"  retomando do lote {start_batch}")

    buf = []
    import time
    t0 = time.time()
    for b in range(start_batch, n_batches):
        lo, hi = b * BATCH, min((b + 1) * BATCH, n_chunks)
        buf.append(embed_chunk_batch(model, tok, flat_chunks[lo:hi]))
        if (b + 1) % CHECKPOINT_EVERY == 0 or b == n_batches - 1:
            shard = np.vstack(buf)
            idx = b // CHECKPOINT_EVERY
            np.save(ckpt_dir / f'emb_{idx:05d}.npy', shard)
            buf = []
            print(f"  lote {b+1}/{n_batches} | ckpt {idx:05d} | {time.time()-t0:.0f}s")

    shards = sorted(ckpt_dir.glob('emb_*.npy'))
    E_chunks = np.vstack([np.load(s) for s in shards])
    assert E_chunks.shape[0] == n_chunks, f"desalinhado: {E_chunks.shape[0]} != {n_chunks}"

    del model
    torch.cuda.empty_cache()

    E_artigo = agrega_chunks_por_artigo(E_chunks, article_of_chunk, n_artigos)
    np.save(PROC / 'emb_bertimbau_corpo_chunked_raw.npy', E_artigo)
    print(f"  Salvo: emb_bertimbau_corpo_chunked_raw.npy {E_artigo.shape}")
    return E_artigo


def extrai_finbert_chunked(textos):
    ckpt_dir = PROC / 'finbert_corpo_chunked_extract'
    ckpt_dir.mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained('lucas-leme/FinBERT-PT-BR')
    model = BertForSequenceClassification.from_pretrained('lucas-leme/FinBERT-PT-BR').to(DEVICE).eval()

    print("  Tokenizando e dividindo em chunks (FinBERT-PT-BR)...")
    flat_chunks, article_of_chunk = prepara_chunks_flat(tok, textos)
    n_chunks = len(flat_chunks)
    n_artigos = len(textos)
    print(f"  {n_artigos:,} artigos -> {n_chunks:,} chunks "
          f"(media {n_chunks/n_artigos:.2f} chunks/artigo)")

    n_batches = (n_chunks + BATCH - 1) // BATCH
    done = sorted(ckpt_dir.glob('probs_*.npy'))
    start_batch = len(done) * CHECKPOINT_EVERY
    if start_batch:
        print(f"  retomando do lote {start_batch}")

    buf_p, buf_e = [], []
    import time
    t0 = time.time()
    for b in range(start_batch, n_batches):
        lo, hi = b * BATCH, min((b + 1) * BATCH, n_chunks)
        p, e = finbert_chunk_batch(model, tok, flat_chunks[lo:hi])
        buf_p.append(p); buf_e.append(e)
        if (b + 1) % CHECKPOINT_EVERY == 0 or b == n_batches - 1:
            idx = b // CHECKPOINT_EVERY
            np.save(ckpt_dir / f'probs_{idx:05d}.npy', np.vstack(buf_p))
            np.save(ckpt_dir / f'emb_{idx:05d}.npy', np.vstack(buf_e))
            buf_p, buf_e = [], []
            print(f"  lote {b+1}/{n_batches} | ckpt {idx:05d} | {time.time()-t0:.0f}s")

    probs_chunks = np.vstack([np.load(s) for s in sorted(ckpt_dir.glob('probs_*.npy'))])
    emb_chunks = np.vstack([np.load(s) for s in sorted(ckpt_dir.glob('emb_*.npy'))])
    assert probs_chunks.shape[0] == n_chunks and emb_chunks.shape[0] == n_chunks

    probs_artigo = agrega_chunks_por_artigo(probs_chunks, article_of_chunk, n_artigos)
    emb_artigo = agrega_chunks_por_artigo(emb_chunks, article_of_chunk, n_artigos)
    np.save(PROC / 'finbert_corpo_chunked_probs_raw.npy', probs_artigo)
    np.save(PROC / 'finbert_corpo_chunked_emb_raw.npy', emb_artigo)
    print(f"  Salvo: finbert_corpo_chunked_probs_raw.npy {probs_artigo.shape}, "
          f"finbert_corpo_chunked_emb_raw.npy {emb_artigo.shape}")
    return probs_artigo, emb_artigo


def main():
    print(f"Device: {DEVICE}")
    corpus = pd.read_parquet(PROC / 'corpus_corpo_bucketed.parquet').reset_index(drop=True)
    textos = corpus['corpo'].tolist()
    print(f"{len(textos):,} corpos a processar (com chunking, sem truncamento)")

    print("\n=== BERTimbau (chunked) ===")
    extrai_bertimbau_chunked(textos)

    print("\n=== FinBERT-PT-BR (chunked) ===")
    extrai_finbert_chunked(textos)

    print("\nExtracao com chunking concluida.")


if __name__ == '__main__':
    main()
