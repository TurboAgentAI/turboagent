"""
RAG Agent with TurboVectorStore — TurboAgent Example

Demonstrates retrieval-augmented generation using TurboQuant's
inner-product-preserving vector search for efficient document retrieval.

Requirements:
    pip install turboagent-ai[torch,native]
"""

import torch

from turboagent import TurboAgent
from turboagent.agents.rag import TurboVectorStore

# --- Step 1: Create and populate the vector store ---

store = TurboVectorStore(embedding_dim=128, chunk_size=256, chunk_overlap=32)

# Simulate document embeddings (in production, use sentence-transformers or similar)
documents = [
    "TurboQuant achieves 4.9x KV cache compression using PolarQuant with Lloyd-Max "
    "codebooks on Beta-distributed coordinates after random rotation.",

    "The QJL (Quantized Johnson-Lindenstrauss) transform provides 1-bit unbiased "
    "inner product estimation on residual vectors, critical for attention fidelity.",

    "For KV cache quantization, K-cache uses Algorithm 1 (MSE-optimal) while "
    "V-cache uses Algorithm 2 (inner-product-optimal) to preserve attention output quality.",

    "Experimental results show that TurboQuant achieves perfect Needle-in-a-Haystack "
    "retrieval at 4x compression on Llama-3.1-8B with contexts up to 104k tokens.",

    "Consumer GPUs like the RTX 4090 (24GB VRAM) can run 70B models with 128k context "
    "when TurboQuant reduces the KV cache from ~20GB to ~4GB.",
]

# Create simple embeddings (replace with real embedding model in production)
embeddings = [torch.randn(128) for _ in documents]
store.add_documents(documents, embeddings, metadata=[{"id": i} for i in range(len(documents))])

print(f"Vector store: {store.size} documents indexed\n")

# --- Step 2: Query the store ---

query_embedding = embeddings[2] + torch.randn(128) * 0.1  # Similar to doc 2
results = store.query(query_embedding, top_k=3)

print("=== Top 3 Retrieved Documents ===")
for r in results:
    print(f"  Score: {r['score']:.4f} | {r['text'][:80]}...")
print()

# --- Step 3: Use retrieved context with TurboAgent ---

# Build context from retrieved documents
context = "\n\n".join(r["text"] for r in results)

print("=== RAG-Augmented Agent Response ===")
print(f"(In production, this would call the LLM with the retrieved context)")
print(f"\nContext ({len(results)} docs, {len(context)} chars):")
print(context[:300])
