RAG with TurboVectorStore
=========================

TurboVectorStore provides retrieval-augmented generation using TurboQuant's
inner-product-preserving quantization for efficient vector search.

Basic Usage
-----------

.. code-block:: python

   import torch
   from turboagent.agents.rag import TurboVectorStore

   store = TurboVectorStore(embedding_dim=768, chunk_size=512)

   # Add documents with pre-computed embeddings
   texts = ["Document chunk 1...", "Document chunk 2..."]
   embeddings = [torch.randn(768), torch.randn(768)]
   store.add_documents(texts, embeddings)

   # Query
   query_emb = torch.randn(768)
   results = store.query(query_emb, top_k=5)

   for r in results:
       print(f"Score: {r['score']:.4f} | {r['text'][:100]}")

Auto-Chunking
-------------

Use ``add_text()`` with an embedding function to automatically chunk and embed:

.. code-block:: python

   def my_embed_fn(chunks):
       # Your embedding model here (e.g., sentence-transformers)
       return [torch.randn(768) for _ in chunks]

   store.add_text(
       "Your long document text here...",
       embed_fn=my_embed_fn,
       metadata={"source": "paper.pdf"},
   )

Combining with TurboAgent
--------------------------

.. code-block:: python

   from turboagent import TurboAgent
   from turboagent.agents.rag import TurboVectorStore

   store = TurboVectorStore(embedding_dim=768)
   # ... populate store ...

   agent = TurboAgent("meta-llama/Llama-3.1-70B-Instruct")

   # Retrieve relevant context
   results = store.query(query_embedding, top_k=3)
   context = "\n".join(r["text"] for r in results)

   response = agent.run(f"Based on this context:\n{context}\n\nAnswer: {question}")
