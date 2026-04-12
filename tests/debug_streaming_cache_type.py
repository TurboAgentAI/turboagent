"""
Debug test: trace what past_key_values object the model returns
after a streaming prefill on Qwen2.5-7B.

The hypothesis: chunk_out.past_key_values after prefill may NOT be
our StreamingDynamicCache, causing decode to run without proper past.

Run:
    python tests/debug_streaming_cache_type.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch

MODEL_PATH = "D:/Projects/BitTorch/models/Qwen2.5-7B"
PROMPT = "The secret code is XK-9. What is the secret code? Answer only the code."

def main():
    from turboagent.backends.torch import TorchEngine
    from turboagent.quant.turboquant import TurboQuantKVCache
    from turboagent.quant.streaming_kv import StreamingDynamicCache

    print(f"Loading Qwen2.5-7B with kv_storage=cpu_streaming...")
    engine = TorchEngine(
        model_id=MODEL_PATH,
        kv_storage="cpu_streaming",
        max_tokens=10,
        temperature=0.0,
        prefill_chunk_size=512,
    )
    print(f"  Layers: {engine._n_layers}  kv_heads: {engine._n_kv_heads}  head_dim: {engine._head_dim}")

    # Show which layers are on which device
    device_counts = {}
    for layer_idx, dev in engine._layer_devices.items():
        key = str(dev)
        device_counts[key] = device_counts.get(key, 0) + 1
    print(f"  Layer device distribution: {device_counts}")

    cache = TurboQuantKVCache(
        bit_mode="turbo3",
        device="cuda",
        head_dim=engine._n_kv_heads * engine._head_dim,
        num_layers=engine._n_layers,
        max_context=4096,
    )

    # ── Monkey-patch generate_chat to intercept current_cache after prefill ──
    original_generate = engine.generate_chat.__func__

    def patched_generate(self, messages, kv_cache, tools=None):
        """
        Patched version that introspects the cache object after each forward pass.
        """
        from typing import Optional, List, Dict, Any, Tuple
        import time

        prompt = self._apply_chat_template(messages)
        input_ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.n_ctx,
        )["input_ids"].to(self.device)

        n_prev, new_input_ids = self._compute_token_diff(input_ids)
        if n_prev > 0 and kv_cache._seq_len > 0:
            n_prev = min(n_prev, kv_cache._seq_len)
            new_input_ids = input_ids[:, n_prev:]

        storage_mode = self._resolve_kv_storage(kv_cache)
        print(f"\n[DEBUG] storage_mode = {storage_mode}")

        past_key_values = None
        if storage_mode == "cpu_streaming":
            past_key_values = self._build_streaming_cache(kv_cache)
            print(f"[DEBUG] built StreamingDynamicCache, id={id(past_key_values)}")
            print(f"[DEBUG] turbo._seq_len at start = {kv_cache._seq_len}")

        n_new = new_input_ids.shape[1]
        chunk_size = self.prefill_chunk_size if self.prefill_chunk_size > 0 else n_new
        print(f"[DEBUG] n_new={n_new}, chunk_size={chunk_size}, n_chunks={(n_new+chunk_size-1)//chunk_size}")

        current_cache = past_key_values

        for chunk_idx, chunk_start in enumerate(range(0, n_new, chunk_size)):
            chunk_end = min(chunk_start + chunk_size, n_new)
            chunk = new_input_ids[:, chunk_start:chunk_end]
            is_last = chunk_end == n_new

            forward_kwargs = dict(
                input_ids=chunk,
                past_key_values=current_cache,
                use_cache=True,
                return_dict=True,
            )

            print(f"[DEBUG] chunk {chunk_idx}: tokens {chunk_start}:{chunk_end}, "
                  f"type(current_cache)={type(current_cache).__name__}")
            print(f"[DEBUG]   turbo._seq_len before chunk = {kv_cache._seq_len}")

            for arg_name in ("logits_to_keep", "num_logits_to_keep"):
                try:
                    with torch.inference_mode():
                        chunk_out = self.model(**forward_kwargs, **{arg_name: 1})
                    break
                except TypeError:
                    continue
            else:
                with torch.inference_mode():
                    chunk_out = self.model(**forward_kwargs)

            print(f"[DEBUG]   type(chunk_out.past_key_values) = {type(chunk_out.past_key_values).__name__}")
            print(f"[DEBUG]   id(chunk_out.past_key_values) = {id(chunk_out.past_key_values)}")
            print(f"[DEBUG]   is same object? {chunk_out.past_key_values is past_key_values}")
            print(f"[DEBUG]   turbo._seq_len after chunk = {kv_cache._seq_len}")

            if hasattr(chunk_out.past_key_values, '_turbo'):
                print(f"[DEBUG]   streaming._turbo._seq_len = {chunk_out.past_key_values._turbo._seq_len}")
            elif hasattr(chunk_out.past_key_values, 'key_cache'):
                n = len(chunk_out.past_key_values.key_cache)
                if n > 0:
                    print(f"[DEBUG]   DynamicCache.key_cache[0].shape = {chunk_out.past_key_values.key_cache[0].shape}")

            current_cache = chunk_out.past_key_values
            if is_last:
                next_token_logits = chunk_out.logits[:, -1, :].clone()

        print(f"\n[DEBUG] After prefill: type(current_cache) = {type(current_cache).__name__}")
        print(f"[DEBUG] After prefill: turbo._seq_len = {kv_cache._seq_len}")
        print(f"[DEBUG] After prefill: is StreamingDynamicCache? {isinstance(current_cache, StreamingDynamicCache)}")

        # Decode loop
        output_ids = []
        for step in range(self.max_new_tokens):
            next_token = self._sample_token(next_token_logits)
            output_ids.append(next_token.item())

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            step_input = next_token.view(1, 1)

            print(f"[DEBUG] Decode step {step}: input_token={next_token.item()!r} "
                  f"({self.tokenizer.decode([next_token.item()])!r}), "
                  f"type(past_kv)={type(current_cache).__name__}, "
                  f"seq_len={kv_cache._seq_len if hasattr(current_cache, '_turbo') else '?'}")

            with torch.inference_mode():
                step_out = self.model(
                    input_ids=step_input,
                    past_key_values=current_cache,
                    use_cache=True,
                )
            current_cache = step_out.past_key_values
            next_token_logits = step_out.logits[:, -1, :]

            print(f"[DEBUG]   after step: type(current_cache) = {type(current_cache).__name__}, "
                  f"turbo._seq_len = {kv_cache._seq_len}")

        if isinstance(current_cache, StreamingDynamicCache):
            actual_cache_seq_len = kv_cache._seq_len
        else:
            actual_cache_seq_len = self._get_cache_seq_len(current_cache)
            self._extract_and_compress_kv(current_cache, kv_cache, actual_cache_seq_len)

        all_ids = torch.cat([input_ids, torch.tensor([output_ids], device=self.device)], dim=1)
        self._prev_input_ids = all_ids

        response_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        metrics = {
            "turn_input_tokens": new_input_ids.shape[1],
            "turn_output_tokens": len(output_ids),
            "total_tokens_cached": actual_cache_seq_len,
            "kv_compressed_mb": kv_cache.memory_usage_gb() * 1000,
        }
        return response_text, metrics

    import types
    engine.generate_chat = types.MethodType(patched_generate, engine)

    # Run test
    messages = [{"role": "user", "content": PROMPT}]
    response, metrics = engine.generate_chat(messages, cache)
    print(f"\nResponse: {response!r}")
    print(f"Tokens cached: {metrics['total_tokens_cached']}")

if __name__ == "__main__":
    main()
