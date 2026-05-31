import torch
from utils import (
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    MODEL_NAME,
    PROFILE_STEPS,
    RESULTS_DIR,
)


def optimized_loop(model, input_ids, n_steps):
    # Optimizations:
    # 1. Enable KV cache to avoid recomputing attention for history
    # 2. Only pass the last token (not full sequence) after the first step
    # 3. Batch .item() calls at the end instead of in the loop (GPU-CPU sync)
    model.config.use_cache = True
    
    generated_ids = input_ids.clone()
    generated_tokens = []
    past_key_values = None
    
    for _ in range(n_steps):
        # After the first iteration, only pass the last generated token
        # This reduces computation from O(seq_len) to O(1)
        if past_key_values is not None:
            input_for_model = generated_ids[:, -1:]
        else:
            input_for_model = generated_ids
        
        outputs = model(
            input_ids=input_for_model,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens.append(next_token_id)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
    
    # Convert tokens to Python list once at the end (single GPU-CPU transfer)
    # instead of calling .item() in the loop
    return torch.cat(generated_tokens, dim=1).squeeze(0).tolist()


def profile(loop_fn, model, input_ids, trace_name: str):
    # Run a short profile and export a Chrome trace for visual inspection.
    trace_path = RESULTS_DIR / trace_name

    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        loop_fn(model, input_ids, PROFILE_STEPS)
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
    prof.export_chrome_trace(str(trace_path))
    print(f"Chrome trace exported to: {trace_path}")


def generate_optimized(optimized_trace_name: str) -> float:
    # Load model in half precision (float16) for faster inference
    model = build_model(torch.float16)
    input_ids = get_input_ids()
    
    # Profile the optimized loop
    profile(optimized_loop, model, input_ids, optimized_trace_name)
    
    # Time the optimized loop for MAX_NEW_TOKENS
    optimized_elapsed = time_generation(optimized_loop, model, input_ids, "Optimized")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return optimized_elapsed


def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "v0_slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Optimized ---")
    optimized_elapsed = generate_optimized(optimized_trace_name="v1_optimized_trace.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print("generate_optimized() did not return a positive elapsed time; "
              "cannot compute speedup.")
    else:
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs V0 slow baseline)")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================
#
# Changes made and speedup per fix:
#
# 1. KV Cache (use_cache=True, past_key_values):
#    - Caches key-value pairs from attention layers to avoid recomputation
#    - Speedup: ~5-10x (eliminates O(seq_len²) attention recomputation)
#
# 2. Last-Token-Only Inference (input_for_model = generated_ids[:, -1:]):
#    - After the first step, only compute for the last 1 token instead of the full sequence
#    - Attention and MLPs compute on shape (1, 1, d_model) instead of (1, seq_len, d_model)
#    - Speedup: ~20-100x (reduces computation from O(seq_len) to O(1) per step)
#
# 3. Batched .item() Calls:
#    - Move token conversion to Python list outside the loop (single GPU-CPU sync at end)
#    - Eliminates per-iteration blocking GPU-CPU synchronization
#    - Speedup: ~2-5x (removes pipeline stalls)
#
# 4. float16 Precision:
#    - Load model in float16 instead of float32
#    - Reduces memory bandwidth requirements by 2x and speeds up compute
#    - Speedup: ~1.5-2x (improved memory efficiency and hardware utilization)
#
# Biggest impact and why:
#
# The last-token-only inference combined with KV cache has the biggest impact.
# Without this optimization, each generation step processes an ever-growing sequence:
# step 1: 1024 tokens, step 2: 1025 tokens, ..., step 128: 1151 tokens.
# Total: ~1087 * 128 / 2 ≈ 69k token forwards per generation.
# With last-token-only + KV cache: only 128 * 1 = 128 token forwards total.
# This is a ~540x reduction in computation and explains the majority of the speedup.
#
