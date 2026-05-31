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
#
# Biggest impact and why:
#
