"""
KV Cache Benchmark — Modal Entry Point
Columbia University HPML Course

Replaces run_benchmark.py for Modal compute.
All methods/, benchmark/, configs/ are identical — only the executor changes.

Usage:
  modal run modal_app.py                            # full benchmark
  modal run modal_app.py --dry-run                  # smoke test (~20 min, ~$1)
  modal run modal_app.py --methods baseline kivi     # specific methods
  modal run modal_app.py --skip-longbench           # skip LongBench eval
  modal run modal_app.py --resume                   # skip completed runs
  modal run modal_app.py --model facebook/opt-6.7b  # swap model

Pull results locally after the run:
  python download_results.py
  python plot_results.py --results results/results.jsonl

Pre-requisites:
  modal secret create huggingface HF_TOKEN=hf_your_token_here
  modal volume create kv-benchmark-results
  modal volume create hf-model-cache
"""

import json
import sys
from pathlib import Path

import modal

# ── MODAL INFRASTRUCTURE ──────────────────────────────────────────────────────

app = modal.App("kv-benchmark")

# Container image: all ML deps + local source code baked in
# Image is rebuilt only when deps or source files change (content-addressed cache)
_base = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.44.2",
        "datasets>=2.18.0",
        "accelerate>=0.27.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
        "rouge-score>=0.1.2",
        "tqdm>=4.65.0",
        "tabulate>=0.9.0",
        "pandas>=2.0.0",
    )
    .add_local_dir(str(_base / "methods"),   "/app/methods")
    .add_local_dir(str(_base / "benchmark"), "/app/benchmark")
    .add_local_dir(str(_base / "configs"),   "/app/configs")
)

# Volumes
# kv-benchmark-results: all JSONL/CSV outputs, survives across runs
# hf-model-cache:       HuggingFace weights cache (~14 GB for LLaMA-2-7B)
#                       downloaded once, reused by every subsequent function call
results_vol    = modal.Volume.from_name("kv-benchmark-results", create_if_missing=True)
model_cache_vol = modal.Volume.from_name("hf-model-cache",      create_if_missing=True)

RESULTS_PATH   = Path("/results")
HF_CACHE_PATH  = Path("/root/.cache/huggingface")


# ── HELPER: shared bootstrap used inside every GPU function ───────────────────

def _bootstrap():
    """Add /app to sys.path so all local imports resolve inside the container."""
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")


def _set_seeds(seed: int = 42):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_method(name: str, cfg: dict):
    from methods.baseline       import BaselineMethod
    from methods.kivi_quant     import KIVIMethod
    from methods.snapkv_eviction import SnapKVMethod
    from methods.topk_selection  import TopKMethod
    from methods.xkv_svd        import XKVMethod

    if name == "baseline":
        return BaselineMethod()
    elif name == "kivi":
        return KIVIMethod(
            bits=cfg.get("bits", 4),
            residual_length=cfg.get("residual_length", 128),
        )
    elif name == "xkv":
        return XKVMethod(
            rank_k=cfg.get("rank_k", 128),
            recompute_interval=cfg.get("recompute_interval", 50),
        )
    elif name == "snapkv":
        return SnapKVMethod(
            budget_ratio=cfg.get("budget_ratio", 0.4),
            sink_size=cfg.get("sink_size", 4),
            observation_window=cfg.get("observation_window", 32),
        )
    elif name == "topk":
        return TopKMethod(
            K=cfg.get("K", 512),
            refresh_interval=cfg.get("refresh_interval", 50),
        )
    else:
        raise ValueError(f"Unknown method: {name}")


def _load_model(model_name: str, device: str = "cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[modal] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    print(f"[modal] Model loaded on {device}.")
    return model, tokenizer


# ── FUNCTION 1: prepare_prompts ───────────────────────────────────────────────
# CPU-only. Tokenises wiki text to build prompt strings.
# Returns serialisable dicts (strings only — no tensors).

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={str(HF_CACHE_PATH): model_cache_vol},
    timeout=900,   # 15 min — dataset download can be slow first time
    memory=8192,
)
def prepare_prompts(config: dict, seed: int = 42) -> dict:
    """
    Load datasets and build all prompt dicts.
    Returns:
      {
        "synthetic":  [ {prompt, seq_len, prompt_id, prompt_type, task}, ... ],
        "wikitext":   [ raw_text_str, ... ],
        "longbench":  { task_name: [ {prompt, seq_len, prompt_id, reference, task}, ... ] }
      }
    """
    _bootstrap()
    _set_seeds(seed)

    import warnings
    from transformers import AutoTokenizer

    model_name = config["model"]["name"]
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Synthetic ─────────────────────────────────────────────────────────────
    synthetic = []
    if config["datasets"]["synthetic"]["enabled"]:
        from datasets import load_dataset

        wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        target_lengths = config["datasets"]["sequence_lengths"]
        n_per_length   = config["datasets"]["synthetic"]["n_per_length"]

        texts = [
            ex["text"] for ex in wiki
            if ex["text"].strip() and not ex["text"].strip().startswith("=")
        ]

        for target_len in target_lengths:
            collected, passage_start = 0, 0
            while collected < n_per_length and passage_start < len(texts):
                combined = ""
                idx = passage_start
                while idx < len(texts):
                    combined += " " + texts[idx]
                    idx += 1
                    ids = tokenizer(combined, return_tensors="pt",
                                    truncation=False)["input_ids"][0]
                    if len(ids) >= target_len:
                        break

                ids = tokenizer(combined, return_tensors="pt",
                                truncation=False)["input_ids"][0]
                if len(ids) < target_len:
                    break

                truncated = tokenizer.decode(ids[:target_len], skip_special_tokens=True)
                prompt    = f"Summarize the following text:\n\n{truncated}\n\nSummary:"
                synthetic.append({
                    "prompt":      prompt,
                    "seq_len":     target_len,
                    "prompt_id":   f"synthetic_{target_len}_{collected}",
                    "prompt_type": "synthetic",
                    "task":        "n/a",
                })
                collected   += 1
                passage_start = idx

        print(f"[modal] {len(synthetic)} synthetic prompts across {target_lengths}")

    # ── WikiText (perplexity) ──────────────────────────────────────────────────
    wikitext_texts = []
    if config["datasets"]["wikitext"]["enabled"]:
        from datasets import load_dataset
        n = config["datasets"]["wikitext"]["n_examples"]
        wt = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        for ex in wt:
            if ex["text"].strip() and not ex["text"].strip().startswith("="):
                wikitext_texts.append(ex["text"].strip())
                if len(wikitext_texts) >= n:
                    break
        print(f"[modal] {len(wikitext_texts)} wikitext examples for PPL")

    # ── LongBench ─────────────────────────────────────────────────────────────
    longbench = {}
    if config["datasets"]["longbench"]["enabled"]:
        from datasets import load_dataset
        tasks     = config["datasets"]["longbench"]["tasks"]
        n_per_task = config["datasets"]["longbench"]["n_per_task"]

        for task in tasks:
            try:
                ds = load_dataset("THUDM/LongBench", task, split="test",
                                  trust_remote_code=True)
                examples = []
                for i, ex in enumerate(ds):
                    if i >= n_per_task:
                        break
                    context  = ex.get("context", "")
                    question = ex.get("input", "")
                    answers  = ex.get("answers", [""])
                    ref = answers[0] if isinstance(answers, list) and answers else str(answers)

                    prompt = f"{context}\n\nQuestion: {question}\nAnswer:"[:4096]
                    examples.append({
                        "prompt":      prompt,
                        "seq_len":     min(len(prompt.split()), 4096),
                        "prompt_id":   f"lb_{task}_{i}",
                        "prompt_type": "longbench",
                        "task":        task,
                        "reference":   ref,
                    })
                longbench[task] = examples
                print(f"[modal]   LongBench {task}: {len(examples)} examples")
            except Exception as e:
                warnings.warn(f"LongBench task '{task}' failed: {e}")

    model_cache_vol.commit()
    return {
        "synthetic":  synthetic,
        "wikitext":   wikitext_texts,
        "longbench":  longbench,
    }


# ── FUNCTION 2: run_baseline ──────────────────────────────────────────────────
# Runs baseline on all prompts to get kv_cache_mb denominators.

@app.function(
    image=image,
    gpu="A100-80GB",
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={
        str(RESULTS_PATH):  results_vol,
        str(HF_CACHE_PATH): model_cache_vol,
    },
    timeout=3600,
    memory=65536,
)
def run_baseline(
    prompts: list,
    config: dict,
    max_new_tokens: int = 200,
    seed: int = 42,
) -> dict:
    """
    Run baseline on every prompt.
    Returns: { prompt_id -> kv_cache_mb }
    """
    _bootstrap()
    _set_seeds(seed)

    from benchmark.metrics import MetricsLogger
    from benchmark.runner  import generate_with_method
    from methods.baseline  import BaselineMethod

    model, tokenizer = _load_model(config["model"]["name"])
    method   = BaselineMethod()
    # Write to a unique file so parallel containers don't clobber each other
    logger   = MetricsLogger(RESULTS_PATH, prefix="baseline")
    kv_cache = {}   # prompt_id -> kv_mb
    baseline_results = []

    for p in prompts:
        _set_seeds(seed)
        try:
            _, metrics = generate_with_method(
                model, tokenizer, method,
                prompt=p["prompt"],
                max_new_tokens=max_new_tokens,
                device="cuda",
            )
            kv_cache[p["prompt_id"]] = metrics["kv_cache_mb"]
            rec = logger.log(
                method="baseline", config={},
                prompt_type=p["prompt_type"],
                run_metrics=metrics,
                baseline_kv_mb=None,
                task=p.get("task", "n/a"),
            )
            baseline_results.append(rec)
            print(
                f"  baseline | seq={p['seq_len']:5d} | "
                f"mem={metrics['peak_memory_gb']:.2f}GB | "
                f"kv={metrics['kv_cache_mb']:.2f}MB | "
                f"tps={metrics['throughput_tps']:.1f}"
            )
        except Exception as e:
            import traceback
            print(f"  [ERROR] baseline {p['prompt_id']}: {e}\n{traceback.format_exc()}")
            kv_cache[p["prompt_id"]] = 1.0   # fallback denominator

    results_vol.commit()
    return {"kv_cache": kv_cache, "results": baseline_results}


# ── FUNCTION 3: run_method ────────────────────────────────────────────────────
# One GPU container per (method, config).  All run in parallel from main().

@app.function(
    image=image,
    gpu="A100-80GB",
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={
        str(RESULTS_PATH):  results_vol,
        str(HF_CACHE_PATH): model_cache_vol,
    },
    timeout=3600,
    memory=65536,
)
def run_method(
    method_name: str,
    method_cfg: dict,
    prompts: list,
    config: dict,
    max_new_tokens: int = 200,
    seed: int = 42,
    baseline_kv_cache: dict = None,
) -> list:
    """
    Load model once; run every prompt for one (method, config).
    Writes to /results/results.jsonl incrementally (crash-safe).
    Returns list of result dicts.
    """
    _bootstrap()
    _set_seeds(seed)

    import traceback
    from benchmark.metrics  import MetricsLogger
    from benchmark.runner   import generate_with_method

    model, tokenizer = _load_model(config["model"]["name"])
    method  = _build_method(method_name, method_cfg)
    # Unique file per (method, config) — avoids volume write races
    import hashlib
    cfg_str = json.dumps(method_cfg, sort_keys=True)
    cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()[:8]
    logger  = MetricsLogger(RESULTS_PATH, prefix=f"{method_name}_{cfg_hash}")
    results = []
    print(f"\n[modal] Starting {method_name} {cfg_str}  ({len(prompts)} prompts)")

    for p in prompts:
        _set_seeds(seed)
        try:
            _, metrics = generate_with_method(
                model, tokenizer, method,
                prompt=p["prompt"],
                max_new_tokens=max_new_tokens,
                device="cuda",
            )
            baseline_mb = (baseline_kv_cache or {}).get(p["prompt_id"])
            rec = logger.log(
                method=method_name,
                config=method_cfg,
                prompt_type=p["prompt_type"],
                run_metrics=metrics,
                baseline_kv_mb=baseline_mb,
                task=p.get("task", "n/a"),
            )
            results.append(rec)
            print(
                f"  {method_name:10s} cfg={cfg_str:35s} | "
                f"seq={p['seq_len']:5d} | "
                f"mem={metrics['peak_memory_gb']:.2f}GB | "
                f"kv={metrics['kv_cache_mb']:.2f}MB | "
                f"ratio={rec['compression_ratio']:.2f}x | "
                f"tps={metrics['throughput_tps']:.1f}"
            )
        except Exception as e:
            import warnings
            print(f"  [ERROR] {method_name} {p['prompt_id']}: {e}\n{traceback.format_exc()}")
            warnings.warn(
                f"Skipping prompt {p['prompt_id']} for {method_name}: method crashed. "
                f"No fallback result recorded."
            )

    results_vol.commit()
    print(f"[modal] Done: {method_name} {cfg_str}")
    return results


# ── FUNCTION 4: run_perplexity (per-method) ───────────────────────────────────

@app.function(
    image=image,
    gpu="A100-80GB",
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={
        str(RESULTS_PATH):  results_vol,
        str(HF_CACHE_PATH): model_cache_vol,
    },
    timeout=3600,
    memory=65536,
)
def run_perplexity(
    method_name: str,
    method_cfg: dict,
    wikitext_texts: list,
    config: dict,
    seed: int = 42,
) -> dict:
    """
    Compute per-method perplexity on WikiText-103.

    All methods (including baseline) use the same split-text protocol:
    prefill first half → apply method → evaluate second half through
    modified KV cache. This ensures PPL values are directly comparable.
    """
    _bootstrap()
    _set_seeds(seed)

    model, tokenizer = _load_model(config["model"]["name"])

    method = _build_method(method_name, method_cfg)
    cfg_str = json.dumps(method_cfg, sort_keys=True)

    # All methods (including baseline) use the same split-text protocol
    # so that perplexity values are directly comparable.
    from benchmark.runner import compute_method_perplexity
    ppl = compute_method_perplexity(
        model, tokenizer, method, wikitext_texts,
        device="cuda", max_length=512,
    )

    print(f"[modal] PPL {method_name} {cfg_str}: {ppl:.3f}")
    return {"method": method_name, "config": method_cfg, "perplexity": ppl}


# ── LOCAL ENTRY POINT ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    dry_run:        bool = False,
    methods:        str  = "",        # comma-separated, e.g. "baseline,kivi"
    skip_longbench: bool = False,
    resume:         bool = False,
    model:          str  = "",        # override model name
    yes:            bool = False,     # skip cost confirmation
):
    """
    Orchestrates the full benchmark:
      1. prepare_prompts  — CPU, one call
      2. run_baseline     — GPU, one call, establishes kv denominators
      3. run_method × N   — GPU, ALL method configs run in PARALLEL
      4. run_perplexity   — GPU, one call
    Results land in modal volume 'kv-benchmark-results'.
    Pull locally with:  python download_results.py
    """
    import yaml
    from tabulate import tabulate

    # ── Load config ───────────────────────────────────────────────────────────
    config_path = Path(__file__).parent / "configs" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if model:
        config["model"]["name"] = model
    if skip_longbench:
        config["datasets"]["longbench"]["enabled"] = False

    if dry_run:
        config["datasets"]["synthetic"]["n_per_length"] = 1
        config["datasets"]["sequence_lengths"]          = [512, 1024]
        config["datasets"]["wikitext"]["n_examples"]    = 5
        config["datasets"]["longbench"]["n_per_task"]   = 2
        max_new_tokens = 20
        print("\n[DRY RUN] Reduced dataset sizes, max_new_tokens=20\n")
    else:
        max_new_tokens = config["model"]["max_new_tokens"]

    seed = config.get("seed", 42)

    # ── Active method configs ─────────────────────────────────────────────────
    requested = set(m.strip() for m in methods.split(",") if m.strip()) if methods else None
    method_order = ["baseline", "kivi", "xkv", "snapkv", "topk"]
    active = []   # [(name, cfg), ...]

    for mname in method_order:
        mcfg = config["methods"].get(mname, {})
        if not mcfg.get("enabled", True):
            continue
        if requested and mname not in requested:
            continue
        if mname == "baseline":
            active.append(("baseline", {}))
        else:
            for cfg in mcfg.get("configs", [{}]):
                active.append((mname, cfg))

    non_baseline = [(n, c) for n, c in active if n != "baseline"]

    # ── Cost estimate ─────────────────────────────────────────────────────────
    n_seq    = len(config["datasets"]["sequence_lengths"])
    n_prompts_per_len = config["datasets"]["synthetic"]["n_per_length"]
    total_prompts = n_seq * n_prompts_per_len

    # A100-80GB: ~60 tok/s decode, 200 tokens ≈ 3-4s/prompt + overhead
    # Parallel: wall time = slowest method, not sum
    est_min_single = total_prompts * 2        # ~2 min per prompt per method
    est_min_wall   = est_min_single + 15      # +15 for model load + baseline serial
    est_cost_usd   = (est_min_wall / 60) * 3.67  # A100 on Modal ~$3.67/hr

    print(f"\nModel:        {config['model']['name']}")
    print(f"GPU:          A100-80GB (Modal)")
    print(f"Methods:      {len(active)} configurations")
    print(f"Seq lengths:  {config['datasets']['sequence_lengths']}")
    print(f"Prompts:      {total_prompts} synthetic")
    print(f"Parallelism:  {len(non_baseline)} method containers run simultaneously")
    print(f"\nEstimated wall time: ~{est_min_wall} min  (~${est_cost_usd:.2f})")
    print("(Methods run in parallel — wall time ≈ slowest single method, not sum)")

    if not yes and not dry_run:
        ans = input("\nContinue? [y/n]: ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return

    # ── PHASE 1: Prepare prompts (CPU) ────────────────────────────────────────
    print("\n── Phase 1: Preparing prompts ─────────────────────────────────────")
    prompt_data = prepare_prompts.remote(config, seed)

    synthetic_prompts = prompt_data["synthetic"]
    wikitext_texts    = prompt_data["wikitext"]
    longbench_data    = prompt_data["longbench"]

    all_prompts = list(synthetic_prompts)
    for task_name, examples in longbench_data.items():
        all_prompts.extend(examples)

    print(f"  {len(synthetic_prompts)} synthetic + "
          f"{sum(len(v) for v in longbench_data.values())} LongBench prompts ready")

    # ── PHASE 2: Baseline (serial, establishes kv denominators) ──────────────
    print("\n── Phase 2: Running baseline ──────────────────────────────────────")
    baseline_data = run_baseline.remote(
        all_prompts, config, max_new_tokens, seed
    )
    baseline_kv_cache = baseline_data["kv_cache"]
    baseline_results  = baseline_data["results"]
    print(f"  Baseline complete. {len(baseline_kv_cache)} kv_cache_mb values captured.")

    # ── PHASE 3: All methods in PARALLEL ──────────────────────────────────────
    print(f"\n── Phase 3: Launching {len(non_baseline)} method configs in parallel ──")

    handles = {}
    for method_name, method_cfg in non_baseline:
        cfg_str = json.dumps(method_cfg, sort_keys=True)
        print(f"  Spawning: {method_name:10s} {cfg_str}")
        handle = run_method.spawn(
            method_name, method_cfg,
            all_prompts, config,
            max_new_tokens, seed,
            baseline_kv_cache,
        )
        handles[(method_name, cfg_str)] = handle

    # Collect results (blocks until all done)
    all_results = list(baseline_results)   # start with baseline
    failed = []
    for (method_name, cfg_str), handle in handles.items():
        try:
            results = handle.get()
            all_results.extend(results)
            print(f"  ✓ {method_name:10s} {cfg_str}  ({len(results)} results)")
        except Exception as e:
            print(f"  ✗ {method_name:10s} {cfg_str}  FAILED: {e}")
            failed.append((method_name, cfg_str))

    # ── PHASE 4: Per-method perplexity (all in parallel) ────────────────────
    ppl_results = {}
    if wikitext_texts and config["datasets"]["wikitext"]["enabled"]:
        print("\n── Phase 4: Per-method perplexity (all in parallel) ──────────────")

        ppl_handles = {}
        for method_name, method_cfg in active:
            cfg_str = json.dumps(method_cfg, sort_keys=True)
            key = f"{method_name} {cfg_str}"
            print(f"  Spawning PPL: {key}")
            h = run_perplexity.spawn(
                method_name, method_cfg,
                wikitext_texts[:20],  # use first 20 for speed
                config, seed,
            )
            ppl_handles[key] = h

        for key, h in ppl_handles.items():
            try:
                res = h.get()
                ppl_results[key] = res["perplexity"]
                print(f"  ✓ {key}  PPL={res['perplexity']:.3f}")
                # Add PPL record to all_results
                all_results.append({
                    "method": res["method"],
                    "config": res["config"],
                    "prompt_type": "wikitext",
                    "task": "n/a",
                    "seq_len": 0,
                    "peak_memory_gb": 0,
                    "kv_cache_mb": 0,
                    "compression_ratio": 1.0,
                    "ttft_ms": 0,
                    "throughput_tps": 0,
                    "per_token_latency_ms": 0,
                    "perplexity": round(res["perplexity"], 4),
                    "task_score": None,
                    "longbench_score": None,
                    "tokens_generated": 0,
                    "timestamp": "",
                })
            except Exception as e:
                print(f"  ✗ {key}  FAILED: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("BENCHMARK COMPLETE")
    print("=" * 65)

    if all_results:
        from collections import defaultdict
        by_method = defaultdict(list)
        for r in all_results:
            if r.get("prompt_type") == "synthetic":
                by_method[r["method"]].append(r)

        table = []
        for mname, rows in sorted(by_method.items()):
            avg_kv   = sum(r["kv_cache_mb"]      for r in rows) / len(rows)
            avg_ratio = sum(r["compression_ratio"] for r in rows) / len(rows)
            avg_tps  = sum(r["throughput_tps"]    for r in rows) / len(rows)
            # Find best PPL for this method
            method_ppls = [r["perplexity"] for r in all_results
                          if r["method"] == mname and r.get("perplexity")]
            ppl_str = f"{min(method_ppls):.2f}" if method_ppls else "—"
            table.append([mname, f"{avg_kv:.1f}", f"{avg_ratio:.2f}x",
                         f"{avg_tps:.1f}", ppl_str])

        print(tabulate(table,
                       headers=["Method", "Avg KV MB", "Compression",
                                "Avg TPS", "PPL"],
                       tablefmt="grid"))

    if failed:
        print(f"\n⚠  {len(failed)} configs failed: {failed}")

    # ── Write merged JSONL locally (no volume race conditions) ─────────────
    local_results_dir = Path(__file__).parent / "results"
    local_results_dir.mkdir(parents=True, exist_ok=True)
    local_jsonl = local_results_dir / "results.jsonl"

    with open(local_jsonl, "w") as f:
        for rec in all_results:
            f.write(json.dumps(rec) + "\n")

    print(f"\n{len(all_results)} results written to {local_jsonl}")
    print("Plot:  python plot_results.py --results results/results.jsonl")
