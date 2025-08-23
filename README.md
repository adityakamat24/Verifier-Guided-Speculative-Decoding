# Verifier-Guided Speculative Decoding

This project explores **speculative decoding** with an extra layer of **verifier guidance** to speed up autoregressive generation while keeping outputs faithful to the target model.

Here’s the big picture:

- A **drafter** proposes multiple next tokens cheaply.
- A **target model** remains the source of truth.
- A **verifier** (lightweight logic + optional scoring head) guides *which* drafted tokens are worth sending to the target for confirmation and *how* we schedule verification.
- Two structural policies—**Tree** and **Cascade**—shape how drafts are organized and checked.

The repo includes an executable notebook, raw benchmark CSV, and a summary figure of results.

---

## Why this exists (the short version)

Speculative decoding gives real speedups by letting a smaller model “draft” ahead. The catch is deciding **which drafts are likely to be accepted by the target** and **how to verify them efficiently**. That decision is what this project focuses on.

Verifier guidance is the idea that you can improve acceptance and scheduling by using:
- inexpensive **signals** (probabilities, entropy, agreement between models),
- **structures** (Tree vs Cascade) that control how drafts flow to the target,
- and **policies** that adapt based on uncertainty.

---

## What this project actually does

### 1) Core loop with a guided verifier

At each decoding step we:

1. Use the drafter `D` to propose a block of `k` tokens and their probabilities.
2. Compute cheap verifier signals `s` for each proposed token (examples below).
3. Filter, order, or regroup proposals using a **Tree** or **Cascade** policy.
4. Ask the target model `T` to verify the ordered set until a mismatch.
5. Accept the verified prefix and continue.

The verifier isn’t a separate heavy model; it’s a light decision layer that runs **before** we spend target compute.

**Typical verifier signals**
- `p_D(y_i)` and entropy for the drafted token `y_i`
- agreement: `p_D(y_i)` vs a cached estimate of `p_T(y_i)` from the previous step
- local constraints: repetition penalties, n-gram bans, top-p/top-k gates
- learned score (optional): a tiny head trained to predict acceptance from drafter features

> The target still makes the final call. The verifier simply **prioritizes** what to check first.

---

## Structural policies

### A) Tree policy
Think of the drafted block as a small branching tree:
- Start with the most promising token (lowest entropy / highest score).
- If it is verified, expand to the next node; if not, backtrack to a sibling.
- This concentrates target calls on the most likely path first.

**When it helps:** the drafter is often “close” to the target; a small number of high-confidence branches cover most acceptance.

### B) Cascade policy
A multi-stage filter:
1. **Stage 1 (cheap):** pure drafter heuristics (entropy, top-k gate, repetition).
2. **Stage 2 (still cheap):** agreement checks and local constraints.
3. **Stage 3 (less cheap):** optional small learned score (single forward of a tiny head).
4. **Target verification:** only for candidates that clear earlier stages.

**When it helps:** you want *predictable* cost by trimming drafts aggressively before any target work.

---

## Algorithm (pseudocode)

```python
def verifier_guided_sd_step(prefix, k, policy):
    # 1) draft tokens with the small model
    drafts, p_d = drafter(prefix, k)  # tokens and probabilities

    # 2) compute fast verifier signals
    signals = compute_signals(prefix, drafts, p_d)  # entropy, agreement, constraints, etc.

    # 3) order/filter/regroup based on policy = {"tree"| "cascade" | "full"}
    candidates = policy_schedule(drafts, signals, policy)

    # 4) verify with target until mismatch
    accepted = []
    for tok in candidates:
        if target_accepts(prefix, tok):  # standard SD acceptance check
            accepted.append(tok)
            prefix = prefix + [tok]
        else:
            break
```
---

## 🔧 Implementation Details

Implementation details live in the notebook:

- **Drafting**: batching + KV-cache reuse  
- **Signal computation**: entropy, agreement, constraints  
- **Tree/Cascade schedulers**: define how drafts are organized before verification  
- **Acceptance**: always target-based, SD-compatible  

---

## 📊 Evaluation and Metrics

Benchmarks are reproducible from:
- `benchmark_analysis.csv` — raw results  
- `Verifier Guided SD (4).ipynb` — end-to-end runs + plots  
- `Verifier Guided SD.png` — summary figure  

**Metrics we track**:
- **Throughput (tokens/sec)**: wall-clock speed while generating full sequences  
- **Relative performance**: normalized vs a fixed reference  
- **Draft acceptance**: mean accepted tokens per verification block  
- **Model-call efficiency**: drafter/target calls per generated sequence  
- **Stability**: std-dev of throughput across runs  

I focus on **how policies change acceptance and scheduling**, not on declaring any baseline “good” or “bad.” The figure below summarizes the current prototype’s behavior across variants.  

![Summary Figure](Verifier%20Guided%20SD.png)

---

## 📂 Repository Layout

- `Verifier Guided SD (4).ipynb` — main notebook with implementation, experiments, plots  
- `benchmark_analysis.csv` — raw benchmark data used in the figures  
- `Verifier Guided SD.png` — combined visualization of results  

---

## ⚠️ Current Limitations (honest notes)

This is a **prototype** aimed at understanding policy design. A few intentional trade-offs and known gaps:

1. **Scheduling is not fully optimized**  
   The balance between drafter steps and target verifications is heuristic and not yet tuned per-sequence.  

2. **Verifier signals are simple**  
   Entropy/agreement/constraints are cheap and fast, but a learned score could be better.  

3. **Single-GPU, Python-level orchestration**  
   There’s room for lower-level optimizations (CUDA streams, fused ops, more aggressive KV-cache management).  

4. **Chunking strategy is basic**  
   Draft verification is mostly linear; more advanced chunk acceptance (verify N at once) is on the roadmap.  

5. **Limited model/dataset sweep**  
   The goal so far was policy behavior, not exhaustive cross-model generalization.  

---

## 🚀 How I Expect This to Improve (near-term roadmap)

1. **Adaptive scheduling**  
   - Adjust draft length `k` and policy choice on the fly using uncertainty signals.  
   - Early-exit rules when confidence is high to avoid unnecessary target calls.  

2. **Learned verifier head**  
   - A tiny classifier trained to predict acceptance from drafter features, calibrated with lightweight distillation from the target.  

3. **Better chunk verification**  
   - Verify multiple tokens as a block when signals agree; fall back to token-wise checks when they don’t.  

4. **System optimizations**  
   - Overlap drafter and target with CUDA streams.  
   - Tighter KV-cache reuse and prefetch.  
   - Batched verifications across sequences for higher hardware utilization.  

5. **Broader evaluation**  
   - Sweep model sizes, prompts, and decoding temperatures.  
   - Include latency-critical settings and streaming output.  

---

```bash
# 1) clone
git clone https://github.com/adityakamat24/verifier-guided-sd.git
cd verifier-guided-sd

# 2) (optional) create a fresh env
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 3) install typical deps
pip install torch torchvision torchaudio \
            transformers \
            numpy pandas matplotlib jupyter

# 4) run the notebook
jupyter notebook "Verifier Guided SD (4).ipynb"
```

# ❓ FAQ

**Is the verifier a second big model?**  
No. It’s a light policy layer (plus an optional tiny head) that ranks/filters drafts before we ask the target.

**Does the target still decide correctness?**  
Yes. Final acceptance always uses the target in a standard SD-compatible way.

**Can this work with my favorite LM?**  
If it supports next-token probabilities and KV-cache, it should be straightforward. The logic is model-agnostic.

---



    # 5) return accepted prefix (possibly empty)
    return accepted
