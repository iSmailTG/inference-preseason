# inference-preseason
6 Weeks Plan for Inference Engineering

# Preseason Plan: 6 Weeks (+1 Buffer Week)

**Purpose:** build the knowledge and the proof needed to enter the arena (the vLLM contribution phase) without wasting maintainers' time.
**Target profile:** inference runtime engineer on vLLM-style engines (the Inferact posting and similar roles).
**Start date:** ____ **Planned end of Week 6:** ____ **Buffer week:** ____

---

## 0. Assumptions and sizing

- **Time:** about 50–60 h/week (six working days of 9–10 h, plus one buffer day). If your real hours are about 30/week, run this as 12 weeks: same order, each week split into two halves. Never cut tests, parity checks, or posts.
- **Reference model:** Qwen3-0.6B. Read every dimension from `config.json`; never hardcode. Expect roughly: hidden 1024, 16 query heads, 8 KV heads, head_dim 128, 28 layers (verify).
- **Where code runs**

| Work | Runs on |
|---|---|
| Unit tests, parity checks, tiny dummy configs | CPU (local, Colab CPU runtime, or Kaggle CPU) |
| Benchmarks, running vLLM | Colab/Kaggle T4 |
| JAX checks on a real chip | Colab/Kaggle TPU session (short checks only) |
| Kernel practice (optional, not required by the posting) | LeetGPU CPU emulation |

- **Not in scope for the preseason:** GPU kernel work, quantization projects, RL experiments, TPU work beyond the capped JAX thread, and any public PR before Week 6.

---

## 1. Rules for all six weeks

1. **Daily loop:** recite, study, code, verify, shadow, commit (template in section 3).
2. **Three kinds of oracle for every module:** a hand-computed value, an independent reference (for example `torch.nn.functional.rms_norm`), and parity against Hugging Face. Parity happens the same day you write the module. Passing shape and gradient checks alone proves nothing.
3. **Hide the solutions.** Do not open a reference implementation until your own version passes, or you have been stuck for 45 minutes. Read nano-vllm and similar projects only after you have attempted a module yourself. Never copy.
4. **Mismatch log.** Every failed parity check goes into `notes/mismatch_log.md` using the template in Appendix C. Bisect first, then fix.
5. **Pull-forward rule.** If you finish a day early, pull forward reading or the next day's study block, not extra scope. Never idle, and never start a new module's code without finishing today's verify block.
6. **Listen first, speak later.**
   - Weeks 1–4: join the vLLM Slack and forum and read only. Skim the issue tracker for 15 minutes a day.
   - Week 5: ask a question only after searching existing issues and docs.
   - Week 6: submit one small contribution.
7. **Weekly reading cadence:** one paper (tied to that week's build), one set of vLLM SIG meeting notes, and one PyTorch-internals slot.
8. **JAX thread:** capped at 4 h/week and taken from Day 6/7 slack. It is the first thing cut if you fall behind.
9. **AI use:** you must be able to rebuild every module from memory (the recite block checks this). Understand every line you commit. For any external PR, read AGENTS.md and CONTRIBUTING first, and label AI-assisted work as the project's policy requires.
10. **Slip rule:** cut stretch items first, never parity tests or the weekly post. If you are more than two days behind at midweek, move the week boundary instead of skipping content. Use the buffer week.

---

## 2. Repository layout (`miniengine`)

```
miniengine/
  miniengine/
    layers.py        # RMSNorm, SwiGLU
    rope.py          # rotary embeddings with explicit positions
    attention.py     # GQA + QK-norm
    model.py         # embeddings, decoder stack, lm_head
    sampling.py
    kv_cache.py      # contiguous cache (Week 2)
    paged_cache.py   # block allocator, block tables (Week 3)
    scheduler.py     # continuous batching (Week 4)
    prefix_cache.py  # hash chain, refcounts, LRU (Week 5)
    engine.py
  tests/
    test_oracles.py
    test_parity_hf.py
    test_cache_equivalence.py
    test_paged.py
    test_scheduler_invariance.py
    test_prefix_cache.py
  jax_port/          # the capped JAX thread
  bench/             # benchmark scripts, so results are reproducible
  notes/             # daily notes, vllm_map.md, mismatch_log.md
  posts/             # blog drafts
  .github/workflows/ci.yml   # runs fast CPU tests on tiny configs
  README.md
```

**Testing convention:** fast tests use tiny random-weight configs (2 layers, small hidden size) and run in seconds on CPU in CI. Full-weight parity tests are marked `slow` and run manually on Colab.

---

## 3. Daily schedule template (about 10 h)

| Block | Minutes | What |
|---|---|---|
| Recite | 45 | Rewrite yesterday's modules from memory, closed book. Then diff against your committed version. |
| Study | 90 | Today's concept: paper, docs, source reading. Write 5 lines of notes. |
| Code | 180 | Today's module, written yourself. |
| Verify | 90 | Oracles, parity, edge cases. Log mismatches. |
| Shadow | 90 | Explain the day out loud, as if teaching. Then answer the day's "break it" questions (below). |
| Commit and notes | 45 | Commit, update `notes/dayN.md`. |
| Buffer | 60 | Overflow, tracker skim (15 min), JAX thread on Day 6/7. |

**"Break it" questions (add one per day):** predict what breaks if you change one detail (remove QK-norm, flip the RoPE pairing, swap the SwiGLU branches, off-by-one the block boundary), then run it and see. This builds the debugging skill the job posting asks for.

**Tolerance starting points:** fp32 module parity `rtol=1e-4, atol=1e-5`. Tune them, then record the final values in your README. For fp16 comparisons against vLLM, use teacher forcing (feed the same tokens) and report max and mean absolute logprob difference plus top-1 agreement. Don't expect exact matches.

---

## 4. Week-by-week plan

### Week 1: One Qwen3 decoder layer

**Goal:** a decoder layer you wrote yourself that matches Hugging Face on real layer-0 weights.
**Ship:** layer parity, oracle tests, CI skeleton, vLLM-vs-yours diff note.

| Day | Build | Verify |
|---|---|---|
| 1 | Layer inventory; RMSNorm; SwiGLU. Also, 10 min: check the live Inferact posting's location terms. | Oracle tests (hand-computed `[1,2,3]` RMSNorm example, `F.rms_norm`, SwiGLU scalar example, branch-swap detector) **plus** HF parity in fp32 |
| 2 | GQA attention with QK-norm; RoPE stubbed as identity (cos=1, sin=0) | HF attention parity. Pin `attn_implementation` and pass an explicit 4D causal mask; otherwise the eager path applies no causal mask and you will chase phantom mismatches. |
| 3 | RoPE with explicit positions (never `arange`), rotate-half pairing | Rotation preserves norm; relative-position property; parity vs HF rotary embedding |
| 4 | Assemble the decoder layer (pre-norm residual structure) | Parity vs HF layer on identical inputs; bisect procedure when it fails |
| 5 | Strict-load real layer-0 weights (names must match the HF checkpoint) | Capture HF layer-0 input with a forward hook (layer 0 takes hidden states, not token ids); compare outputs |
| 6 | Read vLLM's Qwen3 code beside your layer (check where it lives now; model code is being reorganized). Write "what changed and why": merged `qkv_proj` and `gate_up_proj`, fused add-residual in the norm, flattened tokens. Compute KV bytes per token in fp32 and fp16. | Note exists and your formula matches hand arithmetic |
| 7 | README, `pytest` plus GitHub Actions on CPU with tiny config, join vLLM Slack and forum (listen only). JAX: install, hello world, confirm a TPU session starts (1 h, only if on time). | CI green |

**Reading:** Qwen3 model code in HF; vLLM architecture overview; GQA paper (skim); RoPE explanation.
**PyTorch-internals slot:** tensor shapes, views, `contiguous()`, and why `view` vs `reshape` matters in the attention head split.
**Exit check:** layer-0 output matches HF on a real prompt; oracle tests pass; CI green; Day 6 note written.

---

### Week 2: Full model to text

**Goal:** your engine generates text that matches HF, with a contiguous KV cache, and you have compared it against vLLM on a T4.
**Ship:** full model, sampling, cache, vLLM comparison, **Post 1**.

| Day | Build | Verify |
|---|---|---|
| 1 | Embeddings, N decoder layers, final norm, lm_head (tie weights if `tie_word_embeddings` is true in the config); strict-load the full checkpoint | Logits parity vs HF on 3 prompts (fp32, CPU) |
| 2 | Sampling (greedy, temperature, top-k, top-p, seeded) and a generation loop with no cache | Greedy tokens match HF `generate` on 10 prompts x 32 tokens; measure how cost grows |
| 3 | Contiguous KV cache (preallocated per request); separate prefill and decode paths; explicit positions | Cached logits equal uncached logits |
| 4 | Naive static batching with padding and masks | Per-request outputs equal single-request outputs; report pad-token waste |
| 5 | Run vLLM on the T4 with Qwen3-0.6B in fp16. Record the startup log, including KV cache capacity in tokens. | Teacher-forced logprob comparison (see tolerance note); your fp16 KV formula vs the log's token capacity |
| 6 | Internals: `torch.profiler` on the decode step (CPU and T4). Write and publish **Post 1: "One Qwen3 layer: HF vs mine vs vLLM."** | Post published |
| 7 | Buffer. JAX: port RMSNorm and SwiGLU, matching your PyTorch outputs on CPU; one `jit` run on the TPU session. | Arrays match |

**Reading:** the core of HF's generate loop; vLLM quickstart; "Thinking in JAX" (1 h).
**Exit check:** text matches HF; cache equivalence tests pass; Post 1 live.

---

### Week 3: Paged KV cache

**Goal:** a paged cache whose output is identical to the contiguous cache.
**Ship:** block allocator, paged attention in plain PyTorch, memory accounting.

| Day | Build | Verify |
|---|---|---|
| 1 | Read the PagedAttention paper and vLLM's design docs. Write a design note: cache tensor shape `[num_blocks, block_size, n_kv_heads, head_dim]` per layer for K and V, plus invariants. | Design note reviewed by you after a night's sleep |
| 2 | Block allocator (free list), block tables, alloc and free | Tests: exhaustion, reuse, no double free |
| 3 | Write path (scatter new K/V by slot mapping) and read path (gather via block table) | Matches the contiguous cache for a single request |
| 4 | Multiple requests of variable length | Boundary tests at length `k*block_size` and +/-1; random fuzz; paged output equals contiguous |
| 5 | Memory accounting: internal fragmentation, max concurrent requests for a given budget, paged vs contiguous at max length. Read vLLM's KV-cache manager and block pool (check current paths) and take notes. | Numbers in `notes/` |
| 6 | Internals: indexing and gather costs; allocated vs reserved memory in PyTorch's caching allocator on the T4. JAX: port attention (GQA, QK-norm, RoPE with explicit positions), match yours; try `.at[].set()` for a cache update. | JAX attention matches |
| 7 | Buffer, README update | Commit |

**Reading:** PagedAttention (vLLM) paper.
**Exit check:** fuzz tests green; memory-waste numbers recorded; design note compared with vLLM's block manager.

---

### Week 4: Continuous batching

**Goal:** a scheduler where batching never changes outputs.
**Ship:** scheduler, invariance tests, T4 benchmark, **Post 2**.

| Day | Build | Verify |
|---|---|---|
| 1 | Read the Orca paper and vLLM's scheduler overview. Design request states (waiting, running, finished, preempted), a per-step token budget, and admission by free blocks. | Design note |
| 2 | Flattened ragged batch: token ids, positions, sequence ids, slot mapping, block tables. Forward over flattened tokens (a naive per-sequence attention loop is fine). | Equal to per-request outputs |
| 3 | Scheduler loop: each step, decode tokens of running requests plus prefill of waiting ones under the token budget; finish handling; free blocks | Runs a toy workload end to end |
| 4 | Minimal preemption (recompute). Chunked prefill if time allows (otherwise read about it). | **Invariance fuzz:** random arrival times and lengths give tokens identical to running each request alone (greedy) |
| 5 | T4 benchmark: static vs continuous batching on a synthetic mixed-length workload (throughput, p50/p95 latency). Run vLLM on the same workload. | Report ratios and caveats, not absolute claims. Benchmark script in `bench/`. |
| 6 | Write and publish **Post 2: "What continuous batching buys on a T4."** Internals (stretch): CUDA graphs on the decode step with static buffers, or `torch.compile` on a layer; read vLLM's torch.compile integration doc. JAX: read vLLM's TPU docs and status dashboard; note two candidate items for the arena. | Post published |
| 7 | Buffer | Commit |

**Reading:** Orca paper; Sarathi-Serve (optional, concept only).
**Exit check:** invariance fuzz passes; benchmark reproducible from a script; Post 2 live.

---

### Week 5: Prefix caching and the vLLM core map

**Goal:** prefix caching that never changes outputs, and a written map from your engine to vLLM's.
**Ship:** prefix cache, `notes/vllm_map.md`, JAX decoder layer, TPU/AMD/general decision.

| Day | Build | Verify |
|---|---|---|
| 1 | Read vLLM's prefix-caching design doc and the RadixAttention paper. Design: hash chain over full blocks (`hash(prev_hash, tokens)`), refcounts, LRU eviction of unreferenced cached blocks. | Design note |
| 2 | Block hashing, lookup at admission (longest cached prefix of full blocks), refcount increments. A partial last block is never cached. | Unit tests |
| 3 | Eviction under memory pressure, counters (hit rate, evictions) | Identical outputs with and without the cache; shared prefix with different suffixes; eviction mid-flight is safe |
| 4 | Benchmark: shared system-prompt workload, time-to-first-token with and without caching. Compare qualitatively with vLLM prefix caching on the T4. | Script in `bench/` |
| 5 | **vLLM core read:** engine core loop, scheduler, KV-cache manager, model runner (the new runner is now the default and the old one is being removed, so read the new one), sampler. Write `notes/vllm_map.md` mapping each of your modules to vLLM's, with five differences and why they exist (verify all paths). | Map complete |
| 6 | Read-only: how vLLM handles sliding-window and hybrid-model KV caches. JAX: assemble the decoder layer in JAX and match yours. **Decision checkpoint:** TPU, AMD, or generalist (see questions below). | Decision written down |
| 7 | Buffer. Ask your first question in the community only if one is still unanswered after searching. | Commit |

**Decision questions:** Do you enjoy learning JAX's functional style? Would you be happy if TPU became your specialty for three years? If you're lukewarm on both, choose AMD (closer to PyTorch) or stay general.
**Reading:** RadixAttention paper (skim); vLLM design docs on prefix caching.
**Exit check:** all cache-equivalence tests green; vLLM map done; JAX layer matches; decision recorded.

---

### Week 6: Bridge week

**Goal:** a working vLLM dev setup, a small contribution submitted, and an arena plan.
**Ship:** dev setup, first small PR or issue repro, tiny MoE, Post 3, arena plan v1.

| Day | Do | Verify |
|---|---|---|
| 1 | **vLLM dev setup.** Fork the repo and use the Python-only editable install with precompiled wheels (`VLLM_USE_PRECOMPILED=1 uv pip install -U -e .`; check the docs for current flags). Run lint and pre-commit and build the docs locally. Run a few CPU-feasible tests and note which tests need a GPU. | Lint and docs build pass locally |
| 2 | **Tracker triage.** Find three candidate issues (docs, tests, repro). Check for duplicates. Read AGENTS.md and CONTRIBUTING. Read two SIG meeting notes. Draft the comment you will post. | Candidate list with reasons |
| 3 | **Variants day.** Tiny MoE block (top-k routing) with parity against HF's sparse MoE block on a tiny random config (CPU). Read vLLM's fused-MoE concept (read-only). Write a short "shared vs hardware-specific" note. | MoE parity passes |
| 4 | **First contribution.** Submit one small PR or issue repro (docs, test, or repro). Follow the PR template. Disclose AI assistance if the policy requires it. | PR or issue submitted |
| 5 | **Polish.** README with an architecture diagram and results tables, CI green, repo pinned on your profile. Publish **Post 3: "Paged KV and prefix caching: mine vs vLLM's."** Optional bonus note: "One Qwen3 layer in PyTorch vs JAX." | Posts live |
| 6 | **Arena plan v1.** Re-read the roadmap (the new quarter's, if posted). Choose three targets and one candidate anchor non-trivial contribution to discuss in its issue during Arena A. | One-page plan |
| 7 | Retrospective: compare planned vs actual hours, record what you would resize. | Notes |

**Exit check:** see section 5.

---

### Week 7: Buffer (use only if needed)

Use it for slippage in any week. If you are on time, start Arena A early: pick up a second small contribution and begin discussing the anchor target.

---

## 5. Preseason is done when

- [ ] `miniengine` generates text matching HF, with paged KV cache, continuous batching, and prefix caching, all under tests with CI green.
- [ ] You can explain vLLM's request lifecycle from memory and map each step to your code (`notes/vllm_map.md` exists).
- [ ] Three posts are published (or two published plus one drafted).
- [ ] vLLM dev environment works, and one small contribution is submitted.
- [ ] A Qwen3 layer is ported to JAX and matches your PyTorch version.
- [ ] The TPU / AMD / generalist decision is written down.
- [ ] Arena plan v1 exists with three targets and one candidate anchor contribution.
- [ ] You have verified the live job posting's location terms.

---

## 6. How the preseason covers the posting

| Posting requirement | Where it is covered |
|---|---|
| Deep understanding of transformer architectures and variants | Weeks 1–2; tiny MoE and hybrid-cache read in Weeks 5–6 |
| Python and PyTorch internals | Weekly internals slots (views, profiler, indexing and allocator, CUDA graphs and `torch.compile` as stretch) |
| Experience with LLM inference systems | Weeks 2–5: your engine, plus running vLLM yourself |
| Reading and implementing papers | One paper per week |
| Maintainable code and debugging | Tests, CI, mismatch log, "break it" questions |
| KV-cache management and prefix caching (preferred) | Weeks 3 and 5 |
| Open-source contributions (preferred) | Week 6 first PR; Arena A onward |
| Technical blogs (bonus) | Three posts |
| RL, multimodal, integrations | Deferred to the arena |

---

## 7. Handoff to the arena

Arena A (the next two months) is mostly small contributions, learning that produces artifacts, and networking. By the end of it, you should have agreed on one anchor non-trivial target in its issue thread. Arena B closes the posting qualifications and interview prep while the anchor contribution is in review. The depth phase after that is where you build the strongest evidence.

wmctrl -a "Terminal" || gnome-terminal
----

## Appendix A: Oracle test starter (adapt the names to your modules)

```python
import torch, torch.nn.functional as F
from miniengine.layers import RMSNorm, SwiGLU   # adapt to your names

# 1. Worked example: x=[1,2,3], gamma=ones, tiny eps
n = RMSNorm(3, 1e-12)
y = n(torch.tensor([1., 2., 3.]))
torch.testing.assert_close(y, torch.tensor([0.4629, 0.9258, 1.3887]), atol=1e-4, rtol=0)
assert y.mean().item() > 0.5   # no centering happened

# 2. Independent reference (torch >= 2.4)
x = torch.randn(2, 5, 1024)
n = RMSNorm(1024, 1e-6); n.weight.data.uniform_(0.5, 2)
torch.testing.assert_close(n(x), F.rms_norm(x, (1024,), n.weight, 1e-6), rtol=1e-5, atol=1e-6)

# 3. SwiGLU scalar example: gate=2, up=3 -> silu(2)*3 = 5.2848
m = SwiGLU(1, 1)
for lin, v in [(m.gate_proj, 2.), (m.up_proj, 3.), (m.down_proj, 1.)]:
    lin.weight.data.fill_(v)
out = m(torch.ones(1, 1, 1))
torch.testing.assert_close(out, torch.tensor([[[5.2848]]]), atol=1e-3, rtol=0)

# 4. Branch-swap detector: silu(up)*gate would give about 5.7154
assert abs(out.item() - 5.7154) > 0.1
```

## Appendix B: Reading list by week

| Week | Paper or doc | Docs and source |
|---|---|---|
| 1 | GQA (skim), RoPE (RoFormer) | HF Qwen3 code, vLLM architecture overview |
| 2 | (none; focus on code) | HF generate loop, vLLM quickstart, "Thinking in JAX" |
| 3 | PagedAttention (vLLM) | vLLM design docs, KV-cache manager |
| 4 | Orca; Sarathi-Serve (optional) | vLLM scheduler, torch.compile integration, TPU docs and dashboard |
| 5 | RadixAttention (SGLang) | vLLM prefix-caching doc, engine core loop, new model runner |
| 6 | (none) | AGENTS.md, CONTRIBUTING, roadmap, SIG meeting notes |

## Appendix C: Mismatch log template

```
### YYYY-MM-DD: <component>
Symptom:        what differed, by how much
Hypothesis:     what you thought was wrong
Bisect steps:   what you checked, in order
Root cause:     what it actually was
Fix:            what you changed
Lesson:         what to check first next time
```
last 2 days
