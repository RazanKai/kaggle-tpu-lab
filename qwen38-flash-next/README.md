# Qwen3.8-Flash-Next on a free Kaggle TPU v5e-8

**Status: skeleton. No real-weight run has ever completed, so this folder deliberately
publishes no throughput numbers.** Everything the other two recipes in this repo state
as measured — tokens per second, startup time, correctness verification — is unknown
here. Read [`tools/NOTES.md`](tools/NOTES.md) before spending TPU quota on it.

Qwen3.8-Flash-Next (`Qwen4Exp`) is a 125 B-parameter ultra-sparse MoE — 6 B activated per
token — plus a 51 B N-gram embedding table, an early preview of the Qwen4 architecture.
It is the most interesting model in this repo's target hardware class, and the least
proven on it.

## Why it is worth trying

| Property | Value | Consequence on 8 × 16 GB |
|---|---|---|
| Active parameters | 6 B/token | decode reads ~3 GB/token at 4-bit |
| Layer layout | 12 × (3 × Gated DeltaNet → MoE) → 1 × (Qwen Sparse Attention → MoE) | only 12 of 48 layers keep a growing KV cache |
| N-gram embedding | 51 B params, designed to be **offloaded to host RAM** with async prefetch | the model's own design assumes a memory-constrained accelerator |
| Context | 262,144 native, 1M with YaRN | the same hybrid shape as Qwen3.8-27B, which this repo already serves |

The GDN half is the same family as the Qwen3.8-27B recipe's Pallas kernels. The genuinely
new work is QSA (sparse attention with a compressed indexer) and PLE (the N-gram table).

## The obstacle, in one table

Kaggle TPU v5e-8 is 8 × 16 GB = **128 GB of HBM**.

| Checkpoint | Size | Fits? |
|---|---|---|
| `Qwen/Qwen3.8-Flash-Next` (bf16) | 335.3 GiB | ✗ |
| `Qwen/Qwen3.8-Flash-Next-FP8` | 172.8 GiB | ✗ |
| `nvidia/...-NVFP4` | 123.6 GB | ✗ |
| `local-inference-lab/...-NVFP4` | 106.3 GB | ⚠️ the starting point here |
| `beamster/...-Sushi-2.6bpw` | 44.0 GB | ✓ but unreadable by the TPU stack (see below) |

A ~106 GB export leaves ~22 GB for KV cache, activations, XLA workspace and compiled
graphs. The lever that makes this plausible is moving the 51 B N-gram table to host RAM
(~28 GB at 4-bit) — which is what the architecture was designed for, what vLLM
implements as `VLLM_PLE_CPU_OFFLOAD=1`, and what upstream vLLM's own recipe says **is not
supported on TPU**. The recipe therefore enables it and hopes the port honors it.

## What's in this folder

```
kernel/serve_qwen38_flash_next.py     the Kaggle kernel (same 6-step shape as the other recipes)
notebook/…-tpu-serve.ipynb            the same flow as a run-it-yourself notebook
tools/NOTES.md                        the budget arithmetic, the artifact survey, the open questions
```

The engine is **not vendored here.** It is a third-party `tpu-inference` overlay fetched
at a pinned commit and copied over the installed `tpu_inference` at runtime:

- `overlay_repo`: `https://github.com/DQN-Labs/nexus-tpu-fork` (Apache-2.0, a fork of
  `vllm-project/tpu-inference`; the fork ships no LICENSE file, so this repo links to it
  rather than redistributing it)
- `overlay_commit`: `be41c49`

Upstream `tpu-inference` has no Qwen4Exp model at all, and vLLM's recipe marks TPU as
unsupported for this architecture, so nothing serves this model without that overlay.

## Prerequisites

1. A Kaggle account with TPU access (phone-verified) and its ~20 TPU-hours/week.
2. **A weights dataset.** The 106 GB export must be attached as a Kaggle dataset —
   a session's own scratch space cannot hold it, so the Hugging Face fallback in the
   kernel is a formality. See [`tools/NOTES.md` §3](tools/NOTES.md).
3. Then, from the repo root:

```bash
python launch.py serve --model qwen38-flash-next --weights-dataset <owner>/<slug>
```

The defaults are deliberately conservative: 32k context, 8 concurrent sequences,
text-only, no MTP (`--max-model-len` and `--max-num-seqs` are honored; passing
`--max-model-len 262144` means you want the native window and accept the risk).

## What is actually verified

The overlay's own CPU test suite is green on a clean modern stack (`jax 0.11.2`,
`flax 0.12.10`, `torch 2.14.1+cpu`, `transformers 5.18.0`) — **35 passed, 1 skipped**,
where the skip is the on-TPU end-to-end test:

```
pytest tests/models/jax/test_qwen4_exp.py -q     # in the overlay checkout
SKIPPED [1] needs TPU v5e-8 + checkpoint (QWEN4EXP_E2E=1)
```

That suite pins NVFP4 and GPTQ dequantization bit-exact against torch, QSA chunked
prefill against full forward, the GDN norm against upstream, GatedResidual against the
closed form, PLE hash/order isolation, the host-RAM table lookup, and sharding
divisibility. It does **not** test real weights on real hardware — nobody has.

## Known landmines

- **Weight-name mapping at full scale** is unvalidated; the overlay's loader has been
  exercised against mock checkpoints and a single-file inspection path.
- **HBM exhaustion** during load is the most likely wall, and there is a real chance the
  answer is "this model does not fit on v5e-8 in any published format".
- **QSA side caches are not implemented** in the overlay (it recomputes per forward), so
  long context will be slow even if it fits — the 262k selling point is not reachable yet.
- **MTP is a stub.** The overlay has no working multi-token-prediction drafter, and the
  Qwen3.8-27B recipe's experience says spec decoding with hybrid recurrent state is
  fragile on TPU anyway. `mtp_tokens` stays 0.
- **Vision is unsupported** in the overlay, so text-only (`text_only: true`), even though
  the checkpoint is a VLM.
- **GGUF quants are unreadable** by the TPU stack, which rules out the smallest artifacts
  (the 44 GB 2.6 bpw one included) unless someone adds a GGUF reader.
- **No compile cache exists** for these graphs, so this recipe does not attach the 27B
  env dataset; every start compiles cold.

## Credits

- [Qwen](https://github.com/QwenLM/Qwen3.8-Flash-Next) for the model (Apache-2.0 weights).
- [DQN-Labs/nexus-tpu-fork](https://github.com/DQN-Labs/nexus-tpu-fork) for the JAX/TPU
  `Qwen4Exp` implementation, which is the only reason this is conceivable.
- [vLLM](https://github.com/vllm-project/vllm) and
  [tpu-inference](https://github.com/vllm-project/tpu-inference) for the GDN kernels and
  the framework the overlay rides on.
- Kaggle for the free TPUs.

MIT for everything in this repo (see `../LICENSE`); the overlay and the weights keep
their own licenses.
