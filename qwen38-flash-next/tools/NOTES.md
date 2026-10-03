# Working notes — Qwen3.8-Flash-Next on a free Kaggle TPU v5e-8

Engineering notebook for this folder. Everything here is either measured on this
machine (HF API sizes, the overlay's own CPU test suite) or explicitly labelled a
guess. Nothing in this file has been measured on a TPU.

## 1. Why this model is a plausible fit at all

| Property | Value | Why it matters on 8×16 GB |
|---|---|---|
| Active parameters | 6 B/token | decode is bandwidth-bound; 6 B active at 4-bit ≈ 3 GB/token to read |
| Layers | 48, layout 12 × (3 × GDN → MoE) → 1 × (QSA → MoE) | only 12 layers hold a growing KV cache |
| N-gram embedding | 51 B params, bigram/trigram at layer 2 | the architecture's own design intends this to live in **host RAM** with async prefetch |
| Context | 262,144 native, 1M with YaRN | GDN layers compress history instead of growing a cache |
| Vision | present (Qwen3.8 is a VLM) | **not supported** by the TPU overlay; text-only for now |

The 3:1 GDN:QSA ratio is the same shape as Qwen3.8-27B, which this repo already serves
on the same hardware — so the recurrent-attention Pallas kernels that make that recipe
work are the same family. The new work is QSA (the sparse-attention indexer) and PLE
(the N-gram table).

## 2. The capacity arithmetic (the whole problem)

Kaggle TPU v5e-8 = 8 chips × 16 GB = **128 GB HBM**, single host.

| Artifact | Measured size (HF API) | Fits? |
|---|---|---|
| `Qwen/Qwen3.8-Flash-Next` bf16 | 335.3 GiB | ✗ 2.6× |
| `Qwen/Qwen3.8-Flash-Next-FP8` | 172.8 GiB | ✗ 1.35× |
| `nvidia/Qwen3.8-Flash-Next-NVFP4` | 123.6 GB | ✗ no room left for KV/compile |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` | 135.2 GB | ✗ |
| `Minachist/...-INT4-Mixed-AutoRound` | 175.3 GB | ✗ (looks like INT4 + full-precision keepset) |
| `tcclaviger/...-MXFP4-FP8-GPTQ` | 120.3 GB | ✗ marginal |
| `albucino/...-W4A16-FP8PLE` | 120.1 GB | ✗ marginal |
| `Jundot/...-oQ4e-mtp` | 106.3 GB | ⚠️ |
| `local-inference-lab/...-NVFP4` | 106.3 GB, has `hf_quant_config.json` | ⚠️ **chosen starting point** |
| `cafonez/...-HC-Q8` | 92.3 GB | ⚠️ format unknown |
| `pentacoxian-dev/...-IQ3E-Q8D-MTP` | 80.0 GB GGUF | ⚠️ **GGUF: the TPU overlay cannot read it** |
| `beamster/...-Sushi-2.6bpw` | 44.0 GB | ✓ but format unknown |

Sizes are the sum of `*.safetensors`/`*.gguf` from the HF tree API (decimal GB unless
suffixed GiB). The BF16/FP8 numbers match Qwen's and vLLM's published figures.

Two consequences:

1. **4-bit alone is not obviously enough.** ~106 GB of weights on a 128 GB budget
   leaves ~22 GB for KV cache, activations, XLA workspace and the compiled graphs —
   and the N-gram table alone is ~28 GB of that at 4-bit.
2. **The lever is the N-gram table.** Move it to host RAM and the device budget drops to
   roughly 78 GB, leaving ~50 GB of headroom. That is precisely the design the model
   card describes ("offloaded to host memory and overlapped ... asynchronous
   prefetching") and precisely what vLLM implements as `VLLM_PLE_CPU_OFFLOAD=1` —
   **on NVIDIA only**; vLLM's own recipe lists a TPU startup error for this feature.

So the recipe's defaults are: NVFP4 export + `VLLM_PLE_CPU_OFFLOAD=1` + 32k context.
If the host-RAM path does not work on TPU, the fallback is a smaller artifact
(`beamster/...-Sushi-2.6bpw`) plus a GGUF reader — which the TPU overlay does not have.

## 3. Weight delivery — the one thing that needs a human

The kernel can read weights from an attached Kaggle dataset or download from HF. For a
106 GB checkpoint the download path is not viable (a session's own scratch space is
smaller than that), so **the weights must arrive as a mounted dataset**. Three options:

1. `kaggle datasets create` against `local-inference-lab/Qwen3.8-Flash-Next-NVFP4`
   (~106 GB down, ~106 GB up; needs ~110 GB of local scratch — this machine has 112 GB
   free, so it is possible but tight, and it takes hours).
2. Upload the same export through the Kaggle web UI.
3. Use a smaller export if one appears with a readable format (Q4 safetensors + config),
   which would also relax the HBM budget.

Until that dataset exists, `weights_dataset` is a `CHANGEME/...` placeholder and the
kernel will try the HF download and probably run out of disk.

## 4. Status: verified vs assumed

**Verified on this machine (CPU, no TPU).** The overlay at commit `be41c49` (v79) runs
its own test suite green against a modern stack — `jax 0.11.2`, `flax 0.12.10`,
`torch 2.14.1+cpu`, `transformers 5.18.0`:

```
35 passed, 1 skipped   (tests/models/jax/test_qwen4_exp.py)
```

The skip is `test_e2e_tpu`, which needs a TPU and the real checkpoint. What that suite
actually pins down, from its own names: NVFP4 dequant and GPTQ dequant **bit-exact
against a torch reference**, QSA slot rules and top-k selection, QSA chunked prefill ==
full forward, GDN norm against upstream `Qwen3NextRMSNormGated`, GatedResidual
`combine`/`mix` against the closed form, the PLE splitmix64 hash and per-order
isolation, the dilated conv gather, live parameter paths against the loader contract,
sharding specs divisible by TP, and the host-RAM PLE table lookup.

**Assumed, not verified.** Everything about the actual run: that the NVFP4 export's
tensor names match what the loader expects at real scale, that 106 GB of weights plus a
host-resident N-gram table plus KV cache fit, that XLA compiles the QSA Pallas paths,
and that the output is coherent. The overlay's own docs list these as open gaps
(GDN parity against the FLA reference, QSA side caches replaced by per-forward
recompute, MRoPE/vision unsupported, MTP a stub).

**A caveat about the overlay.** Its README documents a file layout (`qsa.py`, `ngram.py`,
`weight_loader.py`, `quant.py`) that the v79 commit has outgrown, and its cited Kaggle
datasets for real weights (`keithtyser/...-nvfp4`, `ram2121/...-gptq-4bit`) return 404 —
consistent with a project that is being actively developed and has not yet published a
real-weight run. Treat the repo as a strong head start and a correctness reference, not
as prior art for the thing this folder is trying to do.

## 5. What the first run is for

Not speed. It is a load-path test, in this order of likelihood of failure:

1. weight-name mapping against the real NVFP4 export at full scale
2. HBM exhaustion during load (the table, the experts, or the staging copies)
3. XLA compile of the QSA/PLE/GatedResidual paths
4. incoherent output (GDN reference vs FLA, QSA selection)

If (2) is the wall, the honest answer may be "this model does not fit on v5e-8 in any
format anyone has published", and the right deliverable is that result with the numbers.

## 6. Open questions to settle before spending quota

- Which export does the overlay's loader actually match? Its docs name NVFP4
  (`modelopt_fp4`) and GPTQ INT4 (group 128, asymmetric). `local-inference-lab/...-NVFP4`
  ships `hf_quant_config.json`, so it is the closest match on paper.
- Does anything on TPU consume `VLLM_PLE_CPU_OFFLOAD`? If not, the recipe needs the
  overlay's own host-table path wired into the runner rather than vLLM's env flag.
- Does Kaggle's TPU image have enough host RAM for a 28–51 GB host buffer in addition to
  the VM's normal footprint?
