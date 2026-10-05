"""
Serve Qwen3.8-Flash-Next (Qwen4Exp, NVFP4) on a Kaggle TPU v5e-8 with vLLM.

This script is pushed to Kaggle as a script kernel by ../launch.py, which fills
in the CFG line below. It also runs standalone with defaults (e.g. pasted into
a Kaggle notebook/script in the UI) — then it just prints instead of using ntfy.

Steps (each one is announced in the log):
  1/6  runtime  — venv with vllm-tpu (pinned, CPU torch) built by uv in ~30 s,
                  then the JAX/TPU Qwen4Exp overlay fetched at a pinned commit
  2/6  cache    — restore a pre-built XLA compile cache, if one is attached
  3/6  weights  — find the mounted NVFP4 weights dataset (or download from HF)
  4/6  server   — start vLLM (TP=8, JAX backend, text-only, no MTP)
  5/6  tunnel   — open a public cloudflared URL (printed before the server is
                  live so you can prepare your client)
  6/6  ready    — READY banner + self-test, then keep serving until
                  keepalive_min elapses

STATUS: no real-weight run of this recipe has ever completed. It is the skeleton
of the port described in ../README.md; the first attempt is expected to die in the
weight loader or in XLA compile. There are deliberately no measured numbers here —
see ../tools/NOTES.md for the memory budget this configuration is aiming at and
for what is verified versus assumed.
"""
import base64
import collections
import struct
import zlib
import glob
import gzip
import importlib.util
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

CFG = None  # __LAUNCHER_CONFIG__  (launch.py replaces this line)

DEFAULTS = {
    "vllm_tpu_version": "0.28.0",
    # Upstream tpu-inference has no Qwen4Exp model at all (vLLM's own recipe lists TPU
    # as unsupported for this architecture). The JAX implementation lives in this
    # third-party overlay and is pinned by commit on purpose: it is still moving.
    "overlay_repo": "https://github.com/DQN-Labs/nexus-tpu-fork",
    "overlay_commit": "be41c49",   # v79 — "host-RAM table live". Bump deliberately.
    "weights_dataset": "aigood/qwen38-flash-next-nvfp4",  # 135 GB: NVFP4 experts + FP8 PLE + bf16 rest
    # Preferred source: a Kaggle *Model*. The same 135 GB checkpoint attaches as a Model in
    # seconds and stalls the session as a Dataset (see ../tools/NOTES.md), and its mounted
    # index is intact where the dataset download API returns a mangled one.
    "model_source": "keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1",
    "env_dataset": "",            # no compile cache exists for this recipe yet
    "hf_model_id": "RadixArk/Qwen3.8-Flash-Next-NVFP4",  # the export the dataset above derives from
    "max_model_len": 32768,        # 32k to start: nothing here is proven at 262k
    "max_num_seqs": 8,
    "mtp_tokens": 0,               # the overlay's MTP head is a stub, not a drafter
    "ple_cpu_offload": True,       # keep the 51B n-gram table in host RAM (the overlay implements it)
    "async_scheduling": None,     # None = vLLM's default
    "reasoning_effort_default": "xhigh",   # server-side default: xhigh | medium | low
    "tool_call_parser": "qwen3_coder",  # matches Qwen3.8's XML tool format; "" disables
    "text_only": True,             # the overlay does not implement the vision tower
    "min_token_bucket": 64,        # smallest padded batch (tokens); 16 = more graphs to compile
    "precompile_workers": 4,       # parallel XLA compile threads (1 = sequential)
    "fast_start": False,           # True: skip precompile and warm shapes after READY
    "keepalive_min": 480,          # auto-shutdown guard (Kaggle TPU caps at 9h anyway)
    "expect_min": 45,              # a guess. First runs are debugging runs, not timing runs.
    "api_key": "",                 # generated if empty
    "ntfy_topic": "",              # optional: publish progress to ntfy.sh/<topic>
    "served_model_name": "qwen3.8-flash-next",
    "verbose": False,              # show every vLLM log line (always saved to vllm.log)
    "build_bundle": False,         # maintainer mode: build the env dataset instead of serving
}
CFG = {**DEFAULTS, **(CFG or {})}
# Notebook flow: drop overrides in a serve_config.json next to this script.
_cfg_file = Path("serve_config.json")
if _cfg_file.exists():
    CFG.update(json.loads(_cfg_file.read_text()))
if not CFG["api_key"]:
    CFG["api_key"] = "sk-" + secrets.token_hex(16)

PORT = 8000
VENV = "/tmp/venv"
PY = f"{VENV}/bin/python"
XLA_CACHE = "/tmp/xla_cache"
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("/tmp")
RAW_LOG = WORK / "vllm.log"          # every line vLLM/pip print, for debugging
CLOUDFLARED = Path("/tmp/cloudflared")
T0 = time.time()
PY_VER = f"{sys.version_info.major}.{sys.version_info.minor}"

os.environ["HF_HOME"] = "/tmp/hf"                 # /kaggle/working is only ~21 GB
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["VLLM_XLA_CACHE_PATH"] = XLA_CACHE
os.environ["MIN_TOKEN_BUCKET"] = str(CFG["min_token_bucket"])
os.environ["NUM_PRECOMPILE_WORKERS"] = str(CFG["precompile_workers"])
if CFG["fast_start"]:
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"
# The venv ships its own libtpu; don't let the image's TPU_LIBRARY_PATH override it.
os.environ.pop("TPU_LIBRARY_PATH", None)

_raw = open(RAW_LOG, "a", buffering=1)


def log(*parts):
    line = time.strftime("[%H:%M:%S] ") + " ".join(str(p) for p in parts)
    print(line, flush=True)
    _raw.write(line + "\n")


def elapsed():
    return f"{int(time.time() - T0) // 60} min {int(time.time() - T0) % 60:02d} s"


def banner(step, title, note=""):
    log("")
    log("=" * 70)
    log(f" STEP {step}/6  {title}" + (f"   ({note})" if note else "") + f"   [{elapsed()} so far]")
    log("=" * 70)


def publish(phase, **extra):
    """Progress event: always logged; also pushed to ntfy if a topic is set."""
    log(f"PHASE {phase}", json.dumps(extra) if extra else "")
    if not CFG["ntfy_topic"]:
        return
    try:
        body = {"topic": CFG["ntfy_topic"], "title": f"kaggle-tpu-lab {phase}",
                "message": json.dumps({"phase": phase, **extra})}
        req = urllib.request.Request("https://ntfy.sh", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        log(f"(ntfy publish failed: {e})")


def sh(cmd, tag, show=None, env=None):
    """Run a command; stream its output to vllm.log (and to the console when
    show/verbose). Returns the exit code."""
    show = CFG["verbose"] if show is None else show
    tail = collections.deque(maxlen=40)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env=env)
    for line in p.stdout:
        line = line.rstrip()
        if not line:
            continue
        tail.append(line)
        _raw.write(f"[{tag}] {line}\n")
        if show:
            print(f"[{tag}] {line[:400]}", flush=True)
    rc = p.wait()
    if rc != 0 and not show:
        log(f"[{tag}] exited with code {rc}; last lines:")
        for ln in list(tail)[-15:]:
            print("    " + ln[:300], flush=True)
    return rc


def find_input(*patterns):
    """Datasets mount at /kaggle/input/<slug> (UI) or /kaggle/input/datasets/<owner>/<slug> (API push)."""
    for pat in patterns:
        hits = glob.glob(f"/kaggle/input/{pat}") + glob.glob(f"/kaggle/input/datasets/*/{pat}")
        if hits:
            return hits[0]
    return None


def find_model_dir():
    """A Kaggle Model source mounts at
    /kaggle/input/models/<owner>/<slug>/<framework>/<variation>/<version>. Return the
    deepest one that actually holds safetensors."""
    for cand in sorted(glob.glob("/kaggle/input/models/*/*/*/*/*"), reverse=True):
        if glob.glob(f"{cand}/*.safetensors"):
            return cand
    return None


def fetch_cloudflared():
    if CLOUDFLARED.exists():
        return
    try:
        urllib.request.urlretrieve(
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            CLOUDFLARED)
        CLOUDFLARED.chmod(0o755)
    except Exception as e:
        log(f"(cloudflared download failed: {e})")


# gzip+base64 of patches/mtp-rollback-v0280.diff; not used by this recipe (MTP is a stub here).
MTP_PATCH_B64 = ""  # unused here: Qwen4Exp ships a shape-correct MTP stub, not a drafter

# gzip+base64 of patches/gdn-dt-bias-ignored.diff; regenerated by tools/embed_patch.py
GDN_PATCH_B64 = "H4sIAEUmw2oC/51Ty27bMBC8+ysWzqFWHMkJ4hioD0UK5AEf4qR1CvQm0OJKYkKTKrmqrRb99y6pykiv4UUQyZ2dmR1KVZaQppUiEDNq2lyZEh2aAmc7K1H72Ys4zH7s0cxzPDSzPaqqplxbIdFlTQfb91SNlJF4gCtRyKvyY5bN54vzspzDxfn5Yj4fpWn6Pjaj6XT6TkbX15BeLC7PFjANn4sr4B2JJSifq8pYhzLfKe+VqSZG7HAJnlwC6SfYWquXIxhBWKqEcd016PLCGoMFKWu47sBNttoWr8zrhXfzvv8YlIGI1leH5ZBaZ+DZtTiaho0TuL9Zf/DwmdlWMANJ+VYJD4VwrgNjYZwNaL4tS3WAbcvTdMhQQkMjHDcgdH6Am1TShNFJLDRf86yAamBEYw7ZU7gOwkigGqG0bi+chEoQ3ws7DovWsbE0oHniM9grhvCq2lklJwPDKbDVk8g7SZIMVvfrx6+3N/nDarNZre/zzbe7u9X3280AxZaRUMbDONaPz2Bfq6KGnaCi5v7j7B8yn3gbe1qWSrXyULWBJzMcwC4XR6cIjbfOR0+80kxes3pnmwYlCIKQgqNkdkV0QVXHBUeVijz8Qmd5YIqyEAo2MY/eThL+Ddx9q4kBT63R3SnUHL8Iuu3e0irsT3SiCkY21lEUEgWo3t8QMI3QaFFgz0d5UiYGKRgRBpn1YKr8n0cIUrLsz94k6U5oj32+TuDh+YmVi5LS+CyO1kx21GSnyRJs6+IlpiPYhXYLvhaOSUUyJFyFNICRa80rlwYkDtRy+SW8r9tDwwBJVNa3CoZAn1HPGS1C2AaM0FeLLqhaZzvdZBwZdOSz39LuWZmzL2chfXnbxJ8/cYoG2cU4N5TZ6C9ZphLyxQQAAA=="  # __EMBEDDED_GDN_PATCH_B64__

# gzip+base64 of patches/ple-spill-mmap.diff; regenerated by tools/embed_patch.py
PLE_SPILL_PATCH_B64 = "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"  # __EMBEDDED_PLE_SPILL_PATCH_B64__


def apply_mtp_patch():
    """Port of upstream PR #3178 (GDN state rollback on rejected draft tokens).
    Without it ANY speculative decoding corrupts outputs on this model."""
    if not MTP_PATCH_B64:
        return True
    Path("/tmp/mtpfix.diff").write_text(
        gzip.decompress(base64.b64decode(MTP_PATCH_B64)).decode())
    origin = subprocess.check_output(
        [PY, "-c", "import importlib.util as u; print(u.find_spec('tpu_inference').origin)"],
        text=True).strip()
    pkg_root = os.path.dirname(os.path.dirname(origin))
    p = subprocess.run(["patch", "-p1", "-d", pkg_root, "-i", "/tmp/mtpfix.diff",
                        "--no-backup-if-mismatch", "-N"], capture_output=True, text=True)
    _raw.write(p.stdout + p.stderr)
    if p.returncode == 0 or "previously applied" in p.stdout:
        return True
    log(p.stdout[-1500:], p.stderr[-500:])
    return False


def runtime_ok():
    r = subprocess.run([PY, "-c", "import importlib.util as u, jax, torch\n"
                        "assert u.find_spec('vllm') and u.find_spec('tpu_inference')\n"
                        "print(jax.__version__, torch.__version__)"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        log(f"   runtime check OK (jax {r.stdout.split()[0]}, torch {r.stdout.split()[1]})")
        return True
    log("   runtime check FAILED:", (r.stderr or r.stdout)[-800:])
    return False


def install_runtime(built=None):
    """Fresh venv with vllm-tpu pinned. CPU torch (what vllm-tpu's own Docker
    image uses) — the default PyPI torch drags in ~3 GB of CUDA libraries that
    a TPU never uses. `built` (a date from the env dataset's manifest) pins the
    dependency resolution to that day so the compile cache keeps matching."""
    ver = CFG["vllm_tpu_version"]
    shutil.rmtree(VENV, ignore_errors=True)
    pin = ["--exclude-newer", f"{built}T23:59:59Z"] if built else []
    log("   building venv with uv" + (f" (packages as of {built})" if built else "") + "...")
    if (sh([sys.executable, "-m", "pip", "install", "-q", "uv"], "pip") == 0
            and sh([sys.executable, "-m", "uv", "venv", VENV, "--python", sys.executable, "-q"], "uv") == 0
            and sh([sys.executable, "-m", "uv", "pip", "install", "--python", PY,
                    "--torch-backend=cpu", *pin, f"vllm-tpu=={ver}"], "uv") == 0):
        return "uv"
    log("   uv failed; falling back to pip (slower)")
    shutil.rmtree(VENV, ignore_errors=True)
    if sh([sys.executable, "-m", "venv", "--without-pip", VENV], "venv") != 0:
        return None
    rc = sh([sys.executable, "-m", "pip", "--python", PY, "install", "-q",
             "--extra-index-url", "https://download.pytorch.org/whl/cpu",
             f"vllm-tpu=={ver}"], "pip")
    return "pip" if rc == 0 else None


OVERLAY_DIR = "/tmp/qwen4exp_overlay"


def install_engine_overlay():
    """Fetch the JAX/TPU Qwen4Exp implementation and copy it over the installed
    tpu_inference. Nothing serves this model without it: upstream tpu-inference has
    no Qwen4Exp, and vLLM's own recipe marks TPU unsupported for this architecture."""
    repo, commit = CFG["overlay_repo"], CFG["overlay_commit"]
    shutil.rmtree(OVERLAY_DIR, ignore_errors=True)
    log(f"   fetching the Qwen4Exp overlay ({repo.rsplit('/', 1)[-1]} @ {commit})...")
    if sh(["git", "clone", "--filter=blob:none", repo, OVERLAY_DIR], "git") != 0:
        return False
    if sh(["git", "-C", OVERLAY_DIR, "checkout", "-q", commit], "git") != 0:
        log(f"   could not check out {commit} — the overlay may have been rebased; "
            f"set overlay_commit to a current SHA")
        return False
    if not apply_engine_patches():
        return False
    spec = importlib.util.find_spec("tpu_inference")
    if spec is None:
        log("   tpu_inference is not installed; nowhere to put the overlay")
        return False
    pkg_dir = os.path.dirname(spec.origin)   # .../site-packages/tpu_inference
    src_root = Path(OVERLAY_DIR, "tpu_inference")
    for item in src_root.rglob("*"):
        dst = Path(pkg_dir, item.relative_to(src_root))
        if item.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dst)
    # Registries are per-process, and `vllm serve` is a fresh interpreter (plus worker
    # processes). The overlay's startup hook is what maps Qwen4ExpForCausalLM to the JAX
    # implementation in each of them; the fork ships it for exactly this reason and
    # expects a .pth in site-packages, whose import line runs at interpreter startup.
    # Copying the files alone is not enough — without this the server cannot resolve the
    # architecture and dies minutes into a scarce TPU slot.
    r = subprocess.run([PY, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                       capture_output=True, text=True)
    purelib = (r.stdout or "").strip()
    if not purelib:
        log("   could not locate site-packages:", (r.stderr or "")[-300:])
        return False
    pth = Path(purelib, "qwen4exp_tpu_startup.pth")
    pth.write_text("import tpu_inference.models.jax.qwen4_exp.startup\n")
    log(f"   startup hook written, so every fresh interpreter registers the arch: {pth}")
    # Prove it on a fresh interpreter before any weights are touched.
    check = (
        "import json, tempfile\n"
        "from pathlib import Path\n"
        "from transformers import AutoConfig\n"
        "d = tempfile.mkdtemp()\n"
        "Path(d, 'config.json').write_text(json.dumps({"
        "'model_type':'qwen4_exp','architectures':['Qwen4ExpForCausalLM'],"
        "'hidden_size':2560,'num_hidden_layers':48,"
        "'text_config':{'model_type':'qwen4_exp','hidden_size':2560,"
        "'num_hidden_layers':48,'hc_count':4}}))\n"
        "c = AutoConfig.from_pretrained(d)\n"
        "assert type(c).__name__ == 'Qwen4ExpConfig', type(c).__name__\n"
        "print('AUTOCONFIG_OK', c.text_config.hidden_size)\n")
    r = subprocess.run([PY, "-c", check], capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    _raw.write("[overlay] " + out.replace("\n", "\n[overlay] ") + "\n")
    if r.returncode != 0:
        log("   overlay verification failed — the server could not load the model:", out[-1200:])
        return False
    if "register() skipped" in out:
        log("   the startup hook ran but model registration was SKIPPED, so vLLM would not "
            "resolve Qwen4ExpForCausalLM to the JAX implementation. Refusing to continue.")
        return False
    log("   overlay verified: arch registered in a fresh interpreter, config parses")
    return True


def apply_engine_patches():
    """Our patches against the overlay, applied to the checkout before it is copied in.

    1. gdn-dt-bias-ignored: GDN's ``dt_bias`` must not be dropped as a "bias". The
       overlay's ``IGNORED_MISSING_SUFFIXES`` contains ``_bias``, which also matches
       ``linear_attn.dt_bias``, and ``is_ignored_missing()`` is consulted in the load
       path before anything that would preserve it. GDN gates its recurrent state with
       ``sigmoid(dt_bias + exp(A_log))``, so losing dt_bias -- 36 tensors, one per
       linear-attention layer, in this checkpoint the *only* names matching ``_bias`` --
       leaves the decay at its zero init and quietly corrupts every GDN layer.

    2. ple-spill-mmap: the n-gram table is dequantized into a bf16 *host* buffer, which
       for 51.2e9 params is 102 GB, on top of the ~51 GB of fp8 shards held while it is
       assembled -- about 154 GB of peak host RAM. Kaggle's container reports 33 GB
       total, so the published path cannot fit. The patch spills the fp8 shards to disk
       and memory-maps them, dequantising only the rows actually gathered, and selects
       itself automatically from MemAvailable, so a large host keeps the dense path.
    """
    patches = [("gdn-dt-bias-ignored.diff", GDN_PATCH_B64,
                "GDN dt_bias is kept (the _bias rule used to drop it)"),
               ("ple-spill-mmap.diff", PLE_SPILL_PATCH_B64,
                "the n-gram table can spill to disk instead of a 102 GB host buffer")]
    if not any(blob for _, blob, _ in patches):
        log("   (no engine patches embedded — serving the overlay as published)")
        return True
    pfdir = Path("/tmp/engine_patches")
    pfdir.mkdir(exist_ok=True)
    for fname, blob, what in patches:
        if not blob:
            continue
        pf = pfdir / fname
        pf.write_bytes(gzip.decompress(base64.b64decode(blob)))
        r = subprocess.run(["patch", "-p1", "-d", OVERLAY_DIR, "-i", str(pf),
                            "--no-backup-if-mismatch", "-N"], capture_output=True, text=True)
        _raw.write(r.stdout + r.stderr)
        if r.returncode != 0 and "previously applied" not in (r.stdout + r.stderr):
            log(f"   engine patch {fname} did NOT apply — the overlay has moved; refusing"
                " to serve a silently-wrong model. Check the patch against the pinned"
                " commit, or bump overlay_commit.",
                (r.stdout or "")[-600:], (r.stderr or "")[-400:])
            return False
        log(f"   engine patch applied: {what}")
    return True


def _safetensors_tensor_names(path):
    """Tensor names from a safetensors header, without importing safetensors (only the
    venv interpreter has it; this helper may run on the kernel's own one)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode("utf-8"))
    hdr.pop("__metadata__", None)
    return list(hdr)


def ensure_usable_checkpoint(model_path):
    """vLLM enumerates safetensors shards from model.safetensors.index.json, and this
    dataset's index is 34 MB of long tensor names — the kind of file that has to be
    readable before anything else can happen.

    Kaggle's file-download API returns it mangled: the first 16 MiB is written twice and
    the middle is dropped (reproduced twice here, byte-identical sha256, both unparseable).
    A mounted dataset copy is probably intact, but a truncated index would abort a run
    minutes into a TPU session, so check it and rebuild from the shard headers if needed.
    Headers are a few hundred KB per shard; no weight data is read or copied here.
    """
    idx = Path(model_path, "model.safetensors.index.json")
    if idx.exists():
        try:
            json.loads(idx.read_text())
            log("   index.json parses; serving the dataset exactly as mounted")
            return model_path
        except Exception as e:  # noqa: BLE001
            log(f"   index.json is unusable ({str(e)[:90]}) -> rebuilding it from headers")
    shards = sorted(Path(model_path).glob("*.safetensors"))
    if not shards:
        log("   no *.safetensors in the mounted dataset — cannot continue")
        return model_path
    t = time.time()
    wmap = {}
    for sp in shards:
        for name in _safetensors_tensor_names(sp):
            wmap[name] = sp.name
    log(f"   read {len(shards)} shard headers -> {len(wmap)} tensors in {int(time.time() - t)} s")
    staged = Path(WORK, "checkpoint")
    staged.mkdir(parents=True, exist_ok=True)
    for f in Path(model_path).iterdir():
        dst = staged / f.name
        if not dst.exists():
            os.symlink(f, dst)
    (staged / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": sum(s.stat().st_size for s in shards)}, "weight_map": wmap}))
    log(f"   staged a repaired checkpoint at {staged} (symlinks; nothing copied)")
    return str(staged)


def tpu_check():
    """Kaggle sometimes starts a "TPU" session with no TPU attached (a CPU-only container; most often on new or
    not-yet-verified accounts). jax then sees one device and vLLM dies minutes later with "Insufficient devices for
    2D mesh: found 1, expected 8" or "No jellyfish device found". Look before installing anything (~20 s)."""
    code = ("import jax\n"
            "try:\n"
            "    d = jax.devices()\n"
            "    print('TPU_CHECK', len(d), d[0].platform, getattr(d[0], 'device_kind', ''))\n"
            "except Exception as e:\n"
            "    print('TPU_CHECK 0 none', str(e).replace(chr(10), ' ')[:200])\n")
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
        out = (r.stdout or "") + (r.stderr or "")
        m = re.search(r"TPU_CHECK (\d+) (\S+)(.*)", out)
    except Exception as e:  # noqa: BLE001
        log(f"   (TPU check skipped: {e})")
        return
    if m is None:
        log("   (TPU check skipped: the image's jax did not answer; details in vllm.log)")
        _raw.write("[tpu-check] " + out.replace("\n", "\n[tpu-check] ") + "\n")
        return
    n, platform, rest = int(m.group(1)), m.group(2), m.group(3).strip()
    if n == 8 and platform == "tpu":
        log(f"   TPU check OK: 8 chips ({rest})")
        return
    if n == 0 and not re.search(r"jellyfish|TPU initialization failed|initialize backend 'tpu'|No TPU|vfio", rest, re.I):
        log(f"   (TPU check inconclusive, continuing: {rest[:160]})")   # only a clear no-TPU signal stops the run
        return
    msg = (f"this session has no working TPU: jax sees {n} {platform} device(s) {rest}. Kaggle sometimes starts a "
           "TPU session without one (most often on new or not-yet-verified accounts); nothing in this notebook can "
           "fix that. Stop the session and start it again. If it repeats, run `import jax; print(jax.device_count())` "
           "in a fresh cell first: it must print 8 before this script is worth running.")
    log("   " + msg)
    publish("failed", step="no-tpu", tail=msg)
    sys.exit(1)


def server_death_report(max_lines=60):
    """What killed vLLM, from vllm.log: the root-cause exception line, the first error block and a hint for the
    causes we have seen. The console tail alone scrolls the cause away. Returns (cause, block, hint)."""
    try:
        lines = [l for l in RAW_LOG.read_text(errors="replace").splitlines() if l.startswith("[vllm]")]
    except Exception:  # noqa: BLE001
        return "", "", ""
    strip = re.compile(r"^\[vllm\] (?:\((?:EngineCore|APIServer|Worker)[^)]*\) )?(?:ERROR|CRITICAL) [\d-]+ [\d:]+ \[[^\]]+\] ?")
    start_pat = re.compile(r"EngineCore (?:failed|encountered|hit)|Traceback \(most recent call last\)|RESOURCE_EXHAUSTED|INVALID_ARGUMENT|NOT_FOUND")
    start = next((i for i, l in enumerate(lines) if start_pat.search(l)), None)
    block = [strip.sub("", l) for l in lines[start:start + max_lines]] if start is not None else []
    cause = ""
    for l in block:                                   # the first traceback's own exception line is the root cause
        t = l.strip()
        if re.match(r"^[\w.]*(?:Error|Exception)\b.*:", t) or re.match(r"^(?:INVALID_ARGUMENT|NOT_FOUND|RESOURCE_EXHAUSTED)", t):
            cause = t
            break
    joined = "\n".join(block)
    hint = ""
    if re.search(r"found 1, expected 8|jellyfish|unexpected worker hostname|TPU initialization failed", joined):
        hint = ("this session has no working TPU (Kaggle sometimes starts one without, most often on new or "
                "not-yet-verified accounts): stop the session and start it again")
    elif "__delitem__" in joined:
        hint = ("a client sent a JSON-mode / structured-output request while MTP and async scheduling are on, which "
                "vllm-tpu 0.28.0 cannot handle: set \"async_scheduling\": false (or \"mtp_tokens\": 0) and run again")
    elif "RESOURCE_EXHAUSTED" in joined or "out of memory" in joined.lower():
        hint = "the TPU ran out of HBM: lower max_model_len or max_num_seqs"
    return cause, joined, hint


def server_died(server, phase, **extra):
    """Log why the vLLM server exited (root cause first), publish it, and stop the kernel."""
    cause, block, hint = server_death_report()
    tail = "\n".join(list(server.tail)[-40:]) if getattr(server, "tail", None) else ""
    log(f"server exited rc={server.returncode}" + (f" — root cause: {cause}" if cause else ""))
    if hint:
        log(f"   -> {hint}")
    if block:
        log("--- first error block from vllm.log ---\n" + block)
    if tail and not block:
        log("--- last output ---\n" + tail)
    log(f"full log: {RAW_LOG}")
    head = (f"root cause: {cause}\n" if cause else "") + (f"{hint}\n" if hint else "")
    publish(phase, rc=server.returncode, cause=cause, hint=hint,
            tail=(head + "\n" + (block or tail)[-2200:]).strip(), **extra)
    sys.exit(1)


# ---------------- 1. runtime ----------------
banner(1, "Python runtime", f"vllm-tpu {CFG['vllm_tpu_version']} + the Qwen4Exp overlay")
tpu_check()
threading.Thread(target=fetch_cloudflared, daemon=True).start()
bundle_root = find_input(CFG["env_dataset"].split("/")[-1]) if CFG["env_dataset"] else None
bundle, manifest = None, {}
if bundle_root:
    # Kaggle may keep the files at the top level or under the kernel's output folder
    hits = glob.glob(f"{bundle_root}/manifest.json") + glob.glob(f"{bundle_root}/*/manifest.json")
    if hits:
        bundle = os.path.dirname(hits[0])
        manifest = json.loads(Path(hits[0]).read_text())
    else:
        bundle = bundle_root
if bundle and Path(bundle, "cloudflared").exists() and not CLOUDFLARED.exists():
    shutil.copy(Path(bundle, "cloudflared"), CLOUDFLARED)
    CLOUDFLARED.chmod(0o755)
if manifest and (manifest.get("python") != PY_VER
                 or manifest.get("vllm_tpu_version") != CFG["vllm_tpu_version"]):
    log(f"   env dataset was built for python {manifest.get('python')} / vllm-tpu "
        f"{manifest.get('vllm_tpu_version')}; this session has python {PY_VER} and wants "
        f"vllm-tpu {CFG['vllm_tpu_version']} -> its compile cache will not match")
    manifest = {}
if not bundle:
    log(f"   env dataset not attached (expected {CFG['env_dataset']}) -> cold compile later")

t = time.time()
publish("install", vllm_tpu=CFG["vllm_tpu_version"])
log("   installer output goes to", RAW_LOG)
runtime = install_runtime(manifest.get("built"))
if runtime is None or not runtime_ok():
    publish("failed", step="install")
    sys.exit(1)
publish("installed", secs=int(time.time() - t), via=runtime)
if not install_engine_overlay():
    publish("failed", step="engine-overlay",
            note="the Qwen4Exp overlay did not install or did not import")
    sys.exit(1)
publish("overlay-applied", commit=CFG["overlay_commit"])
log(f"   runtime ready in {int(time.time() - t)} s")

# ---------------- 2. XLA compile cache ----------------
banner(2, "XLA compile cache")
t = time.time()
cache_tar = (Path(bundle, "xla_cache.tar") if bundle and Path(bundle, "xla_cache.tar").exists()
             else find_input("*/xla_cache*.tar.gz", "xla_cache*.tar.gz"))
cache_dir = find_input("*/*/xla_cache", "*/xla_cache", "xla_cache")
if cache_tar:
    flags = "-xf" if str(cache_tar).endswith(".tar") else "-xzf"
    sh(["tar", flags, str(cache_tar), "-C", "/tmp"], "tar")
elif cache_dir:
    sh(["cp", "-r", cache_dir, "/tmp/"], "cp")
    sh(["chmod", "-R", "u+w", XLA_CACHE], "chmod")
n_entries = len(glob.glob(XLA_CACHE + "/*"))
cache_configs = manifest.get("configs", [])
this_config = [CFG["max_model_len"], CFG["max_num_seqs"], CFG["mtp_tokens"], CFG["text_only"]]
if n_entries:
    covered = (not cache_configs) or (this_config in cache_configs)
    publish("cache-restored", entries=n_entries, secs=int(time.time() - t),
            covers_this_config=covered)
    if not covered:
        log(f"   note: the cache was built for [ctx, seqs, mtp, text_only] in {cache_configs}; "
            f"this run uses {this_config} -> its graphs compile cold (add ~10-15 min)")
    else:
        log("   compiled TPU graphs for this exact config are cached -> fast start")
else:
    publish("cache-missing", note="cold compile: expect ~10 extra minutes")

# ---------------- 3. weights ----------------
banner(3, "Model weights", "NVFP4 experts + FP8 n-gram table + bf16 rest, 135 GB total")
# The n-gram table is never a JAX parameter: the overlay dequantizes the FP8 shards into a
# bf16 HOST buffer (PLE_HOST_TABLES) and gathers rows there. 51.2e9 params x 2 bytes is
# ~102 GB of host RAM, so report what the box actually has before the load starts.
try:
    _mi = dict(l.split(":", 1) for l in Path("/proc/meminfo").read_text().splitlines())
    _kb = lambda k: int(_mi[k].split()[0]) / 1e6  # kB -> GB
    log(f"   host RAM: {_kb('MemTotal'):.0f} GB total, {_kb('MemAvailable'):.0f} GB available "
        f"(the n-gram table wants ~102 GB as a bf16 host buffer)")
except Exception:
    pass
# Two ways in. A Kaggle *Model* attaches this 135 GB checkpoint fine; a Kaggle *Dataset* of
# the same size stalls the session before it ever starts (reproduced at 119 GB across two
# datasets and at 135 GB, while 77 GB mounts in seconds). Prefer the model when one is
# attached, fall back to a dataset, and only then to a Hugging Face download.
model_path = find_model_dir()
if model_path:
    publish("weights-mounted", path=model_path, source="kaggle-model")
else:
    weights_slug = CFG["weights_dataset"].split("/")[-1]
    model_path = find_input(weights_slug)
    if model_path and not os.path.exists(os.path.join(model_path, "config.json")):
        # Kaggle keeps dataset files either at the top level or one directory down
        hits = glob.glob(f"{model_path}/**/config.json", recursive=True)
        if hits:
            model_path = os.path.dirname(hits[0])
    if model_path and os.path.exists(os.path.join(model_path, "config.json")):
        publish("weights-mounted", path=model_path, source="kaggle-dataset")
    else:
        publish("weights-download", model=CFG["hf_model_id"],
                note="attach the Kaggle Model instead: the HF mirror is ~135 GB and will "
                     "not fit in a session's own scratch space")
        t = time.time()
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(CFG["hf_model_id"], allow_patterns=[
            "*.safetensors", "*.json", "*.txt", "tokenizer*", "vocab*", "merges*", "*.jinja"])
        publish("weights-downloaded", secs=int(time.time() - t))
log(f"   weights at {model_path}")
_quant = os.path.join(model_path, "hf_quant_config.json")
if os.path.exists(_quant):
    _q = json.loads(Path(_quant).read_text()).get("quantization", {})
    log(f"   quant config: {_q.get('quant_algo', '?')} (group_size {_q.get('group_size', '?')})")
else:
    log("   note: no hf_quant_config.json next to config.json — if these are bf16 weights "
        "they cannot fit in 128 GB of HBM. See ../tools/NOTES.md")
model_path = ensure_usable_checkpoint(model_path)


# ---------------- 4. vLLM server ----------------
NOISE = ("vllm._C", "metadata.google.internal", "Triton is installed", "Transparent hugepages",
         "Pin memory is not supported", "Expect torch.Tensor", "Inductor compilation",
         "cloud_tpu_init.py", "SyntaxWarning", "Compilation of worker", "AOT lower skipped",
         "torch_dtype", "UserWarning", "warnings.warn", "resource_tracker", "Precompile worker0 sample",
         "Precompile worker0 gather", "Precompile worker0 compute_and_gather")


def server_args(cfg):
    args = [PY, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--tensor-parallel-size", "8",
            "--max-model-len", str(cfg["max_model_len"]),
            "--max-num-seqs", str(cfg["max_num_seqs"]),
            "--port", str(PORT),
            "--api-key", cfg["api_key"],
            "--served-model-name", cfg["served_model_name"],
            # vLLM re-attaches quantization_config from the raw config after AutoConfig
            # parsing and then resolves its CUDA-only quant path (get_config / the
            # auto_gptq and modelopt_fp4 gates). hf-overrides is applied last, so this
            # nulls it and leaves the weights to be dequantized JAX-side at load — which
            # is how the overlay serves them. The overlay also strips it at parse time.
            "--hf-overrides", json.dumps({"quantization_config": None}),
            "--reasoning-parser", "qwen3"]
    if cfg.get("async_scheduling") is not None:
        args.append("--async-scheduling" if cfg["async_scheduling"] else "--no-async-scheduling")
    if cfg["text_only"]:
        # Qwen3.8 is a vision-language checkpoint; we only serve text. This skips
        # the vision tower and roughly halves the number of TPU graphs to compile.
        args += ["--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0})]
    if cfg["mtp_tokens"] > 0:
        args += ["--speculative-config",
                 json.dumps({"method": "mtp", "num_speculative_tokens": cfg["mtp_tokens"]})]
    if cfg["tool_call_parser"]:
        args += ["--enable-auto-tool-choice", "--tool-call-parser", cfg["tool_call_parser"]]
    if cfg["reasoning_effort_default"] != "xhigh":
        # The chat template defaults reasoning_effort to 'xhigh'; ship a copy with a
        # different default so the server-side default changes without client changes.
        tc = json.loads(Path(model_path, "tokenizer_config.json").read_text())
        template = tc["chat_template"].replace(
            "reasoning_effort|default('xhigh')",
            f"reasoning_effort|default('{cfg['reasoning_effort_default']}')")
        Path("/tmp/chat_template.jinja").write_text(template)
        args += ["--chat-template", "/tmp/chat_template.jinja"]
    return args


def n_token_graphs():
    n, b = 1, CFG["min_token_bucket"]
    while b < 2048:  # vLLM's default max_num_batched_tokens on TPU
        b *= 2
        n += 1
    return n


def make_translator():
    """Turns vLLM's firehose into a handful of human lines. Everything raw still
    lands in vllm.log."""
    st = {"graph": 0, "loads": 0, "said": set()}
    n_graphs = n_token_graphs()

    def once(key, msg):
        if key not in st["said"]:
            st["said"].add(key)
            log(msg)

    def tr(line):
        if CFG["verbose"]:
            print(f"[vllm] {line[:500]}", flush=True)
            return
        if any(k in line for k in NOISE):
            return
        m = re.search(r"Loading weights took ([\d.]+) seconds", line)
        if m:
            st["loads"] += 1
            if st["loads"] == 1:
                log(f"   weights read from the dataset in {float(m.group(1)):.0f} s")
            return
        m = re.search(r"load model weights from storage to TPU: ([\d.]+)", line)
        if m:
            if st["loads"] <= 1:
                log(f"   weights sharded across the 8 TPU chips ({float(m.group(1)):.0f} s)")
            else:
                log("   MTP draft head loaded")
            return
        m = re.search(r"KV cache size: ([\d,]+) tokens", line)
        if m:
            log(f"   KV cache fits {m.group(1)} tokens")
            return
        if "Precompile all the subgraphs" in line:
            log(f"   compiling TPU graphs — {n_graphs} text graphs"
                + ("" if CFG["text_only"] else ", the same again for image inputs,")
                + " + helpers (~20 s each if cached, ~1 min if not)")
            return
        m = re.search(r"Precompile worker\d+ backbone --> \{'num_tokens': (\d+)", line)
        if m:
            st["graph"] += 1
            log(f"     graph {st['graph']}/{n_graphs}: batches of {m.group(1)} tokens")
            return
        if "embed_multimodal" in line or "input_embeddings_merger" in line:
            once("vision-enc", "     warming the image encoder (~5 min; \"text_only\": true skips it)")
            return
        if "backbone with embeds" in line:
            once("vision", "     compiling image-input graphs (~3 min)")
            return
        m = re.search(r"Warm-up call pass finished in ([\d.]+) \[secs\] over (\d+) tasks", line)
        if m:
            if float(m.group(1)) > 5:
                log(f"     warm-up run of {m.group(2)} graphs done ({float(m.group(1)):.0f} s)")
            return
        if "Precompile" in line and "drafter" in line:
            once("mtp", "     compiling speculative-decoding (MTP) graphs")
            return
        if "Precompile" in line or "Compilation of" in line:
            once("helpers", "     compiling sampler / helper graphs")
            return
        if "Application startup complete" in line:
            return
        if " ERROR " in line or "Traceback" in line or "Error:" in line or "rror(" in line:
            print(time.strftime("[%H:%M:%S] ") + f"   [vllm] {line[:400]}", flush=True)
    return tr


def launch_server(cfg):
    publish("server-launch", max_model_len=cfg["max_model_len"],
            max_num_seqs=cfg["max_num_seqs"], mtp=cfg["mtp_tokens"],
            text_only=cfg["text_only"], min_token_bucket=cfg["min_token_bucket"])
    tail = collections.deque(maxlen=200)
    env = os.environ.copy()
    env["TPU_BACKEND_TYPE"] = "jax"           # tpu-inference's JAX path, where the overlay lives
    if cfg.get("ple_cpu_offload"):
        env["VLLM_PLE_CPU_OFFLOAD"] = "1"    # keep the 51B n-gram table in host RAM
    p = subprocess.Popen(server_args(cfg), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, env=env)
    tr = make_translator()

    def pump():
        for line in p.stdout:
            line = line.rstrip()
            if line:
                tail.append(line)
                _raw.write(f"[vllm] {line}\n")
                tr(line)
    threading.Thread(target=pump, daemon=True).start()
    p.tail = tail
    return p


def healthy(cfg):
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/models",
                                     headers={"Authorization": f"Bearer {cfg['api_key']}"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def wait_healthy(server, cfg, expect_min):
    t = time.time()
    while time.time() - t < 5400:
        if server.poll() is not None:
            server_died(server, "failed", step="server")
        if healthy(cfg):
            return int(time.time() - t)
        el = int(time.time() - t)
        if el and el % 120 < 6:
            publish("compiling", elapsed_s=el)
            log(f"   ... {el // 60} min into startup (typically ~{expect_min} min)")
        time.sleep(5)
    publish("failed", step="health-timeout", tail="\n".join(list(server.tail)[-60:])[-2500:])
    sys.exit(1)


def stop_server(p):
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=90)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=30)
    time.sleep(10)  # let the TPU runtime free the chips


def completion(cfg, prompt, max_tokens, stream=False, timeout=900):
    body = {"model": cfg["served_model_name"], "prompt": prompt,
            "max_tokens": max_tokens, "temperature": 0.0}
    if stream:
        body.update(stream=True, ignore_eos=True, stream_options={"include_usage": True})
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {cfg['api_key']}"})
    if not stream:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    t0 = time.time(); ttft = None; gen = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            ch = obj.get("choices") or []
            if ch and ch[0].get("text"):
                if ttft is None:
                    ttft = time.time() - t0
                gen += 1
            if obj.get("usage"):
                gen = obj["usage"].get("completion_tokens", gen)
    return ttft, time.time() - t0, gen


def test_png(w=256, h=256, rgb=(200, 30, 30)):
    """A solid-colour PNG without PIL, for the image self-test."""
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def chat(cfg, messages, max_tokens=32, timeout=600):
    body = {"model": cfg["served_model_name"], "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {cfg['api_key']}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def self_test(cfg):
    """Warms the remaining lazy paths, checks an image request when images are
    enabled, and reports single-stream decode speed."""
    tps = None
    try:
        completion(cfg, "Hello", 8)
        txt = completion(cfg, "The capital of France is", 8)["choices"][0]["text"]
        ttft, total, gen = completion(cfg, "Write a short story about a lighthouse.", 192, stream=True)
        tps = (gen - 1) / (total - ttft) if gen > 1 else 0.0
        publish("benchmark", decode_tok_s=round(tps, 1), sanity=txt.strip()[:60])
        log(f"   self-test: {tps:.1f} tok/s single-stream decode; "
            f"'The capital of France is' -> {txt.strip()[:40]!r}")
    except Exception as e:
        publish("benchmark-error", err=str(e)[:200])
    if not cfg["text_only"]:
        try:
            img = "data:image/png;base64," + base64.b64encode(test_png()).decode()
            ans = chat(cfg, [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": img}},
                {"type": "text", "text": "What colour is this image? One word."}]}])
            publish("image-test", answer=ans.strip()[:40])
            log(f"   image request works (a red square -> {ans.strip()[:30]!r})")
        except Exception as e:
            publish("image-test-failed", err=str(e)[:200])
            log(f"   IMAGE REQUEST FAILED: {str(e)[:200]}")
    return tps


def exercise(cfg, quiet=False):
    """Hit the shapes a real client hits: short and long prompts, streaming, a
    small concurrent batch. In build mode this puts their graphs in the cache;
    in fast_start mode it loads them so users don't hit the one-time stalls."""
    steps = [("short prompt", lambda: completion(cfg, "Hello", 8)),
             ("4k-token prompt", lambda: completion(
                 cfg, "The quick brown fox jumps over the lazy dog. " * 400, 16, timeout=1200)),
             ("streaming", lambda: completion(
                 cfg, "Write a short story about a lighthouse.", 64, stream=True))]
    n = min(cfg["max_num_seqs"], 8)
    errs = []

    def one():
        try:
            completion(cfg, "Count from one to twenty in words.", 48)
        except Exception as e:
            errs.append(str(e)[:200])

    def batch():
        ths = [threading.Thread(target=one) for _ in range(n)]
        [t.start() for t in ths]
        [t.join() for t in ths]
    steps.append((f"{n} parallel requests", batch))
    for name, fn in steps:
        t = time.time()
        try:
            fn()
        except Exception as e:
            errs.append(f"{name}: {str(e)[:200]}")
        if not quiet:
            log(f"   warmed: {name} ({time.time() - t:.0f} s)")
    return errs


# ---------------- maintainer mode: build the env dataset ----------------
BUILD_CONFIGS = [
    # Guesses: this recipe has no measured baseline. Build the two configs a first
    # debugging session is most likely to sit in.
    {"max_model_len": 32768, "max_num_seqs": 8, "mtp_tokens": 0, "text_only": True},
    {"max_model_len": 65536, "max_num_seqs": 4, "mtp_tokens": 0, "text_only": True},
]
if CFG["build_bundle"]:
    banner(4, "BUILD MODE", "serving each config once to populate the XLA cache")
    log("   TPU-related env:", {k: v for k, v in os.environ.items() if "TPU" in k or "PJRT" in k})
    results = {}
    for c in BUILD_CONFIGS:
        cfg = {**CFG, **c}
        key = (f"{c['max_model_len']}/{c['max_num_seqs']}/mtp{c['mtp_tokens']}"
               + ("/text" if c["text_only"] else "/mm"))
        server = launch_server(cfg)
        secs = wait_healthy(server, cfg, 30)
        errs = exercise(cfg, quiet=True)
        tps = self_test(cfg)
        stop_server(server)
        results[key] = {"startup_secs": secs, "decode_tok_s": tps, "errors": errs}
        publish("build-config-done", config=key, **results[key])
    # probe: how fast is a start with SKIP_JAX_PRECOMPILE=1 now that the cache is warm?
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"
    cfg = {**CFG, **BUILD_CONFIGS[0]}
    server = launch_server(cfg)
    secs = wait_healthy(server, cfg, 5)
    lat = []
    for i in range(3):
        t = time.time()
        try:
            completion(cfg, ["Hello", "Say hi.", "Name a color."][i], 8, timeout=1800)
            lat.append(round(time.time() - t, 1))
        except Exception as e:
            lat.append(str(e)[:100])
    tps = self_test(cfg)
    stop_server(server)
    os.environ.pop("SKIP_JAX_PRECOMPILE")
    publish("probe-skip-precompile", startup_secs=secs, first_request_secs=lat, decode_tok_s=tps)
    # also a warm re-start of the default config (what users will see)
    cfg = {**CFG, **BUILD_CONFIGS[0]}
    server = launch_server(cfg)
    secs = wait_healthy(server, cfg, 15)
    tps = self_test(cfg)
    stop_server(server)
    publish("probe-warm-restart", startup_secs=secs, decode_tok_s=tps)
    cfg = {**CFG, **BUILD_CONFIGS[2]}
    server = launch_server(cfg)
    secs = wait_healthy(server, cfg, 8)
    tps = self_test(cfg)
    stop_server(server)
    publish("probe-warm-restart-text-only", startup_secs=secs, decode_tok_s=tps)

    out = WORK / "bundle"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    banner(5, "packing the bundle", str(out))
    # (no venv tarball: uv rebuilds the identical env in ~30 s, and Kaggle would
    #  unpack a tar into 100k files anyway — some with '[' in the name, which it rejects)
    sh(["tar", "-cf", str(out / "xla_cache.tar"), "-C", "/tmp", "xla_cache"], "tar")
    fetch_cloudflared()
    if CLOUDFLARED.exists():
        shutil.copy(CLOUDFLARED, out / "cloudflared")
    pkgs = subprocess.run([sys.executable, "-m", "uv", "pip", "list", "--python", PY,
                           "--format=json"], capture_output=True, text=True)
    try:
        pkgs = {d["name"]: d["version"] for d in json.loads(pkgs.stdout)}
    except Exception:
        pkgs = {}
    manifest = {
        "built": time.strftime("%Y-%m-%d"),
        "python": PY_VER,
        "vllm_tpu_version": CFG["vllm_tpu_version"],
        "mtp_patch": "applied at runtime",
        "min_token_bucket": CFG["min_token_bucket"],
        "configs": [[c["max_model_len"], c["max_num_seqs"], c["mtp_tokens"], c["text_only"]]
                    for c in BUILD_CONFIGS],
        "results": results,
        "accelerator": "TPU v5e-8 (Kaggle)",
        "packages": pkgs,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    sizes = {p.name: round(p.stat().st_size / 1e9, 2) for p in out.iterdir()}
    publish("bundle-built", sizes_gb=sizes, results=results)
    sys.exit(0)

# ---------------- 4. launch ----------------
# No measured baseline exists for this recipe (see ../README.md) — this number is a
# guess that keeps the progress lines honest, not a promise.
expect_min = CFG["expect_min"]
if CFG["fast_start"]:
    expect_min = 10
    log("   fast_start: skipping precompile; every new request shape then compiles cold "
        "(~1 min each) unless a matching cache is attached")
banner(4, "Starting vLLM", f"TP=8, ctx {CFG['max_model_len']}, {CFG['max_num_seqs']} seqs, "
       f"MTP k={CFG['mtp_tokens']}, {'text-only' if CFG['text_only'] else 'multimodal'}")
log(f"   expect ~{expect_min} min; progress lines below, full vLLM log in {RAW_LOG}")
server = launch_server(CFG)

# ---------------- 5. tunnel (in parallel with the server start) ----------------
banner(5, "Public URL")
url = None
tunnel = None
for _ in range(60):  # cloudflared download runs in the background from step 1
    if CLOUDFLARED.exists():
        break
    time.sleep(2)
if CLOUDFLARED.exists():
    tunnel = subprocess.Popen([str(CLOUDFLARED), "tunnel", "--url", f"http://127.0.0.1:{PORT}",
                               "--no-autoupdate", "--protocol", "quic"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    pat = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    lines = []

    def pump_cf():
        for line in tunnel.stdout:
            lines.append(line.rstrip())
            _raw.write(f"[cloudflared] {line}")
    threading.Thread(target=pump_cf, daemon=True).start()
    deadline = time.time() + 180
    while time.time() < deadline and url is None:
        for ln in lines:
            m = pat.search(ln)
            if m:
                url = m.group(0).rstrip("/")
                break
        time.sleep(1)
if url:
    log(f"   your endpoint will be  {url}/v1")
    log("   (not live yet — it answers 502 until the READY banner below)")
    publish("tunnel-url", endpoint=f"{url}/v1")
else:
    publish("tunnel-failed", note="server still reachable inside the kernel on :8000")

# ---------------- 6. wait, announce, self-test, keep alive ----------------
startup = wait_healthy(server, CFG, expect_min)
publish("serving", startup_secs=startup)
log("")
log("#" * 70)
log(f"#  READY — the server is live ({elapsed()} after start)")
log(f"#  ENDPOINT : {url + '/v1' if url else 'http://127.0.0.1:8000/v1 (tunnel failed)'}")
log(f"#  API KEY  : {CFG['api_key']}")
log(f"#  MODEL    : {CFG['served_model_name']}   (context {CFG['max_model_len']}, "
    f"{CFG['max_num_seqs']} parallel requests)")
log("#" * 70)
log("#  Try it:")
log(f"#    curl {url + '/v1' if url else 'http://127.0.0.1:8000/v1'}/chat/completions \\")
log(f"#      -H 'Authorization: Bearer {CFG['api_key']}' -H 'Content-Type: application/json' \\")
log("#      -d '{\"model\": \"" + CFG["served_model_name"] + "\", \"messages\": [{\"role\": \"user\", "
    "\"content\": \"Hello!\"}], \"chat_template_kwargs\": {\"reasoning_effort\": \"low\"}}'")
log(f"#  Serving for up to {CFG['keepalive_min']} min, then this cell exits on its own.")
log("#" * 70)
publish("ready", endpoint=(f"{url}/v1" if url else None), api_key=CFG["api_key"],
        model=CFG["served_model_name"], max_model_len=CFG["max_model_len"],
        keepalive_min=CFG["keepalive_min"], startup_secs=startup)

if CFG["fast_start"]:
    banner(6, "Warm-up", "loading the common request shapes; the endpoint is usable meanwhile")
    log("   (fast_start: a request with a new shape waits ~1 min the first time)")
    exercise(CFG)
else:
    banner(6, "Self-test", "one short generation; the endpoint is usable meanwhile")
self_test(CFG)

t_serve = time.time()
while time.time() - t_serve < CFG["keepalive_min"] * 60:
    time.sleep(120)
    if server.poll() is not None:
        server_died(server, "stopped", reason="server-exit")
    up = int((time.time() - t_serve) / 60)
    if up % 10 < 2:
        publish("heartbeat", up_min=up, endpoint=(f"{url}/v1" if url else None))
        log(f"   still serving ({up} min) — {url + '/v1' if url else ''}")
publish("auto-shutdown", served_min=CFG["keepalive_min"])
server.terminate()
sys.exit(0)
