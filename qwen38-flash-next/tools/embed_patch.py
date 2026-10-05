#!/usr/bin/env python3
"""Re-embed every patch in patches/ into kernel/serve_qwen38_flash_next.py
(paths relative to this model folder).

Run after adding or editing a patch:  python qwen38-flash-next/tools/embed_patch.py
"""
import base64
import gzip
import re
from pathlib import Path

# patch file -> the kernel variable that carries it
PATCHES = [
    ("gdn-dt-bias-ignored.diff", "GDN_PATCH_B64"),
    ("ple-spill-mmap.diff", "PLE_SPILL_PATCH_B64"),
]

repo = Path(__file__).resolve().parent.parent
script_path = repo / "kernel" / "serve_qwen38_flash_next.py"
src = script_path.read_text()
for fname, var in PATCHES:
    diff = (repo / "patches" / fname).read_bytes()
    blob = base64.b64encode(gzip.compress(diff, 9)).decode()
    src, n = re.subn(rf'^{var} = .*$',
                     f'{var} = "{blob}"  # __EMBEDDED_{var}__',
                     src, count=1, flags=re.M)
    if n != 1:
        raise SystemExit(f"marker line {var} = ... not found")
    print(f"embedded {fname}: {len(diff)} bytes of diff as {len(blob)} chars of base64")
script_path.write_text(src)
