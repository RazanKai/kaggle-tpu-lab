#!/usr/bin/env python3
"""Re-embed patches/gdn-dt-bias-ignored.diff into kernel/serve_qwen38_flash_next.py
(paths relative to this model folder).

Run after editing the patch file:  python qwen38-flash-next/tools/embed_patch.py
"""
import base64
import gzip
import re
from pathlib import Path

repo = Path(__file__).resolve().parent.parent
diff = (repo / "patches" / "gdn-dt-bias-ignored.diff").read_bytes()
blob = base64.b64encode(gzip.compress(diff, 9)).decode()

script_path = repo / "kernel" / "serve_qwen38_flash_next.py"
src = script_path.read_text()
new, n = re.subn(r'^GDN_PATCH_B64 = .*$',
                 f'GDN_PATCH_B64 = "{blob}"  # __EMBEDDED_GDN_PATCH__',
                 src, count=1, flags=re.M)
if n != 1:
    raise SystemExit("marker line GDN_PATCH_B64 = ... not found")
script_path.write_text(new)
print(f"embedded {len(diff)} bytes of diff as {len(blob)} chars of base64")
