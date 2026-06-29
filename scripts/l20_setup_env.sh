#!/usr/bin/env bash
# L20 (Ada sm_89) environment setup for tokenspeed-kernel.
# Run: ferry run --target l20-fresh -- bash /home/tiger/tokenspeed/scripts/l20_setup_env.sh
set -euo pipefail

WS="${WS:-/home/tiger/tokenspeed}"

export TOKENSPEED_KERNEL_BACKEND=cuda
export FLASHINFER_CUDA_ARCH_LIST="8.9"
export TOKENSPEED_CUDA_ARCH="89"

echo "=== Step 1: install common deps ==="
python3 -m pip install --user \
    "apache-tvm-ffi>=0.1.5,<=0.1.11" \
    numpy packaging PyYAML psutil wheel \
    "tokenspeed-triton>=3.7.10.post20260505" \
    "tokenspeed-proton>=3.7.10.post20260505"

echo "=== Step 2: locate cccl headers ==="
CCCL_INC=$(python3 -c "
import nvidia, pathlib
for base in nvidia.__path__:
    for sub in ['cu13/include/cccl', 'cuda/cccl/include']:
        p = pathlib.Path(base) / sub
        if (p / 'cub').exists() or (p / 'thrust').exists():
            print(p); raise SystemExit
print('')
" 2>/dev/null || echo "")
if [ -z "$CCCL_INC" ]; then
    CCCL_INC=$(python3 -c "
import flashinfer, pathlib
root = pathlib.Path(flashinfer.__file__).parent / 'data'
for p in root.rglob('cccl'):
    if (p / 'cub').exists() or (p / 'thrust').exists():
        print(p); raise SystemExit
print('')
" 2>/dev/null || echo "")
fi
if [ -z "$CCCL_INC" ]; then
    echo "ERROR: cannot find cccl headers" >&2
    exit 1
fi
echo "cccl headers: $CCCL_INC"
export CPLUS_INCLUDE_PATH="$CCCL_INC:${CPLUS_INCLUDE_PATH:-}"
export C_INCLUDE_PATH="$CCCL_INC:${C_INCLUDE_PATH:-}"

echo "=== Step 3: compile tokenspeed-kernel for sm_89 ==="
cd "$WS/tokenspeed-kernel/python"
# --no-deps: skip cuda-thirdparty.txt (Blackwell-only pins not needed on Ada).
# --no-build-isolation: use installed torch/flashinfer/triton.
python3 -m pip install --user --no-build-isolation --no-deps -e . 2>&1 | tail -40

echo "=== Step 4: smoke import ==="
export PYTHONPATH="$WS/tokenspeed-kernel/python:${PYTHONPATH:-}"
python3 << 'PYEOF'
import torch
from tokenspeed_kernel.platform import current_platform
p = current_platform()
print('arch', p.arch_version, 'sm_features', sorted(p.sm_features))
import tokenspeed_kernel
print('tokenspeed_kernel import OK')
from tokenspeed_kernel.ops.attention import mla_decode_with_kvcache, mla_prefill
print('MLA ops import OK')
PYEOF
echo "=== DONE ==="
