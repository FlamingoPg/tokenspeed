#!/usr/bin/env bash
# L20 end-to-end server smoke test with DeepSeek-V2-Lite (MLA + MoE).
#
# Prerequisites (already done on l20-fresh):
#   - tokenspeed-kernel/scheduler/runtime installed (PYTHONPATH)
#   - torch 2.9.1 + flashinfer 0.6.13 + triton
#   - proxy in .bashrc
#
# Usage: ferry run --target l20-fresh -- bash /home/tiger/tokenspeed/scripts/l20_server_smoke.sh
set -euo pipefail

export PYTHONPATH="/home/tiger/tokenspeed/tokenspeed-kernel/python:/home/tiger/tokenspeed/tokenspeed-scheduler/python:/home/tiger/tokenspeed/python:${PYTHONPATH:-}"
export TOKENSPEED_KERNEL_BACKEND=cuda
export FLASHINFER_CUDA_ARCH_LIST="8.9"
export HF_HUB_ENABLE_HF_TRANSFER=1

MODEL_DIR="/home/tiger/models/DeepSeek-V2-Lite-Chat"
PORT=31000

echo "=== Step 1: Download DeepSeek-V2-Lite-Chat (if not present) ==="
if [ ! -f "$MODEL_DIR/config.json" ]; then
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('deepseek-ai/DeepSeek-V2-Lite-Chat', local_dir='$MODEL_DIR')
print('Downloaded to $MODEL_DIR')
" 2>&1 | tail -5
else
    echo "Model already exists at $MODEL_DIR"
fi

echo "=== Step 2: Patch config to DeepseekV3ForCausalLM ==="
python3 -c "
import json
cfg = json.load(open('$MODEL_DIR/config.json'))
cfg['architectures'] = ['DeepseekV3ForCausalLM']
cfg['model_type'] = 'deepseek_v3'
# Remove auto_map so transformers doesn't try remote code loading
cfg.pop('auto_map', None)
json.dump(cfg, open('$MODEL_DIR/config.json','w'), indent=2)
print('Patched config: architectures=%s model_type=%s' % (cfg['architectures'], cfg['model_type']))
"

echo "=== Step 3: Start tokenspeed server ==="
cd /home/tiger/tokenspeed
python3 -m tokenspeed.cli serve \
    --model-path "$MODEL_DIR" \
    --port $PORT \
    --tp-size 1 \
    --trust-remote-code \
    --log-level info &
SERVER_PID=$!

echo "Server PID: $SERVER_PID, waiting for startup..."
# Wait for server to be ready (retry /v1/models)
for i in $(seq 1 120); do
    if curl -s http://localhost:$PORT/v1/models | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
        echo "Server ready after ${i}s!"
        break
    fi
    sleep 1
done

echo "=== Step 4: Send a test request ==="
curl -s http://localhost:$PORT/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "deepseek", "prompt": "Hello, my name is", "max_tokens": 16, "temperature": 0}' \
    2>&1 | python3 -c "import sys,json; r=json.load(sys.stdin); print('Response:', r.get('choices',[{}])[0].get('text','<no text>'))" 2>&1 || echo "Request failed"

echo "=== Step 5: Cleanup ==="
kill $SERVER_PID 2>/dev/null || true
echo "DONE"
