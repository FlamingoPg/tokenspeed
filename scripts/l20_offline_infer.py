#!/usr/bin/env python3
"""L20 offline inference — bypasses SMG gateway.

Constructs ServerArgs + AsyncLLM directly and runs generate(), testing the
full forward path (model load + MLA attention + MoE + sampling) on L20.
"""
from __future__ import annotations

import os
import sys

WS = os.environ.get("WS", "/home/tiger/tokenspeed")
sys.path.insert(0, os.path.join(WS, "tokenspeed-kernel", "python"))
sys.path.insert(0, os.path.join(WS, "tokenspeed-scheduler", "python"))
sys.path.insert(0, os.path.join(WS, "python"))

os.environ.setdefault("TOKENSPEED_KERNEL_BACKEND", "cuda")
os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "8.9")

import asyncio

from tokenspeed.runtime.utils.server_args import ServerArgs
from tokenspeed.runtime.engine.async_llm import AsyncLLM
from tokenspeed.runtime.engine.io_struct import GenerateReqInput


async def run():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/home/tiger/models/DeepSeek-V2-Lite-Chat"
    print(f"=== Constructing ServerArgs for {model_path} ===", flush=True)
    server_args = ServerArgs(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        world_size=1,
        attn_tp_size=1,
        skip_tokenizer_init=False,
    )

    from tokenspeed.runtime.utils.server_args import PortArgs
    port_args = PortArgs.init_new(server_args)
    llm = AsyncLLM(server_args, port_args)
    print("=== Model loaded! ===", flush=True)

    prompts = ["Hello, my name is", "The capital of France is"]
    print(f"=== Generating for {len(prompts)} prompts ===", flush=True)
    obj = GenerateReqInput(
        text=prompts,
        sampling_params={"max_new_tokens": 16, "temperature": 0},
    )
    generator = llm.generate_request(obj)
    results = []
    async for chunk in generator:
        results.append(chunk)
        print(f"  chunk: {str(chunk)[:120]}", flush=True)

    print(f"=== Done, {len(results)} chunks ===")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
