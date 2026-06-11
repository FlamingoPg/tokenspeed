---
doc_type: cleanup-review
refactor: 2026-06-11-glm5-pr-cleanup
status: draft
scope: PR #348 cleanup after first pass
reviewed_at: 2026-06-11
---

# GLM5 PR 清理与 Review

## 当前状态

- 当前分支: `flamingo/glm5_1_pr`
- MR/PR: https://github.com/lightseekorg/tokenspeed/pull/348
- 当前本地清理改动已经 staged 一部分；`glm5.py` 和
  `test_glm5_topk_workspace.py` 还有 unstaged 小改动。
- 代码里已经扫不到 `TOKENSPEED_GLM5_*` 环境变量残留。
- 本地没有发现 Python venv；当前 `python3` 环境没有 `pytest`，所以测试还没跑起来。
- 还有一个未纳入本次 review 的未跟踪文件:
  `.claude-debug-state-glm5-mtp.md`。

## 已清理掉的内容

这轮清理已经基本完成了第一批目标：

- 删除 GLM5 DSA CUDA graph 相关环境变量开关。
- 删除 GLM5 decode top-k / fused projection 的环境变量开关。
- 删除 DeepGEMM/Triton decode threshold 环境变量开关。
- 删除 DSA backend 里只被测试直测、生产路径不用的顶层 helper。
- 删除 `moe/layer.py` 里的未使用 logger。
- 删除 `dsa_topk.py` 报错信息中的环境变量提示。
- 删除部分只服务环境变量或 helper 的测试。

## Review 必修项

这些不是“建议”，是合并前必须修。

### P1: `model_executor.py` 残留已删除 helper 的 import

- 文件: `python/tokenspeed/runtime/execution/model_executor.py`
- 位置: import block 里还在 import `glm5_dsa_cudagraph_min_batch`
- 问题: `cuda_graph_wrapper.py` 已经删除这个 helper，运行时 import
  `model_executor.py` 会失败。
- 处理: 从 import 列表删除 `glm5_dsa_cudagraph_min_batch`。

### P2: `glm5.py` 残留 undefined `topk_impl`

- 文件: `python/tokenspeed/runtime/models/glm5.py`
- 位置: `_write_decode_topk_offsets()` 末尾还留着
  `raise ValueError(... {topk_impl})`
- 问题: `topk_impl` 已经随着环境变量 selector 被删除，这行现在是
  unreachable dead code，但 lint 会报 undefined name。
- 处理: 直接删除这行 unreachable raise。

### P3: `glm5.py` 残留未使用 `os` import

- 文件: `python/tokenspeed/runtime/models/glm5.py`
- 问题: 环境变量逻辑删完后，`os` 已经没有使用。
- 处理: 删除 `import os`。

## 仍建议继续清的项

这些不是当前 blocker，但还是这轮 cleanup 的下一批自然目标。

- `model_executor.py`
  - `_paged_cache_block_table_base_offset_max` 仍只是解包占位，可以改成 `_`。

- `cuda_graph_wrapper.py`
  - `force_eager` / `force_eager_reason` 参数现在基本没有专门调用方了。
    可以再搜一轮调用方，确认后删掉这条参数链路。

- `moe/backends/fp8/deep_gemm.py`
  - threshold 固定为 4 后，`hidden_states.shape[0] <= 4` 会先走 Triton。
    因此 `_DEEP_GEMM_MASKED_MAX_TOKENS`、`_forward_masked_local()`、
    `_masked_local_route_metadata()` 当前看起来不可达。
  - 但测试仍覆盖 `_masked_local_route_metadata()`，删除前要一起改测试。

- B200 MoE config
  - `dtype=fp8_w8a8.json` 和 `dtype=fp8_w8a8_down.json` 内容完全一样。
  - `_down` 文件可以考虑删，但要先验证 fallback 和性能。

- source-text tests
  - `test_glm5_dsa_mtp_verify.py`
  - `test_glm5_nextn.py`
  - `test_idle_forward_inference_mode.py`
  这些测试比较脆，后续有时间可以改成更靠近行为的测试。

## 验证情况

- `rg` 检查:
  - `TOKENSPEED_GLM5_*` 只剩文档引用，代码和测试里已经扫不到。
  - 仍扫到两个残留代码问题:
    - `glm5_dsa_cudagraph_min_batch`
    - `topk_impl`

- `python3 -m py_compile`:
  - 被检查的目标文件语法可编译。
  - 注意: `py_compile` 不会发现“import 已删除符号”或 undefined name 的
    lint 问题。

- `pytest`:
  - 未运行成功。
  - 原因: 当前 `/opt/homebrew/opt/python@3.13/bin/python3.13` 没有安装
    `pytest`。

## 建议下一步顺序

1. 修掉 Review 必修项 P1/P2/P3。
2. 重新跑 `rg`，确认没有 stale import / undefined cleanup 残留。
3. 如果暂时没有项目 venv，就先用可用环境跑 import smoke test。
4. 跑目标测试:
   - `test/runtime/models/test_glm5_topk_workspace.py`
   - `test/runtime/test_moe_deep_gemm_backend.py`
   - `test/runtime/test_server_args_attention_backends.py`
5. 再决定是否删除 DeepGEMM masked-local 路径和重复 B200 `_down` config。
