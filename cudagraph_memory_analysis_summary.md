# ACL Graph profile/capture 内存实验分析总结

## 1. 实验目的

验证以下现象及原因：

- 先执行 `profile_cudagraph_memory()`，再执行正式 `capture_model()` 时，正式 capture 的内存增量较小。
- 跳过 `profile_cudagraph_memory()`、直接执行 `capture_model()` 时，正式 capture 的内存增量较大。

本轮分析使用：

- `vllm_profile.log`：先 profile，再正式 capture。
- `vllm_capture.log`：跳过 profile，直接正式 capture。

## 2. 实验变量一致性

两组实验使用相同配置：

- 模型：`DeepSeek-V2-Lite-W8A8`
- 图模式：`FULL_DECODE_ONLY`
- capture sizes：`[1, 2, 4, 8, 16, 24, 32]`
- `max_model_len=4096`
- `num_gpu_blocks_override=10913`

日志位置：`vllm_profile.log:36`、`vllm_capture.log:36`。

自动计算的 KV block 数并不相同：

- profile 组自动计算为 `10912`。
- skip 组自动计算为 `10919`。

两组最终均被 override 为 `10913`，因此正式 KV cache 大小一致，排除了 KV cache 大小对 capture 内存的影响。

日志位置：

- profile 组 override：`vllm_profile.log:457`
- profile 组最终 KV blocks：`vllm_profile.log:460`
- skip 组 override：`vllm_capture.log:435`
- skip 组最终 KV blocks：`vllm_capture.log:438`

## 3. 核心结果

### 3.1 profile 清理后的残留

profile 完成并执行 Ascend cleanup 后，仍然残留：

| 指标 | 残留量 |
|---|---:|
| device | +46.28 MiB |
| allocated | +4.00 MiB |
| reserved | +20.00 MiB |
| non-allocator | +26.28 MiB |

其中 `allocated` 是 `reserved` 的子集，不能与 `reserved` 重复相加。设备总残留符合：

```text
20.00 MiB reserved + 26.28 MiB non-allocator
= 46.28 MiB device
```

日志位置：

- profile 外层开始快照：`vllm_profile.log:433`
- Ascend cleanup 后快照：`vllm_profile.log:452`
- 外层 `profile_residual`：`vllm_profile.log:453`

### 3.2 正式 capture 增量

外层 `NPUWorker` 对正式 `capture_model()` 的测量结果：

| 模式 | 正式 capture 增量 |
|---|---:|
| 先 profile | 303.21 MiB |
| 直接 capture | 349.05 MiB |
| 两者差值 | 45.84 MiB |

日志位置：

- profile 后正式 capture 外层增量：`vllm_profile.log:520`
- 直接 capture 外层增量：`vllm_capture.log:498`

与 profile 残留对比：

```text
profile 清理后残留                 = 46.28 MiB
直接 capture - profile 后 capture = 45.84 MiB
误差                              = 0.44 MiB
```

误差不到 1%，说明 profile 后残留的内存基本全部在后续正式 capture 中被复用。

## 4. 最终内存状态

使用 `gpu_model_runner.py` 内部、紧邻正式 capture 的快照：

| 指标 | 先 profile | 直接 capture |
|---|---:|---:|
| free | 4606.15 MiB | 4604.97 MiB |
| allocated | 57282.66 MiB | 57282.66 MiB |
| reserved | 57632.00 MiB | 57632.00 MiB |
| non-allocator | 501.85 MiB | 503.03 MiB |

日志位置：

- profile 后正式 capture 内层结束快照：`vllm_profile.log:515`
- 直接 capture 内层结束快照：`vllm_capture.log:493`

最终状态表现为：

- `allocated` 完全相同。
- `reserved` 完全相同。
- free 和 non-allocator 仅相差约 1.18 MiB。

因此，profile 并没有让最终 ACL Graph 真正少占约 46 MiB，而是把约 46 MiB 的分配时间提前到了 profile 阶段。

## 5. 差异集中在第一个最大图

第一个正式捕获的图为 `size=32`。

### 5.1 直接 capture

| 阶段 | device 增量 |
|---|---:|
| warmup | 64.00 MiB |
| capture | 264.00 MiB |
| total | 328.00 MiB |

日志位置：`vllm_capture.log:445-447`。

### 5.2 先 profile 后 capture

| 阶段 | device 增量 |
|---|---:|
| warmup | 38.25 MiB |
| capture | 244.00 MiB |
| total | 282.25 MiB |

日志位置：`vllm_profile.log:467-469`。

两组在第一个图上的差值：

```text
warmup 少分配  = 25.75 MiB
capture 少分配 = 20.00 MiB
total 少分配   = 45.75 MiB
```

完整 capture 的内部差值为：

```text
351.03 MiB - 304.13 MiB = 46.90 MiB
```

第一个图解释了约：

```text
45.75 / 46.90 = 97.5%
```

因此，profile 产生的预热效应几乎全部作用于第一个最大图 `size=32`；后续六个图合计差异约 1.15 MiB。

后续六个图的日志位置：

- profile 组：`vllm_profile.log:474-511`
- 直接 capture 组：`vllm_capture.log:452-489`

## 6. 当前能够确认的预热内存类别

### 6.1 allocator 外内存：约 26 MiB

第一个图的 warmup 阶段：

- 直接 capture 的 `non_allocator` 增量为约 `+20.00 MiB`。
- profile 后正式 capture 的 `non_allocator` 增量为约 `-5.75 MiB`。
- 差值约为 `25.75 MiB`。

这与 profile 清理后的 `non_allocator residual=26.28 MiB` 基本一致。

日志位置：

- profile 后正式 size=32 warmup：`vllm_profile.log:467`
- 直接 size=32 warmup：`vllm_capture.log:445`
- profile cleanup 后残留：`vllm_profile.log:453`

这部分不属于 PyTorch/NPU caching allocator，可能来自：

- ACL/CANN runtime 懒初始化；
- 算子内部 workspace/cache；
- MoE 或通信相关运行时资源；
- kernel/driver 缓存；
- graph runtime 元数据。

当前日志只能确定内存类别，尚不能确定具体对象或具体算子。

### 6.2 allocator reserved 内存：约 20 MiB

第一个图的 capture 阶段：

- 直接 capture 的 `reserved` 增量为 `+262 MiB`。
- profile 后正式 capture 的 `reserved` 增量为 `+242 MiB`。
- 差值正好为 `20 MiB`。

这与 profile 清理后的 `reserved residual=20 MiB` 完全一致。

日志位置：

- profile 后正式 size=32 capture：`vllm_profile.log:468`
- 直接 size=32 capture：`vllm_capture.log:446`
- profile cleanup 后残留：`vllm_profile.log:453`

其中：

- 约 4 MiB 仍计入 active `allocated`。
- 其余约 16 MiB 为 reserved、但不属于 active allocated 的内存。

它可能属于 allocator segment、不可立即释放的碎片或 graph/private pool，但当前日志仍不能确定具体对象。

## 7. cleanup 效果

上游 `GPUModelRunner` cleanup 结束时残留：

| 指标 | 上游 cleanup 后 |
|---|---:|
| device | 68.04 MiB |
| allocated | 8.50 MiB |
| reserved | 42.00 MiB |
| non-allocator | 26.04 MiB |

日志位置：

- 上游 cleanup 后快照：`vllm_profile.log:449`
- 上游 `profile_residual`：`vllm_profile.log:450`

继续执行 Ascend 特有 cleanup 后：

| 指标 | Ascend cleanup 后 |
|---|---:|
| device | 46.28 MiB |
| allocated | 4.00 MiB |
| reserved | 20.00 MiB |
| non-allocator | 26.28 MiB |

日志位置：

- Ascend cleanup 后快照：`vllm_profile.log:452`
- 外层 `profile_residual`：`vllm_profile.log:453`

Ascend cleanup 额外释放约：

```text
68.04 - 46.28 = 21.76 MiB
```

释放的基本都是 allocator 内存；约 26 MiB non-allocator 内存没有被 `reset_graph_params()`、清理 layer KV 引用、`gc.collect()` 或 `empty_cache()` 清除。

## 8. 图内存估算是否准确

profile 的估算过程为：

```text
第一个图               = 328.49 MiB
每个后续图             = 4.00 MiB
图数量                 = 7
估算总量               = 328.49 + 4 × 6
                       = 352.49 MiB
```

日志位置：

- profile size=32 采样：`vllm_profile.log:439-441`
- profile size=24 采样：`vllm_profile.log:445-447`
- 估算外推结果：`vllm_profile.log:448`
- 估算总量：`vllm_profile.log:451`

与直接冷 capture 对比：

```text
估算值                 = 352.49 MiB
直接冷 capture         = 351.03 MiB
误差                   = 1.46 MiB
误差比例               ≈ 0.42%
```

日志位置：

- 直接 capture 内层总增量：`vllm_capture.log:494`
- profile 后 capture 内层总增量：`vllm_profile.log:516`
- 日志中的 estimated/actual 对比：`vllm_profile.log:521`

因此估算值本身非常准确。

如果将估算值与 profile 后的正式 capture 增量比较：

```text
352.49 MiB - 304.13 MiB = 48.36 MiB
```

这个差值不是估算过大，而是比较基线不同：

- 估算值测量冷启动图内存成本。
- profile 后的 `capture_model()` 测量已经预热后的新增成本。

## 9. 已确认结论

1. `profile_cudagraph_memory()` 提前留下约 46 MiB 驻留内存。
2. 后续正式 `capture_model()` 正好少分配约 46 MiB。
3. profile 组和 skip 组最终 capture 完成后的绝对内存状态几乎相同。
4. 差异的约 97.5% 集中在第一个最大图 `size=32`。
5. 约 26 MiB 属于 allocator 外内存，约 20 MiB 属于 allocator reserved 内存。
6. 图内存估算与直接冷 capture 的误差只有约 0.42%，估算本身没有明显偏大。

## 10. 第一轮结束时尚未解决的问题

本轮实验还不能回答以下更细粒度问题：

- 约 26 MiB non-allocator 内存具体由哪个 ACL/CANN、MoE、通信或模型算子初始化产生；
- 约 20 MiB reserved 内存具体属于 `NPUGraph()`、graph private pool、output/workspace，还是 allocator 不可释放 segment；
- 4 MiB active allocation 是否由 attention layer 的 `key_cache/value_cache` 或其他长期引用持有。

因此，本轮实验已经证明“内存被提前支付并在正式 capture 中复用”，但还没有定位到具体对象。后续需要对 eager dummy forward、ACL graph 创建和 cleanup 步骤继续做分阶段消融与快照。

下述问题在第二轮 `setup_only / warmup_only / capture_only / profile` 消融实验中得到了进一步定位，详见第 12 节以后。

## 11. 日志证据索引

| 证据 | profile 日志 | direct capture 日志 |
|---|---|---|
| 模型及图配置 | `vllm_profile.log:36` | `vllm_capture.log:36` |
| 实验模式 | `vllm_profile.log:418` | `vllm_capture.log:418` |
| KV block override | `vllm_profile.log:457` | `vllm_capture.log:435` |
| 最终 KV block 数 | `vllm_profile.log:460` | `vllm_capture.log:438` |
| profile 开始快照 | `vllm_profile.log:433-434` | 不适用 |
| profile size=32 warmup/capture | `vllm_profile.log:439-441` | 不适用 |
| profile size=24 warmup/capture | `vllm_profile.log:445-447` | 不适用 |
| 图内存估算外推 | `vllm_profile.log:448` | 不适用 |
| 上游 cleanup 后残留 | `vllm_profile.log:449-450` | 不适用 |
| Ascend cleanup 后残留 | `vllm_profile.log:452-453` | 不适用 |
| 可用 KV cache 内存 | `vllm_profile.log:455` | `vllm_capture.log:434` |
| 正式 capture 前快照 | `vllm_profile.log:461-462` | `vllm_capture.log:439-440` |
| 正式 size=32 warmup/capture | `vllm_profile.log:467-469` | `vllm_capture.log:445-447` |
| 正式后续六个图 | `vllm_profile.log:474-511` | `vllm_capture.log:452-489` |
| 正式 capture 内层结束/增量 | `vllm_profile.log:515-516` | `vllm_capture.log:493-494` |
| 正式 capture 外层结束/增量 | `vllm_profile.log:519-520` | `vllm_capture.log:497-498` |
| estimated/actual 对比 | `vllm_profile.log:521` | 不适用 |

## 12. 第二轮细粒度消融实验

第二轮使用 `log/` 目录下的四份日志：

| 文件 | 实验模式 |
|---|---|
| `log/vllm_setup.log` | `setup_only` |
| `log/vllm_warmup.log` | `warmup_only` |
| `log/vllm_profile.log` | 完整 `profile` |
| `log/vllm_capture.log` | `capture_only` |

这里的 `log/vllm_capture.log` 表示 `capture_only`，不是第一轮的直接 `skip` 日志。

四种前置模式的含义：

- `setup_only`：只建立临时 KV、graph pool 和 capture context，不执行模型。
- `warmup_only`：执行 setup 和 eager dummy forward，不创建 ACL graph。
- `profile`：执行 setup、eager warmup 和 ACL graph capture。
- `capture_only`：不做 eager warmup，直接执行冷 ACL graph capture。

## 13. 第二轮残留内存总览

| 前置阶段 | device residual | reserved | allocated | non-allocator |
|---|---:|---:|---:|---:|
| setup only | -0.35 MiB | 0 MiB | 0 MiB | -0.35 MiB |
| warmup only | 37.75 MiB | 20 MiB | 4 MiB | 17.75 MiB |
| 完整 profile | 46.07 MiB | 20 MiB | 4 MiB | 26.07 MiB |
| capture only | 45.51 MiB | 20 MiB | 4 MiB | 25.51 MiB |

日志位置：

- setup only：`log/vllm_setup.log:11`
- warmup only：`log/vllm_warmup.log:22`
- 完整 profile：`log/vllm_profile.log:25`
- capture only：`log/vllm_capture.log:14`

由此可以拆分：

```text
纯 setup 最终残留              ≈ 0 MiB
eager warmup 最终残留          ≈ 37.75 MiB
graph capture 额外最终残留      ≈ 46.07 - 37.75
                               = 8.32 MiB
```

`capture_only` 与完整 profile 只相差 0.56 MiB，说明冷 graph capture 内部执行模型时也会完成 eager warmup 中的同类懒初始化。

## 14. 具体预热来源定位

### 14.1 纯 setup 不是最终残留来源

`setup_only` 在上游 cleanup 前暂时占用：

```text
device        = 22.22 MiB
allocated     = 4.50 MiB
reserved      = 22.00 MiB
non-allocator = 0.22 MiB
```

日志位置：`log/vllm_setup.log:1`。

cleanup 中：

- `gc.collect()` 释放 4.50 MiB allocated：`log/vllm_setup.log:8`
- `empty_cache()` 释放 22.00 MiB reserved：`log/vllm_setup.log:10`
- 最终 residual 约为零：`log/vllm_setup.log:11`

因此，临时最小 KV、graph descriptor、graph pool handle 和没有模型执行的 capture context 都不是最终约 46 MiB 残留的主要来源。

### 14.2 约 20 MiB reserved 源于 attention metadata 路径

`warmup_only` 的第一个 `size=32` dummy run 中：

```text
batch setup:
    device=0, allocated=0, reserved=0, non-allocator=0

attention metadata:
    device=+22 MiB
    allocated=+4 MiB
    reserved=+22 MiB
    non-allocator=0
```

日志位置：

- batch setup：`log/vllm_warmup.log:3`
- attention metadata：`log/vllm_warmup.log:5`

完整 profile 中得到相同结果：

```text
device=+22.43 MiB
allocated=+4 MiB
reserved=+22 MiB
non-allocator=+0.43 MiB
```

日志位置：`log/vllm_profile.log:5`。

cleanup 后最终仍保留：

```text
allocated=4 MiB
reserved=20 MiB
```

因此可以将最终 4 MiB active allocation 和约 20 MiB reserved segment 主要归因到 attention metadata 构建路径。对应代码范围包括：

```text
seq_lens.copy_(...)
copy_snapshot_to_gpu(query_start_loc)
commit_block_table(...)
_build_attention_metadata(...)
slot_mapping.gpu.fill_(-1)
```

第二轮日志还不能区分上述操作中的具体哪一个持有 4 MiB active allocation。

### 14.3 约 18 MiB non-allocator 源于第一次 `_model_forward()`

`warmup_only` 的第一次模型 forward：

```text
device=+42 MiB
allocated=+0.13 MiB
reserved=+22 MiB
non-allocator=+20 MiB
```

日志位置：`log/vllm_warmup.log:9`。

完整 profile 中结果相同：`log/vllm_profile.log:9`。

warmup-only 完成全部 cleanup 后仍保留：

```text
non-allocator=17.75 MiB
```

所以约 18～20 MiB allocator 外持久内存由第一次 `_model_forward()` 触发。它不来自 batch/padding、attention metadata、输入/RoPE 准备或 postprocess，因为这些阶段的 non-allocator 增量接近零。

对于当前 DeepSeek + expert parallel 配置，优先嫌疑包括：

- MoE ALLGATHER/HCCL 通信资源；
- ATB/CANN 算子 workspace cache；
- attention 或 MoE kernel 的首次 runtime 初始化；
- stream/event/runtime 元数据。

当前检查点覆盖整个 `_model_forward()`，尚不能指定到其中某一个具体算子。

### 14.4 真正 graph capture 额外留下约 8 MiB non-allocator

完整 profile 与 warmup-only 的最终残留差：

```text
device:        46.07 - 37.75 = 8.32 MiB
reserved:      20.00 - 20.00 = 0 MiB
allocated:      4.00 - 4.00  = 0 MiB
non-allocator: 26.07 - 17.75 = 8.32 MiB
```

因此，在 eager warmup 之外，ACL graph capture 额外留下约 8.3 MiB allocator 外内存，没有额外留下 allocator reserved。

这约 8 MiB 包括：

- profile 对 `size=32` 和 `size=24` 的 graph capture；
- `torch.npu.graph(...)` 上下文；
- graph runtime/driver bookkeeping；
- profiling graph pool 的运行时状态。

### 14.5 `NPUGraph()` 构造本身没有设备内存增量

所有第二轮日志中的 `after_npugraph` 均为：

```text
device=0
allocated=0
reserved=0
non-allocator=0
```

日志位置示例：

- 完整 profile：`log/vllm_profile.log:12`
- capture only：`log/vllm_capture.log:1`

因此可以排除：

```python
aclgraph = torch.npu.NPUGraph()
```

本身是预热内存来源。大量内存分配实际发生在：

```python
with torch.npu.graph(aclgraph, pool=self.graph_pool):
    output = self.runnable(...)
```

完整 profile 的第一次 graph context：

```text
device=+266 MiB
allocated=+249.56 MiB
reserved=+264 MiB
non-allocator=+2 MiB
```

日志位置：`log/vllm_profile.log:13`。

### 14.6 graph entry/弱引用降低 active allocation，但不立即归还设备内存

完整 profile 在 graph entry 保存后：

```text
device=0
allocated=-249.69 MiB
reserved=0
non-allocator=0
```

日志位置：`log/vllm_profile.log:14`。

这说明 output/workspace 的 active tensor 引用被弱引用处理释放，但 backing memory 先回到 allocator/private pool，没有立即增加设备 free memory。

### 14.7 capture-only 会在 graph 内完成同样的懒初始化

`capture_only` 没有 eager warmup，第一次直接进入 graph context：

```text
device=+306 MiB
allocated=+249.69 MiB
reserved=+284 MiB
non-allocator=+22 MiB
```

日志位置：`log/vllm_capture.log:2`。

相比完整 profile 的 graph context，多出的约 20 MiB reserved 和 20 MiB non-allocator，说明 graph 内部的模型执行同时完成了 attention/runtime 懒初始化。

## 15. cleanup 分步骤结论

四组实验表现一致：

### `reset_graph_params()`

所有内存指标变化均为零，例如 `log/vllm_profile.log:18`。它只清理逻辑状态，不释放这里关注的设备内存。

### 清理 `layer.impl.key_cache/value_cache`

所有指标变化也为零，例如 `log/vllm_profile.log:20`。这一步没有立即释放当前残留，说明这些 cache 不是唯一持有者，或者释放需要等待 GC。

### `gc.collect()`

通常释放约 4.50 MiB allocated，但几乎不增加设备 free memory：`log/vllm_profile.log:22`。

说明 tensor 引用被解除，但所在 allocator segment 仍然保留。

### `empty_cache()`

稳定释放：

```text
device=-22 MiB
reserved=-22 MiB
```

日志位置：`log/vllm_profile.log:24`。

最终仍留下 20 MiB reserved 和 4 MiB allocated，意味着这个约 20 MiB segment 中仍有约 4 MiB active allocation，因此整个 segment 无法由 `empty_cache()` 归还；其余约 16 MiB 可能是 segment 内部空闲空间或碎片。

## 16. 正式 capture 的反向验证

以 cleanup 后 residual 约为零的 `setup_only` 作为冷 capture 基线：

| 前置模式 | 正式 capture 增量 | 相对 setup 减少 |
|---|---:|---:|
| setup only | 349.36 MiB | 0 MiB |
| warmup only | 312.60 MiB | 36.76 MiB |
| 完整 profile | 305.40 MiB | 43.96 MiB |
| capture only | 303.50 MiB | 45.86 MiB |

日志位置：

- setup only：`log/vllm_setup.log:27`
- warmup only：`log/vllm_warmup.log:38`
- 完整 profile：`log/vllm_profile.log:41`
- capture only：`log/vllm_capture.log:30`

正式 capture 的减少量与各自前置阶段的 residual 基本吻合，说明这些内存随后确实被正式 capture 复用。

## 17. 更新后的定位结论

第二轮实验将约 46 MiB 进一步拆分为：

```text
约 20 MiB allocator reserved
    主要来源：第一次 attention metadata 构建触发的 allocator segment
    其中约 4 MiB 仍为 active allocation

约 18 MiB non-allocator
    来源：第一次 _model_forward()
    可能是 MoE/HCCL、ATB/CANN 或其他算子 runtime 懒初始化

约 8 MiB non-allocator
    来源：真正的 ACL graph capture 阶段
    包括 graph context、第二个采样图和 runtime/driver bookkeeping

约 0 MiB
    setup、NPUGraph() 构造、batch setup 和 postprocess
```

第二轮后剩余的精确定位问题缩小为：

1. `_build_attention_metadata()` 内哪一个操作持有 4 MiB active allocation；
2. `_model_forward()` 内哪一个具体算子初始化了约 18 MiB non-allocator；
3. graph capture 增加的约 8 MiB non-allocator 在 size=32、size=24 和 runtime bookkeeping 之间如何分布。

## 18. Round 3：attention metadata 子步骤与双图尺寸分析

Round 3 使用聚合日志 `log/memory_round3.log`，比较以下四种前置模式：

- `setup_only`：profile 阶段只做 setup，随后执行正式 warmup/capture，可作为“无 profile 直接 capture”基线；
- `warmup_only`：profile 阶段只执行 eager warmup；
- `profile`：profile 阶段执行 eager warmup 和 ACL graph capture；
- `capture_only`：profile 阶段跳过 eager warmup，直接捕图，用于验证 graph capture 自身的懒初始化效果。

### 18.1 正式 capture 总结果

| 前置模式 | capture device 增量 | allocated | reserved | non-allocator | 相对 setup 减少 |
|---|---:|---:|---:|---:|---:|
| setup only | 361.46 MiB | 4.01 MiB | 308.00 MiB | 53.46 MiB | 0 MiB |
| warmup only | 322.04 MiB | 0.01 MiB | 288.00 MiB | 34.04 MiB | 39.42 MiB |
| 完整 profile | 316.00 MiB | 0.01 MiB | 288.00 MiB | 28.00 MiB | 45.46 MiB |
| capture only | 309.34 MiB | 0.01 MiB | 288.00 MiB | 21.34 MiB | 52.12 MiB |

聚合日志位置：

- setup only：`log/memory_round3.log:684`，原始日志 `vllm_setup_only.log:686`；
- warmup only：`log/memory_round3.log:942`，原始日志 `vllm_warmup_only.log:738`；
- 完整 profile：`log/memory_round3.log:477`，原始日志 `vllm_profile.log:748`；
- capture only：`log/memory_round3.log:213`，原始日志 `vllm_capture_only.log:695`。

`allocated` 是 `reserved` 的子集，不能与 `reserved` 再次相加。由正式 capture 的结果可得：

```text
无 profile 直接 capture - 完整 profile 后 capture
= 361.46 - 316.00
= 45.46 MiB

其中：
eager warmup 的贡献 = 361.46 - 322.04 = 39.42 MiB
预先捕图的额外贡献 = 322.04 - 316.00 = 6.04 MiB
```

因此，本轮条件下正式 capture 减少量约 86.7% 来自 eager warmup，约 13.3% 来自预先捕图。

### 18.2 profile 清理后仍然保留的预热状态

| 前置模式 | residual device | allocated | reserved | non-allocator |
|---|---:|---:|---:|---:|
| setup only | 19.36 MiB | 4.50 MiB | 22.00 MiB | -2.64 MiB |
| warmup only | 62.41 MiB | 8.50 MiB | 42.00 MiB | 20.41 MiB |
| 完整 profile | 65.08 MiB | 8.50 MiB | 42.00 MiB | 23.08 MiB |
| capture only | 67.96 MiB | 8.50 MiB | 42.00 MiB | 25.96 MiB |

日志位置：

- setup only：`log/memory_round3.log:482`；
- warmup only：`log/memory_round3.log:740`；
- 完整 profile：`log/memory_round3.log:275`；
- capture only：`log/memory_round3.log:11`。

warmup only 相对 setup only 在 cleanup 后仍多保留约 20 MiB reserved、4 MiB active allocation 和约 23 MiB non-allocator。说明预热并没有令这些内存消失，而是把它们移动到了正式 `capture_model()` 的测量起点之前，随后由正式 capture 复用。

完整 profile 相对 warmup only 又多保留约 2.67 MiB non-allocator；对应后续正式 capture 额外减少 6.04 MiB non-allocator，说明 ACL graph/runtime 资源除常驻 residual 外还存在复用和分配路径变化。

### 18.3 attention metadata：已定位到 `_build_attention_metadata()`

本轮将 attention metadata 外围操作拆分为：

1. `seq_lens.copy_()`；
2. `query_start_loc` 和 GDN query-start 拷贝；
3. FIA query-start padding；
4. block table commit；
5. `_build_attention_metadata()`；
6. slot mapping fill。

首次直接 capture 的 size=32 结果为：

```text
_build_attention_metadata():
device=+24.00 MiB
allocated=+4.00 MiB
reserved=+22.00 MiB
non-allocator=+2.00 MiB
```

日志位置：`log/memory_round3.log:509`，原始日志 `vllm_setup_only.log:474`。

完整 profile 的首次冷运行结果相近：

```text
device=+22.00 MiB
allocated=+4.00 MiB
reserved=+22.00 MiB
non-allocator=0 MiB
```

日志位置：`log/memory_round3.log:231`，原始日志 `vllm_profile.log:443`。

profile 完成后，正式 capture 再次运行该步骤时变为：

```text
device=+4.00 MiB
allocated=0 MiB
reserved=+2.00 MiB
non-allocator=+2.00 MiB
```

日志位置：`log/memory_round3.log:302`，原始日志 `vllm_profile.log:536`。

除 `_build_attention_metadata()` 外，其余外围子步骤在 size=32 首次运行时均为 0。由此可以确认：

- 约 22 MiB allocator arena 的初始化发生在 `_build_attention_metadata()` 内部；
- 其中约 4 MiB 是 active allocation；
- profile 后再次调用只需要约 2 MiB reserved 扩展，不再产生新的 4 MiB active allocation；
- `seq_lens`、query-start、FIA padding、block table commit 和 slot mapping 不是这部分内存的来源。

下一步需要继续拆分 `_build_attention_metadata()` 内部的 block-table/slot-mapping 获取、`CommonAttentionMetadata` 构造，以及各 attention group 的 `metadata_builder.build()` 或 `build_for_cudagraph_capture()`。

### 18.4 `_model_forward()`：确认约 18–20 MiB non-allocator 懒初始化

无 profile 直接 capture 的首次模型前向：

```text
device=+44.00 MiB
allocated=+0.13 MiB
reserved=+22.00 MiB
non-allocator=+22.00 MiB
```

日志位置：`log/memory_round3.log:518`，原始日志 `vllm_setup_only.log:484`。

完整 profile 的首次冷模型前向：

```text
device=+42.00 MiB
allocated=+0.13 MiB
reserved=+22.00 MiB
non-allocator=+20.00 MiB
```

日志位置：`log/memory_round3.log:240`，原始日志 `vllm_profile.log:453`。

profile 完成后，正式 capture 再执行模型前向：

```text
device=+44.25 MiB
allocated=+0.13 MiB
reserved=+42.00 MiB
non-allocator=+2.25 MiB
```

日志位置：`log/memory_round3.log:311`，原始日志 `vllm_profile.log:546`。

首次模型前向的约 20–22 MiB non-allocator 在预热后只剩约 2–2.25 MiB 新增，证明 compiled model forward 确实初始化并保留了约 18–20 MiB 的 NPU runtime/算子侧非 allocator 资源。

但不能把 `_model_forward()` 的 non-allocator 差值直接与 capture 减少量机械相加：预热后该步骤的 allocator reserved 增量由 22 MiB 变为 42 MiB，说明 allocator pool 的增长时机发生了迁移。可靠结论是“非 allocator runtime 状态已被预热”，而不是单个步骤的 device 增量减少了 20 MiB。

### 18.5 逐层 module hook 未穿透 compiled callable

本轮成功在模型上匹配并注册了 133 个 decoder layer、`self_attn`、`mlp`、`experts` 和 `shared_experts` module hook，例如：

- `log/memory_round3.log:238`：profile 冷运行，`trace_run=0, module_count=133`；
- `log/memory_round3.log:309`：profile 后正式运行，`trace_run=1, module_count=133`；
- `log/memory_round3.log:516`：setup-only 正式运行，`trace_run=0, module_count=133`。

但是聚合日志中不存在任何 `phase=dummy_model_run_*` 事件。这说明实际模型执行走的是已经生成的 compiled callable，没有再次经过运行时注册 hook 的原始 PyTorch module `__call__`。

因此 Round 3 尚不能把约 18–20 MiB non-allocator 进一步归因到某一层 attention、MoE 或其他算子。下一轮不能继续使用运行时 module hook，需要在编译路径之前插桩、拆分 Ascend custom-op 调用，或采用不改变算子初始化语义的 compiled execution 分段方案。

### 18.6 size=32 与 size=24 的 graph context

两个最大 FULL 图的 `after_graph_context` 数据如下：

| 前置模式 | size=32 device | size=32 reserved | size=32 non-allocator | size=24 device | size=24 reserved | size=24 non-allocator |
|---|---:|---:|---:|---:|---:|---:|
| setup only | +266.00 MiB | +264.00 MiB | +2.00 MiB | -20.55 MiB | -24.00 MiB | +3.45 MiB |
| warmup only | +246.00 MiB | +244.00 MiB | +2.00 MiB | -39.26 MiB | -44.00 MiB | +4.74 MiB |
| 完整 profile | +242.00 MiB | +244.00 MiB | -2.00 MiB | -44.00 MiB | -44.00 MiB | 0 MiB |
| capture only | +237.38 MiB | +244.00 MiB | -6.62 MiB | -44.00 MiB | -44.00 MiB | 0 MiB |

日志位置：

- setup size=32：`log/memory_round3.log:522`；size=24：`log/memory_round3.log:551`；
- warmup size=32：`log/memory_round3.log:780`；size=24：`log/memory_round3.log:809`；
- profile size=32：`log/memory_round3.log:315`；size=24：`log/memory_round3.log:344`；
- capture-only size=32：`log/memory_round3.log:51`；size=24：`log/memory_round3.log:80`。

size=24 的负 device/reserved 增量表示该区间复用了 size=32 的 allocator/private pool，并释放了先前的部分 reserved，并不表示捕图产生了“负内存”。各模式必须在相同 checkpoint 上比较。

warmup only 到完整 profile 的正式 capture 总差值为 6.04 MiB，全部体现在 non-allocator。size=32 相同 checkpoint 下降约 4 MiB，size=24 相同 checkpoint 的净变化约 4.74 MiB；两者存在 pool 重用和生命周期重叠，不能直接相加，但共同证明额外预热发生在这两个图尺寸的 ACL graph/runtime 路径中。

### 18.7 Round 3 最终归因

```text
完整 profile 使正式 capture 少增量占用 45.46 MiB

├── eager warmup：39.42 MiB（约 86.7%）
│   ├── _build_attention_metadata() 初始化 allocator arena
│   │   └── 首次约 22 MiB reserved，其中约 4 MiB active
│   └── compiled _model_forward() 初始化 NPU runtime 状态
│       └── 首次约 20–22 MiB non-allocator，预热后新增约 2 MiB
│
└── 预先 ACL graph capture：额外 6.04 MiB（约 13.3%）
    └── size=32、size=24 以及二者共享的 graph/runtime bookkeeping
```

Round 3 后剩余的精确定位问题为：

1. `_build_attention_metadata()` 内部是哪一个 metadata builder 或公共元数据操作创建并持有 4 MiB active allocation；
2. compiled `_model_forward()` 内部约 18–20 MiB non-allocator 属于 attention、MoE、通信还是其他 Ascend runtime 资源；
3. graph 侧约 6 MiB 的后续 capture 减少量中，多少是具体 graph 对象资源，多少是跨 size 共享的 driver/runtime 状态。

## 19. Metadata ownership 对照实验

本轮使用三次独立进程运行：

- `setup_only`：profile 阶段不执行 dummy metadata warmup，作为直接 capture 基线；
- `metadata_keep`：只执行第一张 `FULL size=32` 图的 attention metadata 构建，不执行 `_model_forward()` 和 profile 图 capture，并将返回的 `attn_metadata`、`spec_decode_common_attn_metadata` 引用保留到 profile cleanup 之后；
- `metadata_release`：执行相同的 metadata 构建，但在 cleanup 前主动清除上述两个返回值引用。

`metadata_keep` 在记录 cleanup residual 后，又在同一进程中显式清除保留引用，并执行 `gc.collect()`、NPU synchronize 和 `empty_cache()`。因此，该实验可以直接检验返回的 metadata 对象是否拥有或钉住残余 allocator 内存。

### 19.1 Profile cleanup 后的残余

| 模式 | device residual | allocated residual | reserved residual | non-allocator residual | 日志位置 |
|---|---:|---:|---:|---:|---|
| `setup_only` | -1.80 MiB | 0 MiB | 0 MiB | -1.80 MiB | `vllm-ascend/logs/setup_only.log:435` |
| `metadata_keep` | +20.27 MiB | +4 MiB | +20 MiB | +0.27 MiB | `vllm-ascend/logs/metadata_keep.log:458` |
| `metadata_release` | +17.10 MiB | +4 MiB | +20 MiB | -2.90 MiB | `vllm-ascend/logs/metadata_release.log:456` |

`metadata_keep` 显式释放引用后的增量为：

```text
device=-0.23 MiB
allocated=0 MiB
reserved=0 MiB
non-allocator=-0.23 MiB
```

日志位置：`vllm-ascend/logs/metadata_keep.log:459-460`。

关键事实是：清除返回的 metadata 引用并没有令 4 MiB allocated 或 20 MiB reserved 下降；`metadata_release` 对照组在 cleanup 后也保留了完全相同的 allocator 数值。两组 device residual 相差约 3.17 MiB，全部来自 non-allocator 波动，不影响 allocator ownership 判断。

因此，本轮实验否定了以下归因：

> cleanup 后的 4 MiB active allocation 和 20 MiB reserved 是由 `_build_attention_metadata()` 返回的 Python metadata 对象直接持有。

### 19.2 这 4/20 MiB 确实由 metadata 构建路径提前触发

第一次执行 `_build_attention_metadata()` 时：

| 场景 | device | allocated | reserved | non-allocator | 日志位置 |
|---|---:|---:|---:|---:|---|
| `setup_only` 正式 capture | +24 MiB | +4 MiB | +22 MiB | +2 MiB | `vllm-ascend/logs/setup_only.log:460` |
| `metadata_keep` profile | +22 MiB | +4 MiB | +22 MiB | 0 MiB | `vllm-ascend/logs/metadata_keep.log:431` |
| `metadata_release` profile | +22 MiB | +4 MiB | +22 MiB | 0 MiB | `vllm-ascend/logs/metadata_release.log:429` |

提前执行 metadata profile 后，正式 capture 再次执行相同步骤时变为：

| 场景 | device | allocated | reserved | non-allocator | 日志位置 |
|---|---:|---:|---:|---:|---|
| `metadata_keep` 正式 capture | +4 MiB | 0 MiB | +2 MiB | +2 MiB | `vllm-ascend/logs/metadata_keep.log:485` |
| `metadata_release` 正式 capture | +4 MiB | 0 MiB | +2 MiB | +2 MiB | `vllm-ascend/logs/metadata_release.log:481` |

allocator 数据满足精确的阶段迁移关系：

```text
直接 capture 的首次 metadata 构建：  +4 MiB allocated / +22 MiB reserved

提前 metadata profile 后：
    profile cleanup residual：       +4 MiB allocated / +20 MiB reserved
    正式 capture 再次构建：           0 MiB allocated /  +2 MiB reserved
                                  ----------------------------------------
    合计：                            +4 MiB allocated / +22 MiB reserved
```

这证明 4/20 MiB 是在 `_build_attention_metadata()` 首次执行路径中提前形成，并被正式 capture 复用，而不是测量噪声。

### 19.3 对正式 capture 的影响

| 模式 | 正式 capture 返回时 allocated 增量 | reserved 增量 | 日志位置 |
|---|---:|---:|---|
| `setup_only` | +4 MiB | +306 MiB | `vllm-ascend/logs/setup_only.log:684` |
| `metadata_keep` | 0 MiB | +286 MiB | `vllm-ascend/logs/metadata_keep.log:709` |
| `metadata_release` | 0 MiB | +286 MiB | `vllm-ascend/logs/metadata_release.log:705` |

profile 组的正式 capture 正好少增加 4 MiB allocated 和 20 MiB reserved。三组正式 capture 完成后的绝对值最终一致：`allocated=57282.66 MiB`、`reserved=57632.00 MiB`。因此 profile 没有降低最终总内存，只是把这部分 allocator 成本从正式 capture 阶段提前到了 profile 阶段。

### 19.4 当前能够确认和不能确认的 ownership

当前可以确认：

1. 4 MiB active allocation 和约 20 MiB reserved segment 由第一次 `_build_attention_metadata()` 执行路径触发；
2. 这部分状态在返回 metadata 对象释放后继续存在；
3. 正式 capture 会精确复用这部分 allocator 状态；
4. 所以“profile 后 capture 增量更小”是成本前移，不是最终内存减少。

当前不能确认：

1. 4 MiB active allocation 的最终所有者具体是哪一个全局缓存、metadata builder 字段、NPU 算子 workspace 或 runtime executor cache；
2. 20 MiB reserved 中除 4 MiB active 外的约 16 MiB 只是 allocator arena 的空闲/碎片空间，不能单独对应某个 Python 对象；
3. 仅凭本轮快照不能把它命名为“metadata 内存”。更准确的名称是“由 metadata 构建路径首次触发的持久 allocator/runtime 状态”。

对于本次 DeepSeek-V2-Lite 模型，metadata builder 使用 Ascend MLA 路径。当前优先怀疑点是 `AscendMLAMetadataBuilder.build_decode_metadata()` 中的：

```python
cos, sin = get_cos_and_sin_mla(input_positions, use_cache=True)
```

该调用会执行 NPU advanced indexing，并把结果复制到全局 MLA cos/sin buffer，可能首次初始化索引算子的 workspace 或 executor cache。但本轮日志尚未在 MLA builder 内部继续分段，因此这只是下一轮的首要验证对象，不能作为已经证明的最终归因。

## 20. Round 5 细粒度定位：4 MiB mask 与 20 MiB allocator segment

本节使用 `D:\workspace\vllm-project\log\round5` 下的新日志重新核对，结论取代 19.4 节中“4 MiB 最终所有者尚不能确认”和“优先怀疑 cos/sin”的旧判断。为避免本地 Codex、Markdown 渲染器和 GitHub 对 Windows 绝对路径的解析差异，以下统一使用“实际绝对路径 + 行号”，不再把 Windows 路径包装成 Markdown 链接。

### 20.1 日志行与源码行的正确对应关系

`setup_only` 首次执行正式 warmup 时：

| 阶段 | device | allocated | reserved | non-allocator | 实际日志位置 |
|---|---:|---:|---:|---:|---|
| cos cache index 后 | +4 MiB | 0 MiB | +2 MiB | +2 MiB | `D:\workspace\vllm-project\log\round5\setup_only.log:477` |
| `get_splitfuse_attn_mask()` 后 | +20 MiB | +4 MiB | +20 MiB | 0 MiB | `D:\workspace\vllm-project\log\round5\setup_only.log:486` |
| 外层 `after_attention_mask` | 0 MiB | 0 MiB | 0 MiB | 0 MiB | `D:\workspace\vllm-project\log\round5\setup_only.log:490` |
| metadata 构建总增量 | +24 MiB | +4 MiB | +22 MiB | +2 MiB | `D:\workspace\vllm-project\log\round5\setup_only.log:496` |

`decode_after_attention_mask` 对应的真实业务代码是：

```text
D:\workspace\vllm-project\vllm-ascend\vllm_ascend\attention\mla_v1.py:688
    decode_attn_mask = self.attn_mask_builder.get_splitfuse_attn_mask()

D:\workspace\vllm-project\vllm-ascend\vllm_ascend\attention\mla_v1.py:689
    self._trace_memory("decode_after_attention_mask")
```

因此，第 486 行记录的是执行第 688 行调用前后的差值。日志末尾显示的 `gpu_model_runner.py:6927` 只是 `_log_cudagraph_experiment_delta()` 打印日志的位置，不是产生内存的业务代码位置。

日志中还有一个名字相近的外层检查点 `after_attention_mask`。它发生在 builder 返回以后，此时 mask 已经创建，所以第 490 行为零；不能用该行否定第 486 行的 +20/+4 MiB。

### 20.2 4 MiB active allocation 的所有者

`get_splitfuse_attn_mask()` 的实际实现位于：

```text
D:\workspace\vllm-project\vllm-ascend\vllm_ascend\attention\attention_mask.py:53-58
```

```python
def get_splitfuse_attn_mask(self) -> torch.Tensor:
    if self.chunked_prefill_attn_mask is None:
        self.chunked_prefill_attn_mask = (
            torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8).to(self.device)
        )
    return self.chunked_prefill_attn_mask
```

最终缓存在 `AttentionMaskBuilder.chunked_prefill_attn_mask` 中的 NPU tensor 大小为：

```text
2048 × 2048 × sizeof(int8) = 4,194,304 bytes = 4 MiB
```

它与第 486 行新增的 `allocated=+4.00 MiB` 精确一致。结合调用前后的细粒度检查点，可以把持续存在的 4 MiB active allocation 归到 `chunked_prefill_attn_mask`，而不是 cos/sin 缓存，也不是 `_build_attention_metadata()` 返回的 Python metadata 对象。

该 mask 之所以在释放返回的 metadata 后仍然存在，是因为它还被长期存活的 `AttentionMaskBuilder` 缓存在 `self.chunked_prefill_attn_mask` 字段中。

### 20.3 20 MiB reserved 的准确含义

`reserved=+20 MiB` 不是另一个独立的 20 MiB tensor，也不能与 `allocated=+4 MiB` 相加成 24 MiB。二者是包含关系：

```text
本次调用新增的 allocator reserved segment：约 20 MiB
├── 仍然活跃的 chunked_prefill_attn_mask：4 MiB allocated
└── allocator 内部空闲块、临时 tensor 使用后留下的缓存或碎片：约 16 MiB
```

所以更准确的表述是：首次调用 `get_splitfuse_attn_mask()` 创建并缓存了 4 MiB 的 int8 NPU mask，同时使 NPU caching allocator 扩展了 20 MiB reserved 空间。

### 20.4 profile 如何把这部分成本移出正式 capture

两组 profile 日志在首次创建 mask 时都出现相同增量：

| 模式 | profile 中的 `decode_after_attention_mask` | 日志位置 |
|---|---|---|
| `metadata_keep` | device +20 / allocated +4 / reserved +20 MiB | `D:\workspace\vllm-project\log\round5\metadata_keep.log:456` |
| `metadata_release` | device +20 / allocated +4 / reserved +20 MiB | `D:\workspace\vllm-project\log\round5\metadata_release.log:456` |

进入正式 capture 前的 actual warmup 后，第二次调用直接复用缓存，增量都变为零：

| 模式 | actual warmup 中的 `decode_after_attention_mask` | 日志位置 |
|---|---|---|
| `metadata_keep` | device/allocated/reserved/non-allocator 均为 0 | `D:\workspace\vllm-project\log\round5\metadata_keep.log:547` |
| `metadata_release` | device/allocated/reserved/non-allocator 均为 0 | `D:\workspace\vllm-project\log\round5\metadata_release.log:545` |

正式 capture 返回时也呈现精确的阶段迁移：

| 模式 | allocated 增量 | reserved 增量 | 日志位置 |
|---|---:|---:|---|
| `setup_only`，直接 capture | +4 MiB | +306 MiB | `D:\workspace\vllm-project\log\round5\setup_only.log:720` |
| `metadata_keep`，先 profile | 0 MiB | +286 MiB | `D:\workspace\vllm-project\log\round5\metadata_keep.log:781` |
| `metadata_release`，先 profile | 0 MiB | +286 MiB | `D:\workspace\vllm-project\log\round5\metadata_release.log:779` |

profile 组的正式 capture 恰好少新增 4 MiB allocated 和 20 MiB reserved。这不是最终内存减少，而是 profile 提前触发了 `get_splitfuse_attn_mask()` 的延迟初始化，使正式 capture 复用已经存在的 mask 和 allocator segment。

### 20.5 当前结论

1. 4 MiB active allocation 是 `AttentionMaskBuilder.chunked_prefill_attn_mask`，即一个 `2048 × 2048` 的 int8 NPU attention mask；
2. 20 MiB 是首次构造该 mask 的整个调用使 caching allocator 新增的 reserved segment，其中包含 4 MiB active mask；
3. profile 提前创建并缓存该 mask，所以后续正式 capture 的 `decode_after_attention_mask` 增量为零；
4. `metadata_keep` 与 `metadata_release` 的结果不能证明“这不是 metadata 路径产生的”，它们证明的是“返回的 metadata 对象不是最终 owner”；真正 owner 是 metadata builder 持有的长期缓存字段；
5. cos cache index 仍会单独触发约 2 MiB reserved，但它不是本次 4 MiB active / 20 MiB reserved 差值的主要来源。
