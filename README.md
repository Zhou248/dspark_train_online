# Qwen3.6-35B-A3B DSpark 在线训练（ALLaVA-4V / Ascend A2 16卡）

本目录提供一套不修改 `msModelSpec-Dev` 的在线训练流程。训练时 target
Qwen3.6 vLLM 服务实时生成 verifier hidden states，训练进程读取后立即删除：

```text
ALLaVA-4V
  -> conversations.jsonl
  -> prepare_data (input_ids/loss_mask/messages/token_freq)
  -> target vLLM extract_hidden_states
  -> DSpark online train
  -> checkpoint_best
  -> vLLM target + DSpark speculative decoding
```

在线模式不会像 04a 离线流程那样长期保存每条 `hs_*.safetensors`，但 target
vLLM 必须在整个训练和验证阶段持续运行。

## 1. 脚本说明

| 脚本 | 作用 |
|---|---|
| `common_env.sh` | 路径、模型参数、NPU分配和训练超参数 |
| `00_check_environment.sh` | 检查16张NPU、Python依赖、模型和数据路径 |
| `01_normalize_allava.py` | 流式读取大型 ALLaVA JSON，转换多模态 conversations JSONL |
| `01_prepare_data.sh` | 归一化数据、执行 prepare_data、生成 draft config、审计数据 |
| `02_build_draft_config.py` | 生成3层、1-D RoPE 的纯文本 Qwen3 DSpark decoder config |
| `02_inspect_prepared_data.py` | 检查 token/loss mask，并统计长度和有效 assistant tokens |
| `03_launch_target_vllm.sh` | 启动 target hidden-state vLLM 服务 |
| `04_train_online.sh` | `on-missing=generate` 在线训练 DSpark |
| `run_online_training.sh` | 后台启动 target、训练、退出时清理 target 服务 |
| `05_serve_dspark.sh` | 使用 `checkpoint_best` 启动投机推理服务 |
| `06_test_multimodal.py` | 从 prepared dataset 取一条图片样本测试服务 |

## 2. 默认路径

编辑 `common_env.sh`，至少确认以下三项：

```bash
MSPEC_ROOT=/home/z00909726/msModelSpec-Dev
TARGET_MODEL=/home/z00909726/weights/Qwen3.6-35B-A3B
ALLAVA_JSON=/home/w00608002/models/ALLaVA-4V/allava_laion/ALLaVA-Instruct-LAION-4V.mm.json
```

脚本默认将输出写到：

```text
/home/z00909726/scripts/qwen36_dspark_online/work
```

也可以不改文件，运行时覆盖：

```bash
TARGET_MODEL=/path/to/model \
ALLAVA_JSON=/path/to/ALLaVA.mm.json \
ONLINE_WORK_DIR=/path/to/work \
bash 01_prepare_data.sh
```

新项目只接受 `ONLINE_WORK_DIR` 及其他 `ONLINE_*` 路径覆盖，不继承旧离线项目
导出的 `WORK_DIR/PREPARED_DATA_DIR/HIDDEN_STATES_DIR`，防止覆盖旧训练数据。

## 3. 16卡分配策略

默认配置：

```text
NPU 0-3 : target Qwen3.6 vLLM，TP=4
NPU 8-15: DSpark 8进程FSDP训练
NPU 4-7 : 预留
```

训练必须使用FSDP而不是普通DDP：DDP会在每张卡完整复制模型、参数梯度和优化器
状态，不能解决单卡OOM；FSDP会将这些状态分片到8张卡。当前项目没有暴露可用的
sequence-parallel CLI，因此attention activation仍不会自动除以8，必须同时降低
`MAX_ANCHORS`。

默认：

```bash
ONLINE_TRAIN_NPUS=8,9,10,11,12,13,14,15
ONLINE_TRAIN_NPROC=8
ONLINE_FSDP_SHARD=1
ONLINE_MAX_ANCHORS=512
```

8个训练rank会产生在线hidden-state请求，但target服务由`max_num_seqs=1`限制为
单序列调度。先完成20步smoke；若出现NaN，不要继续扩大训练。

如果512 anchors仍然OOM，继续降到256：

```bash
ONLINE_MAX_ANCHORS=256 MAX_STEPS=20 bash run_online_training.sh
```

## 4. 安装与环境检查

进入服务器上的脚本目录：

```bash
cd /home/z00909726/scripts/qwen36_dspark_online
chmod +x *.sh *.py
bash 00_check_environment.sh
```

脚本假设已经安装并可导入：

```text
torch, torch_npu, transformers, datasets, openai, safetensors, Pillow
```

同时应使用包含 Qwen3.6、DSpark 和 Ascend 适配的 vLLM/vLLM-Ascend 环境。

## 5. 准备 ALLaVA-4V 数据

第一次先处理5,000条，验证完整流程：

```bash
source ./common_env.sh
MAX_SAMPLES=5000 bash 01_prepare_data.sh \
  2>&1 | tee "${LOG_DIR}/prepare_5k.log"
```

该步骤会：

1. 流式读取 ALLaVA JSON，避免一次将整个数组载入内存；
2. 将顶层 `image/images` 与 `<image>` 标记转换为显式本地图片 content part；
3. 校验图片路径存在；
4. 将超过约100万像素或最长边2048的图片等比例缩放到项目工作目录，原图不动；
5. 使用 target processor 生成 `input_ids`、`loss_mask`、`messages`；
6. 保存 `token_freq.pt`，供缩减 draft vocabulary 使用；
7. 生成不含 MRoPE 字段的3层 Qwen3 draft config；
8. 输出序列长度和有效 assistant token 统计。

Qwen多模态 processor 会将大图展开为大量视觉 token。若日志出现
`Mismatch in image token count ... Likely due to truncation`，不要直接把训练长度
提高到16K；默认图片上限用于在 `SEQ_LENGTH=4096` 内为assistant回答保留空间。
需要调整时使用：

```bash
MAX_IMAGE_PIXELS=786432 MAX_IMAGE_SIDE=1536 bash 01_prepare_data.sh
```

默认遇到缺图或格式错误立即停止。若源数据确实包含少量坏行：

```bash
SKIP_INVALID_SOURCE_ROWS=1 MAX_SAMPLES=5000 bash 01_prepare_data.sh
```

不要忽略大量坏行；应先检查 ALLaVA 图片根目录是否正确。

`prepare_data.py --overwrite` 会重建 prepared dataset。改变图片、processor、
`SEQ_LENGTH` 或样本数量后必须重新执行该步骤。

## 6. 启动 target hidden-state vLLM

推荐使用两个终端，以便分别观察 target 和训练日志。

终端A：

```bash
cd /home/z00909726/scripts/qwen36_dspark_online
source ./common_env.sh
bash 03_launch_target_vllm.sh 2>&1 | tee "${LOG_DIR}/target_vllm.log"
```

稳定配置包括：

```text
tensor_parallel_size=4
max_num_seqs=1
enforce_eager
prefix caching off
async scheduling off
AIV off
TASK_QUEUE_ENABLE=0
```

等待以下接口正常：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

不要在 target 服务仍运行时启动最终 DSpark serving；二者会争用同一组卡。

## 7. 在线训练

终端B：

```bash
cd /home/z00909726/scripts/qwen36_dspark_online
source ./common_env.sh
bash 04_train_online.sh 2>&1 | tee "${LOG_DIR}/train_online.log"
```

核心参数：

```text
speculator_type       = dspark
draft layers          = 3
target layer ids      = 2, 20, 37 (+ final layer 40 on vLLM side)
block size            = 7
draft vocabulary      = 32000
markov rank           = 32
max anchors           = 512
distributed           = 8-card FSDP
loss                  = 0.1 CE + 0.9 TV
hidden state dtype    = bfloat16
on_missing            = generate
on_generate           = delete
```

`on_generate=delete` 表示 hidden states 被当前 batch 使用后删除，节省磁盘。训练
中断后可以从 checkpoint 恢复，但缺失 hidden states 会重新在线生成。

也可以一条命令自动管理 target 服务：

```bash
bash run_online_training.sh
```

该命令假设已经完成 `01_prepare_data.sh`。target 日志和训练日志保存在
`work/logs/`。

### 先做最小 smoke test

可以临时限制训练步数：

```bash
MAX_STEPS=20 bash run_online_training.sh
```

也可以使用独立工作目录准备200条数据，完全隔离正式训练产物：

```bash
export ONLINE_WORK_DIR=/home/z00909726/scripts/qwen36_dspark_online/smoke
MAX_SAMPLES=200 bash 01_prepare_data.sh
MAX_STEPS=20 bash run_online_training.sh
```

完整训练前确认：

- target vLLM 没有 NaN；
- train loss 能下降；
- validation 能完成；
- `${CKPT_DIR}/checkpoint_best` 已生成。

## 8. 扩大训练数据

5k只适合验证链路。领域可用版本建议20k–50k高质量样本：

```bash
MAX_SAMPLES=20000 bash 01_prepare_data.sh
bash run_online_training.sh
```

ALLaVA 原始回答不一定完全符合 target Qwen3.6 的实际输出分布。若追求更高
speculative acceptance，后续应使用 target 模型重新生成 assistant responses，
构造 on-policy 数据，再执行 prepare 和训练。

## 9. 启动训练后的 DSpark 推理

先停止 target hidden-state 服务，然后：

```bash
source ./common_env.sh
bash 05_serve_dspark.sh 2>&1 | tee "${LOG_DIR}/dspark_serve.log"
```

服务默认监听 `8100`，并保持 eager 模式。训练的 `BLOCK_SIZE=7` 必须与推理的
`num_speculative_tokens=7` 一致。

另一个终端执行：

```bash
python3 06_test_multimodal.py \
  --prepared-data "${PREPARED_DATA_DIR}" \
  --endpoint http://127.0.0.1:8100/v1 \
  --model "${TARGET_MODEL}" \
  --index 0
```

如果未 source 环境变量，可以直接填写绝对路径。

## 10. 常见问题

### hidden states 出现 NaN

先保持默认单训练进程、`max_num_seqs=1`、eager、关闭 async/AIV/prefix cache。
如果仍出现 NaN，记录失败样本索引和 target vLLM 的第一段 traceback。在线模式
遇到 NaN 会停止训练，这是为了避免使用损坏的 verifier target；不要静默训练。

### Prompt token IDs mismatch

表示 prepare_data 的 processor 输出与 vLLM 对相同 `messages` 的渲染不一致。
确认两端使用同一个 `TARGET_MODEL`、同一份图片以及同一套 processor 文件，修改
数据后重新执行 `01_prepare_data.sh`。

### target vLLM OOM

将 `03_launch_target_vllm.sh` 中 `--gpu-memory-utilization 0.90` 降到0.85，或改用
8张 target 卡：

```bash
TARGET_NPUS=0,1,2,3,4,5,6,7 TARGET_TP=8 bash 03_launch_target_vllm.sh
```

### 训练OOM

日志中3072 anchors的 eager attention softmax单次申请了32.81 GiB，因此默认已经
改为8卡FSDP和512 anchors。若仍OOM：

```bash
ONLINE_MAX_ANCHORS=256 MAX_STEPS=20 bash run_online_training.sh
```

FSDP只分片参数/梯度/优化器，不分片attention矩阵，所以降低anchors仍然必要。
最后才考虑将`SEQ_LENGTH=3072`并重新prepare。确认NPU上的eager attention可运行，
不要直接启用图模式排查训练问题。

### 训练恢复

`scripts/train.py` 默认检查已有 checkpoint 并恢复。若要从头训练，请使用新的
`ONLINE_WORK_DIR/ONLINE_CKPT_DIR`；不要直接删除仍需保留的 checkpoint。
