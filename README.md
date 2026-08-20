# Qwen3.6-35B-A3B DSpark 在线训练（ALLaVA-4V / Ascend A2 16卡）

本目录提供一套不修改 `msModelSpec-Dev` 的在线训练流程。训练时 target
Qwen3.6 vLLM 服务实时生成 verifier hidden states，训练进程读取后立即删除：

```text
ALLaVA-4V
  -> conversations.jsonl
  -> prepare_data (input_ids/loss_mask/messages/token_freq)
  -> serial proxy -> target vLLM extract_hidden_states
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
| `03a_serial_vllm_proxy.py` | 串行化多个FSDP rank的hidden-state请求 |
| `03b_launch_serial_proxy.sh` | 启动本地串行代理（默认8001端口） |
| `04_train_online.sh` | `on-missing=generate` 在线训练 DSpark |
| `run_online_training.sh` | 后台启动 target和代理、训练、退出时清理服务 |
| `05_serve_dspark.sh` | 使用 `checkpoint_best` 启动投机推理服务 |
| `06_test_multimodal.py` | 从 prepared dataset 取一条图片样本测试服务 |
| `07_plot_loss.py` | 从训练tee日志提取loss并生成CSV和PNG曲线 |
| `08_curl_requests.sh` | 用curl发送纯文本和图片理解请求 |

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

8个训练rank会同时产生在线hidden-state请求。只设置`max_num_seqs=1`仍不足以
规避当前Ascend链路的并发NaN，所以训练端默认连接到8001端口的本地代理；代理
每次只向8000端口的target转发一条生成请求，并等待safetensors写入锁释放后才
放行下一条。这会降低hidden-state生成吞吐，但保留8卡FSDP的训练内存分片。

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

手动启动时使用三个终端，以便分别观察target、代理和训练日志。

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

终端B（target ready后启动）：

```bash
cd /home/z00909726/scripts/qwen36_dspark_online
source ./common_env.sh
bash 03b_launch_serial_proxy.sh 2>&1 | tee "${LOG_DIR}/serial_proxy.log"
```

确认训练使用的代理端口正常：

```bash
curl http://127.0.0.1:8001/v1/models
```

不要在 target 服务仍运行时启动最终 DSpark serving；二者会争用同一组卡。

## 7. 在线训练

终端C：

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

`04_train_online.sh`默认请求`http://127.0.0.1:8001/v1`，不要将它改回target的
8000端口，否则8个rank会再次并发请求。`on_generate=delete`表示hidden states
被当前 batch 使用后删除，节省磁盘。训练
中断后可以从 checkpoint 恢复，但缺失 hidden states 会重新在线生成。

也可以一条命令自动管理 target 服务：

```bash
bash run_online_training.sh
```

该命令假设已经完成 `01_prepare_data.sh`。target、serial proxy和训练日志保存
在`work/logs/`。

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

5k只适合验证链路。不要同时大幅增加数据量和epoch；先用20k数据训练3个epoch，
根据validation loss决定是否继续。改变数据规模时使用新的工作目录，避免从旧数据
对应的checkpoint恢复：

```bash
export ONLINE_WORK_DIR=/home/z00909726/scripts/qwen36_dspark_online/work_20k
MAX_SAMPLES=20000 bash 01_prepare_data.sh
EPOCHS=3 LOG_FREQ=1 bash run_online_training.sh
```

如果第3个epoch的validation loss仍明显下降，再扩到5个epoch，训练脚本会从同一
工作目录的checkpoint恢复：

```bash
export ONLINE_WORK_DIR=/home/z00909726/scripts/qwen36_dspark_online/work_20k
EPOCHS=5 LOG_FREQ=1 bash run_online_training.sh
```

如果validation loss持平或上升，应保留`checkpoint_best`，不要继续增加epoch。
20k链路稳定后，可以新建`work_50k`训练50k样本，建议仍从3个epoch开始。

ALLaVA 原始回答不一定完全符合 target Qwen3.6 的实际输出分布。若追求更高
speculative acceptance，后续应使用 target 模型重新生成 assistant responses，
构造 on-policy 数据，再执行 prepare 和训练。

## 9. Loss可视化

训练通过`tee`生成的`train_online_*.log`每步包含结构化train loss，每个验证epoch
包含val loss。生成曲线和明细CSV：

```bash
cd /home/z00909726/scripts/qwen36_dspark_online
python3 07_plot_loss.py \
  /home/z00909726/scripts/qwen36_dspark_online/work_20k/logs/train_online_YYYYMMDD_HHMMSS.log \
  --smooth 20
```

默认在日志旁生成：

```text
train_online_YYYYMMDD_HHMMSS_loss.png
train_online_YYYYMMDD_HHMMSS_loss.csv
```

图中训练loss同时显示浅色原始值和移动平均值，validation loss显示离散点。若环境
缺少绘图库，执行`pip install matplotlib`；CSV仍会先生成。`LOG_FREQ=1`表示每步
记录一次，正式长训练若日志过大可改成`LOG_FREQ=10`。

## 10. 启动训练后的 DSpark 推理

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

也可以用一个脚本连续测试纯文本和多模态图片理解。图片会编码为data URL，因此
客户端图片不要求位于vLLM服务器的`--allowed-local-media-path`目录中：

```bash
cd /home/z00909726/scripts/qwen36_dspark_online
bash 08_curl_requests.sh /home/z00909726/test_images/example.jpg
```

覆盖endpoint、模型名、问题和输出长度：

```bash
API_BASE=http://127.0.0.1:8100/v1 \
MODEL_NAME=/home/z00909726/weights/Qwen3.6-35B-A3B \
TEXT_PROMPT="请解释DSpark。" \
IMAGE_PROMPT="识别图片中的物体和文字。" \
MAX_TOKENS=512 \
bash 08_curl_requests.sh /home/z00909726/test_images/example.jpg
```

## 11. 常见问题

### hidden states 出现 NaN

若各rank在`_maybe_generate_hs -> check_hidden_states`失败，而epoch仍是0%，这是
target生成的verifier hidden states已经含NaN，不是DSpark loss/梯度NaN。
确认8001代理已启动，并在训练日志核对`--vllm-endpoint`指向8001。代理日志应显示
生成请求一条一条完成。若串行后仍只有固定样本NaN，再用该样本索引做单请求重复
诊断，同时检查target vLLM日志的第一段traceback。不要静默跳过NaN，否则各FSDP
rank的batch数可能不一致并造成HCCL死锁。

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
