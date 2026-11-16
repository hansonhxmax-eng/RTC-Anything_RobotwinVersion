# 从零训练 π0 并在 RoboTwin 使用 RTC 异步测评

本项目沿用 RTC-Anything 的 `RealTimeChunkingBuffer`，把异步动作块融合接到 RoboTwin 2.0 的 XPolicyLab 测评客户端。**训练数据和 checkpoint 都为本次任务新建**；不使用之前的 `adjust_bottle` 模型。下面以 RoboTwin 自带的 `place_a2b_left` 为例。原 RTC-Anything 的真机 ROS 任务并没有因此自动变成相同的仿真任务；要做衣物折叠等对应场景，需先实现并验证 RoboTwin 的任务环境。

## 1. 检查新 GPU 和目录

在装有 RoboTwin 2.0 的容器中运行。`RTC_DIR` 改成此仓库在容器里的实际路径：

```bash
export ROBOTWIN_ROOT=/workspace/RoboTwin
export RTC_DIR=/workspace/RTC-Anything_RobotwinVersion
nvidia-smi
test -f "$ROBOTWIN_ROOT/scripts/eval_policy_xpolicylab.py"
test -f "$ROBOTWIN_ROOT/XPolicyLab/policy/Pi_0/openpi/scripts/train.py"
```

若缺少 XPolicyLab，先在 RoboTwin checkout 运行 `git submodule update --init --recursive`，按该版本的安装说明安装仿真环境和 `XPolicyLab/policy/Pi_0` 的 OpenPI uv 环境。数据转换、训练在 OpenPI 环境中执行；仿真评测在 RoboTwin 环境中执行。两者不能假定是同一个 Python 环境。

## 2. 选任务并准备新数据

示例任务为 `place_a2b_left`、`demo_clean`、默认 `aloha_agilex`。先下载该任务的 RoboTwin 2.0 原始演示：

```bash
cd "$ROBOTWIN_ROOT"
bash scripts/download_xpolicylab_data.sh place_a2b_left
find data/demo_clean/place_a2b_left/aloha_agilex/data -name 'episode_*.hdf5' | wc -l
```

XPolicyLab 当前 OpenPI 安装脚本固定使用 LeRobot `0.4.4`，对应数据集 v3.0。在 OpenPI 的 uv 环境中转换成**新**的 LeRobot v3.0 数据集；转换前先检查目标目录不存在，避免覆盖别的实验：

```bash
export REPO_ID=rtc_place_a2b_left_clean
export OPENPI_DIR="$ROBOTWIN_ROOT/XPolicyLab/policy/Pi_0/openpi"
export HF_LEROBOT_HOME="$HOME/.cache/huggingface/lerobot"
test ! -e "$HF_LEROBOT_HOME/$REPO_ID" || { echo 'Dataset exists: choose a new REPO_ID'; exit 1; }
cd "$OPENPI_DIR"
uv run python "$ROBOTWIN_ROOT/XPolicyLab/scripts/transform_lerobot_v30_format.py" \
  'demo_clean.place_a2b_left.aloha_agilex' --repo_id "$REPO_ID"
cat "$HF_LEROBOT_HOME/$REPO_ID/meta/info.json" | head -n 25
```

若你的 checkout 锁定了更早的 LeRobot v2.1，改用 XPolicyLab 官方的 `transform_lerobot_v21_format.py`；先确认 `openpi` 环境里的版本与转换脚本兼容。图像解码沿用官方转换器，不自行交换 RGB/BGR 通道。

## 3. 配置 π0 LoRA 和归一化

默认设置：单张 48GB 卡、`batch_size=32`、`fsdp_devices=1`、训练 30,000 步，每 5,000 步存档。下面脚本会给 XPolicyLab 的 `config.py` 增加独立的 `rtc_robotwin_pi0_lora` 配置，并备份原文件。重复运行会更新同一个配置，不会追加多个副本：

```bash
python "$RTC_DIR/scripts/prepare_robotwin_pi0.py" \
  --robotwin-root "$ROBOTWIN_ROOT" \
  --batch-size 32 --fsdp-devices 1 --steps 30000 --save-interval 5000
export OPENPI_DATA_REPO_ID="$REPO_ID"
cd "$OPENPI_DIR"
uv run scripts/compute_norm_stats.py rtc_robotwin_pi0_lora
test -f "assets/rtc_robotwin_pi0_lora/$REPO_ID/norm_stats.json"
```

双张 24GB 时改为 `--fsdp-devices 2`，训练命令同时改为 `--gpus 0,1`。`batch-size` 必须能被 `fsdp-devices` 整除。这次单张 48GB 直接使用 `batch_size=32` 启动正式训练；启动前核对新数据集、归一化统计和 checkpoint 路径，不另跑 20 步批量测试。更改配置时不要让训练进程运行。

## 4. 新训练、查看进度、断点续训

```bash
export RUN_NAME=rtc_place_a2b_left_lora_seed0
cd "$RTC_DIR"
nohup python scripts/train_robotwin_pi0.py \
  --robotwin-root "$ROBOTWIN_ROOT" --repo-id "$REPO_ID" \
  --run-name "$RUN_NAME" --gpus 0 \
  > "$HOME/${RUN_NAME}.log" 2>&1 &
echo "PID=$!"
```

只有**在该 run 目录已有数字命名的完整 checkpoint**后，才能用相同参数追加 `--resume`。入口会拒绝对已有目录重新启动新训练，也不会传 `--overwrite`：

```bash
pgrep -af 'scripts/train.py'
tail -n 15 "$HOME/${RUN_NAME}.log"
find "$ROBOTWIN_ROOT/XPolicyLab/policy/Pi_0/checkpoints/$RUN_NAME" \
  -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -n
```

训练 loss 衡量模型对训练演示的拟合；不能直接当成仿真成功率或独立验证集 loss。需要真正的 validation loss 时，应划分未参与训练的 episode 并在对应数据集上计算；本流程输出的图会明确标为 **Training loss**。

## 5. 用新 checkpoint 开服务，异步测评并录像

训练完成后选数字 checkpoint（例如 30000），更新 `deploy.yml` 的模型配置、归一化资产 ID 和步数；脚本会先检查它们属于同一个新 run：

```bash
python "$RTC_DIR/scripts/configure_robotwin_deploy.py" \
  --robotwin-root "$ROBOTWIN_ROOT" --run-name "$RUN_NAME" \
  --repo-id "$REPO_ID" --checkpoint-step 30000
```

终端 A，在 OpenPI 策略环境启动模型服务：

```bash
export OPENPI_DATA_REPO_ID="$REPO_ID"
cd "$ROBOTWIN_ROOT/XPolicyLab/policy/Pi_0"
bash setup_eval_policy_server.sh RoboTwin place_a2b_left \
  "$RUN_NAME" arx_x5 joint 0 0 uv 6000 localhost
```

终端 B，在 RoboTwin 仿真环境运行 RTC 异步测评；录像和结果写入 RoboTwin 的 `eval_result/`。先测一个 episode，确认画面、动作维度与结果，再用相同参数测 100 个 episode，保留不同日志：

```bash
cd "$RTC_DIR"
python scripts/eval_robotwin_rtc.py \
  --robotwin-root "$ROBOTWIN_ROOT" --task-name place_a2b_left \
  --ckpt-setting "$RUN_NAME" --port 6000 --test-num 1 \
  --request-interval 8 |& tee "$HOME/${RUN_NAME}_eval1.log"

python scripts/eval_robotwin_rtc.py \
  --robotwin-root "$ROBOTWIN_ROOT" --task-name place_a2b_left \
  --ckpt-setting "$RUN_NAME" --port 6000 --test-num 100 \
  --request-interval 8 |& tee "$HOME/${RUN_NAME}_eval100.log"
```

测评时模拟器和策略服务可分别用不同 GPU：在终端 A 设置 `CUDA_VISIBLE_DEVICES=0`，终端 B 设置 `CUDA_VISIBLE_DEVICES=1`。单卡时两者会争用显存，先完成训练并检查可用显存。`--request-interval 8` 表示每经过约八个仿真动作步启动一次异步预测，模型仍输出 50 步动作块；融合由 `src/robotwin_rtc.py` 调用 `RealTimeChunkingBuffer` 完成。与原 RTC 真机控制频率不同，这里控制时间由 RoboTwin 的 `take_action` 决定。

## 6. 绘图并整理 GitHub 结果

```bash
python scripts/plot_robotwin_metrics.py \
  --train-log "$HOME/${RUN_NAME}.log" \
  --eval-log "$HOME/${RUN_NAME}_eval100.log" \
  --output-dir "$HOME/${RUN_NAME}_metrics"
```

得到 `train_loss.png`、`eval_success.png`、CSV 和 `summary.json`。图表只应在实际训练和测评完成后生成。把代码、配置参数、数据集来源、运行命令、测评设置和结果摘要提交 GitHub；大体积 checkpoint 与视频可以放外部存储并提供链接。成功率若要求与某基准在 ±0.5 个百分点内，100 次独立测评只能以 1 个百分点为单位变化，且抽样波动仍可能远大于 0.5；必须先明确相同任务、评测协议和随机种子，再决定样本量。

## 调参速查

| 想改什么 | 在哪里改 | 影响 |
| --- | --- | --- |
| 训练任务 | 数据下载任务、转换模式、`REPO_ID`、`RUN_NAME`、评测 `--task-name` | 数据、checkpoint 和场景必须一致 |
| 训练步数 | `prepare_robotwin_pi0.py --steps` | 决定目标步数；续训前增大该数，再传 `--resume` |
| 批量 | `--batch-size` | 本次设为 32；只有实际训练遇到显存不足才考虑下调 |
| GPU 数 | `--fsdp-devices` 与训练 `--gpus` | 单 48GB 用 `1`/`0`；双 24GB 用 `2`/`0,1` |
| 保存间隔 | `--save-interval` | 越小可恢复得越细，但 checkpoint 占用更多磁盘 |
| 异步请求间隔 | `eval_robotwin_rtc.py --request-interval` | 越小预测越频繁，负载更高；先用 8 |

更换 `REPO_ID` 后必须重新计算归一化统计，并把测评 `deploy.yml` 的 `repo_id` 改成同一个值。**不要在原 run 上用 `--overwrite`**，否则已有训练 checkpoint 可能被删除。
