# place_a2b_left：π0 + RTC 仿真测评（2026-11-18）

## 配置与训练

- 仿真任务：RoboTwin 2.0 `place_a2b_left`，`demo_clean`，`aloha_agilex`。
- 数据：50 条演示，转换后的 LeRobot 数据集 `rtc_place_a2b_left_clean`，共 7,451 帧。
- 模型：重新从通用 π0 基础权重开始，以 JAX OpenPI 进行 LoRA 微调；训练配置 `rtc_robotwin_pi0_lora`，batch size 32，目标 30,000 步。
- 训练脚本退出码为 0。最终保存的 checkpoint 目录名是 `29999`（训练循环从 0 计数），路径为 `/workspace/RoboTwin/XPolicyLab/policy/Pi_0/checkpoints/rtc_place_a2b_left_lora_seed0/29999`；归一化资产在该目录的 `assets/rtc_place_a2b_left_clean/norm_stats.json`。本实验没有使用先前的 `adjust_bottle` checkpoint。

## 仿真结果

| 评测模式 | 成功 / 总数 | 成功率 | 设置 |
| --- | ---: | ---: | --- |
| RTC 异步动作块融合 | 35 / 100 | 35.0% | 请求间隔 8 个动作步；模型服务端口 6000 |
| 不使用 RTC 的同步 π0 | 33 / 100 | 33.0% | 同一训练 checkpoint，诊断对照 |

两组的差值是 2 个百分点。100 次测评的抽样波动较大，而且专家检查跳过了部分无效随机种子，因此不能据此断言 RTC 提高或降低了成功率。当前可确认的是异步推理与仿真联通，并生成了测评录像；该任务的成功率仍需通过成功和失败视频、动作与数据映射来排查。

另有 1 次异步冒烟测评（0/1）和 10 次同步诊断测评（4/10）；这些不计入上表的正式 100 次测评。

## 曲线

![30,000 步训练 loss](../assets/robotwin/place_a2b_left_train_loss.png)

![100 次异步评测累计成功率](../assets/robotwin/place_a2b_left_eval_success.png)

训练 loss 反映对演示数据的拟合。累计成功率是到第 n 个 episode 为止的成功数除以 n；前期幅度大，终点 35/100 = 35.0%，并不是随着训练步数下降的验证 loss。

## 代表性仿真录像

- [RTC 异步成功 episode 51](../assets/robotwin/place_a2b_left_rtc_success_episode51.mp4)：来自正式 100 次异步测评，12.6 秒；初始 A 在左，机械臂将 A 放到 B 左侧。成功判定对应本次评测的环境标志，视频展示所选片段，不表示衣物折叠、清扫或清洁任务完成。
- 上传压缩包内另有同步 100 条与异步 100 条视频；本仓库目前只收录这一条代表性片段。成功率仍以原始测评日志与结果文件为准。

## 结果文件（仍在 GPU 实例上）

- 异步日志：`/workspace/rtc_place_a2b_left_lora_seed0.eval100.log`。
- 同步对照日志：`/workspace/rtc_place_a2b_left_lora_seed0.sync100.log`。
- 训练日志：`/workspace/rtc_place_a2b_left_lora_seed0.setup.log`。
- 图和数据：`/workspace/rtc_place_a2b_left_lora_seed0_metrics/` 下的 `train_loss.png`、`train_loss.csv`、`eval_success.png`、`eval_success.csv`、`summary.json`。
- 异步评测结果及 MP4：`/workspace/RoboTwin/eval_result/place_a2b_left/Pi_0/demo_clean/rtc_place_a2b_left_lora_seed0/` 的时间戳子目录。

`train_loss.png` 仅表示**训练集 loss**；`eval_success.png` 是仿真 episode 的累计成功率。尚未生成独立验证集的 loss 曲线。两张曲线的 PNG 和一段成功录像已上传；完整录像、曲线 CSV、日志和大体积 checkpoint 尚未上传到 GitHub，销毁租用实例前应下载或另行备份。
