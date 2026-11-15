# RTC Anything on RoboTwin with π0

This fork runs RoboTwin's official XPolicyLab evaluator with RTC Anything's
asynchronous action chunking. The policy server uses XPolicyLab's `Pi_0`
adapter. The simulator and policy server are separate processes. The simulator
steps while the next inference request is in flight; overlapping predictions
are blended at each control step. One episode uses one policy client and one
inference worker. A policy checkpoint trained for RoboTwin's `arx_x5` joint
action layout is required to measure task success; a generic π0 checkpoint
does not establish task performance.

## On the GPU machine

Use your existing Docker image if it already has the RoboTwin simulator,
XPolicyLab and CUDA runtime. Inside the container, check these prerequisites:

```bash
nvidia-smi
python -c 'import sapien, numpy; print("sim imports OK", numpy.__version__)'
test -f /workspace/RoboTwin/scripts/eval_policy_xpolicylab.py
test -f /workspace/RoboTwin/XPolicyLab/policy/Pi_0/model.py
```

If RoboTwin is absent, follow its official `README.md` and installation guide
in a separate checkout, including assets and its pinned XPolicyLab submodule.
Do not mix its simulator's pinned NumPy 1.x environment with the policy's
OpenPI uv environment. Install the π0 policy environment only if absent:

```bash
cd /workspace/RoboTwin/XPolicyLab/policy/Pi_0
bash install.sh
```

Set `ckpt_name` to the existing **RoboTwin-compatible π0 checkpoint** name or
directory. Confirm the corresponding `train_config_name`, normalization
statistics, and checkpoint setting in `Pi_0/deploy.yml`. In one terminal, start
the policy server (replace `YOUR_CHECKPOINT`):

```bash
cd /workspace/RoboTwin/XPolicyLab/policy/Pi_0
bash setup_eval_policy_server.sh RoboTwin place_a2b_left YOUR_CHECKPOINT arx_x5 joint 0 0 uv 6000 localhost
```

In the RoboTwin simulator environment, run one evaluation episode:

```bash
cd /workspace/RTC-Anything_RobotwinVersion
python scripts/eval_robotwin_rtc.py \
  --robotwin-root /workspace/RoboTwin \
  --task-name place_a2b_left \
  --ckpt-setting YOUR_CHECKPOINT \
  --port 6000 --test-num 1 --request-interval 8
```

`--task-name` must be a RoboTwin task in `envs/`, such as `place_a2b_left`;
`demo_clean` is the default task configuration, not a task name.
For a meaningful success rate, use `--test-num 100` only after a one-episode
smoke test and a checkpoint appropriate for that task. RoboTwin writes results
under `eval_result/` in its checkout.

The policy server and simulator need access to the same container network; if
they run on different hosts, set `--host` to the server's reachable address
and bind the server to the appropriate interface. Inference and simulation can
share one GPU if memory permits, but separate GPUs avoid contention.

## Local verification

```bash
python -m unittest discover -s tests -v
```

This checks chunk handling, per-episode reset, error propagation, and
XPolicyLab's dictionary-shaped actions without CUDA. A real RoboTwin rollout
and success rate still require a GPU simulator, π0 server, assets and checkpoint.
