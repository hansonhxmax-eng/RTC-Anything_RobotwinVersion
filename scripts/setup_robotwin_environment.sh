#!/usr/bin/env bash
# Finish RoboTwin 2.0 simulator and JAX OpenPI setup in an existing GPU container.
# This script never downloads demonstration data or starts training.
set -Eeuo pipefail
trap 'printf "[setup failed] line %s: %s\n" "$LINENO" "$BASH_COMMAND" >&2' ERR

ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/workspace/RoboTwin}"
RTC_ROOT="${RTC_ROOT:-/workspace/RTC-Anything_RobotwinVersion}"
OPENPI_ROOT="$ROBOTWIN_ROOT/XPolicyLab/policy/Pi_0/openpi"

test -f "$ROBOTWIN_ROOT/scripts/_download_assets.sh"
test -f "$OPENPI_ROOT/scripts/train.py"
test -f "$RTC_ROOT/scripts/prepare_robotwin_pi0.py"
test -f /usr/share/vulkan/icd.d/nvidia_icd.json
test -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json

source /opt/conda/etc/profile.d/conda.sh
conda activate RoboTwin

echo '[1/6] Preparing the new JAX pi0 training config (no training yet)'
python "$RTC_ROOT/scripts/prepare_robotwin_pi0.py" \
  --robotwin-root "$ROBOTWIN_ROOT" \
  --batch-size 32 --fsdp-devices 1 --steps 30000 --save-interval 5000

echo '[2/6] Installing XPolicyLab into RoboTwin simulator Python'
python -m pip install -e "$ROBOTWIN_ROOT/XPolicyLab"

echo '[3/6] Installing the RoboTwin-pinned cuRobo and Warp'
if [[ ! -d "$ROBOTWIN_ROOT/envs/curobo/.git" ]]; then
  if [[ -e "$ROBOTWIN_ROOT/envs/curobo" ]]; then
    echo 'envs/curobo exists but is not a complete Git checkout; inspect it before retrying' >&2
    exit 1
  fi
  git clone --branch v0.7.8 --depth 1 \
    https://github.com/NVlabs/curobo.git "$ROBOTWIN_ROOT/envs/curobo"
fi
python -m pip install -e "$ROBOTWIN_ROOT/envs/curobo" --no-build-isolation
python -m pip install 'warp-lang==1.12.0' 'setuptools==69.5.1'

echo '[4/6] Applying the two upstream simulator compatibility edits'
python - <<'PY'
from pathlib import Path
import sysconfig

site = Path(sysconfig.get_paths()['purelib'])
patches = {
    site / 'sapien/wrapper/urdf_loader.py': [
        ('with open(urdf_file, "r") as f:', 'with open(urdf_file, "r", encoding="utf-8") as f:'),
        ('with open(srdf_file, "r") as f:', 'with open(srdf_file, "r", encoding="utf-8") as f:'),
    ],
    site / 'mplib/planner.py': [
        ('if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:',
         'if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:'),
    ],
}
for path, replacements in patches.items():
    data = path.read_text(encoding='utf-8')
    for old, new in replacements:
        if old in data:
            data = data.replace(old, new)
        elif new not in data:
            raise RuntimeError(f'Expected compatibility patch target missing in {path}: {old}')
    path.write_text(data, encoding='utf-8')
    print('Patched:', path)
PY

echo '[5/6] Downloading RoboTwin simulation assets and resolving paths'
if [[ -d "$ROBOTWIN_ROOT/assets/background_texture" \
   && -d "$ROBOTWIN_ROOT/assets/embodiments" \
   && -d "$ROBOTWIN_ROOT/assets/objects" ]]; then
  echo 'Assets already extracted; skipping archive download'
else
  cd "$ROBOTWIN_ROOT/assets"
  python _download.py
  for name in background_texture embodiments objects; do
    if [[ -f "$name.zip" ]]; then
      unzip -oq "$name.zip"
      rm -f "$name.zip"
    elif [[ ! -d "$name" ]]; then
      echo "Missing assets/$name and assets/$name.zip" >&2
      exit 1
    fi
  done
fi
cd "$ROBOTWIN_ROOT"
python scripts/update_embodiment_config_path.py

echo '[6/6] Installing OpenPI in its independent Python 3.11 JAX environment'
python -m pip install uv
cd "$OPENPI_ROOT"
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv sync --group lerobot --python 3.11
# --python is necessary: the active RoboTwin Conda environment is Python 3.10.
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv pip install \
  --python "$OPENPI_ROOT/.venv/bin/python" -e "$OPENPI_ROOT"
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv pip install \
  --python "$OPENPI_ROOT/.venv/bin/python" -e "$ROBOTWIN_ROOT/XPolicyLab"

echo '[verify] cuRobo, simulator GPU and JAX GPU'
python - <<'PY'
import torch
from curobo.wrap.reacher.motion_gen import MotionGen
assert torch.cuda.is_available(), 'Simulator Torch cannot see GPU'
print('cuRobo MotionGen import OK; simulator GPU:', torch.cuda.get_device_name(0))
PY
"$OPENPI_ROOT/.venv/bin/python" - <<'PY'
import jax
from openpi.training import config
print('OpenPI JAX devices:', jax.devices())
assert any(d.platform == 'gpu' for d in jax.devices()), 'JAX cannot see GPU'
print('RTC pi0 config:', config.get_config('rtc_robotwin_pi0_lora').name)
PY
echo '[setup complete] Simulator and JAX OpenPI are ready. No training was started.'
