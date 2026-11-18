"""Install an idempotent RoboTwin π0 LoRA config into XPolicyLab.

Run this in the environment used to train and serve the model. The generated
config uses the official arx_x5 observation/action mapping but has its own
dataset, normalization assets, and run name.
"""

from __future__ import annotations

import argparse
from pathlib import Path


CONFIG_NAME = "rtc_robotwin_pi0_lora"
BEGIN = "# RTC_ROBOTWIN_LORA_BEGIN"
END = "# RTC_ROBOTWIN_LORA_END"
ANCHOR = "if len({config.name for config in _CONFIGS}) != len(_CONFIGS):"


def render_config(batch_size: int, steps: int, save_interval: int, fsdp_devices: int) -> str:
    if min(batch_size, steps, save_interval, fsdp_devices) < 1 or batch_size % fsdp_devices:
        raise ValueError("positive batch size, steps, interval and devices required; batch size must divide devices")
    return f'''{BEGIN}
# Based on XPolicyLab's arx_x5 mapping; all checkpoint assets belong to this run.
_rtc_base = next(c for c in _CONFIGS if c.name == "pi0_base_aloha_full_sim_arx-x5_seed_0")
_rtc_model = pi0_config.Pi0Config(
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
)
_CONFIGS.append(dataclasses.replace(
    _rtc_base,
    name="{CONFIG_NAME}",
    model=_rtc_model,
    data=dataclasses.replace(
        _rtc_base.data,
        repo_id=_robodojo_repo_id(),
        assets=AssetsConfig(),
        base_config=DataConfig(prompt_from_task=True),
    ),
    freeze_filter=_rtc_model.get_freeze_filter(),
    batch_size={batch_size},
    fsdp_devices={fsdp_devices},
    num_train_steps={steps},
    ema_decay=None,
    log_interval=100,
    save_interval={save_interval},
    wandb_enabled=False,
))
{END}

'''


def configure(source: str, *, batch_size: int, steps: int, save_interval: int, fsdp_devices: int) -> str:
    if source.count(ANCHOR) != 1:
        raise ValueError("Unsupported XPolicyLab config.py: uniqueness anchor missing or ambiguous")
    if BEGIN in source or END in source:
        if source.count(BEGIN) != 1 or source.count(END) != 1:
            raise ValueError("Malformed RTC config block; inspect config.py before editing")
        start = source.index(BEGIN)
        stop = source.index(END) + len(END)
        if stop > source.index(ANCHOR):
            raise ValueError("RTC config block appears after config validation")
        source = source[:start] + source[stop:].lstrip("\n")
    if source.count('name="pi0_base_aloha_full_sim_arx-x5_seed_0"') != 1:
        raise ValueError("XPolicyLab arx_x5 base config was not found exactly once")
    return source.replace(ANCHOR, render_config(batch_size, steps, save_interval, fsdp_devices) + ANCHOR, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--save-interval", type=int, default=5000)
    args = parser.parse_args()
    path = (args.robotwin_root / "XPolicyLab/policy/Pi_0/openpi/src/openpi/training/config.py").resolve()
    if not path.is_file():
        parser.error(f"XPolicyLab config not found: {path}; initialize RoboTwin submodules first")
    old = path.read_text(encoding="utf-8")
    new = configure(old, batch_size=args.batch_size, steps=args.steps,
                    save_interval=args.save_interval, fsdp_devices=args.fsdp_devices)
    if new != old:
        backup = path.with_name("config.py.before_rtc_robotwin")
        if not backup.exists():
            backup.write_text(old, encoding="utf-8")
        path.write_text(new, encoding="utf-8")
    print(f"{CONFIG_NAME}: batch_size={args.batch_size}, steps={args.steps}, "
          f"save_interval={args.save_interval}, fsdp_devices={args.fsdp_devices}")
    print(f"Updated: {path} (backup: config.py.before_rtc_robotwin)")


if __name__ == "__main__":
    main()
