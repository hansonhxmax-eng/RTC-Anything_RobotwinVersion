"""Check safe config injection and truthful log extraction without a GPU."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_robotwin_pi0 import configure
from plot_robotwin_metrics import parse_evaluation, parse_training


class TrainingSetupTests(unittest.TestCase):
    def test_config_idempotence_and_new_run_identity(self):
        source = ('_CONFIGS = [TrainConfig(name="pi0_base_aloha_full_sim_arx-x5_seed_0")]\n'
                  'if len({config.name for config in _CONFIGS}) != len(_CONFIGS):\n    pass\n')
        kwargs = dict(batch_size=32, steps=30000, save_interval=5000, fsdp_devices=1)
        configured = configure(source, **kwargs)
        self.assertEqual(configure(configured, **kwargs), configured)
        self.assertIn('name="rtc_robotwin_pi0_lora"', configured)
        self.assertIn("assets=AssetsConfig()", configured)
        self.assertIn("fsdp_devices=1", configured)

    def test_invalid_device_batch_combination_fails(self):
        source = ('_CONFIGS = [TrainConfig(name="pi0_base_aloha_full_sim_arx-x5_seed_0")]\n'
                  'if len({config.name for config in _CONFIGS}) != len(_CONFIGS):\n    pass\n')
        with self.assertRaises(ValueError):
            configure(source, batch_size=31, steps=20, save_interval=10, fsdp_devices=2)

    def test_curves_separate_train_loss_from_success(self):
        self.assertEqual(parse_training("Step 0: grad_norm=2.1, loss=0.12\n"
                                        "Step 100: grad_norm=1.3, loss=0.03\n"),
                         [(0, 0.12), (100, 0.03)])
        self.assertEqual(parse_evaluation("Success rate: 1/1 => 100.0%, current seed: 100001\n"
                                          "Success rate: 1/2 => 50.0%, current seed: 100002"),
                         [(1, 1, 100.0), (2, 1, 50.0)])


if __name__ == "__main__":
    unittest.main()
