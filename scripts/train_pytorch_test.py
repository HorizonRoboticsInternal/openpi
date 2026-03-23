import dataclasses
import pathlib
import tempfile
import unittest

import safetensors.torch
import torch

import openpi.models_pytorch.pi0_pytorch as pi0_pytorch
import train_pytorch


class _DummyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 3.0]))
        self.register_buffer("counter", torch.tensor([5], dtype=torch.int64))

    def save_model(self, weight_path, *, only_trainable=True):
        state_dict = pi0_pytorch.get_dedup_state_dict(self,
                                                      only_trainable=only_trainable)
        safetensors.torch.save_file(
            {k: v.detach().cpu() for k, v in state_dict.items()}, weight_path)

    def load_model(self, weight_path, *, only_trainable=True):
        del only_trainable
        state_dict = safetensors.torch.load_file(weight_path)
        self.load_state_dict(state_dict, strict=False)


@dataclasses.dataclass
class _DummyConfig:
    checkpoint_dir: pathlib.Path
    save_interval: int = 1
    num_train_steps: int = 10
    wandb_enabled: bool = False


@dataclasses.dataclass
class _DummyDataConfig:
    norm_stats: object = None
    asset_id: str | None = None


class CpuEmaTrackerTest(unittest.TestCase):

    def test_save_checkpoint_writes_ema_and_raw_weights(self):
        model = _DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        ema_tracker = train_pytorch.CpuEmaTracker(model, decay=0.5)
        model.weight.data.copy_(torch.tensor([5.0, 7.0]))
        model.counter.copy_(torch.tensor([9], dtype=torch.int64))
        ema_tracker.update(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _DummyConfig(checkpoint_dir=pathlib.Path(tmpdir))
            data_config = _DummyDataConfig()

            train_pytorch.save_checkpoint(model,
                                          optimizer,
                                          global_step=1,
                                          config=config,
                                          is_main=True,
                                          data_config=data_config,
                                          ema_tracker=ema_tracker)

            ckpt_dir = config.checkpoint_dir / "1"
            ema_state = safetensors.torch.load_file(ckpt_dir /
                                                    "model.safetensors")
            raw_state = safetensors.torch.load_file(ckpt_dir /
                                                    "train_model.safetensors")
            metadata = torch.load(ckpt_dir / "metadata.pt",
                                  map_location="cpu",
                                  weights_only=False)

            torch.testing.assert_close(ema_state["weight"],
                                       torch.tensor([3.0, 5.0]))
            torch.testing.assert_close(raw_state["weight"],
                                       torch.tensor([5.0, 7.0]))
            torch.testing.assert_close(raw_state["counter"],
                                       torch.tensor([9], dtype=torch.int64))
            self.assertTrue(metadata["checkpoint_uses_ema"])
            self.assertTrue((ckpt_dir / "ema.pt").exists())

    def test_load_checkpoint_restores_raw_weights_and_ema_state(self):
        model = _DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        ema_tracker = train_pytorch.CpuEmaTracker(model, decay=0.5)
        model.weight.data.copy_(torch.tensor([5.0, 7.0]))
        model.counter.copy_(torch.tensor([9], dtype=torch.int64))
        ema_tracker.update(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _DummyConfig(checkpoint_dir=pathlib.Path(tmpdir))
            data_config = _DummyDataConfig()
            train_pytorch.save_checkpoint(model,
                                          optimizer,
                                          global_step=3,
                                          config=config,
                                          is_main=True,
                                          data_config=data_config,
                                          ema_tracker=ema_tracker)

            restored_model = _DummyModel()
            restored_model.weight.data.zero_()
            restored_model.counter.zero_()
            restored_optim = torch.optim.SGD(restored_model.parameters(), lr=0.1)
            global_step, ckpt_dir = train_pytorch.load_checkpoint(
                restored_model,
                restored_optim,
                config.checkpoint_dir,
                device=torch.device("cpu"),
            )

            restored_ema = train_pytorch.CpuEmaTracker(restored_model, decay=0.5)
            loaded = train_pytorch.load_ema_checkpoint(restored_ema, ckpt_dir)

            self.assertEqual(global_step, 3)
            self.assertTrue(loaded)
            torch.testing.assert_close(restored_model.weight,
                                       torch.tensor([5.0, 7.0]))
            torch.testing.assert_close(restored_model.counter,
                                       torch.tensor([9], dtype=torch.int64))
            torch.testing.assert_close(restored_ema.shadow["weight"],
                                       torch.tensor([3.0, 5.0]))


class AverageScalarDictsTest(unittest.TestCase):

    def test_average_scalar_dicts_handles_optional_keys(self):
        infos = [
            {
                "loss": 1.0,
                "learning_rate": 0.1,
                "grad_norm": 2.0,
            },
            {
                "loss": 3.0,
                "learning_rate": 0.3,
                "grad_norm": 4.0,
                "value/ac_pos_frac": 0.25,
            },
            {
                "loss": 5.0,
                "learning_rate": 0.5,
                "grad_norm": 6.0,
                "value/ac_pos_frac": 0.75,
            },
        ]

        avg_info = train_pytorch.average_scalar_dicts(infos)

        self.assertEqual(avg_info["loss"], 3.0)
        self.assertEqual(avg_info["learning_rate"], 0.3)
        self.assertEqual(avg_info["grad_norm"], 4.0)
        self.assertEqual(avg_info["value/ac_pos_frac"], 0.5)


if __name__ == "__main__":
    unittest.main()
