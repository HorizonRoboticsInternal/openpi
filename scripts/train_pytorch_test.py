import dataclasses
import pathlib
import tempfile
import unittest

import safetensors.torch
import torch

import openpi.models_pytorch.pi0_pytorch as pi0_pytorch
import openpi.shared.ema_utils as ema_utils
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

    def load_model(self,
                   weight_path,
                   *,
                   only_trainable=True,
                   allow_missing_keys=False):
        del only_trainable, allow_missing_keys
        state_dict = safetensors.torch.load_file(weight_path)
        self.load_state_dict(state_dict, strict=False)


@dataclasses.dataclass
class _DummyConfig:
    checkpoint_dir: pathlib.Path
    save_interval: int = 1
    num_train_steps: int = 10
    wandb_enabled: bool = False
    ema_decay: float | None = None
    ema_decay_values: tuple[float, ...] | None = None


@dataclasses.dataclass
class _DummyDataConfig:
    norm_stats: object = None
    asset_id: str | None = None


class CpuEmaTrackersTest(unittest.TestCase):

    def test_save_checkpoint_writes_primary_and_secondary_ema_files(self):
        model = _DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        ema_specs = ema_utils.build_ema_specs((0.5, 0.25))
        ema_trackers = train_pytorch.CpuEmaTrackers(model, ema_specs)
        model.weight.data.copy_(torch.tensor([5.0, 7.0]))
        model.counter.copy_(torch.tensor([9], dtype=torch.int64))
        ema_trackers.update(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _DummyConfig(checkpoint_dir=pathlib.Path(tmpdir))
            data_config = _DummyDataConfig()

            train_pytorch.save_checkpoint(model,
                                          optimizer,
                                          global_step=1,
                                          config=config,
                                          is_main=True,
                                          data_config=data_config,
                                          ema_trackers=ema_trackers)

            ckpt_dir = config.checkpoint_dir / "1"
            primary_state = safetensors.torch.load_file(ckpt_dir /
                                                        "model.safetensors")
            secondary_path = ckpt_dir / ema_utils.checkpoint_model_filename_for_decay(
                0.25)
            secondary_state = safetensors.torch.load_file(secondary_path)
            raw_state = safetensors.torch.load_file(ckpt_dir /
                                                    "train_model.safetensors")
            metadata = torch.load(ckpt_dir / "metadata.pt",
                                  map_location="cpu",
                                  weights_only=False)

            torch.testing.assert_close(primary_state["weight"],
                                       torch.tensor([3.0, 5.0]))
            torch.testing.assert_close(secondary_state["weight"],
                                       torch.tensor([4.0, 6.0]))
            torch.testing.assert_close(raw_state["weight"],
                                       torch.tensor([5.0, 7.0]))
            self.assertEqual(metadata["ema_decays"], [0.5, 0.25])
            self.assertEqual(metadata["primary_ema_decay"], 0.5)
            self.assertTrue((ckpt_dir / "ema.pt").exists())
            self.assertTrue(
                (ckpt_dir /
                 ema_utils.checkpoint_shadow_filename_for_decay(0.25)).exists())

    def test_load_ema_checkpoints_restores_exact_match(self):
        model = _DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        ema_specs = ema_utils.build_ema_specs((0.5, 0.25))
        ema_trackers = train_pytorch.CpuEmaTrackers(model, ema_specs)
        model.weight.data.copy_(torch.tensor([5.0, 7.0]))
        ema_trackers.update(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _DummyConfig(checkpoint_dir=pathlib.Path(tmpdir))
            data_config = _DummyDataConfig()
            train_pytorch.save_checkpoint(model,
                                          optimizer,
                                          global_step=3,
                                          config=config,
                                          is_main=True,
                                          data_config=data_config,
                                          ema_trackers=ema_trackers)

            restored_model = _DummyModel()
            restored_optim = torch.optim.SGD(restored_model.parameters(), lr=0.1)
            _, ckpt_dir = train_pytorch.load_checkpoint(restored_model,
                                                        restored_optim,
                                                        config.checkpoint_dir,
                                                        device=torch.device("cpu"))
            restored_trackers = train_pytorch.CpuEmaTrackers(
                restored_model, ema_specs)
            loaded = train_pytorch.load_ema_checkpoints(restored_trackers,
                                                        ckpt_dir)

            self.assertTrue(loaded)
            torch.testing.assert_close(
                restored_trackers.trackers[ema_specs[0].key].shadow["weight"],
                torch.tensor([3.0, 5.0]))
            torch.testing.assert_close(
                restored_trackers.trackers[ema_specs[1].key].shadow["weight"],
                torch.tensor([4.0, 6.0]))

    def test_load_ema_checkpoints_clones_missing_from_primary_shadow(self):
        model = _DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        primary_specs = ema_utils.build_ema_specs((0.5, ))
        primary_trackers = train_pytorch.CpuEmaTrackers(model, primary_specs)
        model.weight.data.copy_(torch.tensor([5.0, 7.0]))
        primary_trackers.update(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _DummyConfig(checkpoint_dir=pathlib.Path(tmpdir))
            data_config = _DummyDataConfig()
            train_pytorch.save_checkpoint(model,
                                          optimizer,
                                          global_step=3,
                                          config=config,
                                          is_main=True,
                                          data_config=data_config,
                                          ema_trackers=primary_trackers)

            resumed_specs = ema_utils.build_ema_specs((0.5, 0.25))
            restored_trackers = train_pytorch.CpuEmaTrackers(model,
                                                             resumed_specs)
            with self.assertLogs(level="WARNING") as logs:
                loaded = train_pytorch.load_ema_checkpoints(
                    restored_trackers, config.checkpoint_dir / "3")

            self.assertTrue(loaded)
            self.assertIn("Initialized them from saved primary shadow",
                          "\n".join(logs.output))
            primary_shadow = restored_trackers.trackers[
                resumed_specs[0].key].shadow["weight"]
            secondary_shadow = restored_trackers.trackers[
                resumed_specs[1].key].shadow["weight"]
            torch.testing.assert_close(primary_shadow, secondary_shadow)


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


class ResumeTargetStepValidationTest(unittest.TestCase):

    def test_validate_resume_target_step_rejects_non_increasing_target(self):
        config = _DummyConfig(checkpoint_dir=pathlib.Path("/tmp"),
                              num_train_steps=1000)

        with self.assertRaisesRegex(ValueError, "absolute final step"):
            train_pytorch.validate_resume_target_step(config, latest_step=15000)

    def test_validate_resume_target_step_accepts_larger_target(self):
        config = _DummyConfig(checkpoint_dir=pathlib.Path("/tmp"),
                              num_train_steps=16000)

        train_pytorch.validate_resume_target_step(config, latest_step=15000)


class FreshOptimizerResumeTest(unittest.TestCase):

    def test_load_checkpoint_can_skip_optimizer_state(self):
        model = _DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        optimizer.zero_grad(set_to_none=True)
        model.weight.grad = torch.tensor([1.0, 1.0])
        optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _DummyConfig(checkpoint_dir=pathlib.Path(tmpdir))
            data_config = _DummyDataConfig()
            train_pytorch.save_checkpoint(model,
                                          optimizer,
                                          global_step=2,
                                          config=config,
                                          is_main=True,
                                          data_config=data_config)

            restored_model = _DummyModel()
            restored_optim = torch.optim.SGD(restored_model.parameters(),
                                             lr=0.1,
                                             momentum=0.9)
            with self.assertLogs(level="WARNING") as logs:
                global_step, _ = train_pytorch.load_checkpoint(
                    restored_model,
                    restored_optim,
                    config.checkpoint_dir,
                    device=torch.device("cpu"),
                    fresh_optimizer_on_resume=True)

            self.assertEqual(global_step, 2)
            self.assertIn("fresh_optimizer_on_resume=True",
                          "\n".join(logs.output))
            self.assertEqual(restored_optim.state, {})


if __name__ == "__main__":
    unittest.main()
