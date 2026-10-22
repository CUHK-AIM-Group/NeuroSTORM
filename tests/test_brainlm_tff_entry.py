"""Exercise the actual BrainLM/TFF encoders through shared training and data."""

from argparse import ArgumentParser, Namespace
from pathlib import Path
import shlex
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch import nn

from datasets.brainlm_dataset import BrainLMDataset
from datasets.data_module import fMRIDataModule
from models.lightning_model import LightningModel
from models.load_model import load_model


def hparams(name, task_type="classification", classes=2, **overrides):
    parser = ArgumentParser(add_help=False)
    for key, default in [("dataset_name", "HCP1200"), ("task_name", "sex"),
                         ("downstream_task_id", 1), ("downstream_task_type", task_type),
                         ("strategy", "auto"), ("print_flops", False)]:
        parser.add_argument("--" + key, default=default)
    parser = LightningModel.add_model_specific_args(parser)
    parser = fMRIDataModule.add_data_specific_args(parser)
    values = vars(parser.parse_args([]))
    values.update(model=name, downstream_task_type=task_type, num_classes=classes,
                  data_type="roi" if name == "brainlm" else "voxel",
                  task_name="age" if task_type == "regression" else "sex",
                  sequence_length=8 if name == "brainlm" else 2,
                  atlas_name="brainlm424", img_size=[16, 16, 16, 2],
                  brainlm_num_rois=8, brainlm_image_size=16, brainlm_patch_size=4,
                  brainlm_hidden_size=32, brainlm_num_layers=1, brainlm_num_heads=4,
                  brainlm_intermediate_size=64, brainlm_dropout=0.,
                  tff_hidden_size=32, tff_num_layers=1, tff_num_heads=4,
                  tff_intermediate_size=64, tff_max_sequence_length=2,
                  num_workers=0, batch_size=2, eval_batch_size=2, limit_training_samples=1.)
    values.update(overrides)
    return values


def inputs(name, batch=1):
    return torch.randn(batch, 1, 8, 8) if name == "brainlm" else torch.randn(batch, 1, 16, 16, 16, 2)


@pytest.mark.parametrize("name", ["brainlm", "tff"])
@pytest.mark.parametrize("task_type,classes", [("classification", 2), ("classification", 3), ("regression", 1)])
def test_actual_encoder_loss_gradients_and_single_item_batch(name, task_type, classes):
    dm = SimpleNamespace(train_dataset=SimpleNamespace(target_values=np.arange(3.).reshape(-1, 1)))
    model = LightningModel(dm, **hparams(name, task_type, classes)).eval()
    assert isinstance(model.output_head, nn.Identity)
    x = inputs(name)
    key = "roi_sequence" if name == "brainlm" else "fmri_sequence"
    batch = {key: x, "target": torch.tensor([1.]), "subject_name": ["s"]}
    logits = model(x)
    assert logits.shape == (1, 3 if classes == 3 else 1)
    loss = model._calculate_loss(batch, mode="train")
    assert torch.isfinite(loss)
    loss.backward()
    connected = [(key, parameter) for key, parameter in model.named_parameters() if parameter.requires_grad]
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for _, parameter in connected)
    assert any(parameter.grad.abs().sum() > 0 for key, parameter in connected if "backbone" in key)
    saved = model.validation_step(batch, 0, 0)[1]
    assert saved[0].shape[0] == saved[1].shape[0] == 1


@pytest.mark.parametrize("name", ["brainlm", "tff"])
def test_complete_encoder_initializer_and_restore_without_initializer(name, tmp_path):
    values = hparams(name)
    original = load_model(name, Namespace(**values))
    path = tmp_path / "initializer.pt"
    prefix = "vit." if name == "brainlm" else "pretrained."
    state = {"module." + prefix + key: value.clone() for key, value in original.backbone.state_dict().items()}
    state["decoder.unused"] = torch.zeros(1)
    torch.save({"state_dict": state}, path)
    values[name + "_checkpoint_path"] = str(path)
    initialized = load_model(name, Namespace(**values))
    for key, expected in original.backbone.state_dict().items():
        torch.testing.assert_close(initialized.backbone.state_dict()[key], expected, rtol=0, atol=0)
    torch.save({"state_dict": {"unrelated": torch.zeros(1)}}, path)
    with pytest.raises(RuntimeError):
        load_model(name, Namespace(**values))
    path.unlink()
    values["test_ckpt_path"] = "framework.ckpt"
    # The full framework restoration supplies model weights separately.
    restored = load_model(name, Namespace(**values))
    restored.load_state_dict(initialized.state_dict(), strict=True)
    x = inputs(name)
    torch.testing.assert_close(restored.eval()(x), initialized.eval()(x), rtol=0, atol=0)


def test_brainlm_roi_scaling_order_center_window_and_invalid_stats(tmp_path):
    folder = tmp_path / "roi" / "a424"
    folder.mkdir(parents=True)
    array = np.arange(80, dtype=np.float32).reshape(8, 10) ** 2
    np.save(folder / "s_a424.npy", array)
    order = np.arange(7, -1, -1)
    stats = tmp_path / "stats.npz"
    np.savez(stats, median=np.arange(8, dtype=np.float32), iqr=np.full(8, 2.), roi_order=order)
    ds = BrainLMDataset(tmp_path, {"s": [0, 1.]}, atlas_name="a424", sequence_length=8,
                        num_rois=8, stats_path=stats, train=False)
    expected = ((array - array.mean(1, keepdims=True) - np.arange(8)[:, None]) / 2.)[order, 1:9] / 5.6430855
    np.testing.assert_allclose(ds[0]["roi_sequence"].numpy()[0], expected, rtol=1e-6)
    torch.testing.assert_close(ds[0]["roi_sequence"], ds[0]["roi_sequence"], rtol=0, atol=0)
    default = BrainLMDataset(tmp_path, {"s": [0, 1.]}, atlas_name="a424", sequence_length=8, num_rois=8)
    assert torch.isfinite(default[0]["roi_sequence"]).all()
    for invalid in [np.zeros((7, 10)), np.zeros((8, 7)), np.full((8, 10), np.nan)]:
        np.save(folder / "s_a424.npy", invalid)
        with pytest.raises(ValueError):
            ds[0]
    np.savez(stats, median=np.zeros(8), iqr=np.zeros(8), roi_order=order)
    with pytest.raises(ValueError, match="positive IQR"):
        BrainLMDataset(tmp_path, {"s": [0, 1.]}, atlas_name="a424", sequence_length=8, num_rois=8, stats_path=stats)


def prepare_data(root, name, values):
    subjects = [str(100001 + i) for i in range(9)]
    rng = np.random.default_rng(32)
    for subject in subjects:
        if name == "brainlm":
            folder = root / "roi" / "brainlm424"
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / (subject + "_brainlm424.npy"), rng.normal(size=(8, 10)).astype(np.float32))
        else:
            folder = root / "img" / subject
            folder.mkdir(parents=True)
            torch.save({"frames": torch.randint(-50, 50, (2, 16, 16, 16), dtype=torch.int8),
                        "scale": 0.1, "num_frames": 2}, folder / "data.pt")
    metadata = root / "metadata"
    metadata.mkdir()
    (metadata / "HCP_1200_gender.csv").write_text("Subject,Gender\n" + "\n".join(s + (",M" if i % 2 else ",F") for i, s in enumerate(subjects)))
    (metadata / "HCP_1200_precise_age.csv").write_text("subject,age,sex\n" + "\n".join(s + f",{20+i},M" for i, s in enumerate(subjects)))
    splits = root / "splits" / "HCP1200"
    splits.mkdir(parents=True)
    (splits / "split_fixed_1.txt").write_text("train\n" + "\n".join(subjects[:3]) + "\nval\n" + "\n".join(subjects[3:6]) + "\ntest\n" + "\n".join(subjects[6:]) + "\n")
    values.update(image_path=str(root), split_root=str(root / "splits"))
    return fMRIDataModule(**values)


@pytest.mark.parametrize("name", ["brainlm", "tff"])
def test_real_data_module_fit_validate_test_and_framework_reload(name, tmp_path):
    values = hparams(name, "regression" if name == "brainlm" else "classification", 1 if name == "brainlm" else 2)
    dm = prepare_data(tmp_path, name, values)
    assert [len(batch["subject_name"]) for batch in dm.val_dataloader()[0]] == [2, 1]
    model = LightningModel(dm, **values)
    initializer = tmp_path / "initializer.pt"
    torch.save(model.model.backbone.state_dict(), initializer)
    values[name + "_checkpoint_path"] = str(initializer)
    model = LightningModel(dm, **values)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=1, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, num_sanity_val_steps=0)
    trainer.fit(model, datamodule=dm)
    assert trainer.global_step == 1
    assert trainer.test(model, datamodule=dm, verbose=False)
    path = tmp_path / "trained.ckpt"
    trainer.save_checkpoint(path)
    initializer.unlink()
    restored = LightningModel.load_from_checkpoint(path, data_module=dm, test_ckpt_path=str(path))
    for key, expected in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], expected, rtol=0, atol=0)
    batch = next(iter(dm.test_dataloader()))
    key = "roi_sequence" if name == "brainlm" else "fmri_sequence"
    torch.testing.assert_close(restored.eval()(batch[key]), model.eval()(batch[key]), rtol=0, atol=0)
    resume_values = dict(values, resume_ckpt_path=str(path))
    resumed = LightningModel(dm, **resume_values)
    next_trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=2, logger=False,
                              enable_checkpointing=False, enable_progress_bar=False,
                              enable_model_summary=False, num_sanity_val_steps=0)
    next_trainer.fit(resumed, datamodule=dm, ckpt_path=str(path))
    assert next_trainer.global_step == 2


@pytest.mark.parametrize("name", ["brainlm", "tff"])
def test_wrong_data_objective_and_cli_runner(name):
    with pytest.raises(ValueError, match="pretraining"):
        load_model(name, Namespace(**hparams(name, pretraining=True)))
    with pytest.raises(ValueError, match="data_type"):
        load_model(name, Namespace(**hparams(name, data_type="fc")))
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash required")
    root = Path(__file__).resolve().parents[1]
    for dataset, task in [("hcp1200", "task1_age"), ("hcptask", "task5")]:
        result = subprocess.run([bash, "scripts/run_experiment.sh", "--model", name, "--dataset", dataset,
                                 "--task", task, "--mode", "train_scratch", "--dry_run"],
                                cwd=root, capture_output=True, text=True, check=True)
        command = shlex.split(next(line for line in result.stdout.splitlines() if line.startswith("python main.py")))
        assert command[command.index("--model") + 1] == name
        assert command[command.index("--sequence_length") + 1] == ("200" if name == "brainlm" else "20")
    result = subprocess.run([bash, "scripts/run_experiment.sh", "--model", name, "--dataset", "hcp1200",
                             "--mode", "pretrain", "--dry_run"], cwd=root, capture_output=True, text=True)
    assert result.returncode != 0


def test_tff_dynamic_batch_and_constant_single_frame():
    model = load_model("tff", Namespace(**hparams("tff"))).eval()
    one = inputs("tff")
    with torch.no_grad():
        torch.testing.assert_close(model(one), model(one.expand(3, -1, -1, -1, -1, -1))[:1], atol=1e-6, rtol=1e-5)
        assert torch.isfinite(model(torch.ones(1, 1, 16, 16, 16, 1))).all()
        with pytest.raises(ValueError, match="frames"):
            model(torch.zeros(1, 1, 16, 16, 16, 3))


def test_brainlm_evaluation_is_deterministic_and_foreground_pooling_has_gradients():
    for readout in ("cls", "foreground_mean"):
        model = load_model("brainlm", Namespace(**hparams("brainlm", brainlm_readout=readout))).eval()
        x = inputs("brainlm").requires_grad_()
        state = torch.random.get_rng_state()
        expected = model(x)
        torch.testing.assert_close(state, torch.random.get_rng_state(), rtol=0, atol=0)
        torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
        expected.square().sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
