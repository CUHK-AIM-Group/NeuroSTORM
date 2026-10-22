"""Shared registry, data and Lightning paths for Brain-JEPA and BAnD."""

from argparse import ArgumentParser, Namespace
from contextlib import nullcontext
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from datasets.brainjepa_dataset import BrainJEPADataset
from datasets.data_module import fMRIDataModule
from models.brainjepa import BrainJEPA, BrainJEPAP01
from models.lightning_model import LightningModel
from models.load_model import load_model


class TinyROIEncoder(nn.Module):
    """Cheap test encoder; production uses the unchanged Brain-JEPA ViT-Base."""

    def __init__(self, gradient_pos_embed, **kwargs):
        super().__init__()
        self.embed_dim = 8
        self.gradient_pos_embed = gradient_pos_embed
        self.patch = nn.Conv2d(1, 8, kernel_size=(1, 16), stride=(1, 16))

    def forward(self, x):
        return self.patch(x).flatten(2).transpose(1, 2)


def small_brainjepa():
    return patch("models.brainjepa.brainjepa_adapter.official_namespace",
                 return_value={"vit_base": TinyROIEncoder})


def hparams(model, task_type='classification', num_classes=2, **overrides):
    parser = ArgumentParser(add_help=False)
    for name, default in [('dataset_name', 'HCP1200'), ('task_name', 'sex'),
                          ('downstream_task_id', 1), ('downstream_task_type', task_type),
                          ('strategy', 'auto'), ('print_flops', False)]:
        parser.add_argument('--' + name, default=default)
    parser = LightningModel.add_model_specific_args(parser)
    parser = fMRIDataModule.add_data_specific_args(parser)
    values = vars(parser.parse_args([]))
    values.update(model=model, downstream_task_type=task_type, num_classes=num_classes,
                  data_type='roi' if model == 'brainjepa' else 'voxel',
                  task_name='age' if task_type == 'regression' else 'sex',
                  band_base_channels=4, band_num_heads=4, band_feedforward_dim=64,
                  band_dropout=0., band_frame_chunk_size=2, band_max_sequence_length=2,
                  num_workers=0, batch_size=2, eval_batch_size=2, limit_training_samples=1.)
    values.update(overrides)
    return values


class VolumeExamples(Dataset):
    def __init__(self):
        self.target_values = np.array([[0.], [1.], [0.]])

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return dict(fmri_sequence=torch.full((1, 16, 16, 16, 2), index / 3.),
                    subject_name=str(index), target=float(index % 2), tr=1., sex=0)


class VolumeData(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_dataset = VolumeExamples()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=2, drop_last=True)

    def val_dataloader(self):
        return [DataLoader(self.train_dataset, batch_size=2)] * 2

    def test_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=2)


class SharedModelEntryTests(unittest.TestCase):
    def test_registry_output_shapes_and_single_sample_loss(self):
        for name in ('brainjepa', 'band'):
            for task_type, classes in [('classification', 2), ('classification', 7), ('regression', 1)]:
                with self.subTest(model=name, task=task_type, classes=classes):
                    context = small_brainjepa() if name == 'brainjepa' else nullcontext()
                    with context:
                        dm = SimpleNamespace(train_dataset=SimpleNamespace(target_values=np.arange(3.).reshape(-1, 1)))
                        model = LightningModel(dm, **hparams(name, task_type, classes)).eval()
                        self.assertIsInstance(model.output_head, nn.Identity)
                        key = 'roi_sequence' if name == 'brainjepa' else 'fmri_sequence'
                        shape = (1, 1, 450, 160) if name == 'brainjepa' else (1, 1, 16, 16, 16, 2)
                        batch = {key: torch.randn(shape), 'target': torch.tensor([1.]), 'subject_name': ['s']}
                        with torch.no_grad():
                            direct = model(batch[key])
                            _, logits, target, _, _ = model._compute_logits(batch)
                            loss = model._calculate_loss(batch, mode='train')
                        self.assertEqual(direct.shape, (1, 7 if classes == 7 else 1))
                        self.assertEqual(target.shape, (1,))
                        self.assertEqual(logits.shape, (1, 7) if classes == 7 else (1,))
                        self.assertTrue(torch.isfinite(loss))
                        saved = model.validation_step(batch, 0, 0)[1]
                        self.assertEqual(saved[0].shape[0], 1)
                        self.assertEqual(saved[1].shape, (1,))

    def test_registry_rejects_wrong_objective_and_data(self):
        for name in ('brainjepa', 'band'):
            with self.subTest(model=name):
                with self.assertRaisesRegex(ValueError, 'pretraining'):
                    load_model(name, Namespace(**hparams(name, pretraining=True)))
                with self.assertRaisesRegex(ValueError, 'data_type'):
                    load_model(name, Namespace(**hparams(name, data_type='fc')))

    def test_pretrained_encoder_and_legacy_adapter(self):
        with tempfile.TemporaryDirectory() as tmp, small_brainjepa():
            original = BrainJEPA()
            path = Path(tmp) / 'pretrained.pt'
            torch.save({'epoch': 299, 'target_encoder': {
                'module.' + key: value for key, value in original.encoder.state_dict().items()
            }}, path)
            loaded = BrainJEPA(checkpoint_path=path, num_outputs=7)
            for key, value in original.encoder.state_dict().items():
                torch.testing.assert_close(loaded.encoder.state_dict()[key], value, rtol=0, atol=0)
            legacy = BrainJEPAP01(path).eval()
            self.assertEqual(legacy(torch.randn(1, 1, 450, 160)).shape, (1,))
            torch.save({'target_encoder': {}}, path)
            with self.assertRaises(RuntimeError):
                BrainJEPA(checkpoint_path=path)

    def test_roi_normalization_sampling_and_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'roi' / 'brainjepa450'
            directory.mkdir(parents=True)
            path = directory / '100001_brainjepa450.npy'
            array = np.random.default_rng(42).normal(size=(450, 500)).astype(np.float32)
            np.save(path, array)
            dataset = BrainJEPADataset(tmp, {'100001': [0, 20.]}, atlas_name='brainjepa450', train=False)
            actual = dataset[0]['roi_sequence']
            signals = torch.from_numpy(array)
            normalized = (signals - signals.mean()) / signals.std()
            expected = normalized[:, torch.linspace(10, 489, 160).long()].unsqueeze(0)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual, dataset[0]['roi_sequence'], rtol=0, atol=0)
            for invalid in [np.zeros((400, 500)), np.zeros((450, 100)), np.zeros((450, 500))]:
                np.save(path, invalid)
                with self.assertRaises(ValueError):
                    dataset[0]

    def test_shared_runner_commands(self):
        bash = shutil.which('bash')
        if bash is None:
            self.skipTest('bash is required for runner verification')
        root = Path(__file__).resolve().parents[1]
        for name, dataset, task in [('brainjepa', 'hcp1200', 'task1_age'), ('band', 'hcptask', 'task5')]:
            with self.subTest(model=name):
                result = subprocess.run([bash, 'scripts/run_experiment.sh', '--model', name,
                                         '--dataset', dataset, '--task', task, '--mode', 'train_scratch', '--dry_run'],
                                        cwd=root, text=True, capture_output=True, check=True)
                command = next(line for line in result.stdout.splitlines() if line.startswith('python main.py'))
                self.assertIn('--model ' + name, command)
                values = hparams(name)
                self.assertIn('--data_type roi', command) if name == 'brainjepa' else self.assertIn('--band_num_heads 8', command)
                self.assertIn('--sequence_length 160' if name == 'brainjepa' else '--sequence_length 40', command)
                self.assertIn('--brainjepa_window_length 480', command) if name == 'brainjepa' else self.assertEqual(values['data_type'], 'voxel')

    def test_lightning_fit_test_and_checkpoint_restore(self):
        for name in ('brainjepa', 'band'):
            with self.subTest(model=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                context = small_brainjepa() if name == 'brainjepa' else nullcontext()
                with context:
                    values = hparams(name, 'regression' if name == 'brainjepa' else 'classification',
                                     1 if name == 'brainjepa' else 2)
                    if name == 'brainjepa':
                        roi = root / 'roi' / 'brainjepa450'
                        roi.mkdir(parents=True)
                        metadata = root / 'metadata'
                        metadata.mkdir()
                        subjects = [str(100001 + i) for i in range(9)]
                        rng = np.random.default_rng(23)
                        for subject in subjects:
                            np.save(roi / (subject + '_brainjepa450.npy'), rng.normal(size=(450, 500)).astype(np.float32))
                        (metadata / 'HCP_1200_gender.csv').write_text('Subject,Gender\n' + '\n'.join(s + ',M' for s in subjects))
                        (metadata / 'HCP_1200_precise_age.csv').write_text('subject,age,sex\n' + '\n'.join(s + f',{20+i},M' for i, s in enumerate(subjects)))
                        splits = root / 'splits' / 'HCP1200'
                        splits.mkdir(parents=True)
                        (splits / 'split_fixed_1.txt').write_text('train\n' + '\n'.join(subjects[:3]) + '\nval\n' + '\n'.join(subjects[3:6]) + '\ntest\n' + '\n'.join(subjects[6:]) + '\n')
                        values.update(image_path=str(root), split_root=str(root / 'splits'), atlas_name='brainjepa450')
                        dm = fMRIDataModule(**values)
                        self.assertEqual([len(batch['subject_name']) for batch in dm.val_dataloader()[0]], [2, 1])
                        initializer = root / 'initializer.pt'
                        torch.save({'target_encoder': BrainJEPA().encoder.state_dict()}, initializer)
                        values['brainjepa_checkpoint_path'] = str(initializer)
                    else:
                        dm = VolumeData()
                    model = LightningModel(dm, **values)
                    trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=1,
                                         logger=False, enable_checkpointing=False,
                                         enable_progress_bar=False, enable_model_summary=False,
                                         num_sanity_val_steps=0)
                    trainer.fit(model, datamodule=dm)
                    self.assertEqual(trainer.global_step, 1)
                    result = trainer.test(model, datamodule=dm, verbose=False)
                    self.assertTrue(result)
                    checkpoint = root / 'trained.ckpt'
                    trainer.save_checkpoint(checkpoint)
                    if name == 'brainjepa':
                        initializer.unlink()
                    restored = LightningModel.load_from_checkpoint(
                        checkpoint, data_module=dm, test_ckpt_path=str(checkpoint))
                    for key, value in model.state_dict().items():
                        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
