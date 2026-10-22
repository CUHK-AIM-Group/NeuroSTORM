"""Complete saved training weights keep the established retrieval features."""
import tempfile
from pathlib import Path
import unittest

import torch

from models.reid_encoders import ROIKernelFeatures
from tasks.reid.checkpoints import load_checkpoint


class BrainGNNCheckpointTests(unittest.TestCase):
    def test_legacy_and_complete_training_weights_have_identical_features(self):
        torch.manual_seed(1234)
        feature_config = dict(num_rois=190, num_communities=8, hidden_dim=32)
        legacy = ROIKernelFeatures(**feature_config).eval()
        classifier = torch.nn.Linear(6080, 867, bias=False)
        hp = dict(model_name="BrainGNN", feature_config=feature_config)
        state = legacy.state_dict()
        complete = dict(state)
        complete["auxiliary_classifier.weight"] = classifier.weight.detach().clone()
        complete_hp = dict(hp, auxiliary_classifier_config=dict(
            in_features=6080, out_features=867, bias=False))
        with tempfile.TemporaryDirectory() as folder:
            old_path, new_path = Path(folder) / "old.ckpt", Path(folder) / "new.ckpt"
            torch.save(dict(hyper_parameters=hp, state_dict=state), old_path)
            torch.save(dict(hyper_parameters=complete_hp, state_dict=complete), new_path)
            old_model, _ = load_checkpoint(old_path, "cpu")
            new_model, _ = load_checkpoint(new_path, "cpu")
            matrices = torch.randn(4, 190, 190)
            with torch.no_grad():
                self.assertTrue(torch.equal(old_model.forward_features(matrices),
                                            new_model.forward_features(matrices)))
            self.assertTrue(torch.equal(new_model.auxiliary_classifier.weight,
                                        classifier.weight))
            self.assertEqual(set(new_model.state_dict()), set(complete))

    def test_declared_classifier_weights_are_required_by_strict_loading(self):
        feature_config = dict(num_rois=190, num_communities=8, hidden_dim=32)
        state = ROIKernelFeatures(**feature_config).state_dict()
        hp = dict(model_name="BrainGNN", feature_config=feature_config,
                  auxiliary_classifier_config=dict(in_features=6080, out_features=867, bias=False))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing_head.ckpt"
            torch.save(dict(hyper_parameters=hp, state_dict=state), path)
            with self.assertRaisesRegex(RuntimeError, "auxiliary_classifier.weight"):
                load_checkpoint(path, "cpu")


if __name__ == "__main__":
    unittest.main()
