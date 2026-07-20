from types import SimpleNamespace
import unittest

import torch

from demo import (
    STATE_CLASS_NAMES,
    _classification_probabilities,
    _task_config,
)


def _args(**overrides):
    values = {
        "phenotype_name": None,
        "phenotype_type": None,
        "num_classes": None,
        "label_scaling_method": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class DemoTaskConfigTests(unittest.TestCase):
    def test_task_config_covers_all_benchmarks(self):
        cases = [
            ("gender", 1, "sex", "classification", 2),
            ("age", 1, "age", "regression", 1),
            ("phenotype", 2, "MMSE_Score", "regression", 1),
            ("diagnosis", 3, "diagnosis", "classification", 2),
            ("retrieval", 4, "fmri_reid", "classification", 1084),
            ("state", 5, "state_classification", "classification", 7),
        ]

        for task, task_id, task_name, task_type, num_classes in cases:
            with self.subTest(task=task):
                hparams = {
                    "task_name": "MMSE_Score" if task == "phenotype" else task_name,
                    "downstream_task_type": task_type,
                    "num_classes": num_classes,
                    "label_scaling_method": "standardization",
                }

                config = _task_config(task, _args(), hparams)

                self.assertEqual(config["downstream_task_id"], task_id)
                self.assertEqual(config["task_name"], task_name)
                self.assertEqual(config["downstream_task_type"], task_type)
                self.assertEqual(config["num_classes"], num_classes)

    def test_single_logit_binary_probabilities_use_sigmoid(self):
        probabilities = _classification_probabilities(torch.tensor([[0.0]]), 2)

        self.assertTrue(torch.allclose(probabilities, torch.tensor([0.5, 0.5])))

    def test_two_logit_binary_probabilities_use_softmax(self):
        probabilities = _classification_probabilities(torch.tensor([[0.0, 1.0]]), 2)

        expected = torch.softmax(torch.tensor([0.0, 1.0]), dim=0)
        self.assertTrue(torch.allclose(probabilities, expected))

    def test_hcp_state_labels_cover_all_seven_classes(self):
        self.assertEqual(
            STATE_CLASS_NAMES,
            (
                "EMOTION", "GAMBLING", "LANGUAGE", "MOTOR",
                "RELATIONAL", "SOCIAL", "WM",
            ),
        )


if __name__ == "__main__":
    unittest.main()
