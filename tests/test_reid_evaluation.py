"""Check retrieval semantics and the optional graph feature path."""
import json
import numpy as np
import pytest
import torch

from tasks.reid.metrics import evaluate, normalize
from tasks.reid.data import load_config, read_fc, scan_ids


def protocol(trials, allowed):
    return {"test": [{"trial": i, "seed": 1234 + i, "query_scans": q, "gallery_scans": g}
                     for i, (q, g) in enumerate(trials)], "fixed400_query_scans": allowed}


def test_trial_mean_and_ap_from_known_hit_ranks():
    scans = ["Aq", "Bq", "A1", "A2", "B1", "B2"]
    subjects = ["A", "B", "A", "A", "B", "B"]
    vectors = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [.6, .8], [.8, .6]])
    p = protocol([(["Aq"], scans[2:]), (scans[:2], scans[2:])], scans[:2])
    metrics, trials, events = evaluate(vectors, subjects, scans, p)
    # A has relevant hits at ranks 1/4; B at ranks 2/3.
    assert events[0]["AP"] == pytest.approx((1 + 2 / 4) / 2)
    assert events[-1]["AP"] == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert metrics["top1"] == pytest.approx(.75)
    assert metrics["mAP"] == pytest.approx(17 / 24)
    assert metrics["query_events"] == 3
    assert [t["queries"] for t in trials] == [1, 2]


def test_ties_follow_original_gallery_order():
    scans = ["Aq", "A1", "B1", "A2", "B2"]
    p = protocol([(["Aq"], scans[1:])], ["Aq"])
    metrics, _, events = evaluate(np.zeros((5, 2)), ["A", "A", "B", "A", "B"], scans, p)
    assert events[0]["gallery_scan_ids"] == scans[1:]
    assert metrics["top1"] == 1
    assert metrics["mAP"] == pytest.approx(5 / 6)


def test_optional_mixed_positive_counts_use_actual_relevant_gallery_count():
    scans = ["Aq", "Bq", "A1", "B1", "B2"]
    subjects = ["A", "B", "A", "B", "B"]
    vectors = np.array([[1., 0.], [0., 1.], [.8, .6], [1., 0.], [.6, .8]])
    p = protocol([(scans[:2], scans[2:])], scans[:2])
    with pytest.raises(ValueError, match="exactly two"):
        evaluate(vectors, subjects, scans, p)
    p["allow_single_positive_gallery"] = True
    metrics, _, events = evaluate(vectors, subjects, scans, p)
    # A's sole positive is rank2; B's positives are ranks1 and3.
    assert events[0]["AP"] == pytest.approx(.5)
    assert events[1]["AP"] == pytest.approx((1. + 2. / 3.) / 2.)
    assert metrics["mAP"] == pytest.approx(2. / 3.)


def test_optional_single_positive_protocol_still_rejects_missing_positive():
    p = protocol([(["Aq"], ["B1", "B2"])], ["Aq"])
    p["allow_single_positive_gallery"] = True
    with pytest.raises(ValueError, match="one or two"):
        evaluate(np.eye(3), ["A", "B", "B"], ["Aq", "B1", "B2"], p)


def test_missing_scan_overlap_and_nonfinite_inputs_are_rejected():
    scans = ["Aq", "A1", "A2"]
    p = protocol([(["Aq"], ["A1", "missing"])], ["Aq"])
    with pytest.raises(KeyError):
        evaluate(np.ones((3, 2)), ["A"] * 3, scans, p)
    p = protocol([(["Aq"], ["Aq", "A1"])], ["Aq"])
    with pytest.raises(ValueError, match="overlapping"):
        evaluate(np.ones((3, 2)), ["A"] * 3, scans, p)
    with pytest.raises(ValueError, match="finite"):
        normalize([[np.nan, 0]])


def test_fc_loading_does_not_change_input_file(tmp_path):
    folder = tmp_path / "123456"
    folder.mkdir()
    original = np.eye(190, dtype=np.float32)
    path = folder / "rfMRI_REST1_LR.npy"
    np.save(path, original)
    actual = read_fc({"fc_root": tmp_path}, ["123456::rfMRI_REST1_LR"])
    assert np.all(actual[0].diagonal() == 0)
    np.testing.assert_array_equal(np.load(path), original)


def test_config_paths_resolve_from_config_location(tmp_path):
    import yaml
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"image_root": "img", "protocol": "fixed400.json"}))
    cfg = load_config(path)
    assert cfg["image_root"] == tmp_path / "img"
    assert cfg["protocol"] == tmp_path / "fixed400.json"


def test_per_node_graph_readout_is_independent_of_batch_members():
    from models.reid_encoders import GraphFeatures
    torch.manual_seed(42)
    model = GraphFeatures(num_rois=8, node_feature_dim=8, hidden_dims=[16, 8],
                          k_neighbors=3, edge_normalization="per_node", dropout=0.).eval()
    data = torch.randn(2, 8, 8)
    with torch.no_grad():
        combined = model.forward_features(data)
        separate = torch.cat([model.forward_features(row[None]) for row in data])
    torch.testing.assert_close(combined, separate, rtol=1e-5, atol=1e-6)
