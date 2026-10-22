"""Check that Fig.5 represents measured query scores and trial statistics."""
import copy
import json
import sys

from matplotlib.figure import Figure
import numpy as np
import pytest

from scripts.task4 import plot_fig5


@pytest.fixture
def ranking_files(tmp_path):
    queries = ["100001::rfMRI_REST1_LR", "100002::rfMRI_REST1_LR"]
    gallery = ["100001::rfMRI_REST2_LR", "100001::rfMRI_REST2_RL",
               "100002::rfMRI_REST2_LR", "100002::rfMRI_REST2_RL"]
    protocol = {"fixed400_query_scans": queries, "test": [
        {"query_scans": queries, "gallery_scans": gallery},
        {"query_scans": queries[:1], "gallery_scans": gallery},
    ]}
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    records = []
    for query, order in [(queries[0], [0, 2, 3, 1]),
                         (queries[1], [0, 2, 3, 1]),
                         (queries[0], [0, 1, 2, 3])]:
        subject, number = plot_fig5.scan_identity(query)
        ranked = []
        for rank, index in enumerate(order):
            candidate, scan_number = plot_fig5.scan_identity(gallery[index])
            ranked.append([candidate, scan_number, .9 - .1 * rank, int(candidate == subject)])
        records.extend([f"{subject} {number}", json.dumps(ranked)])
    for model in plot_fig5.MODELS:
        (tmp_path / f"{model}.txt").write_text("\n".join(records) + "\n", encoding="utf-8")
    return tmp_path, protocol_path


def test_query_scores_and_trial_means_have_distinct_weights(ranking_files):
    rows = plot_fig5.load_ranking_lists(*ranking_files)
    for row in rows:
        points = row["points"]
        assert [point["top1_percent"] for point in points] == [100., 0.]
        assert [point["mAP_percent"] for point in points] == pytest.approx([87.5, 100 * 7 / 12])
        assert row["metrics"] == pytest.approx({"top1": .75, "mAP": 5 / 6})
        assert row["metrics"]["top1"] != np.mean([point["top1_percent"] for point in points]) / 100


def test_inconsistent_relevance_flags_are_rejected(ranking_files):
    directory, protocol = ranking_files
    path = directory / "NeuroSTORM.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    ranked = json.loads(lines[1])
    ranked[0][3] = 0
    lines[1] = json.dumps(ranked)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="relevance flags"):
        plot_fig5.load_ranking_lists(directory, protocol)


def test_drawing_preserves_all_measured_scores_and_method_order(tmp_path, monkeypatch):
    rows = [dict(model=model, metrics=dict(top1=.8, mAP=.7), points=[
        dict(scan_id=f"{100001 + i}::rfMRI_REST1_LR", top1_percent=top1, mAP_percent=ap)
        for i, (top1, ap) in enumerate(zip([0, 100, 100, 100, 100], [0, 50, 100, 100, 100]))
    ]) for model in plot_fig5.MODELS]
    original = copy.deepcopy(rows)
    saved = []
    monkeypatch.setattr(Figure, "savefig", lambda fig, path, **kwargs: saved.append((fig, path)))
    plot_fig5.register_paper_fonts()
    plot_fig5.draw(rows, tmp_path)
    assert rows == original
    assert {path.name for _, path in saved} == {"Fig5_full.png", "Fig5_full.pdf", "Fig5_full.svg"}
    fig = saved[0][0]
    expected_order = ["BrainGNN", "BNT", "LG-GNN", "Com-brainTF", "BrainLM", "TFF", "SwiFT", "NeuroSTORM"]
    for ax, metric in zip(fig.axes, ["top1", "mAP"]):
        assert [label.get_text() for label in ax.get_yticklabels()] == expected_order
        for row in rows:
            model_id = row["model"].lower().replace("-", "")
            actual = []
            for collection in ax.collections:
                if collection.get_gid() in (f"scans-inliers-{metric}-{model_id}",
                                            f"scans-outliers-{metric}-{model_id}"):
                    actual.extend(collection.get_offsets()[:, 0])
            np.testing.assert_array_equal(sorted(actual), sorted(point[metric + "_percent"] for point in row["points"]))


@pytest.mark.parametrize("removed_args", [
    ["--mode", "query"], ["--mode", "simulated"], ["--mode", "both"],
    ["--simulation-seed", "42"], ["--simulation-points", "400"],
])
def test_removed_mode_options_are_rejected(removed_args, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["plot_fig5.py", "--ranking-dir", str(tmp_path), *removed_args])
    with pytest.raises(SystemExit) as exc:
        plot_fig5.main()
    assert exc.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
