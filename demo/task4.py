#!/usr/bin/env python
"""Run one Task4 HF checkpoint, show query progress and update the result table."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction
import json
import math
from numbers import Real
from pathlib import Path, PurePosixPath
import shutil
import statistics
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasks.reid import CHECKPOINT_FILENAMES, MODEL_NAMES



SCAN_NUMBERS = {"rfMRI_REST1_LR": 1, "rfMRI_REST1_RL": 2,
                "rfMRI_REST2_LR": 3, "rfMRI_REST2_RL": 4}


class QueryProgress:
    def __init__(self, scans, protocol, split):
        queries = {scan for trial in protocol[split] for scan in trial["query_scans"]}
        if not queries or not queries.issubset(scans):
            raise ValueError("The scan manifest must contain the protocol queries")
        self.total = len(queries)
        self.completed_scans = 0
        self.prefix = [0]
        for scan in scans:
            self.prefix.append(self.prefix[-1] + (scan in queries))

    def update(self, done, total, **extra):
        """Return newly completed queries; gallery-only scans add no progress."""
        if total != len(self.prefix) - 1 or not self.completed_scans <= done <= total:
            raise ValueError("Progress must count completed scans in manifest order")
        change = self.prefix[done] - self.prefix[self.completed_scans]
        self.completed_scans = done
        return change


def scan_identity(scan):
    subject, sequence = scan.split("::")
    return int(subject), SCAN_NUMBERS[sequence]


def context_for(config, protocol):
    """Prevent separate invocations from mixing protocols or input roots."""
    return {
        "protocol": json.loads(json.dumps(protocol)),
        "split": "test",
        "inputs": {key: str(config[key]) for key in
                   ("image_root", "fc_root", "roi_root", "brainlm_stats", "scan_manifest")
                   if config.get(key)},
        "image_read_workers": config.get("image_read_workers", 4),
    }


def rescore_ranking(path, protocol):
    """Use protocol boundaries and subject IDs; never trust stored hit flags."""
    if len(protocol["test"]) != 5:
        raise ValueError("The Source Data table requires exactly five test trials")
    trials = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for trial in protocol["test"]:
            queries = trial["query_scans"]
            gallery = [scan_identity(s) for s in trial["gallery_scans"]]
            gallery_set = set(gallery)
            if (not queries or len(gallery_set) != len(gallery)
                    or set(queries) & set(trial["gallery_scans"])):
                raise ValueError("Invalid protocol query/gallery lists")
            top1, aps = [], []
            for query in queries:
                try:
                    identity = tuple(int(x) for x in handle.readline().split())
                    ranked = json.loads(handle.readline())
                except (ValueError, TypeError) as error:
                    raise ValueError(f"Missing or malformed ranking for {query}") from error
                if identity != scan_identity(query):
                    raise ValueError(f"Ranking query order differs from the protocol at {query}")
                if not isinstance(ranked, list) or len(ranked) != len(gallery):
                    raise ValueError(f"Wrong gallery size for {query}")
                seen, hits, previous_score = [], [], math.inf
                for rank, row in enumerate(ranked, 1):
                    if not isinstance(row, list) or len(row) != 4:
                        raise ValueError(f"Malformed gallery row for {query}")
                    subject, number, score, flag = row
                    if (not isinstance(subject, int) or isinstance(subject, bool)
                            or not isinstance(number, int) or isinstance(number, bool)
                            or not isinstance(score, Real) or isinstance(score, bool)
                            or not math.isfinite(score) or score > previous_score
                            or flag not in (0, 1)):
                        raise ValueError(f"Invalid or unsorted gallery row for {query}")
                    previous_score = score
                    seen.append((subject, number))
                    relevant = subject == identity[0]
                    if flag != int(relevant):
                        raise ValueError(f"Incorrect stored relevance flag for {query}")
                    if relevant:
                        hits.append(rank)
                if len(set(seen)) != len(seen) or set(seen) != gallery_set:
                    raise ValueError(f"Ranking gallery differs from the shared protocol for {query}")
                valid_positive_counts = (1, 2) if protocol.get("allow_single_positive_gallery") is True else (2,)
                if len(hits) not in valid_positive_counts:
                    raise ValueError(f"Unexpected relevant gallery count for {query}")
                top1.append(int(hits[0] == 1))
                aps.append(sum((Fraction(i, rank) for i, rank in enumerate(hits, 1)),
                               Fraction()) / len(hits))
            trials.append({
                "trial": trial["trial"], "seed": trial["seed"], "queries": len(queries),
                "gallery_scans": len(gallery), "gallery_subjects": len({s for s, _ in gallery}),
                "top1": sum(top1) / len(top1),
                "mAP": float(sum(aps, Fraction()) / len(aps)),
            })
        if any(line.strip() for line in handle):
            raise ValueError("Ranking contains extra queries after the fifth trial")
    return trials


def make_report(model, ranking_path, protocol, context, checkpoint):
    if model not in MODEL_NAMES:
        raise ValueError("Unknown model: " + model)
    trials = rescore_ranking(ranking_path, protocol)
    return {
        "schema_version": 1, "model": model, "context": context,
        "checkpoint": checkpoint, "trials": trials,
        "overall": {key: statistics.mean(t[key] for t in trials) for key in ("top1", "mAP")},
        "std": {key: statistics.pstdev(t[key] for t in trials) for key in ("top1", "mAP")},
        "query_events": sum(t["queries"] for t in trials),
    }


def load_reports(output_dir, context, replacing=None):
    reports = {}
    for model in MODEL_NAMES:
        path = Path(output_dir) / "metrics" / (model + ".json")
        if not path.exists() or model == replacing:
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        if (report.get("schema_version") != 1 or report.get("model") != model
                or report.get("context") != context):
            raise ValueError(f"{path}: incompatible inputs/protocol; use a different --output-dir")
        if not (Path(output_dir) / "rankings" / (model + ".txt")).is_file():
            raise ValueError(f"{model}: saved metrics are missing their ranking list")
        trials = report.get("trials", [])
        if len(trials) != 5 or any(
            isinstance(t.get(key), bool) or not isinstance(t.get(key), Real)
            or not math.isfinite(t[key]) or not 0 <= t[key] <= 1
            for t in trials for key in ("top1", "mAP")
        ):
            raise ValueError(f"{path}: invalid per-trial scores")
        reports[model] = report
    return reports


def write_workbook(path, reports):
    """Write trial scores; missing models remain blank."""
    from openpyxl import Workbook

    book = Workbook()
    book.remove(book.active)
    for sheet_name, metric in (("Top1", "top1"), ("map", "mAP")):
        sheet = book.create_sheet(sheet_name)
        sheet.append(["HCPYA", *MODEL_NAMES])
        for row in range(5):
            sheet.append([f"diameter {row + 1}"] + [None] * len(MODEL_NAMES))
            for column, name in enumerate(MODEL_NAMES, 2):
                if name not in reports:
                    continue
                trial = reports[name]["trials"][row]
                sheet.cell(row + 2, column, trial[metric] * 100)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    book.save(path)
    book.close()


def parse_hf_source(value, model, revision=None):
    """Accept a repository ID/link or a HF blob/resolve checkpoint link."""
    filename, url_revision = None, None
    if "://" in value:
        url = urlsplit(value)
        if (url.scheme not in ("http", "https")
                or url.hostname not in ("huggingface.co", "www.huggingface.co", "hf.co")
                or url.username is not None or url.password is not None):
            raise ValueError("Use a Hugging Face model repository or checkpoint URL")
        parts = url.path.strip("/").split("/")
        if len(parts) < 2 or parts[0] in ("datasets", "spaces"):
            raise ValueError("Expected a Hugging Face model repository URL")
        repo_id = "/".join(unquote(part) for part in parts[:2]).removesuffix(".git")
        if len(parts) > 2:
            if len(parts) < 4 or parts[2] not in ("blob", "resolve", "tree"):
                raise ValueError("Expected a HF repository, blob, resolve or tree URL")
            url_revision = unquote(parts[3])
            remaining = "/".join(unquote(part) for part in parts[4:])
            if parts[2] == "tree":
                filename = (remaining + "/" + CHECKPOINT_FILENAMES[model]) if remaining else None
            elif not remaining:
                raise ValueError("The checkpoint URL is missing its file path")
            else:
                filename = remaining
    else:
        repo_id = value.strip("/")
        if len(repo_id.split("/")) != 2:
            raise ValueError("Use owner/repository or a complete HF checkpoint URL")
    if revision and url_revision and revision != url_revision:
        raise ValueError("--revision conflicts with the revision in the checkpoint URL")
    filename = filename or "task4/" + CHECKPOINT_FILENAMES[model]
    relative = PurePosixPath(filename)
    if (relative.is_absolute() or any(part in ("", ".", "..") for part in filename.split("/"))
            or "\\" in filename or ":" in filename or any(ord(c) < 32 for c in filename)):
        raise ValueError("The HF checkpoint file path must be relative to its repository")
    for name, canonical in CHECKPOINT_FILENAMES.items():
        if relative.name == canonical and model != name:
            raise ValueError(f"The checkpoint URL names {name}, but --model is {model}")
    return {"repo_id": repo_id, "revision": revision or url_revision or "main", "filename": filename}


def download_checkpoint(model, repo_id, revision, directory, filename=None):
    from huggingface_hub import hf_hub_download

    # Let the standard HF downloader handle the requested revision and cache.
    filename = filename or "task4/" + CHECKPOINT_FILENAMES[model]
    path = Path(hf_hub_download(repo_id=repo_id, filename=filename,
                               revision=revision, local_dir=directory)).resolve()
    downloaded = path
    # The native evaluator expects its canonical checkpoint filename.
    # Support custom HF filenames without changing the native loading code.
    if path.name != CHECKPOINT_FILENAMES[model]:
        path = Path(directory).expanduser().resolve() / "selected" / model / CHECKPOINT_FILENAMES[model]
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(downloaded, path)
    return path, {"repo_id": repo_id, "filename": filename,
                  "requested_revision": revision}


def override_inputs(config, data_root=None, image_root=None, fc_root=None, roi_root=None):
    """CLI paths are relative to the shell; explicit roots override data-root."""
    config = dict(config)
    if data_root is not None:
        base = Path(data_root).expanduser().resolve()
        config.update(image_root=(base / "img").resolve(), fc_root=(base / "fc").resolve(),
                      roi_root=(base / "roi").resolve())
    for key, value in (("image_root", image_root), ("fc_root", fc_root), ("roi_root", roi_root)):
        if value is not None:
            config[key] = Path(value).expanduser().resolve()
    return config


def write_effective_config(path, config):
    import yaml

    def serializable(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: serializable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [serializable(item) for item in value]
        return value
    path.write_text(yaml.safe_dump(serializable(config), sort_keys=False), encoding="utf-8")


def check_inputs(config, protocol, model):
    from tasks.reid.data import image_path, scan_ids

    scans = scan_ids(config, "test", protocol)
    expected = {s for t in protocol["test"] for key in ("query_scans", "gallery_scans") for s in t[key]}
    if not scans or not expected.issubset(scans):
        raise ValueError("The scan manifest is missing protocol inputs")
    if model in ("NeuroSTORM", "SwiFT", "TFF"):
        paths = [image_path(config, scan) for scan in scans]
    else:
        root = config["roi_root" if model == "BrainLM" else "fc_root"]
        paths = [root / scan.split("::")[0] / (scan.split("::")[1] + ".npy") for scan in scans]
        if model == "BrainLM":
            paths.append(config["brainlm_stats"])
    # Independent metadata requests overlap on remote/NAS input directories.
    with ThreadPoolExecutor(max_workers=8) as executor:
        missing = next((p for p, exists in zip(paths, executor.map(Path.is_file, paths))
                        if not exists), None)
    if missing is not None:
        raise FileNotFoundError(f"Missing input: {missing}. Set the data paths in demo/task4_config.yaml first.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_NAMES,
                        help="One model per command; run the eight commands sequentially")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--repo", help="HF repository ID or URL; defaults to zxcvb20001/NeuroSTORM")
    source_group.add_argument("--checkpoint", "--checkpoint-url",
                              help="HF checkpoint blob/resolve URL or repository link")
    parser.add_argument("--revision", help="HF branch/tag/commit; defaults to the URL revision or main")
    parser.add_argument("--download-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--data-config", type=Path, default=ROOT / "demo/task4_config.yaml")
    parser.add_argument("--data-root", type=Path, help="Dataset root containing img/, fc/ and roi/")
    parser.add_argument("--image-root", type=Path, help="Override the preprocessed image root")
    parser.add_argument("--fc-root", type=Path, help="Override the FC matrix root")
    parser.add_argument("--roi-root", type=Path, help="Override the ROI time-series root")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--replace", action="store_true", help="Rerun only this model; keep the other models")
    parser.add_argument("--check-only", action="store_true", help="Download and strictly load weights; no data inference or reports")
    args = parser.parse_args(argv)
    from tasks.reid.data import load_config, read_json

    value = args.checkpoint if args.checkpoint is not None else (
        args.repo if args.repo is not None else "zxcvb20001/NeuroSTORM")
    try:
        hf_source = parse_hf_source(value, args.model, args.revision)
    except ValueError as error:
        parser.error(str(error))
    config_path = args.data_config.expanduser().resolve()
    config = override_inputs(load_config(config_path), args.data_root,
                             args.image_root, args.fc_root, args.roi_root)
    protocol = read_json(config["protocol"])
    if len(protocol["test"]) != 5:
        parser.error("The demo requires five test trials")
    context = context_for(config, protocol)
    output = args.output_dir.expanduser().resolve()
    if not args.check_only:
        reports = load_reports(output, context, replacing=args.model if args.replace else None)
        ranking = output / "rankings" / (args.model + ".txt")
        metric_path = output / "metrics" / (args.model + ".json")
        if not args.replace and (ranking.exists() or metric_path.exists()):
            parser.error(f"{args.model} already has results; use --replace to rerun only this model")
        check_inputs(config, protocol, args.model)

    checkpoint, source = download_checkpoint(
        args.model, hf_source["repo_id"], hf_source["revision"],
        args.download_dir.expanduser().resolve(), filename=hf_source["filename"])
    print(f"{args.model}: HF {source['repo_id']}/{source['filename']}", flush=True)
    command = [sys.executable, str(ROOT / "scripts/task4/evaluate_reid.py"),
               "--checkpoint-dir", str(checkpoint.parent), "--models", args.model,
               "--split", "test", "--device", args.device]
    if args.check_only:
        subprocess.run([*command, "--data-config", str(config_path),
                        "--output-dir", str(output), "--check-only"], check=True)
        return

    output.mkdir(parents=True, exist_ok=True)
    # Run inference in a separate empty directory.
    with tempfile.TemporaryDirectory(prefix=f".{args.model}-", dir=output) as temporary:
        temporary = Path(temporary).resolve()
        if temporary.parent != output:
            raise ValueError("Temporary run directory escaped the selected output directory")
        effective_config = temporary / "config.yaml"
        write_effective_config(effective_config, config)
        subprocess.run([*command, "--data-config", str(effective_config),
                        "--output-dir", str(temporary / "inference")], check=True)
        fresh_ranking = temporary / "inference" / (args.model + ".txt")
        report = make_report(args.model, fresh_ranking, protocol, context, source)
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        reports[args.model] = report
        write_workbook(temporary / "performance.xlsx", reports)
        (temporary / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        ranking.parent.mkdir(parents=True, exist_ok=True)
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        fresh_ranking.replace(ranking)
        (temporary / "metrics.json").replace(metric_path)
        (temporary / "performance.xlsx").replace(output / "performance.xlsx")
    print(f"Ranking: {ranking}")
    print(f"Table: {output / 'performance.xlsx'} ({len(reports)}/8 models, 5 trial; missing models stay blank)")
    print("Run the plotting command after all eight models are complete.")


if __name__ == "__main__":
    main()
