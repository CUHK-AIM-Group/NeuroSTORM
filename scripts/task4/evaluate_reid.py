#!/usr/bin/env python
"""Evaluate submitted ReID checkpoints from preprocessed real scan inputs."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def write_ranking_list(path, events):
    scan_numbers = {"rfMRI_REST1_LR": 1, "rfMRI_REST1_RL": 2,
                    "rfMRI_REST2_LR": 3, "rfMRI_REST2_RL": 4}

    def scan_identity(scan):
        subject, sequence = scan.split("::")
        return int(subject), scan_numbers[sequence]

    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            subject, scan = scan_identity(event["query_scan_id"])
            ranked = []
            for gallery_scan, similarity in zip(event["gallery_scan_ids"], event["similarities"]):
                gallery_subject, gallery_number = scan_identity(gallery_scan)
                ranked.append([gallery_subject, gallery_number, similarity, int(gallery_subject == subject)])
            handle.write(f"{subject} {scan}\n")
            handle.write(json.dumps(ranked) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--data-config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/reid/hcpya.yaml")
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true", help="Check strict weight loading without data inference")
    args = parser.parse_args()
    import torch
    from tasks.reid.checkpoints import CHECKPOINT_FILENAMES, MODEL_NAMES, load_checkpoint

    names = list(MODEL_NAMES) if args.models == ["all"] else args.models
    if len(names) != len(set(names)) or any(name not in MODEL_NAMES for name in names):
        parser.error("Use model names: " + ", ".join(MODEL_NAMES))
    if not args.check_only:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            parser.error("Output directory must be empty; each invocation performs fresh inference")
        args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not args.check_only:
        from tasks.reid.data import load_config, read_json, scan_ids
        from tasks.reid.features import extract
        from tasks.reid.metrics import evaluate, normalize

        config = load_config(args.data_config)
        protocol = read_json(config["protocol"])
        scans = scan_ids(config, args.split, protocol)
        if not scans:
            raise ValueError('No scans are configured for this split')
        expected_scans = {s for t in protocol[args.split] for key in ('query_scans', 'gallery_scans') for s in t[key]}
        if not expected_scans.issubset(scans):
            raise ValueError('The scan manifest is missing scans required by the protocol')
        subjects = [s.split("::")[0] for s in scans]

    for name in names:
        path = args.checkpoint_dir / CHECKPOINT_FILENAMES[name]
        if not path.is_file():
            path = args.checkpoint_dir / (name + ".ckpt")
        network, hp = load_checkpoint(path, args.device)
        if hp["model_name"] != name:
            raise ValueError("Checkpoint model name does not match the requested file")
        if args.check_only:
            print(f"{name}: checkpoint loaded successfully", flush=True)
        else:
            last_print = [0.]

            def progress(done, total, **extra):
                if done == total or time.monotonic() - last_print[0] > 30:
                    print(f"{name}: {done / total:.1%}", flush=True)
                    last_print[0] = time.monotonic()

            vectors, _ = extract(network, hp, config, scans, progress)
            vectors = normalize(vectors)
            metrics, _, events = evaluate(vectors, subjects, scans, protocol, args.split)
            write_ranking_list(args.output_dir / (name + ".txt"), events)
            print(f"{name}: Top-1 {metrics['top1'] * 100:.6f}%, mAP {metrics['mAP'] * 100:.6f}%", flush=True)
        del network
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
