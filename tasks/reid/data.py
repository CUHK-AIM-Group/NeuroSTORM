"""ReID inputs; paths are configured independently of checkpoint metadata."""
import json
from pathlib import Path
import h5py
import numpy as np
import yaml


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    for key in ("image_root", "fc_root", "roi_root", "brainlm_stats", "protocol", "scan_manifest"):
        if config.get(key):
            value = Path(config[key]).expanduser()
            config[key] = value if value.is_absolute() else (path.parent / value).resolve()
    return config


def scan_ids(config, split, protocol):
    if config.get("scan_manifest"):
        scans = read_json(config["scan_manifest"])[split]
    else:
        scans = sorted({s for t in protocol[split] for key in ("query_scans", "gallery_scans") for s in t[key]})
    if len(scans) != len(set(scans)):
        raise ValueError("Duplicate scan IDs in the data manifest")
    for scan in scans:
        subject, sequence = scan.split("::")
        if not subject.isdigit() or sequence not in ("rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL"):
            raise ValueError("Invalid scan ID: " + scan)
    return scans


def image_path(config, scan):
    subject, sequence = scan.split("::")
    return config["image_root"] / subject / sequence / "data.h5"


def read_image(path):
    with h5py.File(path, "r") as handle:
        raw = handle["frames"][:]
        scale = float(handle.attrs.get("scale", 1.0))
    if raw.ndim != 4 or not np.isfinite(scale) or not np.isfinite(raw).all():
        raise ValueError("Invalid image data: " + str(path))
    return raw, scale


def read_fc(config, scans):
    arrays = []
    for scan in scans:
        subject, sequence = scan.split("::")
        path = config["fc_root"] / subject / (sequence + ".npy")
        values = np.load(path).astype(np.float32)
        if values.shape != (190, 190) or not np.isfinite(values).all():
            raise ValueError("Invalid FC data: " + str(path))
        np.fill_diagonal(values, 0)
        arrays.append(values)
    return np.stack(arrays)
