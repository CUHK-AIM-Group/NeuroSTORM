"""The frozen feature and scan-aggregation paths for submitted weights."""
from concurrent.futures import ProcessPoolExecutor
from collections import deque
import multiprocessing
import numpy as np
import torch
import torch.nn.functional as F
from .data import image_path, read_image, read_fc
from .metrics import normalize


@torch.no_grad()
def extract_native(network, hp, config, scans, progress):
    from datasets.fmri_datasets import pad_to_96, resize_volume
    device = next(network.parameters()).device
    length, within = hp["sequence_length"], hp["stride_within_seq"]
    duration = length * within
    stride = max(round(hp["stride_between_seq"] * duration), 1)
    batch_size = 3 if hp["model_name"] == "NeuroSTORM" else 6
    features, total_windows = [], 0
    # Bound the read queue: compressed volumes are decoded in separate workers.
    workers = int(config.get('image_read_workers', 4))
    if workers < 1:
        raise ValueError('image_read_workers must be positive')
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as executor:
        pending = deque(executor.submit(read_image, image_path(config, scan)) for scan in scans[:workers])
        for i, scan_id in enumerate(scans):
            raw, scale = pending.popleft().result()
            if i + workers < len(scans):
                pending.append(executor.submit(read_image, image_path(config, scans[i + workers])))
            scan = torch.from_numpy(raw).to(device).float().mul_(scale)
            starts = list(range(0, len(scan) - duration + 1, stride))
            if not starts:
                raise ValueError("Scan is shorter than the model window: " + scan_id)
            windows = []
            for offset in range(0, len(starts), batch_size):
                clips = []
                for start in starts[offset:offset + batch_size]:
                    clip = scan[start:start + duration:within].permute(1, 2, 3, 0).unsqueeze(0)
                    clips.append(resize_volume(pad_to_96(clip), hp["img_size"]))
                windows.append(network(torch.stack(clips)).cpu())
            feature = torch.cat(windows).mean(0)
            features.append(feature)
            total_windows += len(starts)
            del scan, raw
            progress(i + 1, len(scans), windows=total_windows)
    # Preserve the original FP32 scan normalization before FP64 retrieval.
    return F.normalize(torch.stack(features), dim=1).numpy(), total_windows


@torch.no_grad()
def extract(network, hp, config, scans, progress):
    name = hp["model_name"]
    if name in ("NeuroSTORM", "SwiFT"):
        return extract_native(network, hp, config, scans, progress)
    device = next(network.parameters()).device
    if name in ("BrainGNN", "LG-GNN", "BNT", "Com-brainTF"):
        matrices = torch.from_numpy(read_fc(config, scans)).to(device)
        if name in ('BNT', 'Com-brainTF'):
            matrices.diagonal(dim1=-2, dim2=-1).fill_(1.)
        values = []
        batch_size = 16 if name == "BrainGNN" else 32
        fastpath = torch.backends.mha.get_fastpath_enabled()
        try:
            if name in ('BNT', 'Com-brainTF'):
                torch.backends.mha.set_fastpath_enabled(False)
            for start in range(0, len(scans), batch_size):
                values.append(network.forward_features(matrices[start:start + batch_size]).cpu())
                progress(min(start + batch_size, len(scans)), len(scans))
        finally:
            torch.backends.mha.set_fastpath_enabled(fastpath)
        vectors = torch.cat(values)
        if name != "BrainGNN":
            vectors = F.normalize(vectors, dim=1)
        return vectors.numpy(), len(scans)
    if name == "TFF":
        tokens = []
        for i, scan in enumerate(scans):
            tokens.append(network.encode_scan(image_path(config, scan)))
            progress(i + 1, len(scans))
        tokens = torch.stack(tokens)
        features = [network.forward_tokens(tokens[start:start + 16].to(device))[1].cpu()
                    for start in range(0, len(scans), 16)]
        return torch.cat(features).numpy(), len(scans)
    if name == "BrainLM":
        stats = np.load(config["brainlm_stats"], allow_pickle=False)
        recordings = []
        for scan in scans:
            subject, sequence = scan.split("::")
            x = np.load(config["roi_root"] / subject / (sequence + ".npy")).astype(np.float32)
            if x.ndim != 2 or x.shape[0] != 424 or not np.isfinite(x).all():
                raise ValueError("Invalid ROI recording: " + scan)
            x -= x.mean(axis=1, keepdims=True)
            x = (x - stats["median"][:, None]) / stats["iqr"][:, None]
            recordings.append(torch.tensor(x[stats["roi_order"]] / np.float32(5.6430855)))
        requests = [(i, start) for i, x in enumerate(recordings) for start in range(0, x.shape[1] - 199, 200)]
        sums = torch.zeros(len(scans), 768)
        counts = torch.zeros(len(scans))
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        for offset in range(0, len(requests), 16):
            batch = requests[offset:offset + 16]
            x = torch.stack([recordings[i][:, start:start + 200] for i, start in batch])
            image = x[:, None].expand(-1, 3, -1, -1).contiguous().to(device)
            _, z = network.encode_windows(image)
            for (i, _), value in zip(batch, z.cpu()):
                sums[i] += value
                counts[i] += 1
            progress(offset + len(batch), len(requests), unit="windows")
        if not (counts > 0).all():
            raise ValueError("A scan contains no complete BrainLM window")
        # The trained FP64 metric is applied AFTER the FP32 window mean.
        raw = (sums / counts[:, None]).double().numpy()
        return ((raw - network.metric_mean.cpu().numpy()) @ network.metric_matrix.cpu().numpy()), len(requests)
    raise ValueError("Unsupported model: " + name)
