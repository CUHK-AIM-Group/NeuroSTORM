"""Deterministic retrieval on the saved query/gallery lists."""
import numpy as np


def normalize(vectors):
    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Embeddings must be a finite two-dimensional array")
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)


def evaluate(vectors, subjects, scans, protocol, split="test"):
    x = normalize(vectors)
    subjects = np.asarray(subjects, dtype=str)
    scans = np.asarray(scans, dtype=str)
    lookup = {scan: i for i, scan in enumerate(scans)}
    if len(lookup) != len(x) or len(subjects) != len(x):
        raise ValueError("Scan IDs must be unique and aligned with embeddings")
    trials, events = [], []
    allowed = set(protocol["fixed400_query_scans"])
    for trial in protocol[split]:
        queries = [s for s in trial["query_scans"] if split != "test" or s in allowed]
        gallery = trial["gallery_scans"]
        if not queries or len(set(gallery)) != len(gallery) or set(queries) & set(gallery):
            raise ValueError("Invalid or overlapping query/gallery lists")
        qi = np.asarray([lookup[s] for s in queries])
        gi = np.asarray([lookup[s] for s in gallery])
        similarity = x[qi] @ x[gi].T
        order = np.argsort(-similarity, axis=1, kind="stable")
        relevant = subjects[gi][order] == subjects[qi, None]
        positives = relevant.sum(1)
        if protocol.get("allow_single_positive_gallery") is True:
            if not np.all((positives == 1) | (positives == 2)):
                raise ValueError("Each query must have one or two relevant gallery scans")
        elif not np.all(positives == 2):
            raise ValueError("Each query must have exactly two relevant gallery scans")
        top1 = relevant[:, 0].astype(np.float64)
        ap = (np.cumsum(relevant, axis=1) / np.arange(1, len(gi) + 1) * relevant).sum(1) / positives
        trials.append(dict(trial=trial["trial"], seed=trial["seed"], queries=len(qi),
                           top1=float(top1.mean()), mAP=float(ap.mean())))
        for j, scan in enumerate(queries):
            events.append(dict(trial=trial["trial"], seed=trial["seed"], query_scan_id=scan,
                               query_subject_id=subjects[qi[j]], top1=float(top1[j]), AP=float(ap[j]),
                               gallery_scan_ids=scans[gi[order[j]]].tolist(),
                               similarities=similarity[j, order[j]].tolist()))
    if not trials:
        raise ValueError("The protocol contains no trials")
    metrics = {key: float(np.mean([t[key] for t in trials])) for key in ("top1", "mAP")}
    metrics.update({key + "_std": float(np.std([t[key] for t in trials], ddof=0)) for key in ("top1", "mAP")})
    metrics.update(num_scans=len(scans), query_events=len(events),
                   unique_query_scans=len(set(e["query_scan_id"] for e in events)))
    return metrics, trials, events
