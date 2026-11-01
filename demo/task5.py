#!/usr/bin/env python
"""Download one DMT model's five checkpoints, infer Test4 and update Excel."""
import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import statistics
import sys
import tempfile
from urllib.parse import unquote, urlsplit

for _key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key]="2"

import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tasks.reid.data import load_config as base_config, read_json

MODEL_NAMES=("NeuroSTORM","BAnD","SwiFT","BNT","BrainGNN")
SLUGS=dict(zip(MODEL_NAMES,("neurostorm","band","swift","bnt","braingnn")))
SEEDS=tuple(range(1234,1239))
PREDICTION_FIELDS=("method","seed","person","role","label","predicted",
                   "logit_placebo","logit_DMT","P_Placebo","P_DMT","P_true",
                   "correct","P_predicted","selected_epoch","actual_optimizer_step")


def load_config(path,data_root=None):
    path=Path(path).expanduser().resolve()
    config=base_config(path)
    for key in ("data_root","atlas"):
        value=Path(data_root) if key=="data_root" and data_root is not None else Path(config[key])
        value=value.expanduser()
        config[key]=value.resolve() if value.is_absolute() or (key=="data_root" and data_root is not None) else (path.parent/value).resolve()

    return config


def context_for(config,protocol):
    return dict(protocol=protocol,data_root=str(config["data_root"]),atlas=str(config["atlas"]))


def read_predictions(path):
    with Path(path).open(newline="",encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def parse_source(value,model,revision=None):
    prefix=f"task5/{SLUGS[model]}_dmt"
    selected=None
    if "://" in value:
        url=urlsplit(value)
        if url.scheme not in ("http","https") or url.hostname not in ("huggingface.co","www.huggingface.co","hf.co") or url.username or url.password:
            raise ValueError("Use a Hugging Face model repository or DMT bundle URL")
        parts=[unquote(p) for p in url.path.strip("/").split("/")]
        if len(parts)<2 or parts[0] in ("datasets","spaces"):
            raise ValueError("Expected owner/repository")
        repo="/".join(parts[:2]).removesuffix(".git")
        if len(parts)>2:
            if len(parts)<4 or parts[2] not in ("tree","blob","resolve"):
                raise ValueError("Expected a repository, tree, blob or resolve URL")
            selected=parts[3]
            remaining="/".join(parts[4:])
            if remaining:
                folder=remaining if parts[2]=="tree" else str(PurePosixPath(remaining).parent)
                if folder.rstrip("/")!=prefix:
                    raise ValueError(f"{model} requires its five-run bundle at {prefix}")
    else:
        repo=value.strip("/")
    if len(repo.split("/"))!=2 or any(p in ("",".","..") for p in repo.split("/")):
        raise ValueError("Expected owner/repository")
    if revision and selected and revision!=selected:
        raise ValueError("--revision conflicts with the URL revision")

    return dict(repo_id=repo,revision=revision or selected or "main",prefix=prefix)


def download_bundle(model,protocol,source,directory):
    from huggingface_hub import hf_hub_download
    if protocol.get("seeds")!=list(SEEDS):
        raise ValueError("Expected the five training seeds 1234–1238")
    checkpoints=[]
    for run,seed in enumerate(SEEDS,1):
        filename=f"{source['prefix']}/{SLUGS[model]}_dmt_seed{run}.ckpt"
        path=PurePosixPath(filename)
        if path.is_absolute() or ".." in path.parts or chr(92) in filename or ":" in filename:
            raise ValueError("Invalid checkpoint path")
        print(f"{model}: HF checkpoint {run}/5 (training seed {seed})",flush=True)
        local=Path(hf_hub_download(repo_id=source["repo_id"],revision=source["revision"],filename=filename,local_dir=directory)).resolve()
        checkpoints.append((dict(seed=seed,file=filename),local))
    origin=dict(repo_id=source["repo_id"],revision=source["revision"],checkpoints=[row for row,_ in checkpoints])

    return checkpoints,origin


def load_checkpoint(model,selection,path):
    checkpoint=torch.load(path,map_location="cpu",weights_only=False)
    hparams=checkpoint.get("hyper_parameters",{})
    if hparams.get("method_name")!=model or int(hparams.get("seed",-1))!=selection["seed"]:
        raise ValueError("Native checkpoint method/seed differs from its requested run")
    config=hparams.get("trial_config")
    if not isinstance(config,dict) or config.get("method")!=model or config.get("seeds")!=list(SEEDS):
        raise ValueError("Missing or incompatible native training configuration")
    epoch=int(checkpoint.get("epoch",-1))+1
    step=int(checkpoint.get("global_step",-1))
    if epoch<1 or step<0:
        raise ValueError("Missing native checkpoint epoch or optimizer step")
    selection.update(epoch=epoch,actual_optimizer_step=step)
    if model in MODEL_NAMES[:3]:
        tag=checkpoint.get("dmt_self_contained",{})
        if tag.get("format_version")!=1 or tag.get("model")!=model or tag.get("seed")!=selection["seed"] or not checkpoint.get("frozen_encoder_checkpoint",{}).get("state_dict"):
            raise ValueError("Missing embedded encoder or mismatched model/seed")
    return checkpoint


def validate_protocol(protocol):
    if protocol.get("task") != "DMT-HAR-MED" or protocol.get("input") != "paired_pre_post_first40":
        raise ValueError("Expected the paired first40 DMT-HAR-MED protocol")
    if protocol.get("seeds") != list(SEEDS):
        raise ValueError("The same five seeds 1234–1238 must be used")
    people = protocol["subjects"]
    ids = [p["person"] for p in people]
    if len(ids) != 40 or len(set(ids)) != 40:
        raise ValueError("Expected 40 unique subjects")
    for split, size, balance in (("Train", 32, 16), ("Val", 4, 2), ("Test", 4, 2)):
        rows = [p for p in people if p["split"] == split]
        if len(rows) != size or sorted(int(p["label"]) for p in rows) != [0] * balance + [1] * balance:
            raise ValueError(f"Invalid {split} subject counts or class balance")
    scans = [p[k] for p in people for k in ("pre", "post")]
    if len(set(scans)) != 80 or any(Path(s).name != s for s in scans):
        raise ValueError("Scan IDs must be unique, relative names")
    for p in people:
        if p["pre"] != p["person"] + "_ses-01" or p["post"] != p["person"] + "_ses-02":
            raise ValueError("Pre/post scans must belong to the same subject")
    if len(protocol["test_order"]) != 4 or set(p["person"] for p in people if p["split"] == "Test") != set(protocol["test_order"]):
        raise ValueError("Test ordering differs from the subject manifest")
    return protocol


def test_subjects(protocol):
    by_person = {p["person"]: p for p in protocol["subjects"] if p["split"] == "Test"}
    return [by_person[name] for name in protocol["test_order"]]


def load_dataset(config, protocol):
    from datasets.fmri_datasets import HCPTASK as DMT

    subjects = {p[k]: (p.get("sex", "n/a"), int(p["label"]))
                for p in test_subjects(protocol) for k in ("pre", "post")}
    dataset = DMT(
        root=str(config["data_root"]), img_size=[96, 96, 96, 40],
        sequence_length=40, contrastive=False, mae=False,
        stride_between_seq=20, stride_within_seq=1, with_voxel_norm=False,
        shuffle_time_sequence=False, train=False, subject_dict=subjects,
    )
    dataset._blob_cache_max = 1
    if len(dataset) != 8:
        raise ValueError("The protocol requires eight complete pre/post Test scans")
    return dataset


def first40_scans(config, protocol):
    import torch
    from torch.utils.data import DataLoader

    dataset = load_dataset(config, protocol)
    seen = set()
    for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
        name = str(batch["subject_name"][0])
        clip = batch["fmri_sequence"].contiguous().float()
        if name in seen or tuple(clip.shape) != (1, 1, 96, 96, 96, 40) or not torch.isfinite(clip).all():
            raise ValueError(f"Invalid first40 input for {name}")
        seen.add(name)
        yield name, clip


def fc_from_clip(clip, masks):
    """Compute CC200 Pearson correlations and Fisher z in FP64, then return FP32."""
    import numpy as np

    volume = clip.numpy()[0, 0].reshape(-1, 40)
    roi = np.stack([volume[index].mean(axis=0, dtype=np.float64) for index in masks]).astype(np.float32)
    centered = roi.astype(np.float64) - roi.mean(axis=1, dtype=np.float64, keepdims=True)
    lengths = np.sqrt((centered ** 2).sum(axis=1, keepdims=True))
    normalized = np.divide(centered, lengths, out=np.zeros_like(centered), where=lengths > 0)
    fc = np.arctanh(np.clip(normalized @ normalized.T, -1 + 1e-7, 1 - 1e-7))
    np.fill_diagonal(fc, 0)
    if not np.isfinite(fc).all():
        raise ValueError("Non-finite FC input")
    return fc.astype(np.float32)


def atlas_masks(path):
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import zoom

    atlas = nib.load(str(path)).get_fdata().astype(np.int32)
    if atlas.shape != (96, 96, 96):
        atlas = zoom(atlas, [96 / s for s in atlas.shape], order=0).astype(np.int32)
    labels = np.unique(atlas)
    labels = labels[labels > 0]
    if len(labels) != 190:
        raise ValueError("Expected the native CC200 atlas with 190 nonempty ROI labels")
    return [np.flatnonzero(atlas.reshape(-1) == label) for label in labels]

def finite_state(state):
    return all(bool(torch.isfinite(v).all()) for v in state.values() if torch.is_tensor(v))


def configure_runtime():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def build_encoder(model, original, device):
    if model == "BAnD":
        from models.band import BAnD
        encoder = BAnD(in_channels=1, num_classes=1, base_channels=64,
                       transformer_layers=2, num_heads=8, feedforward_dim=2048,
                       dropout=.2, frame_chunk_size=4, max_sequence_length=40)
    else:
        if model == "NeuroSTORM":
            from models.neurostorm import NeuroSTORM
            constructor = NeuroSTORM
        else:
            from models.swift import SwiFT
            constructor = SwiFT
        options = dict(img_size=[96,96,96,40], in_chans=1, embed_dim=36,
                       window_size=[4,4,4,4], first_window_size=[4,4,4,4],
                       patch_size=[6,6,6,1], depths=[2,2,6,2], num_heads=[3,6,12,24],
                       c_multiplier=2, last_layer_full_MSA=True, drop_rate=0.,
                       attn_drop_rate=0., drop_path_rate=0.)
        if model == "NeuroSTORM":
            options["prompt_len"] = 0
        encoder = constructor(**options)
    state = original.get("state_dict", original)
    expected = encoder.state_dict()
    matched = {}
    for key, value in state.items():
        key = key.removeprefix("model.")
        if key not in expected or not torch.is_tensor(value):
            continue
        if value.shape != expected[key].shape:
            if not (model == "SwiFT" and key.startswith("pos_embeds.") and key.endswith(".time_embed")):
                raise ValueError(f"Encoder shape mismatch: {key}")
            target = expected[key]
            if value.ndim != 6 or value.shape[:-1] != target.shape[:-1]:
                raise ValueError(f"Invalid SwiFT temporal adapter: {key}")
            value = torch.nn.functional.interpolate(
                value.reshape(1, value.shape[1], value.shape[-1]).float(),
                size=target.shape[-1], mode="linear", align_corners=False,
            ).reshape(target.shape).to(value.dtype)
        matched[key] = value
    encoder.load_state_dict(matched, strict=True)
    if not finite_state(encoder.state_dict()):
        raise ValueError("Non-finite encoder")
    return encoder.requires_grad_(False).to(device).eval()


class PairedModel(nn.Module):
    def __init__(self, model, config, state):
        super().__init__()
        self.model_name, self.config = model, config
        self.fusion = config["fusion"]
        self.register_buffer("training_people_indices", torch.zeros(32, dtype=torch.long))
        self.backbone = None
        if model in ("NeuroSTORM", "BAnD", "SwiFT"):
            dims = state["head.weight"].shape[1]
            self.register_buffer("feature_mean", torch.zeros(dims))
            self.register_buffer("feature_std", torch.ones(dims))
        elif model == "BNT":
            from models.bnt import BrainNetworkTransformer
            self.backbone = BrainNetworkTransformer(
                num_rois=190, node_feature_size=190, num_classes=2,
                pos_encoding="identity", pos_embed_dim=8, pooling_sizes=[190],
                do_pooling=[False], hidden_size=1024, dropout=0.)
            self.backbone.fc[-1] = nn.Identity()
            dims = 32
        elif model == "BrainGNN":
            from models.braingnn import BrainGNN
            self.backbone = BrainGNN(in_channels=190, num_classes=2, num_rois=190,
                                     pooling_ratio=.5, num_communities=8, dropout=.5)
            self.backbone.fc3 = nn.Identity()
            dims = 512
        else:
            raise ValueError(model)
        self.head = nn.Linear(dims, 2)
        self.load_state_dict(state, strict=True)
        if not finite_state(self.state_dict()):
            raise ValueError("Non-finite native paired model")
        self.eval()

    def forward(self, before, after):
        if self.backbone is None:
            features = after - before
            if self.fusion == "baseline_delta":
                features = torch.cat([before, features], dim=1)
            features = (features - self.feature_mean) / self.feature_std
        else:
            if self.model_name == "BNT":
                representation = self.backbone(torch.cat([before, after], dim=0))[0]
            else:
                from torch_geometric.data import Batch
                graph = Batch.from_data_list(before + after).to(self.head.weight.device)
                representation = self.backbone(graph.x, graph.edge_index, graph.edge_attr, graph.batch, graph.pos)[0]
            pre, post = representation.chunk(2, dim=0)
            features = post - pre
            if self.fusion == "baseline_delta":
                features = torch.cat([pre, features], dim=1)
        return self.head(features)


def graph_from_fc(fc, config):
    from torch_geometric.data import Data
    fc = fc.clone()
    if config.get("roi_fc_space", "fisher") == "correlation":
        fc = torch.tanh(fc)
    fc.fill_diagonal_(0)
    fraction = float(config.get("graph_edge_fraction", 1.))
    upper = torch.triu_indices(190,190,offset=1)
    magnitudes = fc[upper[0],upper[1]].abs()
    active = torch.nonzero(magnitudes > 0, as_tuple=True)[0]
    order = torch.argsort(magnitudes[active], descending=True, stable=True)
    keep = active[order[:max(1, math.ceil(len(active)*fraction))]]
    a,b = upper[:,keep]
    src,dst = torch.cat([a,b]), torch.cat([b,a])
    return Data(x=fc, edge_index=torch.stack([src,dst]),
                edge_attr=fc[src,dst].abs().unsqueeze(-1), pos=torch.eye(190))


def progress(model, done, total, phase):
    filled = round(24 * done / total)
    sys.stdout.write(f"\r{model} {phase} [{'#'*filled}{'-'*(24-filled)}] {100*done/total:.2f}%")
    sys.stdout.flush()
    if done == total:
        print()


@torch.inference_mode()
def infer(model, heads, config, protocol, device):
    people = test_subjects(protocol)
    frozen = model in ("NeuroSTORM", "BAnD", "SwiFT")
    values = {}
    masks = None if frozen else atlas_masks(config["atlas"])
    if not frozen:
        for index, (name, clip) in enumerate(first40_scans(config, protocol), 1):
            values[name] = torch.from_numpy(fc_from_clip(clip, masks))
            if not torch.isfinite(values[name]).all():
                raise ValueError(f"Non-finite representation for {name}")
            progress(model, index, 8, "inputs")
    rows = []
    trial_config = None
    for run_index, (selection, path) in enumerate(heads):
        checkpoint = load_checkpoint(model, selection, path)
        hparams = checkpoint.get("hyper_parameters", {})
        if trial_config is None:
            trial_config = hparams["trial_config"]
        elif hparams["trial_config"] != trial_config:
            raise ValueError("Native checkpoint training configuration differs across seeds")
        expected_train = [int(p["person"].removeprefix("sub-")) - 1
                          for p in protocol["subjects"] if p["split"] == "Train"]
        if list(hparams.get("train_indices", [])) != expected_train:
            raise ValueError("Native checkpoint Train subjects differ from the locked subject split")
        if frozen:
            # Each run uses only its own file: its embedded encoder and native head.
            encoder = build_encoder(model, checkpoint["frozen_encoder_checkpoint"], device)
            values = {}
            for index, (name, clip) in enumerate(first40_scans(config, protocol), 1):
                x = clip.to(device)
                representation = encoder.forward_features(x) if model == "BAnD" else encoder(x).mean(dim=(2,3,4,5))
                values[name] = representation.detach().float().cpu()[0]
                if not torch.isfinite(values[name]).all():
                    raise ValueError(f"Non-finite representation for {name}")
                progress(model, index, 8, f"run {run_index+1}/5 inputs")
            del encoder, x, representation
        paired = PairedModel(model, trial_config, checkpoint["state_dict"])
        # Frozen encoders use the selected GPU; linear heads run in FP32 on CPU.
        head_device = torch.device("cpu") if frozen else device
        paired = paired.to(head_device)
        before = [values[p["pre"]] for p in people]
        after = [values[p["post"]] for p in people]
        if frozen or model == "BNT":
            before = torch.stack(before).to(head_device)
            after = torch.stack(after).to(head_device)
        else:
            before = [graph_from_fc(x, trial_config) for x in before]
            after = [graph_from_fc(x, trial_config) for x in after]
        logits = paired(before, after).detach().float().cpu()
        if logits.shape != (4,2) or not torch.isfinite(logits).all():
            raise ValueError("Invalid native two-logit predictions")
        probabilities = torch.softmax(logits,dim=1)
        for p,z,prob in zip(people,logits,probabilities):
            label = int(p["label"])
            predicted = int(z.argmax())
            rows.append(dict(method=model, seed=selection["seed"], person=p["person"],
                             role="Test", label=label, predicted=predicted,
                             logit_placebo=float(z[0]), logit_DMT=float(z[1]),
                             P_Placebo=float(prob[0]), P_DMT=float(prob[1]),
                             P_true=float(prob[label]), correct=predicted == label,
                             P_predicted=float(prob[predicted]), selected_epoch=selection["epoch"],
                             actual_optimizer_step=int(checkpoint["global_step"])))
        progress(model, (run_index+1)*4, 20, "predictions")
        del paired, checkpoint
    return rows

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_predictions(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def score_predictions(rows, protocol):
    """Use raw logit margins for AUC, avoiding rounded/saturated probabilities."""
    import numpy as np
    from scipy.special import logsumexp, softmax
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    people = test_subjects(protocol)
    expected = {p["person"]: int(p["label"]) for p in people}
    output = []
    if len(rows) != 20:
        raise ValueError("Expected four Test predictions for each of five seeds")
    for seed in SEEDS:
        group = [r for r in rows if int(r["seed"]) == seed]
        if len(group) != 4 or {r["person"] for r in group} != set(expected):
            raise ValueError(f"Missing/duplicate Test people for seed {seed}")
        if any(r["role"] != "Test" or int(r["label"]) != expected[r["person"]] for r in group):
            raise ValueError("Prediction labels or split differ from the fixed protocol")
        z = np.array([[float(r["logit_placebo"]), float(r["logit_DMT"])] for r in group], dtype=np.float64)
        y = np.array([int(r["label"]) for r in group])
        if not np.isfinite(z).all():
            raise ValueError("Non-finite logits")
        p, prediction = softmax(z, axis=1), z.argmax(axis=1)
        for r, pred, prob in zip(group, prediction, p):
            if int(r["predicted"]) != int(pred) or abs(float(r["P_true"]) - prob[int(r["label"])]) > 2e-7:
                raise ValueError("Stored predictions/probabilities disagree with raw logits")
        output.append(dict(seed=seed, n=4, correct=int((prediction == y).sum()),
                           accuracy=float(accuracy_score(y, prediction)),
                           macroF1=float(f1_score(y, prediction, average="macro", zero_division=0)),
                           macroOVRAUC=float(roc_auc_score(y, z[:, 1] - z[:, 0])),
                           NLL=float((logsumexp(z, axis=1) - z[np.arange(4), y]).mean()),
                           Brier_two_class=float(((p - np.eye(2)[y])**2).sum(axis=1).mean())))
    summaries = {}
    for name in ("accuracy", "macroF1", "macroOVRAUC", "NLL", "Brier_two_class"):
        values = [r[name] for r in output]
        summaries[name] = dict(mean=statistics.mean(values), sample_SD=statistics.stdev(values))

    return output, summaries

def make_report(model,rows,protocol,source,context):
    per_seed,summary=score_predictions(rows,protocol)
    return dict(model=model,context=context,seeds=per_seed,summary=summary,
                checkpoint_source=source)


def load_reports(directory,context,replacing=None):
    reports={}
    for model in MODEL_NAMES:
        if model==replacing:
            continue
        path=Path(directory)/"metrics"/(model+"_dmt.json")
        if path.is_file():
            report=read_json(path)
            if report.get("model")!=model or report.get("context")!=context:
                raise ValueError("Existing results use another protocol or input path; choose a new output directory")
            rows=read_predictions(Path(directory)/"predictions"/(model+"_dmt.csv"))
            report["seeds"],report["summary"]=score_predictions(rows,context["protocol"])
            reports[model]=report

    return reports


def write_workbook(directory, reports):
    from openpyxl import Workbook
    from openpyxl.styles import Font

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    book = Workbook()
    book.remove(book.active)
    sheets = {"Accuracy": "accuracy", "macroF1": "macroF1", "macroOVRAUC": "macroOVRAUC",
              "NLL": "NLL", "Brier": "Brier_two_class"}
    for title, metric in sheets.items():
        sheet = book.create_sheet(title)
        sheet.append(["DMT-HAR-MED", *MODEL_NAMES])
        for index, seed in enumerate(SEEDS):
            values = [reports[m]["seeds"][index][metric] if m in reports else None for m in MODEL_NAMES]
            sheet.append([f"seed {seed}", *values])
        for row, summary_name in ((7, "mean"), (8, "sample_SD")):
            sheet.cell(row, 1, summary_name)
            for column, model in enumerate(MODEL_NAMES, 2):
                if model in reports:
                    sheet.cell(row, column, reports[model]["summary"][metric][summary_name])
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        sheet.freeze_panes = "B2"
        sheet.column_dimensions["A"].width = 22
        for column in range(2, 7):
            sheet.column_dimensions[chr(64+column)].width = 19
        for row in sheet.iter_rows(min_row=2, max_row=8, min_col=2, max_col=6):
            for cell in row:
                cell.number_format = "0.000000000000000"
    temporary = directory / "performance.writing.xlsx"
    book.save(temporary)
    temporary.replace(directory / "performance.xlsx")

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",required=True,choices=MODEL_NAMES)
    source_group=parser.add_mutually_exclusive_group()
    source_group.add_argument("--repo",help="HF repository ID or URL")
    source_group.add_argument("--checkpoint","--checkpoint-url",help="HF DMT model folder or checkpoint URL; runs all five seeds")
    parser.add_argument("--revision")
    parser.add_argument("--download-dir",type=Path,default=Path("checkpoints"))
    parser.add_argument("--data-config",type=Path,default=ROOT/"demo/task5_config.yaml")
    parser.add_argument("--data-root",type=Path,help="Prepared DMT root containing img/")
    parser.add_argument("--output-dir",type=Path,default=Path("results/task5"))
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--replace",action="store_true")
    parser.add_argument("--check-only",action="store_true",help="Download and strictly load all five files without inference or reports")
    args=parser.parse_args(argv)
    config=load_config(args.data_config,args.data_root)
    protocol=validate_protocol(read_json(config["protocol"]))
    context=context_for(config,protocol)
    output=args.output_dir.expanduser().resolve()
    try:
        source=parse_source(args.checkpoint or args.repo or "zxcvb20001/NeuroSTORM",args.model,args.revision)
    except ValueError as error:
        parser.error(str(error))
    reports={}
    if not args.check_only:
        csv_path=output/"predictions"/(args.model+"_dmt.csv")
        report_path=output/"metrics"/(args.model+"_dmt.json")
        if not args.replace and (csv_path.exists() or report_path.exists()):
            parser.error(f"{args.model} already has results; use --replace to rerun only this model")
        reports=load_reports(output,context,args.model if args.replace else None)
        for person in test_subjects(protocol):
            for key in ("pre","post"):
                folder=config["data_root"]/"img"/person[key]
                if not folder.is_dir() or not any((folder/f).is_file() for f in ("data.pt","frame_0.pt")):
                    parser.error(f"Missing prepared Test scan: {folder}; set data_root in task5_config.yaml")
        if args.model in ("BNT","BrainGNN") and not config["atlas"].is_file():
            parser.error(f"Missing CC200 atlas: {config['atlas']}")
    configure_runtime()
    if not args.check_only and args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; source env.sh and select an available GPU")
    checkpoints,origin=download_bundle(args.model,protocol,source,args.download_dir.expanduser().resolve())
    if args.check_only:
        trial_config=None
        for selected,path in checkpoints:
            checkpoint=load_checkpoint(args.model,selected,path)
            if trial_config is None:
                trial_config=checkpoint["hyper_parameters"]["trial_config"]
            elif checkpoint["hyper_parameters"]["trial_config"]!=trial_config:
                raise ValueError("Native checkpoint training configuration differs across seeds")
            paired=PairedModel(args.model,trial_config,checkpoint["state_dict"])
            if args.model in MODEL_NAMES[:3]:
                encoder=build_encoder(args.model,checkpoint["frozen_encoder_checkpoint"],torch.device("cpu"))
                del encoder
            del paired,checkpoint
        print(f"{args.model}: all five checkpoints loaded strictly; no inference",flush=True)
        return
    rows=infer(args.model,checkpoints,config,protocol,torch.device(args.device))
    report=make_report(args.model,rows,protocol,origin,context)
    reports[args.model]=report
    output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{args.model}-",dir=output) as temporary:
        temporary=Path(temporary)
        write_predictions(temporary/"predictions.csv",rows)
        write_json(temporary/"metrics.json",report)
        write_workbook(temporary,reports)
        if args.replace and (csv_path.exists() or report_path.exists()):
            history=output/"history"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")/args.model
            history.mkdir(parents=True)
            for old in (csv_path,report_path):
                if old.is_file():shutil.copy2(old,history/old.name)
        csv_path.parent.mkdir(parents=True,exist_ok=True)
        report_path.parent.mkdir(parents=True,exist_ok=True)
        (temporary/"predictions.csv").replace(csv_path)
        (temporary/"metrics.json").replace(report_path)
        (temporary/"performance.xlsx").replace(output/"performance.xlsx")
    for run,item in enumerate(report["seeds"],1):
        print(f"run {run}, seed {item['seed']}: Accuracy {100*item['accuracy']:.2f}%")
    mean=report["summary"]["accuracy"]["mean"]*100
    sd=report["summary"]["accuracy"]["sample_SD"]*100
    print(f"{args.model}: Accuracy {mean:.2f}% ± {sd:.2f}% s.d.",flush=True)
    print(f"Updated {output/'performance.xlsx'} ({len(reports)}/5 models)")


if __name__=="__main__":
    main()
