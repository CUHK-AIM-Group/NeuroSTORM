import os
import torch


def load_ckpt(exp_id, root_dir):
    path = os.path.join(root_dir, exp_id, "last.ckpt")
    return torch.load(path, map_location="cpu")


def get_prev_args(ckpt_path, args):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    ignored_args_list = ["data_dir", "default_root_dir", "max_epochs", "resume_ckpt_path"]
    _is_rank0 = (
        os.environ.get("NEUROSTORM_IS_WORKER") != "1"
        and int(os.environ.get("LOCAL_RANK", "0")) == 0
    )
    if _is_rank0:
        print(
            f"Resuming from checkpoint; the following args are taken from CLI, "
            f"not the checkpoint: {ignored_args_list}"
        )
    for k, v in ckpt["hyper_parameters"].items():
        if k in ignored_args_list:
            continue
        setattr(args, k, v)
    return args
