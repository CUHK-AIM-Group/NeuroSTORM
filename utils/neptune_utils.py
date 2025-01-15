import os
import torch


def load_ckpt(exp_id, root_dir):
    path = os.path.join(root_dir, exp_id, "last.ckpt")
    return torch.load(path, map_location="cpu")


def get_prev_args(ckpt_path, args):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    ignored_args_list = ["data_dir", "default_root_dir", "max_epochs", "resume_ckpt_path", "adjust_thresh"]
    print(f"Warning: You have to specify the following arguments list when you are running the process: {ignored_args_list}")
    for k, v in ckpt["hyper_parameters"].items():
        if k in ignored_args_list:
            continue
        setattr(args, k, v)
    return args
