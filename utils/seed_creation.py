import os
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader, Subset
from data_preprocess_and_load.datasets2 import S1200
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from parser import str2bool
import pandas as pd

split_file_path = './data/splits/S1200/seed_candidate.txt'
dataset_name = 'S1200'
img_path = '/mnt/ssd/processed/S1200'
train_split = 0.7
val_split = 0.15


def get_dataset(dataset_name):
    if dataset_name == "S1200":
        return S1200
    else:
        raise NotImplementedError

def save_split(split_file_path, sets_dict):
    with open(split_file_path, "w+") as f:
        for name, subj_list in sets_dict.items():
            f.write(name + "\n")
            for subj_name in subj_list:
                f.write(str(subj_name) + "\n")

def convert_subject_list_to_idx_list(self, train_names, val_names, test_names, subj_list):
    subj_idx = np.array([str(x[1]) for x in subj_list])
    train_idx = np.where(np.in1d(subj_idx, train_names))[0].tolist()
    val_idx = np.where(np.in1d(subj_idx, val_names))[0].tolist()
    test_idx = np.where(np.in1d(subj_idx, test_names))[0].tolist()
    return train_idx, val_idx, test_idx

Dataset = get_dataset(dataset_name)

dataset = Dataset(root = img_path)

subject_list = dataset.data

S = np.unique([x[1] for x in subject_list])
S_train = int(len(S) * 0.7)
S_val = int(len(S) * 0.15)
S_train = np.random.choice(S, S_train, replace=False)
remaining = np.setdiff1d(S, S_train)
S_val = np.random.choice(remaining, S_val, replace=False)
S_test = np.setdiff1d(S, np.concatenate([S_train, S_val]))

save_split(split_file_path, {"train_subjects": S_train, "val_subjects": S_val, "test_subjects": S_test})

# Evaluating the split

meta_data = pd.read_csv(os.path.join(img_path, "metadata", "HCP_1200_precise_age.csv"))

S_train = S_train.astype(np.int64)
S_val = S_val.astype(np.int64)
S_test = S_test.astype(np.int64)

meta_data_train = meta_data.query("subject in @S_train")
meta_data_val = meta_data.query("subject in @S_val")
meta_data_test = meta_data.query("subject in @S_test")

print("Train set age mean: ", meta_data_train["age"].mean())
print("Train set age std: ", meta_data_train["age"].std())
print("Val set age mean: ", meta_data_val["age"].mean())
print("Val set age std: ", meta_data_val["age"].std())
print("Test set age mean: ", meta_data_test["age"].mean())
print("Test set age std: ", meta_data_test["age"].std())

# number of sexes in each set
print(meta_data_train["sex"].value_counts())
print(meta_data_val["sex"].value_counts())
print(meta_data_test["sex"].value_counts())
