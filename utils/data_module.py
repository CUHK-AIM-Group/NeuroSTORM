import os
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from datasets.fmri_datasets import HCP1200, ABCD, UKB, Cobre, ADHD200, UCLA, HCPEP, HCPTASK, GOD, MOVIE, TransDiag
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from .parser import str2bool

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import random


def select_elements(S, n):
    level_count = defaultdict(int)
    for value in S.values():
        level_count[value[1]] += 1

    total_elements = sum(level_count.values())
    level_quota = {level: int(n * count / total_elements) for level, count in level_count.items()}

    remaining = n - sum(level_quota.values())
    levels = sorted(level_count.keys(), key=lambda x: -level_count[x])
    for i in range(remaining):
        level_quota[levels[i % len(levels)]] += 1

    selected_elements = []
    for level in level_quota:
        elements_of_level = [k for k, v in S.items() if v[1] == level]
        selected_elements.extend(random.sample(elements_of_level, level_quota[level]))

    S_prime = {k: S[k] for k in selected_elements}

    return S_prime


class fMRIDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # generate splits folder
        if self.hparams.pretraining:
            split_dir_path = f'./data/splits/{self.hparams.dataset_name}/pretraining'
        else:
            split_dir_path = f'./data/splits/{self.hparams.dataset_name}'
        os.makedirs(split_dir_path, exist_ok=True)
        self.split_file_path = os.path.join(split_dir_path, f"split_fixed_{self.hparams.dataset_split_num}.txt")
        
        self.setup()

    def get_dataset(self):
        if self.hparams.dataset_name == "HCP1200":
            return HCP1200
        elif self.hparams.dataset_name == "ABCD":
            return ABCD
        elif self.hparams.dataset_name == 'UKB':
            return UKB
        elif self.hparams.dataset_name == 'Cobre':
            return Cobre
        elif self.hparams.dataset_name == 'ADHD200':
            return ADHD200
        elif self.hparams.dataset_name == 'UCLA':
            return UCLA
        elif self.hparams.dataset_name == 'HCPEP':
            return HCPEP
        elif self.hparams.dataset_name == 'GOD':
            return GOD 
        elif self.hparams.dataset_name == 'HCPTASK':
            return HCPTASK
        elif self.hparams.dataset_name == 'MOVIE':
            return MOVIE
        elif self.hparams.dataset_name == 'TransDiag':
            return TransDiag
        else:
            raise NotImplementedError

    def convert_subject_list_to_idx_list(self, train_names, val_names, test_names, subj_list):
        subj_idx = np.array([str(x[1]) for x in subj_list])
        S = np.unique([x[1] for x in subj_list])
        print('unique subjects:',len(S))  
        train_idx = np.where(np.in1d(subj_idx, train_names))[0].tolist()
        val_idx = np.where(np.in1d(subj_idx, val_names))[0].tolist()
        test_idx = np.where(np.in1d(subj_idx, test_names))[0].tolist()
        
        return train_idx, val_idx, test_idx
    
    def save_split(self, sets_dict):
        with open(self.split_file_path, "w+") as f:
            for name, subj_list in sets_dict.items():
                f.write(name + "\n")
                for subj_name in subj_list:
                    f.write(str(subj_name) + "\n")
                    
    def determine_split_randomly(self, S):
        np.random.seed(0)
        S_keys = list(S.keys())
        S_train = int(len(S_keys) * self.hparams.train_split)
        S_val = int(len(S_keys) * self.hparams.val_split)
        
        if self.hparams.downstream_task_type == 'classification':
            S_train = select_elements(S, S_train)
            S_remaining = {k: v for k, v in S.items() if k not in S_train}
            S_train_keys = list(S_train.keys())
        else:
            S_train_keys = np.random.choice(S_keys, S_train, replace=False)
        
        remaining_keys = np.setdiff1d(S_keys, S_train_keys)

        if self.hparams.downstream_task_type == 'classification':
            S_val = select_elements(S_remaining, S_val)
            S_val_keys = list(S_val.keys())
            if self.hparams.val_split + self.hparams.train_split < 1:
                S_test = {k: v for k, v in S_remaining.items() if k not in S_val}
                S_test_keys = list(S_test.keys())
        else:
            S_val_keys = np.random.choice(remaining_keys, S_val, replace=False)
            if self.hparams.val_split + self.hparams.train_split < 1:
                S_test_keys = np.setdiff1d(S_keys, np.concatenate([S_train_keys, S_val_keys]))
        
        if self.hparams.val_split + self.hparams.train_split < 1:
            self.save_split({"train_subjects": S_train_keys, "val_subjects": S_val_keys, "test_subjects": S_test_keys})
            return S_train_keys, S_val_keys, S_test_keys
        else:
            self.save_split({"train_subjects": S_train_keys, "val_subjects": S_val_keys, "test_subjects": S_val_keys})
            return S_train_keys, S_val_keys, S_val_keys
    
    def load_split(self):
        subject_order = open(self.split_file_path, "r").readlines()
        subject_order = [x[:-1] for x in subject_order]
        train_index = np.argmax(["train" in line for line in subject_order])
        val_index = np.argmax(["val" in line for line in subject_order])
        test_index = np.argmax(["test" in line for line in subject_order])
        train_names = subject_order[train_index + 1 : val_index]
        val_names = subject_order[val_index + 1 : test_index]
        test_names = subject_order[test_index + 1 :]

        return train_names, val_names, test_names

    def prepare_data(self):
        # This function is only called at global rank==0
        return
    
    # filter subjects with metadata and pair subject names with their target values (+ sex)
    def make_subject_dict(self):
        img_root = os.path.join(self.hparams.image_path, 'img')
        final_dict = dict()

        if self.hparams.dataset_name == "HCP1200":
            subject_list = os.listdir(img_root)
            meta_data = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "HCP_1200_gender.csv"))
            meta_data_residual = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "HCP_1200_precise_age.csv"))
            if self.hparams.task_name == 'sex': task_name = 'Gender'
            elif self.hparams.task_name == 'age': task_name = 'age'
            # MMSE_Score Social_Task_Random_Perc_TOM CogTotalComp_Unadj Emotion_Task_Acc Language_Task_Acc Strength_Unadj 
            elif self.hparams.downstream_task_id == 2: task_name = self.hparams.task_name
            else: raise NotImplementedError()

            print('downstream_task_id = {}, task_name = {}'.format(self.hparams.downstream_task_id, task_name))

            if task_name == 'Gender':
                meta_task = meta_data[['Subject',task_name]].dropna()
            elif task_name == 'age':
                meta_task = meta_data_residual[['subject',task_name,'sex']].dropna()
                meta_task = meta_task.rename(columns={'subject': 'Subject'})
            elif self.hparams.downstream_task_id == 2:
                meta_task = meta_data[['Subject', task_name, 'Gender']].dropna()  
            
            for subject in subject_list:
                if int(subject) in meta_task['Subject'].values:
                    if task_name == 'Gender':
                        target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0]
                        target = 1 if target == "M" else 0
                        sex = target
                    elif task_name == 'age':
                        target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0]
                        sex = meta_task[meta_task["Subject"]==int(subject)]["sex"].values[0]
                        sex = 1 if sex == "M" else 0
                    elif self.hparams.downstream_task_id == 2:
                        target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0]
                        sex = meta_task[meta_task["Subject"]==int(subject)]["Gender"].values[0]
                        sex = 1 if sex == "M" else 0
                    final_dict[subject] = [sex, target]
            
            print('Load dataset HCP1200, {} subjects'.format(len(final_dict)))
            
        elif self.hparams.dataset_name == "ABCD":
            subject_list = [subj for subj in os.listdir(img_root)]
            
            meta_data = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "abcd-rest.csv"))
            if self.hparams.task_name == 'sex': task_name = 'sex'
            elif self.hparams.downstream_task == 'age': task_name = 'age'
            else: raise ValueError('downstream task not supported')
           
            if task_name == 'sex':
                meta_task = meta_data[['subjectkey', task_name]].dropna()
            else:
                meta_task = meta_data[['subjectkey', task_name, 'sex']].dropna()
            
            for subject in subject_list:
                if subject in meta_task['subjectkey'].values:
                    target = meta_task[meta_task["subjectkey"]==subject][task_name].values[0]
                    if task_name == 'sex':
                        target = 1 if target == "M" else 0
                    sex = meta_task[meta_task["subjectkey"]==subject]["sex"].values[0]
                    sex = 1 if sex == "M" else 0
                    final_dict[subject] = [sex, target]
            
            print('Load dataset ABCD, {} subjects'.format(len(final_dict)))
        
        elif self.hparams.dataset_name == "Cobre":
            subject_list = [subj for subj in os.listdir(img_root)]
            
            meta_data = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "cobre-rest.csv"))
            if self.hparams.task_name == 'sex': task_name = 'sex'
            elif self.hparams.task_name == 'age': task_name = 'age'
            elif self.hparams.task_name == 'diagnosis': task_name = 'dx'
            else: raise ValueError('downstream task not supported')
           
            if task_name == 'sex':
                meta_task = meta_data[['subject_id', task_name]].dropna()
            else:
                meta_task = meta_data[['subject_id', task_name, 'sex']].dropna()
            
            for subject in subject_list:
                if subject in meta_task['subject_id'].values:
                    target = meta_task[meta_task["subject_id"]==subject][task_name].values[0]
                    if task_name == 'sex':
                        target = 1 if target == "M" else 0
                    elif task_name == 'dx':
                        if target == 'Schizophrenia_Strict': target = 0
                        elif target == 'Schizoaffective': target = 1
                        elif target == 'No_Known_Disorder': target = 2
                        elif target == 'Bipolar_Disorder': target = 3
                        else: 
                            import ipdb; ipdb.set_trace()
                        
                    sex = meta_task[meta_task["subject_id"]==subject]["sex"].values[0]
                    sex = 1 if sex == "male" else 0
                    final_dict[subject] = [sex, target]
            
            print('Load dataset Cobre, {} subjects'.format(len(final_dict)))
        
        elif self.hparams.dataset_name == "ADHD200":
            subject_list = [subj for subj in os.listdir(img_root)]
            
            meta_data = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "adhd200-rest.csv"))
            if self.hparams.task_name == 'sex': task_name = 'Gender'
            elif self.hparams.task_name == 'age': task_name = 'Age'
            elif self.hparams.task_name == 'diagnosis': task_name = 'DX'
            else: raise ValueError('downstream task not supported')
           
            if task_name == 'sex':
                meta_task = meta_data[['subject_id', task_name]].dropna()
            else:
                meta_task = meta_data[['subject_id', task_name, 'Gender']].dropna()
            
            for subject in subject_list:
                if int(subject) in meta_task['subject_id'].values:
                    target = meta_task[meta_task["subject_id"]==int(subject)][task_name].values[0]
                    if task_name == 'DX':
                        if target == 'pending': continue
                        
                        target = int(target)
                        target = 1 if target > 0 else 0
                        
                    sex = meta_task[meta_task["subject_id"]==int(int(subject))]["Gender"].values[0]
                    sex = int(sex)
                    final_dict[subject] = [sex, target]
            
            print('Load dataset ADHD200, {} subjects'.format(len(final_dict)))
        
        elif self.hparams.dataset_name == "UCLA":
            subject_list = [subj for subj in os.listdir(img_root)]
            
            meta_data = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "ucla-rest.csv"))
            if self.hparams.task_name == 'sex': task_name = 'gender'
            elif self.hparams.task_name == 'age': task_name = 'age'
            elif self.hparams.task_name == 'diagnosis': task_name = 'diagnosis'
            else: raise ValueError('downstream task not supported')
           
            if task_name == 'sex':
                meta_task = meta_data[['subject_id', task_name]].dropna()
            else:
                meta_task = meta_data[['subject_id', task_name, 'gender']].dropna()
            
            for subject in subject_list:
                if subject in meta_task['subject_id'].values:
                    target = meta_task[meta_task["subject_id"]==subject][task_name].values[0]
                    if task_name == 'gender':
                        target = 1 if target == "M" else 0
                    elif task_name == 'diagnosis':
                        if target == 'CONTROL': target = 0
                        elif target == 'SCHZ': target = 1
                        elif target == 'BIPOLAR': target = 2
                        elif target == 'ADHD': target = 3
                        else: 
                            import ipdb; ipdb.set_trace()
                        
                    sex = meta_task[meta_task["subject_id"]==subject]["gender"].values[0]
                    sex = 1 if sex == "M" else 0
                    final_dict[subject] = [sex, target]
            
            print('Load dataset UCLA, {} subjects'.format(len(final_dict)))
        
        elif self.hparams.dataset_name == "HCPEP":
            subject_list = [subj for subj in os.listdir(img_root)]
            
            meta_data = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "hcpep-rest.csv"))
            if self.hparams.task_name == 'sex': task_name = 'sex'
            elif self.hparams.task_name == 'age': task_name = 'interview_age'
            elif self.hparams.task_name == 'diagnosis': task_name = 'phenotype'
            else: raise ValueError('downstream task not supported')
           
            if task_name == 'sex':
                meta_task = meta_data[['subject_id', task_name]].dropna()
            else:
                meta_task = meta_data[['subject_id', task_name, 'sex']].dropna()
            
            for subject in subject_list:
                if int(subject[-4:]) in meta_task['subject_id'].values:
                    target = meta_task[meta_task["subject_id"]==int(subject[-4:])][task_name].values[0]
                    if task_name == 'sex':
                        target = 1 if target == "M" else 0
                    elif task_name == 'phenotype':
                        if target == 'Control': target = 0
                        elif target == 'Patient': target = 1
                        else:
                            import ipdb; ipdb.set_trace()
                        
                    sex = meta_task[meta_task["subject_id"]==int(subject[-4:])]["sex"].values[0]
                    sex = 1 if sex == "M" else 0
                    final_dict[subject] = [sex, target]
            
            print('Load dataset HCPEP, {} subjects'.format(len(final_dict)))
        
        elif self.hparams.dataset_name == "GOD":
            subject_list = [subj for subj in os.listdir(img_root)]
            
            meta_data = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "god_label.csv"))
            if self.hparams.downstream_task_id == 4: task_name = 'class'
            else: raise ValueError('downstream task not supported')
           
            meta_task = meta_data[['subject_id', task_name]].dropna()
            
            for subject in subject_list:
                if subject in meta_task['subject_id'].values:
                    target = meta_task[meta_task["subject_id"]==subject][task_name].values[0]
                    if task_name == 'sex':
                        target = 1 if target == "M" else 0
                    elif task_name == 'class':
                        target = target - 1
                        if target >= 150:
                            import ipdb; ipdb.set_trace()
                        
                    sex = 0
                    final_dict[subject] = [sex, target]

            category_count = defaultdict(int)
            for subject_id, (gender, category) in final_dict.items():
                category_count[category] += 1

            categories_to_delete = {category for category, count in category_count.items() if count < 40}
            final_dict = {subject_id: [gender, category] for subject_id, (gender, category) in final_dict.items() if category not in categories_to_delete}

            unique_categories = sorted(set(category for gender, category in final_dict.values()))
            category_mapping = {old_category: new_category for new_category, old_category in enumerate(unique_categories)}

            for subject_id in final_dict:
                final_dict[subject_id][1] = category_mapping[final_dict[subject_id][1]]

            print('Load dataset GOD, {} subjects, {} classes'.format(len(final_dict), len(unique_categories)))

        elif self.hparams.dataset_name == "UKB":
            subject_list = [subj for subj in os.listdir(img_root)]

            meta_data = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "ukb-rest.csv"))
            if self.hparams.task_name == 'sex': task_name = 'sex'
            elif self.hparams.task_name == 'age': task_name = 'age'
            else: raise ValueError('downstream task not supported')
           
            if task_name == 'sex':
                meta_task = meta_data[['subject_id', task_name]].dropna()
            else:
                meta_task = meta_data[['subject_id', task_name, 'sex']].dropna()
            
            for subject in subject_list:
                if int(subject) in meta_task['subject_id'].values:
                    target = meta_task[meta_task["subject_id"]==int(subject)][task_name].values[0]
                    if task_name == 'sex':
                        target = int(target)
                        
                    sex = meta_task[meta_task["subject_id"]==int(subject)]["sex"].values[0]
                    sex = int(sex)
                    final_dict[subject] = [sex, target]
            
            print('Load dataset UKB, {} subjects'.format(len(final_dict)))
        
        elif self.hparams.dataset_name == "HCPTASK":
            subject_list = [subj for subj in os.listdir(img_root)]

            if self.hparams.downstream_task_id == 5: task_name = 'classification'
            else: raise ValueError('downstream task not supported')

            state_to_label = {'EMOTION': 0, 'GAMBLING': 1, 'LANGUAGE': 2, 'MOTOR': 3, 'RELATIONAL': 4, 'SOCIAL': 5, 'WM': 6}

            for subject in subject_list:
                state = subject.split('_')[-2]
                state = state_to_label[state]

                sex = 0
                final_dict[subject] = [sex, state]
            
            print('Load dataset HCPTASK, {} subjects'.format(len(final_dict)))

        elif self.hparams.dataset_name == "MOVIE":
            subject_list = [subj for subj in os.listdir(img_root)]
            parent_dir = os.path.dirname(os.path.abspath(img_root))
            metadata_dir = os.path.join(parent_dir, 'metadata')
            participants_file = os.path.join(metadata_dir, 'participants.tsv')

            try:
                df = pd.read_csv(participants_file, sep='\t')
            except Exception as e:
                import ipdb; ipdb.set_trace()

            participant_id = df.iloc[:, 0]
            group = df.iloc[:, -1]
            participant_dict = pd.Series(group.values, index=participant_id).to_dict()

            if self.hparams.downstream_task_id == 5: task_name = 'classification'
            else: raise ValueError('downstream task not supported')

            for subject in subject_list:
                subject_id = subject.split('_')[0]
                if participant_dict[subject_id] == 'Control':
                    final_dict[subject] = [0, 0]
                else:
                    final_dict[subject] = [0, 1]
            
            print('Load dataset MOVIE, {} subjects'.format(len(final_dict)))
        
        if self.hparams.dataset_name == "TransDiag":
            subject_list = os.listdir(img_root)

            # diagnosis, clinical_variables
            if self.hparams.task_name == 'diagnosis': 
                csv_file = self.hparams.task_name + '.csv'
            else:
                csv_file = 'clinical_variables.csv'

            # import chardet
            # with open(os.path.join(self.hparams.image_path, "metadata", csv_file), 'rb') as file:
            #     result = chardet.detect(file.read())
            #     print(result)

            meta_data = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", csv_file), encoding='ISO-8859-1')
            if self.hparams.task_name == 'diagnosis': 
                task_name = 'diagnosis'
                # label_name = 'Group'
                label_name = 'Diagnostic_Category_Code'
            else: task_name = 'clinical_variables'

            print('downstream_task_id = {}, task_name = {}'.format(self.hparams.downstream_task_id, task_name))
            target_counts = defaultdict(int)

            if task_name == 'diagnosis':
                meta_task = meta_data[['subjectkey', label_name, 'sex']].dropna()
            elif task_name == 'clinical_variables':
                label_name = self.hparams.task_name
                meta_task = meta_data[['subjectkey', label_name]].dropna()
            
            for subject in subject_list:
                subject_id = subject[:4] + '_' + subject[4:-5]
                if subject_id in meta_task['subjectkey'].values:
                    if task_name == 'diagnosis':
                        sex = meta_task[meta_task["subjectkey"]==subject_id]['sex'].values[0]
                        if sex == 'F': sex = 0
                        else: sex = 1

                        target = meta_task[meta_task["subjectkey"]==subject_id][label_name].values[0]
                        if label_name == 'Diagnostic_Category_Code':
                            if target in [0, 5, 7]: continue
                            elif target == 8: target = 0
                            elif target == 6: target = 5
                        elif label_name == 'Group':
                            if target == 'Patient': target = 1
                            else: target = 0
                    else:
                        sex = 0
                        target = meta_task[meta_task["subjectkey"]==subject_id][label_name].values[0]
                        if target == 'n/a':
                            import ipdb; ipdb.set_trace()
                            continue
                    
                    target_counts[target] += 1
                    # print('sex = {}, target = {}'.format(sex, target))
                    final_dict[subject] = [sex, target]
            
            for category, count in target_counts.items():
                print(f"Target {category}: {count}")
            print('Load dataset TransDiag, {} subjects'.format(len(final_dict)))

        return final_dict

    def setup(self, stage=None):
        Dataset = self.get_dataset()
        params = {
                "root": self.hparams.image_path,
                "sequence_length": self.hparams.sequence_length,
                "contrastive": self.hparams.use_contrastive,
                "contrastive_type": self.hparams.contrastive_type,
                "mae": self.hparams.use_mae,
                "stride_between_seq": self.hparams.stride_between_seq,
                "stride_within_seq": self.hparams.stride_within_seq,
                "with_voxel_norm": self.hparams.with_voxel_norm,
                "downstream_task_id": self.hparams.downstream_task_id,
                "task_name": self.hparams.task_name,
                "shuffle_time_sequence": self.hparams.shuffle_time_sequence,
                "label_scaling_method": self.hparams.label_scaling_method,
                "dtype": 'float16'}
        
        subject_dict = self.make_subject_dict()
        
        if os.path.exists(self.split_file_path):
            train_names, val_names, test_names = self.load_split()
        else:
            train_names, val_names, test_names = self.determine_split_randomly(subject_dict)
        
        if self.hparams.bad_subj_path:
            bad_subjects = open(self.hparams.bad_subj_path, "r").readlines()
            for bad_subj in bad_subjects:
                bad_subj = bad_subj.strip()
                if bad_subj in list(subject_dict.keys()):
                    print(f'removing bad subject: {bad_subj}')
                    del subject_dict[bad_subj]
        
        if self.hparams.limit_training_samples:
            selected_num = int(self.hparams.limit_training_samples * len(train_names))
            train_names = np.random.choice(train_names, size=selected_num, replace=False, p=None)
        
        train_dict = {key: subject_dict[key] for key in train_names if key in subject_dict}
        val_dict = {key: subject_dict[key] for key in val_names if key in subject_dict}
        test_dict = {key: subject_dict[key] for key in test_names if key in subject_dict}
        
        self.train_dataset = Dataset(**params, subject_dict=train_dict, use_augmentations=False, train=True)
        self.val_dataset = Dataset(**params, subject_dict=val_dict, use_augmentations=False, train=False) 
        self.test_dataset = Dataset(**params, subject_dict=test_dict, use_augmentations=False, train=False)
        
        print("number of train subjects:", len(train_dict))
        print("number of val subjects:", len(val_dict))
        print("number of test subjects:", len(test_dict))
        print("number of train samples:", len(self.train_dataset.data))
        print("number of val samples:", len(self.val_dataset.data))  
        print("number of test samples:", len(self.test_dataset.data))
        
        # DistributedSampler is internally called in pl.Trainer
        def get_params(train):
            return {
                "batch_size": self.hparams.batch_size if train else self.hparams.eval_batch_size,
                "num_workers": self.hparams.num_workers,
                "drop_last": True,
                "pin_memory": False,
                "persistent_workers": (train and (self.hparams.strategy == 'ddp')),
                "shuffle": train
            }
        
        self.train_loader = DataLoader(self.train_dataset, **get_params(train=True))
        self.val_loader = DataLoader(self.val_dataset, **get_params(train=False))
        self.test_loader = DataLoader(self.test_dataset, **get_params(train=False))

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return [self.val_loader, self.test_loader]

    def test_dataloader(self):
        return self.test_loader

    def predict_dataloader(self):
        return self.test_dataloader()

    @classmethod
    def add_data_specific_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=True, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("DataModule arguments")
        group.add_argument("--dataset_split_num", type=int, default=1)
        group.add_argument("--label_scaling_method", default="standardization", choices=["minmax","standardization"], help="label normalization strategy for a regression task (mean and std are automatically calculated using train set)")
        group.add_argument("--image_path", default=None, help="path to image datasets preprocessed for SwiFT")
        group.add_argument("--bad_subj_path", default=None, help="path to txt file that contains subjects with bad fMRI quality")
        group.add_argument("--train_split", default=0.9, type=float)
        group.add_argument("--val_split", default=0.1, type=float)
        group.add_argument("--batch_size", type=int, default=4)
        group.add_argument("--eval_batch_size", type=int, default=8)
        group.add_argument("--img_size", nargs="+", default=[96, 96, 96, 20], type=int, help="image size (adjust the fourth dimension according to your --sequence_length argument)")
        group.add_argument("--sequence_length", type=int, default=20)
        group.add_argument("--stride_between_seq", type=int, default=1, help="skip some fMRI volumes between fMRI sub-sequences")
        group.add_argument("--stride_within_seq", type=int, default=1, help="skip some fMRI volumes within fMRI sub-sequences")
        group.add_argument("--num_workers", type=int, default=8)
        group.add_argument("--with_voxel_norm", type=str2bool, default=False)
        group.add_argument("--shuffle_time_sequence", action='store_true')
        group.add_argument("--limit_training_samples", type=float, default=None, help="use if you want to limit training samples")

        return parser
