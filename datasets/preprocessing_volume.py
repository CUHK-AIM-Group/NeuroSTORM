from monai.transforms import LoadImage
import torch
import os
import time
from multiprocessing import Process, Queue
import argparse
import torch.nn.functional as F
import numpy as np


def select_middle_96(vector):
    start_index, end_index = [], []
    for i in range(3):
        if vector.shape[i] > 96:
            start_index.append((vector.shape[i] - 96) // 2)
            end_index.append(start_index[-1] + 96)
        else:
            start_index.append(0)
            end_index.append(vector.shape[i])

    if len(vector.shape) == 3:
        result = vector[start_index[0]:end_index[0], start_index[1]:end_index[1], start_index[2]:end_index[2]]
    elif len(vector.shape) == 4:
        result = vector[start_index[0]:end_index[0], start_index[1]:end_index[1], start_index[2]:end_index[2], :]
    
    return result

def resize_to_96(vector):
    min_dim = min(vector.shape[:-1])
    resize_radio = 96 / min_dim
    new_size = (int(vector.shape[0] * resize_radio), int(vector.shape[1] * resize_radio), int(vector.shape[2] * resize_radio))

    if len(vector.shape) == 4:
        vector_permuted = vector.permute(3, 0, 1, 2)
        vector_unsqueezed = vector_permuted.unsqueeze(0)
    elif len(vector.shape) == 3:
        vector_unsqueezed = vector.unsqueeze(0).unsqueeze(0)
    output_tensor = F.interpolate(vector_unsqueezed, size=new_size, mode='trilinear', align_corners=True)
    if len(vector.shape) == 4:
        vector_squeezed = output_tensor.squeeze()
        vector = vector_squeezed.permute(1, 2, 3, 0)
    elif len(vector.shape) == 3:
        vector = output_tensor.squeeze()

    vector = select_middle_96(vector)
    return vector

def read_data(dataset_name, delete_after_preprocess, filename, load_root, save_root, subj_name, count, queue=None, scaling_method=None, fill_zeroback=False):
    print("processing: " + filename, flush=True)
    path = os.path.join(load_root, filename)
    try:
        data = LoadImage()(path)
    except:
        print('{} open failed'.format(path))
        return None
    
    save_dir = os.path.join(save_root, subj_name)
    isExist = os.path.exists(save_dir)
    if not isExist:
        os.makedirs(save_dir)
    
    if dataset_name in ['ukb', 'abcd', 'hcp', 'hcpd', 'hcpep', 'hcptask', 'movie']:
        data = select_middle_96(data)
    elif dataset_name in ['adhd200', 'cobre', 'ucla', 'god', 'transdiag']:
        data = resize_to_96(data)

    if dataset_name in ['adhd200', 'god', 'hcp', 'hcpd', 'ukb', 'hcptask', 'transdiag']:
        background = data==0
    else:
        if dataset_name in ['abcd', 'cobre', 'hcpep']:
            mask_path = path[:-19] + 'brain_mask.nii.gz'
        elif dataset_name == 'ucla':
            mask_path = path[:-14] + 'brainmask.nii.gz'
        elif dataset_name == 'movie':
            mask_path = path[:-4] + '_brainmask.nii.gz'

        try:
            background = LoadImage()(mask_path)
        except:
            print('mask open failed')
            return None

        if dataset_name == 'movie':
            background = background==0
            background = select_middle_96(background)
    
    data[background] = 0
    if scaling_method == 'z-norm':
        global_mean = data[data>0].mean()
        if dataset_name == 'movie':
            global_std = 10
        else:
            global_std = data[data>0].std()
        data_temp = (data - global_mean) / global_std
    elif scaling_method == 'minmax':
        data_temp = (data - data[~background].min()) / (data[~background].max() - data[~background].min())

    data_global = torch.empty(data.shape)
    data_global[background] = data_temp[~background].min() if not fill_zeroback else 0
    data_global[~background] = data_temp[~background]

    data_global = data_global.type(torch.float16)
    data_global_split = torch.split(data_global, 1, 3)

    for i, TR in enumerate(data_global_split):
        torch.save(TR.clone(), os.path.join(save_dir, "frame_"+str(i)+".pt"))
    
    if delete_after_preprocess:
        os.remove(path)
        print('delete {}'.format(path))

def main():
    parser = argparse.ArgumentParser(description='Process image data.')
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset name')
    parser.add_argument('--load_root', type=str, required=True, help='directory to load data from')
    parser.add_argument('--save_root', type=str, required=True, help='directory to save data to')
    parser.add_argument('--delete_after_preprocess', action='store_true', help='delete nii file after preprocess')
    parser.add_argument('--delete_nii', action='store_true', help='if you did not delete after preprocess, you can use it to delete nii file')
    parser.add_argument('--num_processes', type=int, default=1, help='number of processes to use')
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    load_root = args.load_root
    save_root = args.save_root
    scaling_method = 'z-norm'

    filenames = os.listdir(load_root)
    os.makedirs(os.path.join(save_root, 'img'), exist_ok=True)
    os.makedirs(os.path.join(save_root, 'metadata'), exist_ok=True)
    save_root = os.path.join(save_root, 'img')
    
    queue = Queue() 
    count = 0

    for filename in sorted(filenames):
        if not (filename.endswith('.nii.gz') or filename.endswith('.nii')) or 'mask' in filename or 'imagery' in filename:
            continue

        # Determine subject name based on dataset
        subj_name = determine_subject_name(dataset_name, filename)
        if args.delete_nii:
            handle_delete_nii(load_root, save_root, filename, subj_name)
        else:
            try:
                count += 1
                if args.num_processes == 1:
                    read_data(dataset_name, args.delete_after_preprocess, filename, load_root, save_root, subj_name, count, queue, scaling_method)
                else:
                    processes = []
                    p = Process(target=read_data, args=(dataset_name, args.delete_after_preprocess, filename, load_root, save_root, subj_name, count, queue, scaling_method))
                    processes.append(p)
                    p.start()
                    if count % args.num_processes == 0:
                        for p in processes:
                            p.join()
                        processes = []
            except Exception as e:
                print(f'encountered problem with {filename}: {e}')
        
        # if args.num_processes > 1:
        #     for p in processes:
        #         p.join()


def determine_subject_name(dataset_name, filename):
    if dataset_name in ['abcd', 'cobre']:
        return filename.split('-')[1][:-4]
    elif dataset_name == 'adhd200':
        return filename.split('_')[2]
    elif dataset_name == 'god':
        return filename[:6] + '_' + filename.split('perception_')[1][:6]
    elif dataset_name == 'hcp':
        return filename[:-7]
    elif dataset_name == 'hcpd':
        return filename[:10]
    elif dataset_name == 'hcpep':
        return filename[:8]
    elif dataset_name == 'ucla':
        return filename[:9]
    elif dataset_name == 'ukb':
        return filename.split('.')[0]
    elif dataset_name == 'hcptask':
        return filename.split('.')[0]
    elif dataset_name == 'movie':
        return filename.split('.')[0]
    elif dataset_name == 'transdiag':
        return filename.split('_task')[0].split('-')[1]

def handle_delete_nii(load_root, save_root, filename, subj_name):
    path = os.path.join(load_root, filename)
    save_dir = os.path.join(save_root, subj_name)

    if os.path.isdir(save_dir):
        print(f'{subj_name} has {len(os.listdir(save_dir))} slices, save_dir is {save_dir}')
        os.remove(path)
    else:
        print(f'{save_dir} is empty, if you still want to delete nii file, uncomment the following code')
        # os.remove(path)

if __name__=='__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('\nTotal', round((end_time - start_time) / 60), 'minutes elapsed.')
