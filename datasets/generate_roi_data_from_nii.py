import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm
import argparse
from multiprocessing import Pool


def load_nii_files(base_dir, exclude_keyword='mask'):
    datasets = {}
    for root, dirs, _ in os.walk(base_dir):
        for dir_name in dirs:
            dataset_path = os.path.join(root, dir_name)
            nii_files = []
            for file_name in os.listdir(dataset_path):
                if file_name.endswith('.nii.gz') and exclude_keyword not in file_name:
                    file_path = os.path.join(dataset_path, file_name)
                    nii_files.append(file_path)
            
            if len(nii_files) != 0:
                datasets[dir_name] = nii_files

    return datasets

def load_atlas_file(atlas_dir, filename):
    atlas_path = os.path.join(atlas_dir, filename)
    if os.path.exists(atlas_path):
        return nib.load(atlas_path).get_fdata()
    else:
        print(f"File {filename} not found in {atlas_dir}")
        return None

def resize_atlas_labels(atlas_labels, target_shape):
    zoom_factors = [t / a for t, a in zip(target_shape, atlas_labels.shape)]
    resized_labels = zoom(atlas_labels, zoom_factors, order=0)
    return resized_labels.astype(np.int32)

def process_single_fmri_file(args):
    fmri_file, atlases, output_dir, dataset_name = args

    base_name = os.path.basename(fmri_file)
    all_exist = True
    exist_dict = {}
    for atlas_name in atlases.keys():
        output_file = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_{atlas_name}.npy")
        if os.path.exists(output_file):
            print(f"Skipping {output_file} as it already exists.")
            exist_dict[atlas_name] = True
        else:
            exist_dict[atlas_name] = False
            all_exist = False

    if all_exist:
        return

    try:
        fmri_data = nib.load(fmri_file).get_fdata()
        num_frames = fmri_data.shape[-1]
    except:
        print("Failed to load fMRI data: {}".format(fmri_file))
        os.remove(fmri_file)
        return

    for atlas_name, atlas_data in atlases.items():
        if exist_dict[atlas_name]:
            continue

        if atlas_data.shape != fmri_data.shape[:-1]:
            atlas_labels = resize_atlas_labels(atlas_data, fmri_data.shape[:-1])
            atlases[atlas_name] = atlas_labels
        else:
            atlas_labels = atlas_data
        unique_labels = np.unique(atlas_labels)
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background (0)
    
        averages = []
        for frame in range(num_frames):
            frame_data = fmri_data[..., frame]
            frame_averages = []
            for label in unique_labels:
                mask = atlas_labels == label
                if np.any(mask):
                    mean_value = frame_data[mask].mean()
                    frame_averages.append(mean_value)
                else:
                    frame_averages.append(0)
            averages.append(frame_averages)

        averages = np.array(averages).T
        save_averages(averages, fmri_file, atlas_name, output_dir, dataset_name)

def save_averages(averages, fmri_file, atlas_name, output_dir, dataset_name):
    dataset_output_dir = output_dir
    os.makedirs(dataset_output_dir, exist_ok=True)
    base_name = os.path.basename(fmri_file)
    output_file = os.path.join(dataset_output_dir, f"{os.path.splitext(base_name)[0]}_{atlas_name}.npy")
    np.save(output_file, averages)

def process_fmri_files(fmri_files, atlases, output_dir, dataset_name, num_processes):
    args = [
        (fmri_file, atlases, output_dir, dataset_name)
        for fmri_file in fmri_files
    ]

    if num_processes == 1:
        for arg in tqdm(args, desc=f'Processing {dataset_name}'):
            process_single_fmri_file(arg)
    else:
        with Pool(processes=num_processes) as pool:
            list(tqdm(pool.imap(process_single_fmri_file, args), total=len(args), desc=f'Processing {dataset_name}'))

def main():
    parser = argparse.ArgumentParser(description='Process fMRI data with selected atlases and datasets.')
    parser.add_argument('--atlas_names', type=str, nargs='+', required=True, help='Names of the atlases to use')
    parser.add_argument('--dataset_names', type=str, nargs='+', required=True, help='Names of the datasets to process')
    parser.add_argument('--fmri_dir', type=str, required=True, help='Directory containing fMRI data')
    parser.add_argument('--atlas_dir', type=str, required=True, help='Directory containing atlas data')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output files')
    parser.add_argument('--num_processes', type=int, default=1, help='Number of processes to use')
    args = parser.parse_args()

    # Load all atlas files
    atlases = {name: load_atlas_file(args.atlas_dir, name + '.nii.gz') for name in args.atlas_names}

    # Load fMRI datasets
    fmri_datasets = load_nii_files(args.fmri_dir)

    # Process each selected dataset with each selected atlas
    for dataset_name in args.dataset_names:
        if (dataset_name in fmri_datasets) and fmri_datasets[dataset_name]:
            files = fmri_datasets[dataset_name]
            process_fmri_files(files, atlases, args.output_dir, dataset_name, args.num_processes)

if __name__ == '__main__':
    main()
