import os
import time
import torch

def read_all_pt_files(directory):
    data_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".pt"):
            file_path = os.path.join(directory, filename)
            data = torch.load(file_path)
            data_list.append(data)
    return data_list

def save_combined_pt_file(data_list, output_file):
    combined_data = torch.cat(data_list)
    torch.save(combined_data, output_file)

def measure_time(func, *args):
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    return end_time - start_time, result

directory = './data/UKB_MNI_to_TRs_minmax/img/1098228'
combined_file = './data/UKB_MNI_to_TRs_minmax/img/combined.pt'

for i in range(100):
    # Measure time to read all .pt files individually
    time_individual, data_list = measure_time(read_all_pt_files, directory)
    print(f"Time to read all .pt files individually: {time_individual:.4f} seconds")

    # Measure time to read the combined .pt file
    time_combined, combined_data = measure_time(torch.load, combined_file)
    print(f"Time to read the combined .pt file: {time_combined:.4f} seconds")
