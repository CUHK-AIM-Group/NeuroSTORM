import os
import pandas as pd
from tqdm import tqdm

folder_path = './data/GOD_clean'
output_file = './data/GOD_clean/god_label.csv'

combined_df = pd.DataFrame(columns=['subject_id', 'class', 'onset', 'duration'])

file_list = [file_name for file_name in os.listdir(folder_path) if 'perception' in file_name and file_name.endswith('.tsv')]

for file_name in tqdm(file_list, desc="Processing files"):
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path, sep='\t')
    df_filtered = df[df['event_type'] == 'stimulus'].copy()
    subject_id_prefix = file_name[:6] + '_' + file_name[-17:-11] + '_'
    df_filtered.loc[:, 'subject_id'] = subject_id_prefix + df_filtered['trial_no'].astype(str)
    df_filtered = df_filtered[['subject_id', 'category_index', 'onset', 'duration']]
    df_filtered.columns = ['subject_id', 'class', 'onset', 'duration']
    df_filtered['class'] = df_filtered['class'].astype(int)
    combined_df = pd.concat([combined_df, df_filtered], ignore_index=True)

combined_df.to_csv(output_file, index=False)
