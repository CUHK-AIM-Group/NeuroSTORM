'''
This script downloads data from the Human Connectome Project - 1200 subjects release.
'''
# Import packages
import argparse
import os
import boto3
from boto3.s3.transfer import TransferConfig
import pickle

# Make module executable
from tqdm import tqdm


SERIES_MAP = {}


def length_transform(length):
    if length > 1024 * 1024 * 1024:
        print("length:{:.2f}GB".format(length / 1024 / 1024 / 1024))
    elif length > 1024 * 1024:
        print("length:{:.2f}MB".format(length / 1024 / 1024))
    elif length > 1024:
        print("length:{:.2f}KB".format(length / 1024))
    else:
        print("length:{}B".format(length))


class download_process():
    def __init__(self, length):
        self.length = length
        self.bar = tqdm(range(int(length / 1024)))

    def precess(self, chunk):
        self.bar.update(int(chunk / 1024))

    def close(self):
        self.bar.close()


def download_prs(chunk):
    print(chunk)


def process_subjects(subjects, bucket, config):
    tbar = tqdm(total=len(subjects))

    for subject_id in subjects:
        for object in bucket.objects.filter(Prefix='{}/{}/'.format(s3_prefix, subject_id)):
            l = len('HCP_1200/{}/'.format(subject_id))
            if l >= len(object.key):
                continue
            filename = object.key[l:]

            SERIES_MAP['fmri_1'] = 'MNINonLinear/T1w.nii.gz'
            SERIES_MAP['fmri_2'] = 'MNINonLinear/T2w.nii.gz'

            if filename in SERIES_MAP.values():
                output_name = os.path.join(out_dir, '{}_{}.nii.gz'.format(subject_id, filename.split('/')[-1]))
                if not os.path.exists(output_name):
                    bucket.download_file(object.key, output_name, Config=config)

        print('subject {} is downloaded'.format(subject_id))
        tbar.update(1)

    tbar.close()


def run_process(single_process_func, subjects, bucket, config, cpu_worker_num=16):
    from multiprocessing import Process

    item_count = len(subjects)
    interval = item_count / cpu_worker_num
    process_pool = []

    for i in range(cpu_worker_num):
        start_index = int(i*interval)
        end_index = int((i+1)*interval)
        if i == cpu_worker_num - 1:
            process_pool.append(Process(target=single_process_func, args=(subjects[start_index:], bucket, config,)))
            print('Process {}: {} - end'.format(i, start_index))
        else:
            process_pool.append(Process(target=single_process_func, args=(subjects[start_index:end_index], bucket, config,)))
            print('Process {}: {} - {}'.format(i, start_index, end_index))
    [p.start() for p in process_pool]
    [p.join() for p in process_pool]
    [p.close() for p in process_pool]


if __name__ == '__main__':
    # Init arparser
    parser = argparse.ArgumentParser(description=__doc__)

    # Required arguments
    parser.add_argument('--id', type=str)
    parser.add_argument('--key', type=str)
    parser.add_argument('--out_dir', required=True, type=str, help='Path to local folder to download files to')
    parser.add_argument('--save_subject_id', action='store_true')
    parser.add_argument('--cpu_worker', type=int, default=1)

    args = parser.parse_args()

    if not args.save_subject_id:
        with open('all_pid.pkl', 'rb') as f:
            subjects = pickle.load(f)

    out_dir = os.path.abspath(args.out_dir)

    s3_bucket_name = 'hcp-openaccess'
    s3_prefix = 'HCP_1200'
    aws_access_key_id = args.id
    aws_secret_access_key = args.key
    s3 = boto3.resource('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    bucket = s3.Bucket(s3_bucket_name)

    GB = 1024 ** 3
    config = TransferConfig(max_concurrency=500, multipart_threshold=int(0.01 * GB), multipart_chunksize=int(0.01 * GB))

    if args.save_subject_id:
        import csv
        
        subjects = set()
        csv_name = os.path.join('hcp.csv')
        with open(csv_name, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            row = next(csv_reader)

            for row in csv_reader:
                subject_id = row[0]
                release = row[1]
                subjects.add(subject_id)

                # if not os.path.exists(out_dir + '/' + subject_id):
                #     print('Could not find %s, creating now...' % out_dir + '/' + subject_id)
                #     os.makedirs(out_dir + '/' + subject_id)
        
        subjects = list(subjects)
        with open('all_pid.pkl', 'wb') as f:
            pickle.dump(subjects, f, pickle.HIGHEST_PROTOCOL)
    else:
        if args.cpu_worker == 1:
            process_subjects(subjects, bucket, config)
        elif args.cpu_worker > 1:
            run_process(process_subjects, subjects, bucket, config, args.cpu_worker)
    