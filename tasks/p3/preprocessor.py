import json
import os

from tasks.p3.p3 import P3_TASK_LIST
from datasets import load_dataset
from tqdm import tqdm


def write_file(split, task_name, dataset):
    if not os.path.exists(os.path.join(save_dir, task_name)):
        os.mkdir(os.path.join(save_dir, task_name))
    write_file_path = os.path.join(save_dir, task_name, split+".json")

    write_file = open(write_file_path, "w")
    for item in dataset:
        line=json.dumps(item)
        write_file.write(line + "\n")
    write_file.close()


save_dir = "/dataset/fd5061f6/yanan/data/preprocessed_t0"
data_dir="/dataset/fd5061f6/yanan/data/P3/data"

t0_task_names = P3_TASK_LIST
print(len(t0_task_names))
for task_name in tqdm(t0_task_names):

    dataset_dict = load_dataset("./tasks/p3/p3.py", task_name, cache_dir=data_dir)
    print(dataset_dict)

    if "train" in dataset_dict:
        write_file("train", task_name, dataset_dict["train"])

    if "validation" in dataset_dict:
        write_file("validation", task_name, dataset_dict["validation"])

    if "test" in dataset_dict:
        write_file("test", task_name, dataset_dict["test"])