import h5py
import numpy as np
import json
from huggingface_hub import hf_hub_download
import os

def preprocessing(repo_id, dataset_filename, train_split_percentage = 0.8):
    print("Downloading dataset")
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    print(f"Directory {data_dir} created.")
    dataset_path = hf_hub_download(
        repo_id=repo_id, 
        filename=dataset_filename, 
        repo_type="dataset", 
        local_dir=data_dir
    )

    num_sims = 256
    train_split = int(train_split_percentage * num_sims)
    train_idx = list(range(0, train_split))
    test_idx = list(range(train_split, num_sims))

    print("Calculating minimum and maximum values for normalization")
    global_min, global_max = float("inf"), float("-inf")
    dataset_path = "data/valid/KolmFlow_valid_256.h5"
    with h5py.File(dataset_path, "r") as f:
        u_data = f["valid/u"]

        for sim_id in train_idx:
            data_sim = u_data[sim_id][:]

            local_min = np.nanmin(data_sim)
            local_max = np.nanmax(data_sim)

            if local_min < global_min:
                global_min = local_min

            if local_max > global_max:
                global_max = local_max

    preprocessing_results = {
        "dataset_path": dataset_path,
        "g_min": float(global_min),
        "g_max": float(global_max),
        "train_idx": train_idx,
        "test_idx": test_idx
    }

    with open("preprocessing_info.json", "w") as f:
        json.dump(preprocessing_results, f)
    
    print(f"Preprocessing ready training Min: {global_min}, training Max: {global_max}, train_idx: {len(train_idx)}")

if __name__ == "__main__":
    repo_id = "ayz2/temporal_pdes"
    dataset_filename = "valid/KolmFlow_valid_256.h5"
    train_split = 0.8
    preprocessing(repo_id=repo_id, dataset_filename=dataset_filename, train_split_percentage=train_split)