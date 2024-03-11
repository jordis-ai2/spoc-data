import gzip
import json
import os
from tqdm import tqdm
import prior
import urllib.request
from typing import Sequence

try:
    from prior import LazyJsonDataset
except:
    raise ImportError("Please update the prior package (pip install --upgrade prior).")


def process_and_load_data(split):
    base_url = "https://prior-datasets.s3.us-east-2.amazonaws.com/vida-objaverse-asset-labelmismatch-ref/"
    filename = f"bad_scene_ids_{split}.jsonl.gz"

    if not os.path.exists(filename):
        try:
            response = urllib.request.urlopen(base_url)
            urllib.request.urlretrieve(
                base_url + filename,
                "./" + filename,
            )
        except urllib.error.URLError as e:
            print(f"Error: {split} not found")
            return []

    reconstructed_data = {}
    with gzip.open(filename, "rt") as f:
        for line in f:
            data = json.loads(line)
            for key, value in data.items():
                reconstructed_data[int(key)] = json.dumps(value)

    max_index = max(reconstructed_data.keys())
    result_list = [
        reconstructed_data.get(i, "null") for i in range(0, max_index + 1)
    ]
    return result_list


def load_dataset() -> prior.DatasetDict:
    """Load the houses dataset."""
    data = {}

    for split in ["train", "val", "test"]:
        asset_id_data = process_and_load_data(split)

        data[split] = LazyJsonDataset(
            data=asset_id_data, dataset="vida-additional-references", split=split
        )
    return prior.DatasetDict(**data)
