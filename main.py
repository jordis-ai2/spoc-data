import gzip
import os
from tqdm import tqdm
import prior
import urllib.request
from typing import Sequence

try:
    from prior import LazyJsonDataset
except:
    raise ImportError("Please update the prior package (pip install --upgrade prior).")


TASK_TYPES = [
    "ObjectNavType",
    "ObjectNavRoom",
    "ObjectNavOpenVocab",
    "ObjectNavRelAttribute",
    "ObjectNavAffordance",
    "ObjectNavLocalRef",
    "RoomNav",
    "PickupType",
    "FetchType",
    "SimpleExploreHouse",
]

def process_and_load_data(split, task_type):
    base_url = (
        "https://prior-datasets.s3.us-east-2.amazonaws.com/vida-benchmark-oct22-closed-oct31trashandplantfixes/"
    )
    # set the open type files as test, everything is _val
    if split == "test":
        filename = f"opentype_minival_Oct22_{task_type.lower()}_val.jsonl.gz"
    elif split == "val":
        filename = f"closedtype_minival_Oct22_fixedlemma_filterasset_{task_type.lower()}_val.jsonl.gz"
    else:
        raise ValueError(f"Unknown split {split}")

    if not os.path.exists(filename):
        try:
            response = urllib.request.urlopen(base_url)
            urllib.request.urlretrieve(
                base_url + filename,
                "./" + filename,
            )
        except urllib.error.URLError as e:
            print(f"Error: {task_type} for {split} not found")
            return []

    with gzip.open(filename, "rt") as f:
        tasks = [line for line in tqdm(f, desc=f"Loading {split}")]

    return tasks


def load_dataset(task_types: Sequence[str] = TASK_TYPES) -> prior.DatasetDict:
    """Load the houses dataset."""
    data = {}

    for split in ["val", "test"]:
        split_task_list = []
        for task in task_types:
            tasks = process_and_load_data(split, task)
            split_task_list.extend(tasks)

        data[split] = LazyJsonDataset(
            data=split_task_list, dataset="vida-benchmark", split=split
        )
    return prior.DatasetDict(**data)
