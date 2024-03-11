import gzip
import os
from tqdm import tqdm
import prior
import urllib.request

try:
    from prior import LazyJsonDataset
except:
    raise ImportError("Please update the prior package (pip install --upgrade prior).")


def load_dataset() -> prior.DatasetDict:
    """Load the houses dataset."""
    base_url = (
        "https://prior-datasets.s3.us-east-2.amazonaws.com/vida-RL-training-data/"
    )
    data = {}
    for split, size in zip(("train", "val"), (65395, 1804)):
        if split == "train":
            filename = "PickupType_train.jsonl.gz"
        else:
            filename = "PickupType_val.jsonl.gz"
        if not os.path.exists(filename):
            try:
                response = urllib.request.urlopen(base_url)
                urllib.request.urlretrieve(
                    base_url + filename,
                    "./" + filename,
                    )
            except urllib.error.URLError as e:
                print(f"Error: PickupType for {split} not found")
                return []

        with gzip.open(filename, "rt") as f:
            tasks = [line for line in tqdm(f, desc=f"Loading {split}")]
        data[split] = LazyJsonDataset(data=tasks, dataset="PickupType", split=split)
    return prior.DatasetDict(**data)
