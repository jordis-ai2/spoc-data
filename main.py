import gzip
import json
import os
import urllib
import urllib.request

import prior
from filelock import FileLock

# See prior-datasets S3 bucket
ANNOTATIONS_URL = "https://prior-datasets.s3.us-east-2.amazonaws.com/objaverse-thor/object-annotations/objaverse_plus_thor_subset_ai_merged__2023-10-26.json.gz"
CLOSEST_MAPPING_URL = "https://prior-datasets.s3.us-east-2.amazonaws.com/objaverse-thor/object-annotations/closest_objaverse_plus_mapping_2023_09_06.json.gz"
SYNSET_TO_BEST_LEMMA_URL = "https://prior-datasets.s3.us-east-2.amazonaws.com/objaverse-thor/object-annotations/wn2022_synset_to_best_lemma_2023-10-26.json.gz"


def load_dataset(which_dataset: str = "annotations") -> prior.DatasetDict:
    if which_dataset == "annotations":
        dataset_url = ANNOTATIONS_URL
    elif which_dataset == "closest_mapping":
        dataset_url = CLOSEST_MAPPING_URL
    elif which_dataset == "synset_to_best_lemma":
        dataset_url = SYNSET_TO_BEST_LEMMA_URL
    else:
        raise ValueError(f"Unknown dataset {which_dataset}")

    file_name = dataset_url.split("/")[-1]
    with FileLock(file_name + ".lock"):
        if not os.path.exists(file_name):
            try:
                urllib.request.urlretrieve(dataset_url, file_name)
            except:
                try:
                    os.remove(file_name)
                except:
                    pass
                raise

    with gzip.open(file_name, "r") as f:
        thor_ds = json.load(f)

    data = prior.Dataset(data=thor_ds, dataset="objaverse-plus", split="train")
    return prior.DatasetDict(data)
