"""Synchronous Parquet streaming with deterministic buffer shuffling."""

import random
from contextlib import closing
from itertools import islice

from huggingface_hub import HfApi, HfFileSystem
from huggingface_hub.hf_api import RepoFile
import pyarrow.parquet as pq

STREAM_VERSION = "synchronous-parquet-v1"


def parquet_rows(paths, open_file):
    """Read one shard at a time; close both readers even on generator.close()."""
    for path in paths:
        with open_file(path, "rb") as handle:
            with pq.ParquetFile(handle, pre_buffer=False) as parquet:
                for batch in parquet.iter_batches(batch_size=256, use_threads=False):
                    yield from batch.to_pylist()


def shuffled_rows(rows, seed, buffer_size):
    if buffer_size < 1:
        raise ValueError("shuffle_buffer must be positive.")
    rng = random.Random(seed)
    with closing(rows):
        buffer = list(islice(rows, buffer_size))
        for row in rows:
            index = rng.randrange(len(buffer))
            candidate, buffer[index] = buffer[index], row
            yield candidate
        rng.shuffle(buffer)
        yield from buffer


def stream_documents(dataset, subset, revision, seed, buffer_size):
    """Read only train shards of the two supported, pinned source layouts."""
    if dataset == "wikimedia/wikipedia" and subset == "20231101.sr":
        prefix = subset
    elif dataset == "HuggingFaceFW/fineweb-2" and subset == "srp_Cyrl":
        prefix = f"data/{subset}/train"
    else:
        raise ValueError(f"Unsupported Parquet source: {dataset}/{subset}")
    files = sorted(
        entry.path
        for entry in HfApi().list_repo_tree(
            dataset, path_in_repo=prefix, revision=revision, repo_type="dataset"
        )
        if isinstance(entry, RepoFile)
        and entry.path.endswith(".parquet")
        and (dataset != "wikimedia/wikipedia" or entry.path.rsplit("/", 1)[-1].startswith("train-"))
    )
    if not files:
        raise ValueError(f"No train Parquet shards found for {dataset}/{subset}@{revision}")
    random.Random(seed).shuffle(files)
    paths = [f"datasets/{dataset}@{revision}/{path}" for path in files]
    filesystem = HfFileSystem(block_size=4 * 1024 * 1024)
    with closing(shuffled_rows(parquet_rows(paths, filesystem.open), seed, buffer_size)) as rows:
        yield from rows
