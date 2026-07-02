"""MQAR data generation.

Copied (near-verbatim) from `zoology/data/multiquery_ar.py`. Each example places
`num_kv_pairs` key->value bindings up front, then queries a subset of those keys; the
model must recall the matching value. Non-query positions are filled with random
distractor tokens. Labels are `-100` (ignored by the loss/metric) except at answer
positions. The large vocabulary (8192) is, per the paper, important for separating
architectures.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from .config import MQARTaskConfig


def multiquery_ar(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    power_a: float = 0.01,
    num_kv_pairs: int = 8,
    num_passes: int = 1,
    random_non_queries: bool = True,
):
    """Generate (inputs, labels) for the multi-query associative recall task.

    Verbatim port of zoology.data.multiquery_ar.multiquery_ar (HazyResearch/zoology).
    """
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    assert num_kv_pairs * 2 * num_passes + num_kv_pairs * 2 <= input_seq_len

    np.random.seed(seed)

    context_size = num_kv_pairs * 2 * num_passes

    # keys / values are drawn from disjoint halves of the vocabulary
    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    keys_unshuffled = np.tile(key_choices, (num_examples, 1))
    keys = np.apply_along_axis(np.random.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs)

    values_unshuffled = np.tile(value_choices, (num_examples, 1))
    values = np.apply_along_axis(np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs)

    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values
    kvs = np.tile(kvs, (1, num_passes))

    # power-law gap distribution between a key-value pair and its query
    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a - 1)
    p = p / p.sum()

    x = np.stack([np.arange(space, dtype=int)] * num_examples)
    gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs)

    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, (gaps * 2), values=keys, axis=1)
    examples = np.concatenate([kvs, queries], axis=1)

    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=values, axis=1)

    inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])

    if random_non_queries:
        inputs[inputs == 0] = torch.randint(vocab_size, size=inputs.shape)[inputs == 0]

    return inputs, labels


def build_dataloaders(task: MQARTaskConfig, seed: int, batch_size: int, test_batch_size: int,
                      drop_last: bool = False):
    """Build reproducible train/test loaders with disjoint train/test seeds.

    Returns (train_dl, test_dl, fingerprint) where `fingerprint` is an md5 of the test
    set, so dataset identity is verifiable across machines.

    `drop_last` drops the ragged final *train* batch so every train step has the same shape
    -- required for torch.compile's CUDA-graph (`reduce-overhead`) capture.
    """
    MAX_SEED = 2 ** 32
    rng = np.random.RandomState(seed)
    train_seed = int(rng.randint(0, MAX_SEED // 2))
    test_seed = int(rng.randint(MAX_SEED // 2, MAX_SEED))

    train_inputs, train_labels = multiquery_ar(
        vocab_size=task.vocab_size, num_examples=task.num_train_examples,
        input_seq_len=task.input_seq_len, seed=train_seed,
        power_a=task.power_a, num_kv_pairs=task.num_kv_pairs,
        random_non_queries=task.random_non_queries,
    )
    test_inputs, test_labels = multiquery_ar(
        vocab_size=task.vocab_size, num_examples=task.num_test_examples,
        input_seq_len=task.input_seq_len, seed=test_seed,
        power_a=task.power_a, num_kv_pairs=task.num_kv_pairs,
        random_non_queries=task.random_non_queries,
    )

    fingerprint = hashlib.md5(test_inputs.numpy().tobytes() + test_labels.numpy().tobytes()).hexdigest()

    train_ds = torch.utils.data.TensorDataset(train_inputs, train_labels)
    test_ds = torch.utils.data.TensorDataset(test_inputs, test_labels)

    g = torch.Generator().manual_seed(seed)  # reproducible shuffling
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                           generator=g, drop_last=drop_last)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=test_batch_size, shuffle=False)
    return train_dl, test_dl, fingerprint


def build_eval_dataloader(task: MQARTaskConfig, seed: int, batch_size: int):
    """Test-set-only loader for a post-training generalization eval.

    The eval seed is derived from (seed, input_seq_len, num_kv_pairs), so every grid cell gets a
    distinct, reproducible test set. Generation runs under a forked torch RNG because
    `random_non_queries` draws from the *global* torch generator -- without the fork the test set
    would depend on how much RNG state training consumed, and would differ across runs.

    Returns (test_dl, fingerprint).
    """
    digest = hashlib.md5(f"eval-{seed}-{task.input_seq_len}-{task.num_kv_pairs}".encode()).digest()
    eval_seed = int.from_bytes(digest[:4], "little")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(eval_seed)
        inputs, labels = multiquery_ar(
            vocab_size=task.vocab_size, num_examples=task.num_test_examples,
            input_seq_len=task.input_seq_len, seed=eval_seed,
            power_a=task.power_a, num_kv_pairs=task.num_kv_pairs,
            random_non_queries=task.random_non_queries,
        )

    fingerprint = hashlib.md5(inputs.numpy().tobytes() + labels.numpy().tobytes()).hexdigest()
    test_ds = torch.utils.data.TensorDataset(inputs, labels)
    return torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False), fingerprint
