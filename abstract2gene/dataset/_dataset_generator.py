"""Convert labeled abstracts to anchor-positive-negative triplets.

Training methods need a dataset with keys: "Anchor", "Positive", "Negative" but
the original dataset is not in this form and putting it in this form would
require coupling abstracts together.
"""

__ALL__ = ["dataset_generator"]

import numpy as np
from datasets import Dataset, Features, Value

from . import mutators
from ._utils import lol_to_csc


def dataset_generator(
    dataset: Dataset,
    label: str = "gene",
    max_labels: int = 10,
    batch_size: int = 32,
    n_batches: int = 100,
    augment_label: float = 0,
    seed: int = 0,
    sep_token: str = "[SEP]",
) -> Dataset:
    """Convert labeled abstracts to anchor-positive-negative triplets.

    Does not produce hard-negatives. Negatives a randomly sampled from any
    other label than the anchor.

    Max labels is the maximum number of labels an individual sample can be
    given. As the number of samples increase it losses specificity for that
    label and can't be seen as a positive example.

    Additionally unlabeled data is dropped since it cannot have a positive
    example without a label.

    """
    rng = np.random.default_rng(seed=seed)
    if augment_label > 0:
        dataset = mutators.augment_labels(
            dataset, "gene", augment_label, rng.integers(0, 9999, (1,))[0]
        )

    labels = lol_to_csc(dataset[label])

    overlabeled = labels.sum(axis=1) > max_labels
    unlabeled = labels.sum(axis=1) == 0
    sample_mask = np.logical_not(np.logical_or(unlabeled, overlabeled))
    sub_ds = dataset.select(np.arange(len(dataset))[sample_mask])
    samples = [
        title + sep_token + abstract
        for title, abstract in zip(sub_ds["title"], sub_ds["abstract"])
    ]
    labels = labels[sample_mask, :]
    labels = labels[:, labels.sum(axis=0) > 2]
    probs = np.log(labels.sum(axis=0))
    probs = probs / probs.sum()

    if batch_size > labels.shape[1]:
        RuntimeError(
            "Not enough unique labels to generate a full batch."
            + f" Lower batch size to less than {labels.shape[1]}"
        )

    n_labels = labels.shape[1]

    anchors = []
    positives = []
    negatives = []
    for batch_i in range(n_batches):
        mix_idx = rng.permuted(np.arange(len(samples)))
        batch_samps = [samples[i] for i in mix_idx]
        batch_labels = labels[mix_idx, :]
        for i, label in enumerate(rng.choice(n_labels, batch_size, p=probs)):
            label_mask = batch_labels[:, [label]].toarray().squeeze()
            positive_idx = np.arange(len(batch_samps))[label_mask]
            negative_idx = np.arange(len(batch_samps))[
                np.logical_not(label_mask)
            ]

            anchors.append(batch_samps[positive_idx[0]])
            positives.append(batch_samps[positive_idx[1]])
            negatives.append(
                batch_samps[negative_idx[i % np.logical_not(label_mask).sum()]]
            )

    feats = Features(
        {k: Value(dtype="string") for k in ["anchor", "positive", "negative"]}
    )
    return Dataset.from_dict(
        {"anchor": anchors, "positive": positives, "negative": negatives},
        features=feats,
    )
