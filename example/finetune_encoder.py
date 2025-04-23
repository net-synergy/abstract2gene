import argparse
import json
import os

import datasets
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

import example._config as cfg
from abstract2gene.data import encoder_path
from abstract2gene.dataset import dataset_generator, mutators

CHKPT_PATH = "models/"

seed = cfg.seeds["finetune_encoder"]


def load_dataset(
    files: list[str],
    batch_size: int,
    n_batches: int,
    mask: str | list[str] | None,
    labels: str | list[str],
    seed_generator: rng,
    permute_prob: float = 0,
) -> dict[str, Dataset]:
    dataset = datasets.load_dataset(
        f"{cfg.hf_user}/pubtator3_abstracts", data_files=files
    )["train"]

    log("Converting genes to human orthologs:")
    log("  Before conversion:")
    log(f"    {len(dataset.features["gene"].feature.names)} unique genes")
    log(f"    {len([g for gs in dataset["gene"] for g in gs])} total genes")
    dataset = mutators.translate_to_human_orthologs(
        dataset, max_cpu=cfg.max_cpu
    )
    log("  After conversion:")
    log(f"    {len(dataset.features["gene"].feature.names)} unique genes")
    log(f"    {len([g for gs in dataset["gene"] for g in gs])} total genes")
    log("")

    if mask is not None:
        dataset = mutators.mask_abstract(
            dataset,
            mask,
            permute_prob,
            seed=seed_generator(),
            max_cpu=cfg.max_cpu,
        )

    if isinstance(labels, str):
        labels = [labels]

    return {
        lab: dataset_generator(
            dataset,
            seed=seed_generator(),
            batch_size=batch_size,
            n_batches=n_batches,
        )
        for lab in labels
    }


def finetune(
    model_name: str,
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    learning_rate: float,
    warmup_ratio: float,
):
    model = SentenceTransformer(cfg.models[model_name])
    loss = MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=f"models/{model_name}",
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        fp16=False,
        bf16=True,
        batch_sampler=BatchSamplers.BATCH_SAMPLER,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        logging_steps=50,
        logging_first_step=True,
        logging_dir="logs",
    )

    evaluator = TripletEvaluator(
        anchors=test_dataset["anchor"],
        positives=test_dataset["positive"],
        negatives=test_dataset["negative"],
        batch_size=batch_size,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    trainer.train()
    model.save_pretrained(encoder_path(f"{model_name}-abstract2gene"))


if __name__ == "__main__":
    if not os.path.exists(os.path.join("results", "hyperparameters.json")):
        raise RuntimeError("No hyperparameters found. Run model selection.")

    with open(os.path.join("results", "hyperparameters.json"), "r") as js:
        hyperparams = json.load(js)

    n_steps = 10_000
    n_test_steps = 50
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_steps",
        default=n_steps,
        type=int,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--n_test_steps",
        default=n_test_steps,
        type=int,
        help="Number of test steps.",
    )
    args = parser.parse_args()
    n_steps = args.n_steps
    n_test_steps = args.n_test_steps

    for model_name, params in hyperparams:
        batch_size = params["per_device_train_batch_size"]
        warmup_ratio = params["warmup_ratio"]
        learning_rate = params["learning_rate"]

        train_dataset = load_dataset(
            cfg.EMBEDDING_TRAIN_FILES,
            batch_size=batch_size,
            n_batches=n_steps,
            seed=seed,
        )

        test_dataset = load_dataset(
            cfg.TEST_FILES,
            batch_size=batch_size,
            n_batches=n_test_steps,
            seed=seed + 1,
        )

        finetune(
            model_name,
            train_dataset,
            test_dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
        )
