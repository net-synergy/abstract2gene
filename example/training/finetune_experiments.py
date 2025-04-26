"""Experiment on fine-tuning the best performing encoding models.

Collects the chosen models from the embedding model selection test by reading
in the resulting hyperparameters file and performing the following fine
turnings for each of the models tracked in that file.

Masking is performed on the dataset (as well as in the selection experiment) to
ensure the model isn't relying on the name of genes (or diseases) to determine
the relevant gene but is instead learning actual patterns in the text.

Experiments
-----------
Four separate experiments are perform in this file to determine the best
approach for fine-tuning.

Experiment 1: Control (No masking).
  Fine tune a model without masking genes or disease. Used as a baseline to
  compare performance of masked models. Otherwise trained exactly like
  experiment 1.

Experiment 1: Predict Genes Masking Genes only.
  Fine tune using only a dataset with gene labels. The dataset's abstracts have
  their gene symbols masked (based on PubTator3's annotations).

Experiment 2: Predict Genes Masking Genes and Disease.
  Like experiment 1, but in addition to masking genes, mask diseases. This is
  intended to prevent the model from learning the strong relationship between
  diseases and the set of genes that are frequently studied with them. Without
  this additional masking there is a fear the model is using disease name as a
  proxy for related genes.

Experiment 3: Alternate Predicting Genes and Disease Masking Both.
  Use two datasets for training. Both datasets have genes and diseases masked
  from their abstracts. One dataset uses genes as labels the other uses disease
  as labels. Since training plateaus when using only genes for labels, try
  training with a related to task after the plateaus to see if it can push the
  weights in the right direction. Training is broken into three phases: initial
  gene training, disease training, and final gene training. Switch to disease
  after gene training plateaus, then back to gene to hone it into the correct
  task again.

Experiment 4: Predict Genes Masking and Permuting Genes
  Instead of using only masking, permute some percent (25%) of genes. All genes
  are corrupted. But some are corrupted by replacing them with another gene at
  random. Previous experiments show training with masked abstracts causes
  performance on unmasked abstracts to degrade. This intends on showing the
  model genes it is used to them, but, make them uninformative so it doesn't
  try to use them in predictions.

All experiments use the same number of total training steps and are evaluated
on the same evaluators (gene labels + no masking and gene labels + masking gene
and disease) to ensure fair comparisons. Models already trained on scientific
literature have likely already developed an understanding of gene names. To
improve generalizing to abstracts without gene symbols, the goal is to remove
the models reliance on gene symbols for predicting. The goal of this
fine-tuning is not only to improve the similarity between the embeddings
generated for abstracts with the same genes but also to get the accuracy the
same between masked and unmasked abstracts (gene symbols and disease symbols
are no longer helping the model guess the genes).

"""

import argparse
import json
import os
from typing import Sequence

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import (
    SentenceEvaluator,
    SequentialEvaluator,
    TripletEvaluator,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

import example._config as cfg
from abstract2gene.data import encoder_path
from example._logging import log, set_log
from example.training._utils import load_dataset, make_seed_generator

LOG_PATH = "logs/"
EXPERIMENT = "finetune_experiments"

set_log(EXPERIMENT)
seed = cfg.seeds[EXPERIMENT]
seed_generator = make_seed_generator(seed)


def finetune(
    model_name: str,
    experiment_name: str,
    train_dataset: Dataset | Sequence[Dataset],
    test_dataset: Dataset | dict[str, Dataset],
    batch_size: int,
    learning_rate: float,
    warmup_ratio: float,
    seed: int,
    data_seed: int,
):
    model = SentenceTransformer(cfg.models[model_name])
    loss = MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=f"models/{model_name}_{experiment_name}",
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        fp16=False,
        bf16=True,
        num_train_epochs=1,
        batch_sampler=BatchSamplers.BATCH_SAMPLER,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=250,
        logging_steps=250,
        logging_first_step=True,
        logging_dir=f"{LOG_PATH}/{model_name}/{experiment_name}",
        seed=seed,
        data_seed=data_seed,
    )

    if isinstance(test_dataset, Dataset):
        evaluator: SentenceEvaluator = TripletEvaluator(
            anchors=test_dataset["anchor"],
            positives=test_dataset["positive"],
            negatives=test_dataset["negative"],
            batch_size=batch_size,
        )
        log(f"Initial accuracy: {evaluator(model)["cosine_accuracy"]}")
    else:
        evaluator = SequentialEvaluator(
            [
                TripletEvaluator(
                    anchors=ds["anchor"],
                    positives=ds["positive"],
                    negatives=ds["negative"],
                    batch_size=batch_size,
                    name=k,
                )
                for k, ds in test_dataset.items()
            ],
        )
        log("Initial accuracy:")
        for k in test_dataset:
            log(f"  {k}: {evaluator(model)[k + "_cosine_accuracy"]}")

    if isinstance(train_dataset, Dataset):
        train_dataset = [train_dataset]

    for ds in train_dataset:
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=ds,
            loss=loss,
            evaluator=evaluator,
        )

        trainer.train()

    if isinstance(test_dataset, Dataset):
        log(f"Final eval: {evaluator(model)["cosine_accuracy"]}")
    else:
        log("Final eval:")
        for k in test_dataset:
            log(f"  {k}: {evaluator(model)[k + "_cosine_accuracy"]}")

    model.save_pretrained(
        encoder_path(f"{model_name}-{experiment_name}-abstract2gene")
    )


if __name__ == "__main__":
    if not os.path.exists(os.path.join("results", "hyperparameters.json")):
        raise RuntimeError("No hyperparameters found. Run model selection.")

    with open(os.path.join("results", "hyperparameters.json"), "r") as js:
        hyperparams = json.load(js)

    models = list(hyperparams.keys())
    experiments = [1, 2, 3, 4, 5]
    n_steps = 10_000
    n_test_steps = 100

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
    parser.add_argument(
        "--models", default=models, nargs="*", help="The models to finetune."
    )
    parser.add_argument(
        "--experiments",
        default=experiments,
        type=int,
        nargs="*",
        help="Which experiments to run.",
    )

    args = parser.parse_args()
    n_steps = args.n_steps
    n_test_steps = args.n_test_steps
    models = args.models
    experiments = args.experiments

    for model in models:
        if model not in list(hyperparams.keys()):
            raise RuntimeError(f"No hyperparameters stored for {model}.")

    for experiment in experiments:
        if experiment > 5:
            raise RuntimeError("Experiments should be less than 5.")

    for model_name in models:
        params = hyperparams[model_name]
        model = SentenceTransformer(cfg.models[model_name])

        log(f"Training {model_name}:")
        batch_size = params["per_device_train_batch_size"]
        warmup_ratio = params["warmup_ratio"]
        learning_rate = params["learning_rate"]

        test_dataset = {
            "unmasked": load_dataset(
                cfg.TEST_FILES,
                model,
                batch_size=batch_size,
                n_batches=n_test_steps,
                mask=None,
                labels="gene",
                seed_generator=seed_generator,
            )["gene"],
            "genes_masked": load_dataset(
                cfg.TEST_FILES,
                model,
                batch_size=batch_size,
                n_batches=n_test_steps,
                mask="gene",
                labels="gene",
                seed_generator=seed_generator,
            )["gene"],
            "genes_and_disease_masked": load_dataset(
                cfg.TEST_FILES,
                model,
                batch_size=batch_size,
                n_batches=n_test_steps,
                mask=["gene", "disease"],
                labels="gene",
                seed_generator=seed_generator,
            )["gene"],
        }

        if 1 in experiments:
            log("\nExperiment 1:")
            train_dataset = load_dataset(
                cfg.EMBEDDING_TRAIN_FILES,
                model,
                batch_size=batch_size,
                n_batches=n_steps,
                mask=None,
                labels="gene",
                seed_generator=seed_generator,
            )["gene"]

            finetune(
                model_name,
                "unmasked",
                train_dataset,
                test_dataset,
                batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_ratio=warmup_ratio,
                seed=seed_generator(),
                data_seed=seed_generator(),
            )

        if 2 in experiments:
            log("\nExperiment 2:")
            train_dataset = load_dataset(
                cfg.EMBEDDING_TRAIN_FILES,
                model,
                batch_size=batch_size,
                n_batches=n_steps,
                mask="gene",
                labels="gene",
                seed_generator=seed_generator,
            )["gene"]

            finetune(
                model_name,
                "gene_only",
                train_dataset,
                test_dataset,
                batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_ratio=warmup_ratio,
                seed=seed_generator(),
                data_seed=seed_generator(),
            )

        if 3 in experiments:
            log("\nExperiment 3:")
            train_dataset = load_dataset(
                cfg.EMBEDDING_TRAIN_FILES,
                model,
                batch_size=batch_size,
                n_batches=n_steps,
                mask=["gene", "disease"],
                labels="gene",
                seed_generator=seed_generator,
            )["gene"]

            finetune(
                model_name,
                "gene_and_disease",
                train_dataset,
                test_dataset,
                batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_ratio=warmup_ratio,
                seed=seed_generator(),
                data_seed=seed_generator(),
            )

        if 4 in experiments:
            log("\nExperiment 4:")
            train_dataset = load_dataset(
                cfg.EMBEDDING_TRAIN_FILES,
                model,
                batch_size=batch_size,
                n_batches=n_steps // 3,
                mask=["gene", "disease"],
                labels=["gene", "disease"],
                seed_generator=seed_generator,
            )

            finetune(
                model_name,
                "multi_phase",
                (
                    train_dataset["gene"],
                    train_dataset["disease"],
                    train_dataset["gene"],
                ),
                test_dataset,
                batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_ratio=warmup_ratio,
                seed=seed_generator(),
                data_seed=seed_generator(),
            )

        if 5 in experiments:
            log("\nExperiment 5:")
            train_dataset = load_dataset(
                cfg.EMBEDDING_TRAIN_FILES,
                model,
                batch_size=batch_size,
                n_batches=n_steps,
                mask="gene",
                permute_prob=0.25,
                labels="gene",
                seed_generator=seed_generator,
            )["gene"]

            finetune(
                model_name,
                "permute",
                train_dataset,
                test_dataset,
                batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_ratio=warmup_ratio,
                seed=seed_generator(),
                data_seed=seed_generator(),
            )
