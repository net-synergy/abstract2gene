"""Compare models and hyperparamaters."""

import argparse
import json
import os

import datasets
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

import example._config as cfg
from abstract2gene.dataset import dataset_generator, mutators
from example._logging import log, set_log

EXPERIMENT = "embedding_model_selection"
n_steps = 100
n_trials = 20

seed = cfg.seeds[EXPERIMENT]
set_log(EXPERIMENT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_steps",
        default=n_steps,
        type=int,
        help="Number of steps per batch.",
    )
    parser.add_argument(
        "--n_trials",
        default=n_trials,
        type=int,
        help="Number of trials to perform per model.",
    )
    args = parser.parse_args()
    n_steps = args.n_steps
    n_trials = args.n_trials


def load_dataset(
    files: list[str],
    batch_size: int,
    n_batches: int,
    mask: str | list[str] | None,
    seed: int,
) -> datasets.Dataset:
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
        dataset = mutators.mask_abstract(dataset, mask, max_cpu=cfg.max_cpu)

    return dataset_generator(
        dataset,
        seed=seed,
        batch_size=batch_size,
        n_batches=n_batches,
    )


def hpo_search_space(trial):
    return {
        "per_device_train_batch_size": trial.suggest_int(
            "per_device_train_batch_size", 8, 64
        ),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0, 0.3),
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-6, 1e-4, log=True
        ),
    }


def hpo_loss_init(model):
    return MultipleNegativesRankingLoss(model)


def hpo_compute_objective(metrics):
    return metrics["eval_cosine_accuracy"]


training_args = SentenceTransformerTrainingArguments(
    output_dir="models",
    fp16=False,
    bf16=True,
    batch_sampler=BatchSamplers.BATCH_SAMPLER,
    eval_strategy="no",
    save_strategy="no",
    logging_dir="logs",
)

dataset_train = load_dataset(cfg.EMBEDDING_TRAIN_FILES, 64, n_steps, 0)
dataset_train = dataset_train.remove_columns("negative")
dataset_test = load_dataset(cfg.TEST_FILES, 64, 50, seed)

evaluator = TripletEvaluator(
    anchors=dataset_test["anchor"],
    positives=dataset_test["positive"],
    negatives=dataset_test["negative"],
)

log("Pre fine-tuning accuracy")
for name, model in cfg.models.items():
    original_model = SentenceTransformer(model)
    log(f"{name}: {evaluator(original_model)["cosine_accuracy"]}")

# ## Select model
log("\nTraining")
for name, model in cfg.models.items():

    def hpo_model_init() -> SentenceTransformer:
        return SentenceTransformer(model)

    print(name)
    trainer = SentenceTransformerTrainer(
        model=None,
        args=training_args,
        train_dataset=dataset_train,
        loss=hpo_loss_init,
        model_init=hpo_model_init,
        evaluator=evaluator,
    )

    best_trial = trainer.hyperparameter_search(
        hp_space=hpo_search_space,
        compute_objective=hpo_compute_objective,
        n_trials=n_trials,
        direction="maximize",
        backend="optuna",
    )

    log(f"{name}: {best_trial.objective}")


## Test winner further.
# After running the above, mpnet and pubmedncl came out as the best model to
# fine-tune.
dataset_train = load_dataset(
    cfg.EMBEDDING_TRAIN_FILES, 64, n_steps * 4, seed + 1
)
dataset_train = dataset_train.remove_columns("negative")
winners = ["MPNet", "PubMedNCL"]
hyperparams: dict[str, dict] = {}

print("\nFurther training")
for name in winners:

    def hpo_winner_init() -> SentenceTransformer:
        return SentenceTransformer(cfg.models[name])

    print(name)
    trainer = SentenceTransformerTrainer(
        model=None,
        args=training_args,
        train_dataset=dataset_train,
        loss=hpo_loss_init,
        model_init=hpo_winner_init,
        evaluator=evaluator,
    )

    best_trial = trainer.hyperparameter_search(
        hp_space=hpo_search_space,
        compute_objective=hpo_compute_objective,
        n_trials=30,
        direction="maximize",
        backend="optuna",
    )
    hyperparams[name] = best_trial.hyperparameters

    log(f"{name}: {best_trial.objective}")
    log("Parameters:")
    for k, v in best_trial.hyperparameters.items():
        log(f"  {k}: {v}")

if not os.path.exists("results"):
    os.mkdir("results")

with open(os.path.join("results", "hyperparameters.json"), "w") as js:
    json.dump(hyperparams, js)
