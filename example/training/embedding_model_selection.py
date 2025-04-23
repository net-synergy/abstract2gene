"""Compare models and search for hyperparamaters.

Performs mini fine-tunings for different models to determine which models can
be trained best for the abstract2gene purposes.

Each model gets repeated trials of fine-tuning to find the best hyperparameters
for fine-tuning that model suing a random search. The single best trial for
each model is compared to find the model that performs the best.

For the top models, a second experiment with more and longer trials is used to
get a better prediction of the hyperparamters that should be used in the final
fine-tuning.
"""

import argparse
import json
import os

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

import example._config as cfg
from example._logging import log, set_log
from example.training._utils import load_dataset, make_seed_generator

EXPERIMENT = "embedding_model_selection"

set_log(EXPERIMENT)
seed = cfg.seeds[EXPERIMENT]
seed_generator = make_seed_generator(seed)

n_steps = 300
n_trials = 20
n_test_steps = 50
save_hyperparameters = True
models = list(cfg.models.keys())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_steps",
        default=n_steps,
        type=int,
        help="Number of steps per batch.",
    )
    parser.add_argument(
        "--n_test_steps",
        default=n_test_steps,
        type=int,
        help="Number of steps per batch.",
    )
    parser.add_argument(
        "--n_trials",
        default=n_trials,
        type=int,
        help="Number of trials to perform per model.",
    )
    parser.add_argument(
        "--models", default=models, nargs="*", help="The models to test."
    )
    parser.add_argument(
        "--save",
        default=save_hyperparameters,
        type=bool,
        help="If true, store the determined hyperparameters in a JSON file.",
    )

    args = parser.parse_args()
    n_steps = args.n_steps
    n_trials = args.n_trials
    models = args.models
    save_hyperparameters = args.save

    for model in models:
        if model not in list(cfg.models.keys()):
            RuntimeError(
                f"{model} not in known models. Make sure it's"
                + " spelled exactly like in the experiments/_config.py"
                + " file or add to the a2g.toml file."
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
    num_train_epochs=1,
    eval_strategy="no",
    save_strategy="no",
    logging_dir="logs/_tmp",
    seed=seed_generator(),
    data_seed=seed_generator(),
)

dataset_train = load_dataset(
    cfg.EMBEDDING_TRAIN_FILES,
    64,
    n_steps,
    labels="gene",
    mask=["gene", "disease"],
    seed_generator=seed_generator,
)
dataset_train = dataset_train["gene"].remove_columns("negative")
dataset_test = load_dataset(
    cfg.TEST_FILES,
    64,
    n_test_steps,
    labels="gene",
    mask=None,
    seed_generator=seed_generator,
)["gene"]

evaluator = TripletEvaluator(
    anchors=dataset_test["anchor"],
    positives=dataset_test["positive"],
    negatives=dataset_test["negative"],
)

## Select model
log("Pre fine-tuning accuracy")
for name, model in cfg.models.items():
    original_model = SentenceTransformer(model)
    log(f"{name}: {evaluator(original_model)["cosine_accuracy"]}")

hyperparams: dict[str, dict] = {}

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
    hyperparams[name] = best_trial.hyperparameters

    log(f"{name}: {best_trial.objective}")
    log("Parameters:")
    for k, v in best_trial.hyperparameters.items():
        log(f"  {k}: {v}")

log("")

if save_hyperparameters:
    if not os.path.exists("results"):
        os.mkdir("results")

    file_name = os.path.join("results", "hyperparameters.json")
    if os.path.exists(file_name):
        with open(file_name, "r") as js:
            new_hyperparams = hyperparams
            hyperparams = json.load(js)
            hyperparams.update(new_hyperparams)

    with open(file_name, "w") as js:
        json.dump(hyperparams, js)
