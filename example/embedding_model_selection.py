"""Compare models and hyperparamaters."""

import datasets
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from abstract2gene.dataset import dataset_generator, mutators
from example import config as cfg

CHKPT_PATH = "models/"
N_STEPS = 100
N_TRIALS = 20


def load_dataset(
    files: list[str], batch_size: int, n_batches: int, seed: int
) -> datasets.Dataset:
    dataset = datasets.load_dataset(
        "dconnell/pubtator3_abstracts", data_files=files
    )["train"]
    dataset = mutators.mask_abstract(dataset, "gene", max_cpu=20)

    return dataset_generator(
        dataset, seed=seed, batch_size=batch_size, n_batches=n_batches
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


args = SentenceTransformerTrainingArguments(
    output_dir="models",
    fp16=False,
    bf16=True,
    batch_sampler=BatchSamplers.BATCH_SAMPLER,
    eval_strategy="no",
    save_strategy="no",
    logging_dir="logs",
)

# dataset_train = load_dataset(cfg.EMBEDDING_TRAIN_FILES, 64, N_STEPS, 0)
# dataset_train = dataset_train.remove_columns("negative")
dataset_test = load_dataset(cfg.TEST_FILES, 64, 50, 0)

evaluator = TripletEvaluator(
    anchors=dataset_test["anchor"],
    positives=dataset_test["positive"],
    negatives=dataset_test["negative"],
)

# print("Pre fine-tuning accuracy")
# for name, model in cfg.MODELS.items():
#     print(name)
#     original_model = SentenceTransformer(model)
#     print(evaluator(original_model))

# ## Select model
# print("\nTraining")
# for name, model in cfg.MODELS.items():

#     def hpo_model_init() -> SentenceTransformer:
#         return SentenceTransformer(model)

#     print(name)
#     trainer = SentenceTransformerTrainer(
#         model=None,
#         args=args,
#         train_dataset=dataset_train,
#         loss=hpo_loss_init,
#         model_init=hpo_model_init,
#         evaluator=evaluator,
#     )

#     best_trial = trainer.hyperparameter_search(
#         hp_space=hpo_search_space,
#         compute_objective=hpo_compute_objective,
#         n_trials=N_TRIALS,
#         direction="maximize",
#         backend="optuna",
#     )

#     print(best_trial)
#     print("")


## Test winner further.
# After running the above, ernie and pubmedncl came out as the best model to
# fine-tune.
dataset_train = load_dataset(cfg.EMBEDDING_TRAIN_FILES, 64, N_STEPS * 4, 1)
dataset_train = dataset_train.remove_columns("negative")
winners = ["ernie", "pubmedncl"]

print("\nFurther training")
for name in winners:

    def hpo_winner_init() -> SentenceTransformer:
        return SentenceTransformer(cfg.MODELS[name])

    print(name)
    trainer = SentenceTransformerTrainer(
        model=None,
        args=args,
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

    print(best_trial)
    print("")
