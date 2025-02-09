"""Compare models and hyperparamaters."""

import itertools

from datasets import Dataset, load_from_disk
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from abstract2gene.data import dataset_path
from abstract2gene.dataset import dataset_generator

# TEMP: Change to full dataset. Requires modifying to work with a datasetdict.
DATASET = "bioc_finetune"
MODELS = {
    "ernie": "nghuyong/ernie-2.0-base-en",
    "mpnet": "microsoft/mpnet-base",
    "bert": "google-bert/bert-base-uncased",
    "specter": "sentence-transformers/allenai-specter",
}
N_STEPS = 100
CHKPT_PATH = "models/"


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


dataset_dict = load_from_disk(dataset_path(DATASET)).train_test_split(
    train_size=0.9, seed=0, shuffle=True
)

dataset_train = dataset_generator(dataset_dict["train"], seed=0, batch_size=64)
dataset_train = dataset_train.remove_columns("negative")
dataset_test = dataset_generator(dataset_dict["test"], seed=0, batch_size=64)

args = SentenceTransformerTrainingArguments(
    output_dir="models",
    fp16=False,
    bf16=True,
    batch_sampler=BatchSamplers.BATCH_SAMPLER,
    eval_strategy="no",
    save_strategy="no",
    logging_dir="logs",
)

n_eval = 64 * 50
eval_examples = list(itertools.islice(dataset_test, n_eval))
eval_kwds = {k: [ex[k] for ex in eval_examples] for k in dataset_test.features}

evaluator = TripletEvaluator(
    anchors=eval_kwds["anchor"],
    positives=eval_kwds["positive"],
    negatives=eval_kwds["negative"],
)

for name in MODELS:
    print(name)
    original_model = SentenceTransformer(MODELS[name])
    print(evaluator(original_model))

## Select model
n_train = 64 * N_STEPS
train_dataset = Dataset.from_list(
    list(itertools.islice(dataset_train, n_train))
)

for name in MODELS:

    def hpo_model_init() -> SentenceTransformer:
        return SentenceTransformer(MODELS[name])

    trainer = SentenceTransformerTrainer(
        model=None,
        args=args,
        train_dataset=train_dataset,
        loss=hpo_loss_init,
        model_init=hpo_model_init,
        evaluator=evaluator,
    )

    best_trial = trainer.hyperparameter_search(
        hp_space=hpo_search_space,
        compute_objective=hpo_compute_objective,
        n_trials=20,
        direction="maximize",
        backend="optuna",
    )

    print(name)
    print(best_trial)
    print("")


## Test winner further.
# After running the above, specter came out as the best model to fine-tune.
n_train = 64 * N_STEPS * 4
train_dataset = Dataset.from_list(
    list(itertools.islice(dataset_train, n_train))
)


def hpo_specter_init() -> SentenceTransformer:
    return SentenceTransformer(MODELS["specter"])


trainer = SentenceTransformerTrainer(
    model=None,
    args=args,
    train_dataset=train_dataset,
    loss=hpo_loss_init,
    model_init=hpo_specter_init,
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
