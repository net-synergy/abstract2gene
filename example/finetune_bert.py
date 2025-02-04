import itertools

from datasets import Dataset, IterableDataset, load_from_disk
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from abstract2gene.data import default_data_dir
from abstract2gene.dataset import dataset_generator, dataset_path

MODELS = {
    "ernie": "nghuyong/ernie-2.0-base-en",
    "mpnet": "microsoft/mpnet-base",
    "bert": "google-bert/bert-base-uncased",
    "specter": "sentence-transformers/allenai-specter",
}

CHKPT_PATH = "models/"


def load_data(
    name: str, split: float = 0.8, batch_size: int = 32
) -> dict[str, IterableDataset]:
    # TODO: Replace load_from_disk -> load_dataset when pushed to HF hub.
    dataset_dict = load_from_disk(dataset_path(name)).train_test_split(
        train_size=split, seed=0, shuffle=True
    )

    return {
        "train": dataset_generator(
            dataset_dict["train"], seed=0, batch_size=batch_size
        ),
        "test": dataset_generator(
            dataset_dict["test"], seed=0, batch_size=batch_size
        ),
    }


def finetune(
    model_name: str,
    train_data: IterableDataset,
    test_data: IterableDataset,
    batch_size: int,
    n_steps: int,
    learning_rate: float,
    warmup_ratio: float,
):
    model = SentenceTransformer(MODELS[model_name])
    loss = MultipleNegativesRankingLoss(model)

    eval_steps = min(n_steps, 500)
    args = SentenceTransformerTrainingArguments(
        output_dir=f"models/{model_name}",
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        fp16=False,
        bf16=True,
        batch_sampler=BatchSamplers.BATCH_SAMPLER,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        max_steps=n_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        save_total_limit=2,
        logging_steps=eval_steps,
        logging_first_step=True,
        logging_dir="logs",
    )

    n_eval = batch_size * 50
    eval_examples = list(itertools.islice(test_data, n_eval))
    eval_kwds = {
        k: [ex[k] for ex in eval_examples] for k in test_data.features
    }

    evaluator = TripletEvaluator(
        anchors=eval_kwds["anchors"],
        positives=eval_kwds["positives"],
        negatives=eval_kwds["negatives"],
        batch_size=batch_size,
    )

    n_train = batch_size * n_steps
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(
            list(itertools.islice(train_data, n_train))
        ),
        loss=loss,
        evaluator=evaluator,
    )

    trainer.train()
    model.save_pretrained(
        default_data_dir(f"models/{model_name}-abstract-genes")
    )


if __name__ == "__main__":
    batch_size = 64
    n_steps = 10000
    warmup_ratio = 0.22
    learning_rate = 1.4e-5
    dataset_name = "bioc_finetune"
    model = "specter"

    data_dict = load_data(dataset_name, split=0.9, batch_size=batch_size)

    finetune(
        model,
        data_dict["train"].remove_columns("negatives"),
        data_dict["test"],
        batch_size=batch_size,
        n_steps=n_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
    )
