import os

import torch
from datasets import load_from_disk
from transformers import AutoModel, AutoTokenizer

from abstract2gene.data import default_data_dir

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true "
    "--xla_gpu_enable_triton_gemm=true "
    "--xla_gpu_graph_level=0 "
)

os.environ.update(
    {
        "NCCL_LL128_BUFFSIZE": "-2",
        "NCCL_LL_BUFFSIZE": "-2",
        "NCCL_PROTO": "SIMPLE,LL,LL128",
    }
)


def tokenize_and_embed(
    examples: dict[str, list], rank: int | None
) -> dict[str, list]:
    device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
    model.to(device)
    inputs = tokenizer(
        examples["abstract"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        return {"embedding": model(**inputs).last_hidden_state[:, 0, :]}


model_id = "allenai/specter"
tokenizer = AutoTokenizer.from_pretrained(model_id)
mask_token = tokenizer.special_tokens_map["mask_token"]
model = AutoModel.from_pretrained(model_id)


save_path = os.path.join(default_data_dir("datasets"), "bioc")
dataset = load_from_disk(save_path)
dataset = dataset.select(range(100)).map(
    tokenize_and_embed,
    batched=True,
    batch_size=10,
    remove_columns="abstract",
    num_proc=1,
    with_rank=True,
    desc="Embed Abstracts",
)
