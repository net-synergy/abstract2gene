import os

from datasets import load_from_disk
from sentence_transformers import SentenceTransformer

from abstract2gene.data import dataset_path

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

DATASET = "bioc_small"


model = SentenceTransformer("sentence-transformers/allenai-specter")
save_path = dataset_path(DATASET)
dataset = load_from_disk(save_path)

dataset = dataset.select(range(1000)).map(
    lambda examples: {"embedding": model.encode(examples["abstract"])},
    batched=True,
    batch_size=10,
    remove_columns="abstract",
    num_proc=1,
    desc="Embed Abstracts",
)

len(dataset["embedding"][0])
