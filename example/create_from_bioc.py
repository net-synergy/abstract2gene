import os

from abstract2gene.dataset import bioc2dataset
from abstract2gene.storage import default_data_dir, set_cache_dir, set_data_dir

set_cache_dir("/disk4/david/cache")
set_data_dir("/disk4/david/share")

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

dataset = bioc2dataset([0], batch_size=10)

save_path = os.path.join(default_data_dir("dataset"), "bioc")
if os.path.isdir(save_path):
    for f in os.listdir():
        os.unlink(f)
    os.rmdir(save_path)

dataset.save_to_disk(save_path, max_shard_size="1GB")
