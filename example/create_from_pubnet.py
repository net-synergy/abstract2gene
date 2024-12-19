import os

import numpy as np
from pubnet import PubNet, sanitize, text_transformations
from pubnet.download import from_pubmed

import abstract2gene as a2g
from abstract2gene.data import pubtator
from abstract2gene.dataset import net2dataset

GENE_ANNOTATIONS = "pubtator"
GRAPH_NAME = f"{GENE_ANNOTATIONS}_genes"
START = 1200
N_FILES = 3

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

node_list = [
    {"name": "publication", "value": "date"},
    "abstract",
]

net = from_pubmed(
    range(START, START + N_FILES),
    node_list,
    "genes",
    load_graph=True,
    overwrite=True,
)


def remove_duplicate_publications(net: PubNet, node_name: str) -> PubNet:
    """Remove publications with duplicate entries in a node.

    Sometimes a revised publication can lead to multiple entries for a single
    PMID. This is rare enough to exclude the PMID from the net rather than try
    to separate out the edges and save the PMID as the more recent entry.

    Node should have an edge set with Publications.

    """
    edge = net.get_edge(node_name, "Publication")
    (values, counts) = np.unique_counts(edge["Publication"])
    unique_pubs = values[counts == 1]

    if unique_pubs.shape[0] == values.shape[0]:
        return net

    return net[unique_pubs]


net = remove_duplicate_publications(net, "Abstract")

a2g.data.download(f"{GENE_ANNOTATIONS}_genes")
pubtator.add_gene_edges(net, replace=True)

sanitize.abstract(net)
text_transformations.specter(net, "Abstract", batch_size=128, max_tokens=512)

dataset = net2dataset(net, min_occurrences=50)
dataset.save("pubnet_pubtator")
