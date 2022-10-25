import sys
import time
from functools import reduce

import abstract2gene as a2g
from pubnet import from_dir


def message(*args):
    print(*args, file=sys.stderr)


def elapsed_time_factory():
    t = time.time()

    def elapsed_time():
        nonlocal t
        t_elapsed = time.time() - t
        t = time.time()
        return f"{t_elapsed:.2f}s"

    return elapsed_time


elapsed_time = elapsed_time_factory()

# NOTE: graphs headers manually changed, see pubmedparser issue #6
data_dir = "/home/voidee/data/abstracts_small"
nodes = ("Abstract", "Descriptor", "Chemical", "Publication")
edges = (
    ("Publication", "Abstract"),
    ("Publication", "Chemical"),
    ("Publication", "Descriptor"),
)

message("Loading network")
publications = from_dir("Publication", nodes, edges, data_dir=data_dir)
message(f"Network loaded in {elapsed_time()}\n")

message("Finding AD publications")
ad_publications = publications.containing(
    "Descriptor", "DescriptorName", "Alzheimer Disease"
)
n_publications = ad_publications["Publication"].shape[0]
message(f"Found {n_publications} AD publications in {elapsed_time()}\n")

elapsed_time()
message("Looking for gene symbols in abstracts.")
ad_publications = a2g.genes.attach(ad_publications)
n_abstracts_with_symbols = ad_publications["Publication", "Gene"].shape[0]
message(
    f"Found {n_abstracts_with_symbols} AD abstracts with gene symbols in"
    f" {elapsed_time()}.\n"
)
print(f"Total AD publications: {n_publications}")
print(
    "Number of AD related abstracts with gene symbols:"
    f" {n_abstracts_with_symbols}"
)
ad_publications.plot_distribution("Gene", "GeneSymbol", threshold=2)
