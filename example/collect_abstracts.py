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
# Based on looking at the gene distribution plot
excluded_genes = ["CA1", "HR", "SCD", "LBP", "CA3", "CA4", "CBS", "GC", "STAR"]
ad_publications = a2g.genes.attach(ad_publications, exclude=excluded_genes)
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

total_word_freq = a2g.nlp.freq_dist(ad_publications)
high_freq_words = total_word_freq.most_common(100)

# TODO: set when using larger dataset
specificity_threshold = 50
minimum_publications = 3


def gene_indices(nodes):
    dist = ad_publications["Gene", "Publication"].distribution("Gene")
    return dist > minimum_publications


ad_publications = ad_publications.where("Gene", lambda x: gene_indices)
high_specificity_words = set()
for gene in ad_publications["Gene"]["GeneSymbol"]:
    gene_publications = ad_publications.containing("Gene", "GeneSymbol", gene)
    gene_word_freq = a2g.nlp.freq_dist(
        gene_publications, exclude=high_freq_words
    )
    n_words_ratio = total_word_freq.N() / gene_word_freq.N()
    for word in gene_word_freq.keys():
        specificity = n_words_ratio * (
            gene_word_freq[word] / total_word_freq[word]
        )
        if specificity > specificity_threshold:
            high_specificity_words.add(word)
