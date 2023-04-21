import functools
import sys
import time
from functools import reduce

import abstract2gene as a2g
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pubnet
from nltk.corpus import stopwords
from pubnet import from_dir
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def message(*args):
    print(*args, file=sys.stderr)


def get_SS_Publications(net):
    """Get the Social Science Publications from the PubNet network"""
    message("Finding Social Science publications")
    ss_publications = net.containing(
        "Descriptor",
        "DescriptorName",
        [
            "Health behavior",
            "Health education",
            "Health knowledge, attitudes, practice",
            "Health promotion",
            "Health services accessibility",
            "Health status disparities",
            "Patient compliance",
            "Patient education as topic",
            "Patient participation",
            "Social determinants of health",
            "Social support",
            "Stigma, social",
            "Disease outbreaks",
            "Epidemics",
            "Pandemics",
            "Risk factors",
            "Public health",
            "Health policy",
            "Health equity",
            "Health literacy",
        ],
    )
    n_publications = ss_publications["Publication"].shape[0]
    message(f"Found {n_publications} publications in {elapsed_time()}\n")
    return ss_publications


def get_AD_Publications_With_Genes(net):
    """Get the Alzheimer's Disease Publications from the PubNet network"""
    message("Finding AD publications")
    ad_publications = net.containing(
        "Descriptor", "DescriptorName", "Alzheimer Disease"
    )
    n_publications = ad_publications["Publication"].shape[0]
    message(f"Found {n_publications} publications in {elapsed_time()}\n")
    excluded_genes = [
        "CA1",
        "HR",
        "SCD",
        "LBP",
        "CA3",
        "CA4",
        "CBS",
        "GC",
        "STAR",
    ]
    ad_publications = a2g.genes.attach(
        ad_publications, exclude=excluded_genes
    )

    total_word_freq = a2g.nlp.freq_dist(ad_publications)
    high_freq_words = total_word_freq.most_common(100)

    # TODO: set when using larger dataset
    # specificity_threshold = 50
    minimum_occurances = 3

    def gene_indices(nodes):
        dist = ad_publications["Gene", "Publication"].distribution("Gene")
        return dist > minimum_occurances

    ad_publications = ad_publications.where("Gene", gene_indices)

    return ad_publications


def get_Embeddings(ss_net, ad_net):
    ss_abstracts = ss_net["Abstract"]["AbstractText"].array
    gen_abstracts = ad_net["Abstract"]["AbstractText"].array
    gen_abstracts = [reduce(lambda x, y: x + y, gen_abstracts)]

    abstracts = np.concatenate((ss_abstracts, gen_abstracts))

    tokenizer = functools.partial(
        a2g.nlp.getTokens, exclude=stopwords.words("english")
    )

    vectorizer = TfidfVectorizer(tokenizer=tokenizer, token_pattern=None)

    embeddings = vectorizer.fit_transform(abstracts)

    return embeddings


def get_Embedding_From_Documents(abstracts):
    tokenizer = functools.partial(
        a2g.nlp.getTokens, exclude=stopwords.words("english")
    )

    vectorizer = TfidfVectorizer(tokenizer=tokenizer, token_pattern=None)

    embeddings = vectorizer.fit_transform(abstracts)

    return embeddings


def get_Cosine_Similarites(embeddings1, embeddings2):
    """Get the cosine similarity between two networks"""

    scores = cosine_similarity(embeddings1, embeddings2)
    return scores


def elapsed_time_factory():
    t = time.time()

    def elapsed_time():
        nonlocal t
        t_elapsed = time.time() - t
        t = time.time()
        return f"{t_elapsed:.2f}s"

    return elapsed_time


if __name__ == "__main__":
    elapsed_time = elapsed_time_factory()

    # NOTE: graphs headers manually changed, see pubmedparser issue #6
    data_dir = "/mnt/c/Users/georgs2/pubnet/share"
    nodes = ("Abstract", "Descriptor", "Chemical", "Publication")
    edges = (
        ("Publication", "Abstract"),
        ("Publication", "Chemical"),
        ("Publication", "Descriptor"),
    )

    message("Loading network")
    publications = from_dir("Publication", nodes, edges, data_dir=data_dir)
    message(f"Network loaded in {elapsed_time()}\n")

    nltk.download("stopwords")

    ss_publications = get_SS_Publications(publications)
    ss_freq_dist = a2g.nlp.freq_dist(
        ss_publications, exclude=stopwords.words("english")
    )
    most_common = ss_freq_dist.most_common(50)
    print(most_common)

    # sort in-place from highest to lowest
    most_common.sort(key=lambda x: x[1], reverse=True)

    words = list(zip(*most_common))[0]
    freqs = list(zip(*most_common))[1]
    x_pos = np.arange(len(words))

    plt.bar(x_pos, freqs, align="center")
    plt.xticks(x_pos, words, rotation="vertical", size=6)
    plt.ylabel("50 Most Common Word Frequencies")
    plt.tight_layout()
    plt.savefig("50_most_common_word_frequencies.png")

    ad_publications = get_AD_Publications_With_Genes(publications)
    genes = ad_publications["Gene"]["GeneSymbol"].array
    genes = genes.tolist()
    genes.sort()
    ss_abstracts = ss_publications["Abstract"]["AbstractText"].array
    abstracts = ss_abstracts

    num_ss = len(ss_publications["Abstract"]["AbstractText"].array)
    for gene in genes:
        gene_publications = ad_publications.containing(
            "Gene", "GeneSymbol", gene
        )
        gene_abstracts = gene_publications["Abstract"]["AbstractText"].array
        gene_abstracts = [reduce(lambda x, y: x + y, gene_abstracts)]
        abstracts = np.concatenate((abstracts, gene_abstracts))

    embeddings = get_Embedding_From_Documents(abstracts)
    # # print(embeddings.shape)
    # # print(embeddings)
    # # print(type(embeddings))

    scores = get_Cosine_Similarites(embeddings[:num_ss], embeddings[num_ss:])
    print(scores.shape)
    print(scores)

    max_scores = [(genes[i], np.max(scores[:, i])) for i in range(50)]

    max_scores.sort(key=lambda x: x[1], reverse=True)
    print(max_scores)

    gene_names = list(zip(*max_scores))[0]
    gene_scores = list(zip(*max_scores))[1]
    x_pos2 = np.arange(len(gene_names))

    plt.clf()
    plt.bar(x_pos2, gene_scores, align="center")
    plt.xticks(x_pos2, gene_names, rotation="vertical", size=6)
    plt.ylabel("50 Maximum Cosine Similarities")
    plt.tight_layout()
    plt.savefig("max_cosine_similarities.png")

    # message("Embeddings Shape: ", embeddings.shape)
    # message("First social science abstract embeddings: ")
    # print(embeddings[0])
    # message("Last social science abstract embeddings: ")
    # print(embeddings[399])
    # message("First AD abstract embeddings: ")
    # print(embeddings[400])
    # message("Cosine Similarity between the first abstract and all other abstracts:")
    # scores = cosine_similarity(embeddings[0], embeddings[1:])
    # print(scores)

    # message("Social Science tokens for first abstract:")
    # tok = a2g.nlp.getTokens(ss_publications["Abstract"]["AbstractText"].array[0], exclude=stopwords.words('english'))
    # print(tok)

    # nltk.download('stopwords')
    # total_word_freq = a2g.nlp.freq_dist(ss_publications, exclude=stopwords.words('english'))
    # highest_freq = total_word_freq.most_common(1000)
    # print(highest_freq)
