import concurrent.futures

import pandas as pd

from .data.hgnc import gene_symbols
from .nlp import tokenize

symbols = gene_symbols()


def attach(net, exclude=None):
    """Parse abstracts to find genes related to publications and add the genes
    as a new graph in the network.

    Arguments
    ---------
    net : PubNet, the publication network to parse.
    exculde : list, optional list of genes to exclude from search.
    """

    abstracts = net["Abstract"]
    gene_symbols = [sym for sym in symbols if sym not in exclude]

    gene_list = _collect_abstract_gene_symbols(abstracts, gene_symbols)
    wide_edges = _gene_publication_edges(net, gene_list)
    gene_nodes, gene_edges = _wide_to_relational(wide_edges)

    net.add_node("Gene", gene_nodes)
    net.add_edge(
        ("Gene", "Publication"),
        gene_edges,
    )
    return net


def _collect_genes(args):
    abstract, gene_symbols = args
    words = tokenize(abstract)
    return [w for w in set(words) if w in gene_symbols]


def _collect_abstract_gene_symbols(abstracts, gene_symbols):
    args = (
        (abstract, gene_symbols)
        for abstract in abstracts["AbstractText"].array
    )
    with concurrent.futures.ProcessPoolExecutor() as executor:
        abstract_genes = list(
            executor.map(
                _collect_genes,
                args,
            )
        )
    return zip(abstracts["AbstractId"].array, abstract_genes)


def _gene_publication_edges(net, gene_list):
    pub_abs_edges = net["Abstract", "Publication"]
    edges = []
    for abstract_id, genes in gene_list:
        if len(genes) > 0:
            publication_id = pub_abs_edges["Publication"][
                pub_abs_edges.isin("Abstract", abstract_id)
            ]
            assert publication_id.shape[0] == 1, (
                "Found multiple publication IDs for an individual"
                f" abstract.\n\n{publication_id} claim {abstract_id}"
            )
            publication_id = publication_id[0]

        for gene in genes:
            edges.append([publication_id, gene])

    return pd.DataFrame(edges, columns=[net["Publication"].id, "GeneSymbol"])


def _wide_to_relational(wide_edges):
    node_index = set()
    for sym in wide_edges["GeneSymbol"].array:
        node_index.add(sym)

    node_index = {symbol: (id + 1) for id, symbol in enumerate(node_index)}
    nodes = pd.DataFrame(
        {
            "GeneId:ID(Gene)": node_index.values(),
            "GeneSymbol": node_index.keys(),
        }
    )
    edges = pd.DataFrame(
        (
            (pubId, node_index[sym])
            for pubId, sym in zip(
                wide_edges["PublicationId"], wide_edges["GeneSymbol"]
            )
        ),
        columns=[":START_ID(PublicationId)", ":END_ID(GeneId)"],
    )

    return nodes, edges
