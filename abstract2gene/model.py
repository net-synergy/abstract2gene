import nltk

from .data.hgnc import gene_symbols
from .nlp import normalize, tokenize


def abstract_features(abstract, word_list):
    """Returns a logical lists indicating which words in word_list are in the
    abstract."""

    symbols = gene_symbols()
    abstract_words = tokenize(abstract)
    abstract_words = [normalize(w) for w in abstract_words if w not in symbols]
    # abstract_words = tokenize(normalize(abstract))
    return {w: (w in abstract_words) for w in word_list}
