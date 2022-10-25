import nltk


def tokenize(text):
    "Split raw text into a list of words."

    return nltk.regexp_tokenize(text, r"[a-zA-Z]\w+(?:[-']\w+)*")


def normalize(text):
    "Clean up words to be in a consistent form."

    return text.lower()


def freq_dist(net, exclude=None):
    all_abstracts = " ".join(net["Abstract"]["AbstractText"].array)
    all_words = tokenize(normalize(all_abstracts))
    if exclude:
        all_words = [w for w in all_words if w not in exclude]

    return nltk.FreqDist(all_words)
