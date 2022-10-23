import nltk


def tokenize(text):
    "Split raw text into a list of words."

    return nltk.regexp_tokenize(text, r"\w+(?:[-']\w+)*")
