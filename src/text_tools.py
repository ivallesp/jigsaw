import nltk
from src.general_utilities import flatten
from nltk import ngrams
import numpy as np
from collections import Counter


def remove_substrings(s, substrings):
    for substr in substrings:
        s = s.replace(substr, "")
    return s


class TokenEvaluator:
    def __init__(self, n_grams=1):
        self.n_grams = n_grams

    def calculate_ngrams(self, lists_of_tokens):
        if self.n_grams > 1:
            lists_of_grams = list(map(lambda x: list(ngrams(x, self.n_grams)), lists_of_tokens))
        else:
            lists_of_grams = lists_of_tokens
        return lists_of_grams

    def fit(self, list_of_real_sentences):
        lists_of_tokens = list(map(lambda x: nltk.word_tokenize(x, "english"), list_of_real_sentences))
        lists_of_tokens = map(lambda s: [w.lower() for w in s], lists_of_tokens)
        fdist = Counter(flatten(self.calculate_ngrams(lists_of_tokens)))
        items = list(map(lambda x: x[0], filter(lambda x:x[1] >= 2, dict(fdist).items()))) # Remove hapaxes
        self.unique_items = set(items)

    def evaluate(self, list_of_generated_sentences):
        lists_of_tokens = list(map(lambda x: nltk.word_tokenize(x, "english"), list_of_generated_sentences))
        if self.n_grams == 1:
            lists_of_tokens = list(map(lambda s: [w.lower() for w in s if len(w) > 2], lists_of_tokens))
            # only consider tokens >2 characters
        else:
            lists_of_tokens = list(map(lambda s:[w.lower() for w in s], lists_of_tokens))

        lists_of_grams = self.calculate_ngrams(lists_of_tokens)
        
        if self.n_grams > 1: # Remove n_grams composed of one-char tokens
            lists_of_grams = list(map(lambda s: list(filter(lambda ng: sum([len(w) for w in ng])>self.n_grams, s)),
                                      lists_of_grams))
        lists_of_grams = list(filter(lambda x:len(x), lists_of_grams))
        # Remove the empty lists (e.g. bigrams of 1 token)
        accuracies = [np.mean(list(map(lambda w:w in self.unique_items, s))) for s in lists_of_grams]
        return accuracies


class CharacterLevelEncoder:
    def __init__(self, allowed_symbols):
        self.code_dict = dict(zip(allowed_symbols, range(len(allowed_symbols))))
        self.unknown_code = len(self.code_dict)
        self.start_code = len(self.code_dict) + 1
        self.end_code = len(self.code_dict) + 2
        self.inverse_code_dict = {value: key for key, value in self.code_dict.items()}
        self.inverse_code_dict[self.unknown_code] = "<UNK>"

    def transform(self, x):
        code = [self.start_code] + list(map(lambda k: self.code_dict.get(k, self.unknown_code), x)) + [self.end_code]
        return code

    def inverse_transform(self, code):
        code = code[1:-1]
        text = " ".join(list(map(lambda k: self.inverse_code_dict.get(k), code)))
        return text


