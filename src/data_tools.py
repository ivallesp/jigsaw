import re
import os
import pandas as pd
import numpy as np
from src.common_paths import *
from src.data_dictionaries import contractions, word_replacements
from src.text_tools import CharacterLevelEncoder
from src.general_utilities import batching
from collections import Counter


def replace(s, dictionary):
    dictionary_re = re.compile('(%s)' % '|'.join(dictionary.keys()))
    def replace(match):
        return dictionary[match.group(0)]
    return dictionary_re.sub(replace, s)

# TODO: Fix seed!!!
def train_dev_test_split(df, train_percent=.6, dev_percent=.2):
    m = len(df)
    train_end = int(train_percent * m)
    dev_end = int(dev_percent * m) + train_end
    train = df.iloc[:train_end]
    validate = df.iloc[train_end:dev_end]
    test = df.iloc[dev_end:]
    return train, validate, test


def load_train_data():
    df = pd.read_csv(os.path.join(get_data_path(), "train.csv"))
    df, allowed_symbols, cle = preprocess(df)
    df_train, df_dev, df_test = train_dev_test_split(df, train_percent=.8, dev_percent=0.1)
    return df_train, df_dev, df_test, allowed_symbols, cle


def load_scoring_data():
    _, _, _, allowed_symbols, cle = load_train_data()
    df = pd.read_csv(os.path.join(get_data_path(), "test.csv"))
    df, _, _ = preprocess(df, allowed_symbols=allowed_symbols, cle=cle)
    return df, allowed_symbols, cle


def preprocess(df, allowed_symbols=None, cle=None):
    df["comment_text"] = (df.comment_text
                          .str.replace('"', '')  # Remove quote symbols
                          .str.replace("\n{1,}", ". ")  # Remove \n
                          .str.replace("\ \.", ".")  # Fix " ."
                          .str.replace("\?{1,}", "?")  # Remove repeated quest. marks
                          .str.replace("\!{1,}", "!")  # Remove repeated excl. marks
                          .str.replace(" {1,}- {1,}", ", ")  # Remove dashes as punctuation
                          .str.replace(",\.", ".").str.strip()  # Normalize ",."
                          .str.replace(r",([a-z0-9])", r", \1")  # Normalize ",###" to ", ###"
                          .str.replace("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "[IP]")  # Normalize IPs
                          .str.replace(r"https?\:\/\/(www\.)?(([a-z0-9]*)\.){0,}([a-z]*)\.([a-z]{0,4}).*?(\ |$)",
                                       r"[URL|\4] ")  # Normalize urls (get domain)
                          .str.replace(r"\'{2,}", "")  # Remove repeated quote marks
                          .str.replace("\ {1,}", " ")  # Remove repeated spaces
                          .str.replace("(\.{1,})|(\.\ ){1,}", ".")  # Remove repeated dots
                          .str.replace(r"^[\.\,\:\-]{0,}\ {0,}",
                                       "").str.strip()  # Remove symbols from the begining
                          .str.replace(r"[\.\,\:\-]{1,}$", "").str.strip()  # Remove symbols from the end
                          .str.replace(r"[=\~\*]", "").str.strip()  # Remove certain symbols
                          .str.lower()
                          .map(lambda x: replace(x, contractions))  # Expand english contractions
                          .map(lambda x: replace(x, word_replacements))  # Replace words
                         )
    if not allowed_symbols:
        allowed_symbols = list(map(lambda x: x[0], Counter(" ".join(df.comment_text)).most_common(72)))
    if not cle:
        cle = CharacterLevelEncoder(allowed_symbols)
    df["code"] = df.comment_text.map(cle.transform)
    df = df.sample(frac=1, random_state=655321).reset_index(drop=True)
    return df, allowed_symbols, cle


def get_batcher(df, b_size=16, train=True):
    columns_target = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    if train:
        pool = [df.id.values.tolist(), df.code.map(np.matrix).values.tolist(), df[columns_target].values.tolist()]
        batcher = batching(pool, b_size)
        for element in batcher:
            max_len = max(map(lambda x:x.shape[1], element[1]))
            batch = np.row_stack(list(map(lambda x: np.pad(np.array(x)[0], (0,max_len-x.shape[1]), mode="constant"), element[1])))
            targets = np.row_stack(element[2])
            yield element[0], batch, targets
    else:
        pool = [df.id.values.tolist(), df.code.map(np.matrix).values.tolist()]
        batcher = batching(pool, b_size)
        for element in batcher:
            max_len = max(map(lambda x:x.shape[1], element[1]))
            batch = np.row_stack(list(map(lambda x: np.pad(np.array(x)[0], (0,max_len-x.shape[1]), mode="constant"), element[1])))
            yield element[0], batch