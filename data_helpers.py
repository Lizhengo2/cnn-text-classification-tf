import numpy as np
import re
import os
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(data_path, vocab_path, max_length):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    unused_word_list = ["<unk>", "<pun>", "<num>"]
    x_text = []
    y = []
    vocab_file_in_words = os.path.join(vocab_path, "vocab_in_words")
    vocab_file_in_letters = os.path.join(vocab_path, "vocab_in_letters")
    data_file = os.path.join(data_path, "train_in_lm")

    id2token_in_words, id2token_in_letters = {}, {}
    token2id_in_words, token2id_in_letters = {}, {}
    with open(vocab_file_in_words, mode="r") as f:
        for line in f:
            token, id = line.strip().split("##")
            id = int(id)
            id2token_in_words[id] = token
            token2id_in_words[token] = id
    print("in words vocabulary size =", str(len(token2id_in_words)))
    in_words_count = len(token2id_in_words)

    with open(vocab_file_in_letters, mode="r") as f:
        for line in f:
            token, id = line.strip().split("##")
            id = int(id)
            id2token_in_letters[id] = token
            token2id_in_letters[token] = id
    print("in letters vocabulary size =", str(len(token2id_in_letters)))
    in_letters_count = len(token2id_in_letters)

    with open(data_file, mode="r") as f:
        for line in f:
            output_words, input_chars = line.strip().split("|#|")
            for (word, chars) in zip(output_words.split(), input_chars.split("#")):
                if word not in unused_word_list and word in token2id_in_words:
                    char_list = chars.split()[1:]
                    one_x_text = [token2id_in_letters[char] for char in char_list][:max_length] \
                                 + [0] * (max_length - len(char_list))
                    x_text.append(one_x_text)

                    one_y = [0 for _ in range(in_words_count)]
                    one_y[token2id_in_words[word]] = 1
                    y.append(one_y)

    return np.array(x_text), np.array(y), in_words_count, in_letters_count


def batch_iter(data, batch_size, data_size, num_batches_per_epoch):

    # Shuffle the data at each epoch

    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield data[start_index:end_index]
