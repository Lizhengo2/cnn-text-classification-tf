#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re


class DataUtility:
    def __init__(self, vocab_file_in_words=None, vocab_file_in_letters=None):

        self.bos_str = "<bos>"
        self.eos_str = "<eos>"
        self.unk_str = "<unk>"
        self.num_str = "<num>"
        self.pun_str = "<pun>"
        self.pad_id = 0

        if vocab_file_in_words and vocab_file_in_letters:
            self.id2token_in_words, self.id2token_in_letters = {}, {}
            self.token2id_in_words, self.token2id_in_letters = {}, {}
            with open(vocab_file_in_words, mode="r") as f:
                for line in f:
                    token, id = line.strip().split("##")
                    id = int(id)
                    self.id2token_in_words[id] = token
                    self.token2id_in_words[token] = id

            self.in_words_count = len(self.token2id_in_words)

            with open(vocab_file_in_letters, mode="r") as f:
                for line in f:
                    token, id = line.strip().split("##")
                    id = int(id)
                    self.id2token_in_letters[id] = token
                    self.token2id_in_letters[token] = id

            self.in_letters_count = len(self.token2id_in_letters)

    def letters2ids(self, letters_array):
        max_length = 20
        return [[self.token2id_in_letters.get(letter.lower(), self.token2id_in_letters[self.unk_str])
                for letter in letters if len(letter) > 0][:max_length] + [self.pad_id] * (max_length - len(letters))
                for letters in letters_array]

    def ids2inwords(self, ids_in):
        return [self.id2token_in_words.get(int(id), self.unk_str) for id in ids_in]

    def data2ids_line(self, data_line):
        data_line_split = re.split("\\|#\\|", data_line)
        letters_line = data_line_split[0].replace(" ","").split("\t")
        words_line = data_line_split[1].strip().split("\t")
        letters_ids = self.letters2ids(letters_line)
        return words_line, letters_line, letters_ids

    def sentence2ids(self, sentence):
        words = sentence.split()
        letters_ids = self.letters2ids(words)
        return letters_ids
