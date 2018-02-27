#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import numpy as np
from collections import defaultdict

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

    def word2id(self, word):
        if re.match("^[a-zA-Z]$", word) or (word in self.token2id_in_words):
            word_out = word
        else:
            if re.match("^[+-]*[0-9]+.*[0-9]*$", word):
                word_out = self.num_str
            else:
                if re.match("^[^a-zA-Z0-9']*$", word):
                    word_out = self.pun_str
                else:
                    word_out = self.unk_str
        rid = self.token2id_in_words.get(word_out, -1)
        if rid == -1:
            if self.fullvocab_set and word_out in self.fullvocab_set:
                return self.token2id_in_words[self.unk_str2]
            else:
                return self.token2id_in_words[self.unk_str]
        return rid

    def words2ids(self, words):
        return [self.bos_id] + [self.word2id(word) for word in words if len(word) > 0][:78] + [self.eos_id] + [self.pad_id] * (78 - len(words))

    def letters2ids(self, letters_array):
        max_length = 20
        return [[self.token2id_in_letters.get(letter.lower(), self.token2id_in_letters[self.unk_str])
                                  for letter in letters][:max_length] + [self.pad_id] * (max_length - len(letters))
                                  for letters in letters_array]

    def outword2id(self, outword):
        return self.token2id_out.get(outword, self.token2id_out[self.unk_str])

    def ids2outwords(self, ids_out):
        return [self.id2token_out.get(id, self.unk_str) for id in ids_out]

    def ids2inwords(self, ids_in):
        return [self.id2token_in_words.get(int(id), self.unk_str) for id in ids_in]

    def data2ids_line(self, data_line):
        data_line_split = re.split("\\|#\\|", data_line)
        letters_line = data_line_split[0].replace(" ","").split("\t")
        raw_words_line = data_line_split[1].strip().split("\t")
        words_line = []
        for i in range(len(raw_words_line)):
            # if raw_words_line[i].lower() != letters_line[i]:
            #     words_line.append(letters_line[i])
            # else:
            words_line.append(raw_words_line[i])
        words_ids = self.words2ids(words_line)
        letters_ids = self.letters2ids(letters_line)
        words_num = len(words_line)
        letters_num = [len(letter) for letter in letters_line]
        return raw_words_line, words_line, letters_line, words_ids, letters_ids, words_num, letters_num

    def sentence2ids(self, sentence):
        words = sentence.split()
        chars_array = [[char for char in word] for word in words]
        letters_ids = self.letters2ids(chars_array)
        return letters_ids
