#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import re


def InputEngineTest(test_file, vocab_file_out):
    id2token_out = {}
    token2id_out = {}
    with open(vocab_file_out, mode="r") as f:
        for line in f:
            token, id = line.split("##")
            id = int(id)
            id2token_out[id] = token
            token2id_out[token] = id
    print("out vocabulary size =", str(len(token2id_out)))

    with open(test_file, "r") as f:
        count = 0.0
        top1 = 0.0
        top3 = 0.0

        count1 = 0.0
        top11 = 0.0
        top31 = 0.0

        count2 = 0.0
        top12 = 0.0
        top32 = 0.0
        for line in f:
            word_line, letter_line, result_line = line.strip().split("|#|")
            word_line = re.split('\\s+',word_line)
            letter_line = re.split('\\s+',letter_line)
            result_line = result_line.split("|")
            # if len(word_line) == 1:
            #     continue
            for (word, letter, result) in zip(word_line, letter_line, result_line):
                result = re.split('\\s+', result)

                if word.lower() != letter and word in token2id_out:
                    count += 1
                    if word in result[:1]:
                        top1 += 1
                    if word in result:
                        top3 += 1
                if word.lower() == letter and word in token2id_out:
                    count1 += 1
                    if word in result[:1]:
                        top11 += 1
                    if word in result:
                        top31 += 1
                if word in token2id_out:
                    count2 += 1
                    if word in result[:1]:
                        top12 += 1
                    if word in result:
                        top32 += 1

    f.close()
    print(count,top1,top3)
    print(top1/count,top3/count)
    print(count1, top11, top31)
    print(top11 / count1, top31 / count1)
    print(count2, top12, top32)
    print(top12 / count2, top32 / count2)


if __name__ == "__main__":

    args = sys.argv
    test_file_in = args[1]
    vocab_file_out = args[2]
    InputEngineTest(test_file_in, vocab_file_out)

