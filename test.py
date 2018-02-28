#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import os
import tensorflow as tf
from data_utility_dynamic import DataUtility
from text_cnn import TextCNN
import time

class InputEngineRnn:

    def __init__(self, model_path, vocab_path):

        vocab_file_in_words = os.path.join(vocab_path, "vocab_in_words")
        vocab_file_in_letters = os.path.join(vocab_path, "vocab_in_letters")

       # model_file = os.path.join(model_path,model_name)

        self._data_utility = DataUtility(vocab_file_in_words=vocab_file_in_words,
                                         vocab_file_in_letters=vocab_file_in_letters)
        with tf.Graph().as_default():
            #initializer = tf.random_uniform_initializer(-self._config.init_scale, self._config.init_scale)

            self.model_test = TextCNN(sequence_length=20,
                                        num_classes=self._data_utility.in_words_count,
                                        vocab_size=self._data_utility.in_letters_count,
                                        embedding_size=128,
                                        filter_sizes=[3,4,5],
                                        num_filters=128,
                                        l2_reg_lambda=0.0)

            gpu_config = tf.ConfigProto()
            gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.48
            self._sess = tf.Session(config=gpu_config)
            with self._sess.as_default():
                checkpoint_file = tf.train.latest_checkpoint(model_path)
                # Do not restore sparse weights from pretrain phase
                restore_variables = dict()
                for v in tf.trainable_variables():
                    print("restore:", v.name)
                    restore_variables[v.name] = v
                saver = tf.train.Saver(restore_variables)
                saver.restore(self._sess, checkpoint_file)

            self._fetches = {
                "topk": self.model_test.predictions,
                "probability": self.model_test.top_probs
            }

    def predict(self, sentence):
        global probabilities, top_k_predictions
        inputs = self._data_utility.sentence2ids(sentence)
        print(inputs)
        feed_values = {self.model_test.input_x:inputs, self.model_test.dropout_keep_prob: 1.0}

        vals = self._sess.run(self._fetches,feed_dict=feed_values)
        top_k_predictions = vals["topk"]
        probabilities = vals["probability"]

        print(top_k_predictions)
        print(probabilities)
        words_list = self._data_utility.ids2inwords(top_k_predictions)
        probs_list = [str(prob) for prob in probabilities]

        return words_list, probs_list

    def predict_data(self, sentence):
        global probabilities, top_k_predictions
        words, keys, inputs = self._data_utility.data2ids_line(sentence)
        if '' in keys:
            return None
        # print(inputs)
        feed_values = {self.model_test.input_x:inputs, self.model_test.dropout_keep_prob: 1.0}

        vals = self._sess.run(self._fetches,feed_dict=feed_values)
        top_k_predictions = vals["topk"]
        probabilities = vals["probability"]

        # print(top_k_predictions)
        # print(probabilities)
        words_list = self._data_utility.ids2inwords(top_k_predictions)
        # probs_list = [str(prob) for prob in probabilities]

        return words, keys, words_list

    def predict_file(self, test_file_in, test_file_out):
        testfilein = open(test_file_in, "r")
        testfileout = open(test_file_out, 'w')
        t1 = time.time()

        for sentence in testfilein:
            sentence = sentence.rstrip()
            result = self.predict_data(sentence)
            if result is not None:
                words, keys, out_words = result

                print(" ".join(words) + "|#|" + " ".join(keys) + "|#|" +
                      "|".join(out_words))
                testfileout.write(" ".join(words) + "|#|" + " ".join(keys) + "|#|" +
                                  "|".join(out_words) + "\n")

        t2 = time.time()
        print(t2 - t1)
        testfilein.close()
        testfileout.close()

if __name__ == "__main__":
    args = sys.argv

    model_path = args[1]
    #model_name = args[2]
    vocab_path = args[2]
    phase = args[3]
    engine = InputEngineRnn(model_path, vocab_path)

    if phase == "online":
        while True:
            sentence = input("please enter sentence:")
            if sentence == "quit()":
                exit()
            if sentence == "":
                continue
            words, probs = engine.predict(sentence.strip())
            words = " ".join(words)
            probs = " ".join(probs)
            print(sentence)
            print(words)
            print(probs)
    if phase == "file":
        test_file_in = args[4]
        test_file_out = "test_result"
        engine.predict_file(test_file_in, test_file_out)