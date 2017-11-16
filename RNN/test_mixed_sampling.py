from __future__ import print_function
import tensorflow as tf
import numpy as np

import csv
import sys

from rnn_model import Rnn_Model
import config
import operator
import argparse
import codecs
import nltk
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

#Inspired from https://github.com/sherjilozair/char-rnn-tensorflow


def sample(config):
    #get test text
    with codecs.open(config.test_input_file, "r", encoding='utf-8') as f:
        text = f.read()

    words = nltk.tokenize.word_tokenize(text)

    # Load training vocabulary from csv file
    char_ids = {}
    for char, id in csv.reader(open("char_ids.csv", encoding='utf-8')):
        char_ids[char] = id
    # Load character counts from csv file
    char_counts = {}
    for char, count in csv.reader(open("char_counts.csv", encoding='utf-8')):
        char_counts[char] = count

    # Same procedure for word tags
    tag_ids = {}
    for tag, id in csv.reader(open("tag_ids.csv", encoding='utf-8')):
        tag_ids[tag] = id

    config.sample_prime = '.'

    vocab_size_tags = len(tag_ids)
    vocab_size_chars = len(char_ids)

    # Initialize neural network
    config.language_modeling = "simplified_word"
    with tf.variable_scope(config.language_modeling):
        rnn_model_tags = Rnn_Model(is_training=False, vocab_size=vocab_size_tags, config=config)
    config.language_modeling = "character"
    with tf.variable_scope(config.language_modeling):
        rnn_model_chars = Rnn_Model(is_training=False, vocab_size=vocab_size_chars, config=config)

    correct_words = 0

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # Restore trained model
        saver_character = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='character')) #Define the scopes for the saver in order to not have conflicting variables
        saver_simplified_word = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='simplified_word'))
        ckpt_tags = tf.train.get_checkpoint_state(config.checkpoint_dir_simplified_word_model)
        ckpt_chars = tf.train.get_checkpoint_state(config.checkpoint_dir_character_model)

        saver_character.restore(sess, ckpt_chars.model_checkpoint_path)
        saver_simplified_word.restore(sess, ckpt_tags.model_checkpoint_path)

        # Initial internal network state
        state_tags = sess.run(rnn_model_tags.cell.zero_state(1, tf.float32))
        state_chars = sess.run(rnn_model_chars.cell.zero_state(1, tf.float32))

        # Compute network state after feeding prime text
        for word in nltk.tokenize.word_tokenize(config.sample_prime[:-1]):
            x = np.zeros((1, 1))
            x[0, 0] = tag_ids[word]
            feed_dict = {rnn_model_tags.input_data: x, rnn_model_tags.initial_state: state_tags}
            [state_tags] = sess.run([rnn_model_tags.final_state], feed_dict)

        # Compute network state after feeding prime text
        for char in config.sample_prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = char_ids[char]
            feed_dict = {rnn_model_chars.input_data: x, rnn_model_chars.initial_state: state_chars}
            [state_chars] = sess.run([rnn_model_chars.final_state], feed_dict)

        # Keep track of overall sampled text
        sampled_text = config.sample_prime

        #TODO change back: last_char = config.sample_prime[-1]

        state_tags_before_word = state_tags
        state_chars_before_word = state_chars

        last_word = '.' #just start with a dot -> makes it easy to start new sentence
        last_tag = '.'

        word_number = 0

        #produce ? words & measure performance
        while word_number < 1000:
            print("Testing on word number "+str(word_number))
            correct_next_word = words[word_number]

            states_after_word = []
            sampled_words = []

            # STEP 1: SAMPLE POS-TAG ON WORD LEVEL
            last_char = ' '

            state_tags = state_tags_before_word
            x = np.zeros((1, 1))

            x[0, 0] = tag_ids[last_tag]

            feed_dict = {rnn_model_tags.input_data: x, rnn_model_tags.initial_state: state_tags}
            [word_probabilities, state_tags] = sess.run([rnn_model_tags.probabilities, rnn_model_tags.final_state], feed_dict)
            next_word_probabilities = word_probabilities[0]

            # Sample from probability distribution
            t = np.cumsum(next_word_probabilities)
            s = np.sum(next_word_probabilities)
            sample = (int(np.searchsorted(t, np.random.rand(1) * s)))

            tags = list()
            for i in tag_ids.keys():
                tags.append(i)
            predicted_tag = tags[sample]
            print("Predicted Tag: "+predicted_tag)

            last_tag = predicted_tag

            states_after_word.append(state_tags)
            #sampled_words.append(predicted_word)

            tries = 0
            # STEP 2: SAMPLE WORD ON CHARACTER LEVEL (filter by predicted Pos_Tag)
            #sample 5 words, 1 max prob word
            last_char = ' '

            state_chars = state_chars_before_word

            sampled_word = ""
            # add characters until space appears
            while True:
                # Feed characters into network one by one
                x = np.zeros((1, 1))
                x[0, 0] = char_ids[last_char]

                feed_dict = {rnn_model_chars.input_data: x, rnn_model_chars.initial_state: state_chars}
                [char_probabilities, state_chars] = sess.run([rnn_model_chars.probabilities, rnn_model_chars.final_state],
                                                       feed_dict)
                next_char_probabilities = char_probabilities[0]

                # Sample from probability distribution
                t = np.cumsum(next_char_probabilities)
                s = np.sum(next_char_probabilities)
                sample = (int(np.searchsorted(t, np.random.rand(1) * s)))

                # Append prediction to sampled text
                chars = list()
                for i in char_ids.keys():
                    chars.append(i)
                predicted_char = chars[sample]

                last_char = predicted_char

                if predicted_char == ' ' and sampled_word != '':
                    tries += 1
                    # test the tag of predicted word
                    text_with_word = sampled_text + sampled_word
                    tokenized_text = nltk.word_tokenize(text_with_word)
                    tagged_text = nltk.pos_tag(tokenized_text)
                    print(tagged_text[-1][1])
                    if tagged_text[-1][1] == predicted_tag:
                        states_after_word.append(state_chars)
                        sampled_words.append(sampled_word)
                        break
                    #if 100 samples don't contain correct Tag, just use any sample
                    elif tries == config.maximum_tries:
                        states_after_word.append(state_chars)
                        sampled_words.append(sampled_word)
                        print(str(config.maximum_tries)+" tries but no sample with correct POS tag")
                        break

                sampled_word += predicted_char

            sampled_text += sampled_words[0] + ' '
            for c in correct_next_word:
                #compute new state (state before next word)
                x[0, 0] = char_ids[c]
                feed_dict = {rnn_model_chars.input_data: x, rnn_model_chars.initial_state: state_chars_before_word}
                state_chars_before_word = sess.run([rnn_model_chars.final_state], feed_dict)
            #add blank between words
            x[0, 0] = char_ids[' ']
            feed_dict = {rnn_model_chars.input_data: x, rnn_model_chars.initial_state: state_chars_before_word}
            state_chars_before_word = sess.run([rnn_model_chars.final_state], feed_dict)
            print(words[0:word_number])
            print(sampled_text)
            print('\n')
            word_number += 1

            #COUNT CORRECT WORDS
            if words[word_number] == sampled_words[0]:
                correct_words += 1
            print("Correctly predicted words: "+str(correct_words))

if __name__ == '__main__':
    sample(config=config.Standard_Config())