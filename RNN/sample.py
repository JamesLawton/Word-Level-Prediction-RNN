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

#Inspired from https://github.com/sherjilozair/char-rnn-tensorflow


def sample(config):
    #create frequency distribution
    with codecs.open(config.train_input_file, "r", encoding='utf-8') as f:
        text = f.read()

    words = nltk.tokenize.word_tokenize(text)
    fdist = nltk.FreqDist(words)
    text_length = len(text)

    # Load training vocabulary from csv file
    char_ids = {}
    for char, id in csv.reader(open("char_ids.csv", encoding='utf-8')):
        char_ids[char] = id
    # Load character counts from csv file
    char_counts = {}
    for char, count in csv.reader(open("char_counts.csv", encoding='utf-8')):
        char_counts[char] = count

    #Same procedure for words
    word_ids = {}
    for word, id in csv.reader(open("word_ids.csv", encoding='utf-8')):
        word_ids[word] = id
    # Load word counts from csv file
    word_counts = {}
    for word, count in csv.reader(open("word_counts.csv", encoding='utf-8')):
        word_counts[word] = count

    # Same procedure for word tags
    tag_ids = {}
    for tag, id in csv.reader(open("tag_ids.csv", encoding='utf-8')):
        tag_ids[tag] = id

    config.sample_prime = '.'

    if config.language_modeling == 'word':
        vocab_size = len(word_ids)
    elif config.language_modeling == 'simplified_word':
        vocab_size = len(tag_ids)
    else:
        vocab_size = len(char_ids)

    # Initialize neural network
    rnn_model = Rnn_Model(is_training=False, vocab_size=vocab_size, config=config)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # Restore trained model
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(config.checkpoint_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)

        # Initial internal network state
        state = sess.run(rnn_model.cell.zero_state(1, tf.float32))

        if config.language_modeling == 'word' or config.language_modeling == 'simplified_word':
            # Compute network state after feeding prime text
            for word in nltk.tokenize.word_tokenize(config.sample_prime[:-1]):
                x = np.zeros((1, 1))
                x[0, 0] = word_ids[word]
                feed_dict = {rnn_model.input_data: x, rnn_model.initial_state: state}
                [state] = sess.run([rnn_model.final_state], feed_dict)
        else:
            # Compute network state after feeding prime text
            for char in config.sample_prime[:-1]:
                x = np.zeros((1, 1))
                x[0, 0] = char_ids[char]
                feed_dict = {rnn_model.input_data: x, rnn_model.initial_state: state}
                [state] = sess.run([rnn_model.final_state], feed_dict)

        # Keep track of overall sampled text
        sampled_text = config.sample_prime

        #TODO change back: last_char = config.sample_prime[-1]

        state_before_word = state

        last_word = '.' #just start with a dot -> makes it easy to start new sentence

        #keep producing text until user interrupts
        while True:

            states_after_word = []
            sampled_words = []

            #sample 5 words, 1 max prob word
            for w in range(5):
                if config.language_modeling == 'word' or config.language_modeling == 'simplified_word':
                    if (w == 5):
                        sample_use_max_prob = True
                    else:
                        sample_use_max_prob = False

                    state = state_before_word
                    x = np.zeros((1, 1))
                    if config.language_modeling == 'word':
                        x[0, 0] = word_ids[last_word]
                    else:
                        x[0, 0] = tag_ids[last_word]

                    feed_dict = {rnn_model.input_data: x, rnn_model.initial_state: state}
                    [word_probabilities, state] = sess.run([rnn_model.probabilities, rnn_model.final_state], feed_dict)
                    next_word_probabilities = word_probabilities[0]

                    # Sample next character either by max prob (might result in repetitive texts) or by sampling from the softmax probability distribution
                    if sample_use_max_prob:
                        # Pick most probably character instead of sampling
                        sample = np.argmax(next_word_probabilities)
                    else:
                        # Sample from probability distribution
                        t = np.cumsum(next_word_probabilities)
                        s = np.sum(next_word_probabilities)
                        sample = (int(np.searchsorted(t, np.random.rand(1) * s)))

                    # Append prediction to sampled text
                    if config.language_modeling == 'word':
                        words = list()
                        for i in word_ids.keys():
                            words.append(i)
                        predicted_word = words[sample]
                    else:
                        words = list()
                        for i in tag_ids.keys():
                            words.append(i)
                        predicted_word = words[sample]

                    last_word = predicted_word

                    states_after_word.append(state)
                    sampled_words.append(predicted_word)

                else:
                    last_char = ' '

                    state = state_before_word
                    sampled_words.append("")
                    # add characters until space appears
                    while True:

                        # Feed characters into network one by one
                        x = np.zeros((1, 1))
                        x[0, 0] = char_ids[last_char]

                        feed_dict = {rnn_model.input_data: x, rnn_model.initial_state: state}
                        [char_probabilities, state] = sess.run([rnn_model.probabilities, rnn_model.final_state],
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

                        if predicted_char == ' ' and sampled_words[w] != '':
                            states_after_word.append(state)
                            break

                        sampled_words[w] += predicted_char

            #let user select next word
            print('\n\n'+sampled_text+'\n')
            for i in range(len(sampled_words)-1):
                print(str(i)+") "+sampled_words[i]+'\t\t\t'+str(fdist[sampled_words[i]])+"/"+str(text_length))
            print("M) " + sampled_words[len(sampled_words)-1]+'\t\t\t'+str(fdist[sampled_words[len(sampled_words)-1]])+"/"+str(text_length))
            input_var = input("Choose next word: ")
            while input_var != 'M' and (int(input_var) < 0 or int(input_var) > 4):
                input_var = input("Wrong input! Choose next word: ")
            if(input_var == 'M'):
                state_before_word = states_after_word[len(sampled_words)-1]
                sampled_text += sampled_words[len(sampled_words)-1] + ' '
            else:
                state_before_word = states_after_word[int(input_var)]
                sampled_text += sampled_words[int(input_var)]+' '


if __name__ == '__main__':
    sample(config=config.Standard_Config())