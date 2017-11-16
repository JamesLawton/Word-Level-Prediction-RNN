from __future__ import print_function
import tensorflow as tf
import numpy as np
from flexx import app, event, ui
import csv
import sys
from rnn_model import Rnn_Model
import config
import operator
import argparse
import codecs
import nltk
#from user_interface import User_interface
import tkinter as tk

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

        #keep producing text until user interrupts
        while True:
            states_chars_after_word = []
            sampled_words = []
            sampled_tags = []

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
            sample_list = []


            while len(sample_list) < 5:
            #Restrict sampling to non-punctuation chars
                prob_num = int(np.searchsorted(t, np.random.rand(1) * s))
                while prob_num == 23 or prob_num == 3:
                    prob_num = int(np.searchsorted(t, np.random.rand(1) * s))
                if prob_num not in sample_list:
                    sample_list.append(prob_num)

            tags = list()
            for i in tag_ids.keys():
                tags.append(i)

            predicted_tag = []

            for i in sample_list:
                predicted_tag.append(tags[i])
            print(predicted_tag)
            #last_tag = predicted_tag

            #states_chars_after_word.append(state_tags)
            #sampled_words.append(predicted_word)

            w = 0
            tries = 0
            # STEP 2: SAMPLE WORD ON CHARACTER LEVEL (filter by predicted Pos_Tag)
            #sample 5 words, 1 max prob word
            while w < 6:
                last_char = ' '

                state_chars = state_chars_before_word

                # TODO filter words by predicted Pos-Tag

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
                        if tagged_text[-1][1] in predicted_tag:
                            states_chars_after_word.append(state_chars)
                            #Only increase w if the sampled word is not in already in the list.
                            if sampled_word not in sampled_words:
                                sampled_words.append(sampled_word)
                                for words in predicted_tag:
                                    sampled_tags.append(words)
                                w += 1
                                tries = 0
                        #if 100 samples don't contain correct Tag, just use any sample
                        elif tries == 100:
                            print(predicted_tag)
                            print(tagged_text[-1][1])
                            print("WRONG")
                            states_chars_after_word.append(state_chars)
                            if sampled_word not in sampled_words:
                                sampled_words.append(sampled_word)
                                sampled_tags.append(tagged_text[-1][1])
                                w += 1
                                tries = 0
                                print("100 tries but no sample with correct POS tag")
                        break

                    sampled_word += predicted_char


            print(sampled_words)



            """
            class User_interface(ui.Widget):

                def init(self):
                    with ui.HBox(flex = 0.5):

                        self.b1 = ui.Button(flex=.25, text="testing1")
                        self.b2 = ui.Button(flex=.25, text="testing2")
                        self.b3 = ui.Button(flex=.25, text="testing3")
                        self.b4 = ui.Button(flex=.25, text="testing4")
                        self.b5 = ui.Button(flex=.25, text="testing5")
                        #for i in range(len(sampled_words) - 1):

                        self.buttonlabel = ui.Label(text = sampled_text)


                @event.connect('b1.mouse_click')
                def _button1_click(self, *events):
                    self.b1.text = "Updated text"
                    self.buttonlabel.text = 'Clicked on a button'
                @event.connect('b2.mouse_click')
                def _button2_click(self, *events):
                    self.b2.text = "Upasdfassd text"

                @event.connect('b3.mouse_click')
                def _button3_click(self, *events):
                    self.b3.text = "sfdatext"

                @event.connect('b4.mouse_click')
                def _button4_click(self, *events):
                    self.b4.text = "more text"

                @event.connect('b5.mouse_click')
                def _button5_click(self, *events):
                    self.b5.text = "fasdf text"

            w3 = app.launch(User_interface)
            app.run()
            """
            print('\n\n'+sampled_text+'\n')
            if (config.user_selects_words):
                for i in range(len(sampled_words)-1):
                    print(str(i)+") "+sampled_words[i]+'\t\t\t'+str(fdist[sampled_words[i]])+"/"+str(text_length))
                input_var = input("Choose next word: ")
                while (int(input_var) < 0 or int(input_var) > 4):
                    input_var = input("Wrong input! Choose next word: ")
            else:
                input_var = 0
            sampled_text += sampled_words[int(input_var)]+' '

            state_chars_before_word = states_chars_after_word[int(input_var)]
            last_tag = sampled_tags[int(input_var)]




if __name__ == '__main__':
    sample(config=config.Standard_Config())
