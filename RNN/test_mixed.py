from __future__ import print_function
import tensorflow as tf
import numpy as np
import csv
from rnn_model import Rnn_Model
import config
import codecs
import nltk
import random
import collections
import string
import matplotlib as plt

#Inspired from https://github.com/sherjilozair/char-rnn-tensorflow


def sample(config):


    #create frequency distribution
    with codecs.open(config.train_input_file, "r", encoding='utf-8') as f:
        text = f.read()
    words = nltk.tokenize.word_tokenize(text)
    word_counter = collections.Counter(words)
    word_counts = sorted(word_counter.items(), key=lambda x: -x[1])
    words_dictionary = list(list(zip(*word_counts))[0])
    iteration_count = 0

    with codecs.open(config.test_input_file, "r", encoding='utf-8') as f:
        test_text = f.read()
    test_words = nltk.tokenize.word_tokenize(test_text)
    tagged_test_text = nltk.pos_tag(test_words)

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

        attempts_for_correct_word = 0
        count_mixed_prediction_better = 0
        count_char_prediction_better = 0
        count_dict_prediction_better = 0
        count_no_prediction_better = 0
        total_tries_char = 0
        total_tries_mixed = 0
        total_tries_dict = 0

        while True: # test sentences until user stops program

            # Initial internal network state
            state_tags = sess.run(rnn_model_tags.cell.zero_state(1, tf.float32))
            state_chars = sess.run(rnn_model_chars.cell.zero_state(1, tf.float32))

            state_chars_before_word = state_chars

            sampled_text = ""

            #test for a specific part of the test text
            start_index = random.randint(0, len(test_words)-20)
            #feed words into tag and character networks
            for i in range(20):
                #feed words into tag network
                x = np.zeros((1, 1))
                x[0, 0] = tag_ids[tagged_test_text[start_index+i][1]]
                feed_dict = {rnn_model_tags.input_data: x, rnn_model_tags.initial_state: state_tags}
                [word_probabilities, state_tags] = sess.run([rnn_model_tags.probabilities, rnn_model_tags.final_state],
                                                        feed_dict)
                #feed in words (character by character) into character network
                for c in test_words[start_index+i]:
                    # Feed characters into network one by one
                    x = np.zeros((1, 1))
                    x[0, 0] = char_ids[c]
                    feed_dict = {rnn_model_chars.input_data: x, rnn_model_chars.initial_state: state_chars}
                    [state_chars] = sess.run(
                        [rnn_model_chars.final_state],
                        feed_dict)
                sampled_text += ' '+test_words[start_index+i] if not test_words[start_index+i].startswith("'") and test_words[start_index+i] not in string.punctuation else test_words[start_index+i]

            next_word_probabilities = word_probabilities[0]
            # Sample from probability distribution
            t = np.cumsum(next_word_probabilities)
            s = np.sum(next_word_probabilities)

            if not config.count_tries_for_right_word:
                sample = (int(np.searchsorted(t, np.random.rand(1) * s)))
                tags = list()
                for i in tag_ids.keys():
                    tags.append(i)
                predicted_tag = tags[sample]
            else:
                sample_list = []
                while len(sample_list) < 5:
                    # Restrict sampling to non-punctuation chars
                    prob_num = int(np.searchsorted(t, np.random.rand(1) * s))
                    if prob_num not in sample_list:
                        sample_list.append(prob_num)

                tags = list()
                for i in tag_ids.keys():
                    tags.append(i)

                predicted_tag = []

                for i in sample_list:
                    predicted_tag.append(tags[i])



            correct_word = test_words[start_index + 20]

            tries = 0
            print("\nSentence: "+sampled_text)
            if (config.count_tries_for_right_word):
                print("ACTUAL WORD: "+correct_word)
            right_tag_found = False
            right_dict_word_found = False

            # keep on sampling words until right word appears
            if (config.count_tries_for_right_word):
                character_count_until_right_word = 0
                mixed_count_until_right_word = 0
                dict_count_until_right_word = 0
                flag_for_character = False
                flag_for_mixed = False
                flag_for_dict = False
            prediction_character = ""
            prediction_mixed = ""
            prediction_dict = ""


            # SAMPLE NEXT WORD
            # try words until break because right tag / too many tries
            while (not config.count_tries_for_right_word and tries < 100 and not (right_tag_found and right_dict_word_found)) or (config.count_tries_for_right_word and tries < 3000):
                last_char = ' '

                state_chars = state_chars_before_word

                sampled_word = ""
                # add characters until word is completed
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


                    #WORD COMPLETED
                    if predicted_char == ' ' and sampled_word != '':
                        # test the tag of predicted word
                        text_with_word = sampled_text + sampled_word
                        if config.count_tries_for_right_word:
                            character_count_until_right_word += 1
                            if sampled_word == correct_word and flag_for_character == False:
                                print("CORRECT WORD FOUND BY CHAR MODEL AFTER "+str(character_count_until_right_word)+"tries")
                                flag_for_character = True
                                print("0) CHARACTER MODEL PREDICTION: " + sampled_word)
                                total_tries_char += character_count_until_right_word
                            if character_count_until_right_word == 500 and flag_for_character == False:
                                total_tries_char += 500
                                print("CHAR TEST")
                        if (tries == 0 and not config.count_tries_for_right_word):
                            #print("0) CHARACTER MODEL PREDICTION: "+sampled_word)
                            if sampled_word == correct_word:
                                count_char_prediction_better += 1
                            if (not config.count_tries_for_right_word):
                                prediction_character = sampled_word
                        tries += 1
                        tokenized_text = nltk.word_tokenize(text_with_word)
                        tagged_text = nltk.pos_tag(tokenized_text)
                        #check if correct tag
                        if (tagged_text[-1][1] == predicted_tag and not right_tag_found) or (tagged_text[-1][1] in predicted_tag and config.count_tries_for_right_word):
                            if (not config.count_tries_for_right_word):
                                prediction_mixed = sampled_word
                                #print("1) MIXED MODEL PREDICTION: "+sampled_word)
                            if config.count_tries_for_right_word:
                                if (sampled_word == correct_word and flag_for_mixed == False):
                                    print("CORRECT WORD FOUND BY MIXED MODEL AFTER " + str(mixed_count_until_right_word) + "tries")
                                    print("1) MIXED MODEL PREDICTION: "+sampled_word)
                                    total_tries_mixed += mixed_count_until_right_word
                                    flag_for_mixed = True
                            right_tag_found = True
                            if sampled_word == correct_word:
                                count_mixed_prediction_better += 1
                            if config.count_tries_for_right_word:
                                mixed_count_until_right_word += 1
                        if config.count_tries_for_right_word and flag_for_mixed == False and mixed_count_until_right_word == 500:
                            total_tries_mixed += 500
                        #check if word in dictionary (of training text)
                        if sampled_word in words_dictionary and (not right_dict_word_found or config.count_tries_for_right_word):
                            if (not config.count_tries_for_right_word):
                                prediction_dict = sampled_word
                                #print("2) CHAR MODEL WITH DICTIONARY: "+sampled_word)
                            if config.count_tries_for_right_word:

                                if (sampled_word == correct_word and flag_for_dict == False):
                                    print("CORRECT WORD FOUND BY DICT MODEL AFTER " + str(
                                        dict_count_until_right_word) + "tries")
                                    print("2) CHAR MODEL WITH DICTIONARY: " + sampled_word)
                                    flag_for_dict = True
                                    total_tries_dict += dict_count_until_right_word

                            right_dict_word_found = True
                            if sampled_word == correct_word:
                                count_dict_prediction_better += 1
                            if config.count_tries_for_right_word:
                                dict_count_until_right_word += 1
                        if config.count_tries_for_right_word and dict_count_until_right_word == 500 and flag_for_dict == False:
                            total_tries_dict += 500
                            print("DICT TEST")
                        break
                    else:
                        sampled_word += predicted_char


            if not right_tag_found:
                # if 100 samples don't contain correct Tag, just use any sample
                #print("1) MIXED MODEL PREDICTION (correct tag not found): "+sampled_word)
                prediction_mixed = sampled_word
                if config.count_tries_for_right_word:
                    total_tries_mixed += 500
            if not right_dict_word_found:
                # if 100 samples don't contain correct Tag, just use any sample
                #print("2) CHAR MODEL WITH DICT PREDICTION (correct dict word not found): "+sampled_word)
                prediction_dict = sampled_word
                if config.count_tries_for_right_word:
                    total_tries_dict += 500
            iteration_count += 1

            if (config.count_tries_for_right_word):
                print("Amount of sentences: " + str(iteration_count))
                print ("Total tries for character: " + str(total_tries_char))
                print("Total tries for mixed: " + str(total_tries_mixed))
                print("Total tries for dictionary: " + str(total_tries_dict))



            if not config.count_tries_for_right_word:
                permutation = [0,1,2]
                random.shuffle(permutation)
                predictions = [prediction_character+"c", prediction_mixed+"m", prediction_dict+"d"]
                input_var = input(
                    "Make an appropriate choice for the next word!\n0) "+predictions[permutation[0]]+"\n1) "+predictions[permutation[1]]+"\n2) "+predictions[permutation[2]]+"\n3) ***None***\n4) STOP\n")
                while (int(input_var) < 0 or int(input_var) > 4):
                    input_var = input(
                        "What could be the next word? (0 = character, 1 = mixed, 2 = char_dict, 3 = none, 4 = STOP): ")
                if int(input_var) == permutation[0]:
                    correct_word = prediction_character
                elif int(input_var) == permutation[1]:
                    correct_word = prediction_mixed
                elif int(input_var) == permutation[2]:
                    correct_word = prediction_dict
                elif int(input_var) == 3:
                    count_no_prediction_better += 1
                else:
                    print("Stopping test program.")
                    break
                if correct_word == prediction_character:
                    count_char_prediction_better += 1
                if correct_word == prediction_mixed:
                    count_mixed_prediction_better += 1
                if correct_word == prediction_dict:
                    count_dict_prediction_better += 1
                print("\nCharacter vs.Mixed vs.Dict score: "+str(count_char_prediction_better)+" - "+str(count_mixed_prediction_better)+" - "+str(count_dict_prediction_better)+"; ***None*** score: "+str(count_no_prediction_better));





if __name__ == '__main__':
    sample(config=config.Standard_Config())
