import config
from rnn_model import Rnn_Model
import tensorflow as tf
import numpy as np
import csv
import codecs
import argparse


def test(config):

    # Load test text
    with codecs.open(config.test_input_file, "r", encoding='utf-8') as f:
        text = f.read()

    # Start at char 5000 to initialize the last5000chars oracle
    text = text[5000:min(config.test_text_length+5000, len(text))]
    current_word = ""
    last_5000_chars = text[:5000]

    # Load training vocabulary from csv file
    char_ids = {}
    for char, id in csv.reader(open("char_ids.csv")):
        char_ids[char] = id

    # Initialize neural network
    rnn_model = Rnn_Model(is_training=False, vocab_size=len(char_ids), config=config)

    # Initialize test performance indicators
    test_cross_entropy_loss = 0.0
    chars_with_prob_over_50 = 0
    correct_char_count = [0.0]*len(char_ids)
    wrong_char_count = [0.0]*len(char_ids)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # Restore trained model
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(config.checkpoint_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)

        state = sess.run(rnn_model.cell.zero_state(1, tf.float32))


        text_pointer = 0
        states = []
        weights_i, weights_j, weights_f, weights_o = [], [], [], []
        biases_i, biases_j, biases_f, biases_o = [], [], [], []


        #iterate over test-text and check whether prediction was correct
        for index in range(len(text)-1):

            if index%1000 == 0:
                print("Testing on char "+str(index)+" of "+str(len(text)))

            current_char = text[text_pointer]
            # Get word that contains current char / store last 5000 chars: for error type visualization
            if current_char.isalpha():
                if current_word == "":
                    temp_pointer = text_pointer
                    while temp_pointer < len(text) and text[temp_pointer].isalpha():
                        current_word += text[temp_pointer]
                        temp_pointer += 1
            else:
                current_word = ""
            last_5000_chars += current_char
            if len(last_5000_chars) > 5000:
                last_5000_chars = last_5000_chars[1:]

            # Feed characters into network one by one
            x = np.zeros((1, 1))
            x[0, 0] = char_ids[current_char]
            feed_dict = {rnn_model.input_data: x, rnn_model.initial_state: state}
            [char_probabilities, state] = sess.run([rnn_model.probabilities, rnn_model.final_state], feed_dict)


            # Probability assigned to the correct character
            next_char_probabilities = char_probabilities[0]
            correct_next_char = text[text_pointer+1]
            correct_char_id = (int)(char_ids[correct_next_char])
            probability_for_correct_char = next_char_probabilities[correct_char_id]

            # Store the internal state in order to visualize the single cells' activation lateron
            if(config.visualize_cell_activations):
                states.append(state)

            # Add character loss to overall cross entropy loss
            test_cross_entropy_loss -= np.log(probability_for_correct_char)

            # Count to correctly/falsely classified characters
            if probability_for_correct_char < 0.5:
                wrong_char_count[correct_char_id] += 1
            else:
                chars_with_prob_over_50 += 1
                correct_char_count[correct_char_id] += 1

            text_pointer += 1

    # Compute and print overall test cross-entropy loss
    relative_test_cross_entropy_loss = test_cross_entropy_loss/((float)(len(text)-1))
    print("Test Cross Entropy Loss per character: "+str(relative_test_cross_entropy_loss))
    print(str(chars_with_prob_over_50)+" of "+str(len(text)-1)  +" were predicted correctly with probability >= 0.5")


if __name__ == '__main__':
    test(config=config.Standard_Config())