import os

import tensorflow as tf
from preprocessor import Preprocessor
from rnn_model import Rnn_Model
import config
import datetime

#Inspired from https://github.com/sherjilozair/char-rnn-tensorflow

def train(config):
    with tf.variable_scope(config.language_modeling):

        starttime = datetime.datetime.now()

        #Fetch input info from preprocessor module
        preprocessor = Preprocessor(train_file=config.train_input_file, validation_file=config.validation_input_file, test_file=config.test_input_file, batch_size = config.batch_size, sequence_length = config.sequence_length, char_id_file="char_ids.csv", word_id_file="word_ids.csv", tag_id_file="tag_ids.csv", char_counts_file="char_counts.csv", word_counts_file="word_counts.csv", language_modeling=config.language_modeling)
        vocabulary_size = preprocessor.vocabulary_size
        num_batches = preprocessor.num_batches

        # Initialize neural network
        rnn_model = Rnn_Model(is_training=True, vocab_size=vocabulary_size, config=config)

        with tf.Session() as sess:

            saver = tf.train.Saver(tf.global_variables())

            if (config.load_model):
                print('Loading Model...')
                ckpt = tf.train.get_checkpoint_state(config.checkpoint_dir)
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Model Restored')
            else:
                sess.run(tf.global_variables_initializer())
            # For early stopping: track the validation losses of the last x epochs
            minimum_validation_loss = float("inf")
            epochs_without_improvement = 0


            for epoch_number in range(config.num_epochs):
                print('starting epoch %d...'%epoch_number)

                #As stated in the paper, decrease the learning rate by a decay factor each epoch, starting after 10 epochs
                if epoch_number > config.lr_decay_start_epoch:
                    sess.run(tf.assign(rnn_model.learning_rate, config.learning_rate * (config.learning_rate_decay ** epoch_number)))

                #Get current network configuration
                state = sess.run(rnn_model.initial_state)

                for batch_number in range(num_batches):
                    batch_inputs, batch_targets = preprocessor.get_next_batch()

                    #Feed inputs, targets and configuration from previous time step (for LSTM) into network
                    feed_dict = {rnn_model.input_data: batch_inputs, rnn_model.targets: batch_targets}
                    if config.rnn_type == 'lstm':
                        for i, (c, h) in enumerate(rnn_model.initial_state):
                            feed_dict[c] = state[i].c
                            feed_dict[h] = state[i].h

                    # For tensorboard visualizations
                    if epoch_number % 10 == 0 and batch_number == 0 or epoch_number % 10 == 0 and batch_number > 2000: #must first initialize summaries, but if it's ran every batch training is significantly slower
                        summaries = tf.summary.merge_all()
                    summary, train_loss, state, _ = sess.run(
                        [summaries, rnn_model.cost, rnn_model.final_state, rnn_model.train_op], feed_dict)
                    if epoch_number % 10 == 0 and batch_number == 0 or epoch_number % 10 == 0 and batch_number > 2000:
                        writer = tf.summary.FileWriter(
                            os.path.join('logs', 'test'), flush_secs = 60)
                        writer.add_graph(sess.graph)



                        writer.add_summary(summary, epoch_number * num_batches + batch_number)
                        writer.flush()
                    print("overall batch {}/{} (epoch {}), train_loss = {:.3f}".format(epoch_number * num_batches + batch_number, config.num_epochs * num_batches,epoch_number, train_loss))

                    # Store checkpoint
                    if (epoch_number * num_batches + batch_number) % config.save_interval == 0 or (epoch_number == config.num_epochs - 1 and batch_number == num_batches - 1):
                        checkpoint_path = os.path.join('checkpoints', 'model.ckpt')
                        saver.save(sess, checkpoint_path+str(batch_number), global_step=epoch_number * num_batches + batch_number)
                        print("model stored in " + checkpoint_path)

                print("epoch finished. starting validation (for early stopping). overall training time: "+str(datetime.datetime.now()-starttime))
                ts1 = datetime.datetime.now()

        endtime = datetime.datetime.now()
        delta = endtime - starttime
        print("overall training time: "+str(delta))

if __name__ == '__main__':
    train(config=config.Standard_Config())