import tensorflow as tf

#Inspired from https://github.com/sherjilozair/char-rnn-tensorflow

class Rnn_Model():

   def __init__(self, is_training, vocab_size, config):
        with tf.variable_scope(config.language_modeling):
            print('initializing rnn model...')

            # For testing / sampling mode, just 1 character is fed into the network at a time
            if not is_training:
                config.sequence_length = 1
                config.dropout_keep_prob = 1
                config.batch_size = 1

            # One network layer
            def rnn_cell():
                if config.rnn_type == 'lstm':
                    rnn_cell = tf.contrib.rnn.BasicLSTMCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True)
                elif config.rnn_type == 'gru':
                    rnn_cell = tf.contrib.rnn.GRUCell(config.hidden_size)
                else:  # vanilla
                    rnn_cell = tf.contrib.rnn.BasicRNNCell(config.hidden_size)
                if is_training and config.dropout_keep_prob < 1:
                    return tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=config.dropout_keep_prob)
                return rnn_cell

            # Multiple network layers stacked
            self.cell = tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

            # Initial internal network state
            self.initial_state = self.cell.zero_state(config.batch_size, tf.float32)

            # Inputs, targets and outputs
            self.input_data = tf.placeholder(tf.int32, [config.batch_size, config.sequence_length])
            self.targets = tf.placeholder(tf.int32, [config.batch_size, config.sequence_length])
            embedding = tf.get_variable("embedding"+config.language_modeling, [vocab_size, config.hidden_size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)
            if is_training and config.dropout_keep_prob < 1:
                inputs = tf.nn.dropout(inputs, config.dropout_keep_prob)
            inputs = tf.split(inputs, config.sequence_length, 1)
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
            outputs, state = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, self.initial_state, self.cell)
            output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])

            # Trainable parameters
            softmax_w = tf.get_variable("softmax_w"+config.language_modeling, [config.hidden_size, vocab_size], dtype=tf.float32)
            softmax_b = tf.get_variable("softmax_b"+config.language_modeling, [vocab_size], dtype=tf.float32)

            # Computation of logits, softmax probabilities and loss tensor
            self.logits = tf.matmul(output, softmax_w) + softmax_b
            loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([self.logits],[tf.reshape(self.targets, [-1])],[tf.ones([config.batch_size * config.sequence_length])])
            self.cost = tf.reduce_sum(loss) / config.batch_size / config.sequence_length
            self.probabilities = tf.nn.softmax(self.logits)

            tf.summary.scalar('training loss', self.cost)

            # Gradient clipping to fight exploding gradients
            trainable_vars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_vars), config.max_grad_norm)

            # Instead of AdamOptimizer, use SGD with rmsProp for optimization as described in the paper
            self.learning_rate = tf.Variable(config.learning_rate, trainable=False)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=config.rms_prop_decay)

            # Optimization step
            self.train_op = optimizer.apply_gradients(zip(grads, trainable_vars), global_step=tf.contrib.framework.get_or_create_global_step())

            # New internal network state
            self.final_state = state