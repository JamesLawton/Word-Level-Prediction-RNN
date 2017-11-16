
class Standard_Config(object):
    """Standard rnn config."""

    count_tries_for_right_word = False
    user_selects_words = True

    #CHARACTER OR WORD LEVEL MODELING
    language_modeling = 'simplified_word' # character | word | simplified_word
    maximum_tries = 10 # for mixed_sampling / testing -> if no word with correct POS_TAG is sampled, just take a random one

    # FILES
    train_input_file = 'input_data/books_1_to_6.txt'
    test_input_file = 'input_data/book_7_test.txt'
    validation_input_file = 'input_data/book_7_validation.txt'
    checkpoint_dir='checkpoints'
    checkpoint_dir_character_model = 'checkpoints_character_model'
    checkpoint_dir_simplified_word_model = 'checkpoints_simplified_word_model'
    log_dir = 'logs'

    # ARCHITECTURE
    hidden_size = 512     # 64 | 128 | 256 | 512
    num_layers = 3       # 1 | 2 | 3
    batch_size = 50
    rnn_type = 'lstm'     # vanilla | lstm | gru

    # TRAINING
    dropout_keep_prob = 0.8
    num_epochs = 100
    save_interval = 500
    max_grad_norm = 5.0   # for gradient clipping
    sequence_length = 50
    learning_rate = 0.002
    learning_rate_decay = 0.95
    rms_prop_decay = 0.95
    lr_decay_start_epoch = 10
    max_epochs_without_improvement = 3  # for early stopping
    early_stopping = True
    load_model = False
    #configure early stopping

    # SAMPLING
    sampled_text_length = 20000
    sample_use_max_prob = False
    sample_prime = ''

    #TESTING
    test_text_length = 50000
    visualize_cell_activations = False
    visualize_gate_saturations = False
    visualize_error_oracles = False
    visualize_char_probabilities = True