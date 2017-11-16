import codecs
import collections
import numpy as np
import csv
import nltk

#Inspired from https://github.com/sherjilozair/char-rnn-tensorflow

class Preprocessor():

    def __init__(self, train_file, validation_file, test_file, batch_size, sequence_length, char_id_file, word_id_file, tag_id_file, char_counts_file, word_counts_file, language_modeling):

        self.batch_size = batch_size

        # Read all 3 texts (training, test, validation) in order to setup vocabulary
        print('reading file...')
        with codecs.open(train_file, "r", encoding='utf-8') as f:
            training_text = f.read()
        with codecs.open(validation_file, "r", encoding='utf-8') as f:
            validation_text = f.read()
        with codecs.open(test_file, "r", encoding='utf-8') as f:
            test_text = f.read()

        if language_modeling == 'word':
            print("WORD LEVEL MODELING")
        elif language_modeling == 'simplified_word':
            print("SIMPLIFIED WORD LEVEL MODELING")
        else:
            print("CHARACTER LEVEL MODELING")

        # Generate vocabulary and character count file
        overall_text = training_text + validation_text + test_text

        if language_modeling == 'simplified_word':
            tokenized_text = nltk.word_tokenize(overall_text)
            tagged_text = nltk.pos_tag(tokenized_text)

            # print(tagged_text)

            # for word, tag in tagged_text:
            #    print(word + ": " + nltk.help.upenn_tagset(tag))

            text_as_ids = []
            tag_ids = {}

            for word, tag in tagged_text:
                if tag in tag_ids:
                    text_as_ids.append(tag_ids[tag])
                else:
                    tag_ids[tag] = len(tag_ids)
                    text_as_ids.append(tag_ids[tag])
            # Save TAG_IDS to csv file (careful with windows: might have formatting problems!)
            w = csv.writer(open(tag_id_file, "w", encoding='utf-8'), lineterminator='\n')
            for tag, id in tag_ids.items():
                w.writerow([tag, id])

        counter = collections.Counter(overall_text)
        char_counts = sorted(counter.items(), key=lambda x: -x[1])
        chars_, _ = zip(*char_counts)
        words = nltk.word_tokenize(overall_text)
        word_counter = collections.Counter(words)
        word_counts = sorted(word_counter.items(), key=lambda x: -x[1])
        words_, _ = zip(*word_counts)
        char_ids = dict(zip(chars_, range(len(chars_))))
        word_ids = dict(zip(words_, range(len(word_counts))))
        # Save char_ids to csv file (careful with windows: might have formatting problems!)
        w = csv.writer(open(char_id_file, "w", encoding = 'utf-8'), lineterminator = '\n')
        for char, id in char_ids.items():
            w.writerow([char, id])
        # Save word_ids to csv file (careful with windows: might have formatting problems!)
        w = csv.writer(open(word_id_file, "w",  encoding = 'utf-8'), lineterminator = '\n')
        for word, id in word_ids.items():
            w.writerow([word, id])
        # Save character counts to dict (for sampling first character)
        w = csv.writer(open(char_counts_file, "w", encoding = 'utf-8'), lineterminator = '\n')
        for char, count in dict(char_counts).items():
            w.writerow([char, count])
        # Save word counts to dict (for sampling first character)
        w = csv.writer(open(word_counts_file, "w",  encoding = 'utf-8'), lineterminator = '\n')
        for word, count in dict(word_counts).items():
            w.writerow([word, count])

        # Size of vocabulary
        if language_modeling == 'word':
            self.vocabulary_size = len(words_)
        elif language_modeling == 'simplified_word':
            self.vocabulary_size = len(tag_ids)
        else:
            self.vocabulary_size = len(chars_)

        # Create training batches
        print('generating batches...')
        if language_modeling == 'word':
            training_words = nltk.word_tokenize(training_text)
            inputs = []
            for training_word in training_words:
                if training_word in word_ids:
                    inputs.append(word_ids[training_word])
            inputs = np.array(inputs)
            print('input training text has {} words and a vocabulary size of {}'.format(len(training_words),
                                                                                        self.vocabulary_size))
        elif language_modeling == 'simplified_word':
            tokenized_training_text = nltk.word_tokenize(training_text)
            tagged_training_text = nltk.pos_tag(tokenized_training_text)

            for word, tag in tagged_training_text:
                text_as_ids.append(tag_ids[tag])
            inputs = np.array(text_as_ids)
            print('input training text has {} words and a vocabulary size of {}'.format(len(inputs),
                                                                                        self.vocabulary_size))
        else:
            inputs = np.array(list(map(char_ids.get, training_text)))
            print('input training text has {} characters and a vocabulary size of {}'.format(len(training_text),
                                                                                             self.vocabulary_size))
        self.num_batches = int(inputs.size / (batch_size * sequence_length))
        inputs = inputs[:self.num_batches * batch_size * sequence_length]
        targets = np.copy(inputs)
        targets[:-1] = inputs[1:]
        targets[-1] = inputs[0]
        self.input_batches = np.split(inputs.reshape(batch_size, -1), self.num_batches, 1)
        self.target_batches = np.split(targets.reshape(batch_size, -1), self.num_batches, 1)
        self.batch_pointer = 0

    # Fetch next training batch
    def get_next_batch(self):
        print('processing batch %d of %d...'%(self.batch_pointer,self.num_batches))
        batch_input, batch_target = self.input_batches[self.batch_pointer], self.target_batches[self.batch_pointer]
        self.batch_pointer += 1
        if self.batch_pointer == self.num_batches:
            self.batch_pointer = 0
        return batch_input, batch_target


    # Fetch next validation batch
    def get_next_validation_batch(self):
        print('processing batch %d of %d...' % (self.validation_batch_pointer,self.validation_num_batches))
        val_batch_input, val_batch_target = self.validation_input_batches[self.validation_batch_pointer], self.validation_target_batches[self.validation_batch_pointer]
        self.validation_batch_pointer += 1
        if self.validation_batch_pointer == self.validation_num_batches:
            self.validation_batch_pointer = 0
        return val_batch_input, val_batch_target