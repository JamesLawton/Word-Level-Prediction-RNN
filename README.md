# Write like your favorite author - A word-level prediction RNN

Typically a word-level RNN utilizes a similar strategy of a character RNN, which takes training text and 
predicts the next most likely character according to the previously generated characters. Since the characters
sample space is small and consistent for the majority of training texts, this type of approach works well. Problems occur
when the sample space of possible predictions is much larger as in the word-level case. For this reason, this implementation
utilizes two complementary RNNs, a traditional character model RNN and a parts of speech tagger RNN that creates a representation
of the author's writing style and semantics according to the training text.

In order to use the RNN, simply modify the config file with your corresponding training text filepath.
Then train each RNN individually by setting the appropriate configuration. Once trained, copy the checkpoint files to the corresponding subfolder.
Once the RNNs have been trained, run the sample.py file to begin writing like your favorite author!