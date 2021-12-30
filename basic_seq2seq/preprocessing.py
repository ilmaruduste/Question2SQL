from . import encoding
from . import decoding

import tensorflow as tf
import numpy as np
import pandas as pd
import os

class Preprocessor():

    def tf_lower_and_split_punct(self, text):
        text = tf.strings.lower(text)
        # Keep space, a to z, and select punctuation.
        text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
        # Add spaces around punctuation.
        text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
        # Strip whitespace.
        text = tf.strings.strip(text)
        text = tf.strings.join(['[START]', text, '[END]'], separator=' ')

        return text

    def load_and_preprocess_data(self, data_json_path, concatenate_table_data = False):
        print(f"Loading and preprocessing data from {data_json_path}!")
        dataset_json = pd.read_json(data_json_path)

        if concatenate_table_data:
            print("Concatenating table and column names to each input query!")
            table_data = pd.read_json('spider/tables.json')
            joined_data = pd.merge(left = dataset_json, right = table_data, on = "db_id", how = "left")
            
            # The next line concatenates the table name and column names to each input question
            joined_data['concatenated_input'] = joined_data.apply(
                lambda x: x['question'] + 
                    ', '.join(x['table_names_original']) + 
                    ', '.join([column_name[1] for column_name in x['column_names_original']]), 
                axis = 1)

            input = np.array(joined_data['concatenated_input'])
            target = np.array(joined_data['query'])
        else:
            input = np.array(dataset_json['question'])
            target = np.array(dataset_json['query'])

        self.input, self.target = self.tf_lower_and_split_punct(input), self.tf_lower_and_split_punct(target)

    def create_text_processors(self, max_vocab_size = 5000):
        print("Creating text processors!")
        input_text_processor = tf.keras.layers.TextVectorization(
            standardize=self.tf_lower_and_split_punct,
            max_tokens=max_vocab_size)
        input_text_processor.adapt(self.input)

        output_text_processor = tf.keras.layers.TextVectorization(
            standardize=self.tf_lower_and_split_punct,
            max_tokens=max_vocab_size)
        output_text_processor.adapt(self.target)

        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor


    def create_encoder_decoder(self, embedding_dim = 256, units = 1024):
        print("Creating encoder and decoder!")
        encoder = encoding.Encoder(self.input_text_processor.vocabulary_size(),
                        embedding_dim, units)

        decoder = decoding.Decoder(self.output_text_processor.vocabulary_size(),
                    embedding_dim, units)

        self.encoder = encoder
        self.decoder = decoder

    def create_dataset(self, BATCH_SIZE = 16):
        BUFFER_SIZE = len(self.input)

        dataset = tf.data.Dataset.from_tensor_slices((self.input, self.target))
        dataset.shuffle(BUFFER_SIZE)
        dataset = dataset.batch(BATCH_SIZE)

        self.dataset = dataset
    
    def create_word_index(self):
        from tf.keras.layers import TextVectorization

        vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=200)
        text_ds = tf.data.Dataset.from_tensor_slices(self.input).batch(128)
        vectorizer.adapt(text_ds)
        voc = vectorizer.get_vocabulary()
        word_index = dict(zip(voc, range(len(voc))))

        self.voc = voc
        self.word_index = word_index

        return word_index

    # https://keras.io/examples/nlp/pretrained_word_embeddings/
    def create_glove_embeddings_index(self):
        embeddings_index = {}
        glove_filepath = os.path.join('./glove/', 'glove.42B.300d.txt')
        print(f"glove_filepath: {glove_filepath}")
        with open(glove_filepath) as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs
        self.embeddings_index = embeddings_index
        return embeddings_index

    def create_glove_embeddings_matrix(self):
        self.create_word_index()
        num_tokens = len(self.voc) + 2
        embedding_dim = 100
        hits = 0
        misses = 0

        # Prepare embedding matrix
        embedding_matrix = np.zeros((num_tokens, embedding_dim))
        for word, i in self.create_word_index().items():
            embedding_vector = self.create_glove_embeddings_index().get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                misses += 1
        print("Converted %d words (%d misses)" % (hits, misses))

        self.embedding_matrix = embedding_matrix
        return embedding_matrix
