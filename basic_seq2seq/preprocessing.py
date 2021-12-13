from . import encoding
from . import decoding

import tensorflow as tf
import numpy as np
import pandas as pd

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

    def load_and_preprocess_data(self, data_json_path):
        print(f"Loading and preprocessing data from {data_json_path}!")
        train_spider = pd.read_json(data_json_path)
        input = np.array(train_spider['question'])
        target = np.array(train_spider['query'])

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

    def create_sample_dataset(self):
        BUFFER_SIZE = len(self.input)
        BATCH_SIZE = 64

        dataset = tf.data.Dataset.from_tensor_slices((self.input, self.target))
        dataset.shuffle(BUFFER_SIZE)
        dataset = dataset.batch(BATCH_SIZE)

        self.sample_dataset = dataset