from . import shape_check
import tensorflow as tf
import numpy as np
import os

class Encoder(tf.keras.layers.Layer):
  def __init__(self, input_vocab_size, embedding_dim, enc_units, embedding_matrix = None):
    super(Encoder, self).__init__()
    self.enc_units = enc_units
    self.input_vocab_size = input_vocab_size
    self.embedding_matrix = embedding_matrix

    # The embedding layer converts tokens to vectors
    if self.embedding_matrix is not None:

      print("Using pretrained GLoVe for Embedding layer!")
      self.embedding = tf.keras.layers.Embedding(
          self.input_vocab_size,
          embedding_dim,
          embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
          trainable=False,
)

    else:
      print("Using Keras' Embedding layer!")
      self.embedding = tf.keras.layers.Embedding(self.input_vocab_size,
                                               embedding_dim)

    # The GRU RNN layer processes those vectors sequentially.
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   # Return the sequence and state
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, tokens, state=None):
    shape_checker = shape_check.ShapeChecker()
    shape_checker(tokens, ('batch', 's'))

    # 2. The embedding layer looks up the embedding for each token.
    vectors = self.embedding(tokens)
    shape_checker(vectors, ('batch', 's', 'embed_dim'))

    # 3. The GRU processes the embedding sequence.
    #    output shape: (batch, s, enc_units)
    #    state shape: (batch, enc_units)
    output, state = self.gru(vectors, initial_state=state)
    shape_checker(output, ('batch', 's', 'enc_units'))
    shape_checker(state, ('batch', 'enc_units'))

    # 4. Returns the new sequence and its state.
    return output, state