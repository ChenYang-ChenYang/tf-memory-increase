import tensorflow as tf
from tensorflow.python.keras import layers

class Encoder(layers.Layer):
    """
    encoder layer with biLSTM
    """

    def __init__(self, embedding, hidden_size, batch_size):
        super(Encoder, self).__init__()
        self.embedding = embedding
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.bilstm = layers.Bidirectional(layers.CuDNNLSTM(self.hidden_size, return_sequences=True, return_state=True),
                                           merge_mode='concat')

    def call(self, inputs, pre_state=None):
        """
        encode a batch of input
        Args:
            inputs: to be encoded input, shape is (batch_size, time_steps)
            pre_state: initial state, structure is [fw_h, fw_c, bw_h, bw_c]
        Returns:
            bi-directional LSTM encoder output(sequence), shape is (batch_size, time_steps, 2*hidden_size)
            bi-directional LSTM encoder state, a list containing state_h and state_c,
            both state shape are (batch_size, 2*hidden_size)
        """
        inputs = self.embedding(inputs)
        output = self.bilstm(inputs, initial_state=pre_state)
        sequence = output[0]
        state = output[1:]
        encoder_fw_h = state[0]
        encoder_fw_c = state[1]
        encoder_bw_h = state[2]
        encoder_bw_c = state[3]
        encoder_final_state_h = tf.concat((encoder_fw_h, encoder_bw_h), 1)
        encoder_final_state_c = tf.concat((encoder_fw_c, encoder_bw_c), 1)
        encoder_final_state = [encoder_final_state_h, encoder_final_state_c]

        return sequence, encoder_final_state

    def initialize_fw_hidden_state(self):
        return tf.zeros((self.batch_size, self.hidden_size))

    def initialize_bw_hidden_state(self):
        return tf.zeros((self.batch_size, self.hidden_size))

    def initialize_fw_cell_state(self):
        return tf.zeros((self.batch_size, self.hidden_size))

    def initialize_bw_cell_state(self):
        return tf.zeros((self.batch_size, self.hidden_size))

class Decoder(layers.Layer):
    """
    decoder layer with LSTM
    """

    def __init__(self, intent_size):
        super(Decoder, self).__init__()
        self.intent_dense = layers.Dense(intent_size, activation='sigmoid', name='intent_output_dense')

    def call(self, encoder_outputs_sequence, encoder_final_state, **kwargs):
        """
        decode the encoder output to get intent prediction
        """
        intent_logits = self.intent_dense(encoder_final_state[0])
        return intent_logits


class TestModel(tf.keras.Model):
    """
    model with encoder and decoder layers
    """
    def __init__(self, vocab_size, embedding_size, hidden_size, batch_size, intent_size):
        super(TestModel, self).__init__()
        self.embedding = layers.Embedding(input_dim= vocab_size,
                                               output_dim= embedding_size,
                                               embeddings_initializer='uniform',
                                               name='embedding'
                                               )
        # initialize encoder and decoder
        self.encoder = Encoder(self.embedding, hidden_size, batch_size)
        self.decoder = Decoder(intent_size)

    def call(self, inputs):
        """
        call encoder to encode input and decode to get result
        """
        encoder_fw_h = self.encoder.initialize_fw_hidden_state()
        encoder_fw_c = self.encoder.initialize_fw_cell_state()
        encoder_bw_h = self.encoder.initialize_bw_hidden_state()
        encoder_bw_c = self.encoder.initialize_bw_cell_state()
        encoder_state = [encoder_fw_h, encoder_fw_c, encoder_bw_h, encoder_bw_c]
        sequence, encoder_final_state = self.encoder(inputs, encoder_state)
        intent_logits = self.decoder(sequence, encoder_final_state)
        return intent_logits
