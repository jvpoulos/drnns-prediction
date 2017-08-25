from keras.layers.core import Layer  
from keras import backend as K

class Attention(Layer):
    def __init__(self, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.

        Put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        super(Attention, self).__init__(**kwargs)


    def compute_mask(self, x, input_mask=None):
        # Do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        a = K.softmax(x)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1],)