import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec

class SelfAttention(Layer):
    def __init__(self, num_heads=1, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.kernel = None  # Initialize kernel to None

    def build(self, input_shape): # BUILD method to create weights properly
        self.kernel = self.add_weight(
            name='attention_kernel',
            shape=(input_shape[-1], self.num_heads),
            initializer='uniform',
            trainable=True
        )

    def call(self, inputs):
        attention_scores = tf.matmul(inputs, self.kernel)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        
        # Transpose inputs to make dimensions compatible
        output = tf.matmul(attention_weights, tf.transpose(inputs, perm=[0, 2, 1]))
        return output

    def get_config(self):
        config = super().get_config()
        config.update({'num_heads': self.num_heads}) # save num_heads
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config) # load num_heads

    def compute_output_spec(self, input_spec):
        if not isinstance(input_spec, InputSpec):
            raise ValueError("This layer only supports InputSpec input.")
        
        shape = tf.TensorShape([input_spec.shape[0], self.num_heads, input_spec.shape[1]])
        
        self.output_spec = InputSpec(shape=shape)
