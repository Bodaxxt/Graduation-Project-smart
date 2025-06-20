
# self_attention.py

import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable()
class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # مثلاً نضع الأوزان هنا
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel"
        )
        super().build(input_shape)

    def call(self, inputs):
        # هنا حط اللوجيك بتاع الـ attention
        # مثال بسيط:
        attention_scores = tf.matmul(inputs, self.kernel)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        output = tf.matmul(attention_weights, inputs)
        return output

    def get_config(self):
        config = super(SelfAttention, self).get_config()
        config.update({"units": self.units})
        return config
