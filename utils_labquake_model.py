"""
© 2022. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.
"""

import numpy as np
import tensorflow as tf
from kernel_attention import KernelAttention

class Conv1DLayer(tf.keras.layers.Layer):
  """Conv1D/Conv1DT layer"""
  def __init__(self, filters=32, kernel_size=3, strides=1, padding='same', dilation=1, use_transpose=False):
    super(Conv1DLayer, self).__init__()
    
    if not use_transpose:
      self.conv_1 = tf.keras.layers.Conv1D(filters=filters, 
                                         kernel_size=kernel_size, 
                                         strides=strides, 
                                         padding=padding, 
                                         dilation_rate=dilation, 
                                         activation=None, 
                                         use_bias=True, 
                                         kernel_initializer='he_normal')
    else:
      self.conv_1 = tf.keras.layers.Conv1DTranspose(filters=filters, 
                                         kernel_size=kernel_size, 
                                         strides=strides, 
                                         padding=padding, 
                                         dilation_rate=dilation, 
                                         activation=None, 
                                         use_bias=True, 
                                         kernel_initializer='he_normal')

  def call(self, inputs, training=False):
    x = self.conv_1(inputs)
    return x

class ResConv1DLayer(tf.keras.layers.Layer):
  """Residual Conv1D layer for music waveform encoder as in Jukebox: A generative model for music"""
  def __init__(self, filters=32, dilation=1):
    super(ResConv1DLayer, self).__init__()
    
    self.conv_1 = Conv1DLayer(filters=filters, 
                              kernel_size=3, 
                              strides=1, 
                              padding='same', 
                              dilation=dilation, 
                              use_transpose=False)
    
    self.conv_2 = Conv1DLayer(filters=filters, 
                              kernel_size=1, 
                              strides=1, 
                              padding='valid', 
                              dilation=1, 
                              use_transpose=False)
    
    self.act_1 = tf.keras.layers.Activation('relu')
    self.act_2 = tf.keras.layers.Activation('relu')
    
  def call(self, inputs, training=False):
    x = self.act_1(inputs)
    x = self.conv_1(x)
    x = self.act_2(x)
    x = self.conv_2(x)
    return x + inputs

class ResConv1DStack(tf.keras.layers.Layer):
  """Residual Conv1D layers Stack for music waveform encoder as in Jukebox: A generative model for music"""
  def __init__(self, filters=32, num_res_layers=4, dilation_growth_rate=3, reverse_dilation=False):
    super(ResConv1DStack, self).__init__()
    
    self.conv_layers = []
    for ii in range(num_res_layers):
      self.conv_layers.append(
        ResConv1DLayer(filters=filters, dilation=dilation_growth_rate ** ii)
      )
    
    if reverse_dilation:
      self.conv_layers = self.conv_layers[::-1]
    
  def call(self, inputs, training=False):
    x = inputs
    for ii, conv_layer in enumerate(self.conv_layers):
      x = conv_layer(x, training)
    return x

class DownScale(tf.keras.layers.Layer):
  """Downscaling layer"""
  def __init__(self, filters=32, kernel_size=4, strides=2, use_act='relu'):
    super(DownScale, self).__init__()
    
    self.conv_1 = Conv1DLayer(filters=filters, kernel_size=kernel_size, strides=1)
    
    if use_act == 'leaky_relu':
      self.activation_1 = tf.keras.layers.LeakyReLU()
    else:
      self.activation_1 = tf.keras.layers.Activation(use_act)
    
    self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.maxpool = tf.keras.layers.MaxPooling1D(pool_size=strides, strides=strides, padding='same')
    
  def call(self, inputs, training=False):
    x = self.conv_1(inputs, training)
    x = self.activation_1(x)
    x = self.layernorm(x)
    x = self.maxpool(x)
    return x

class DownScaleJukebox(tf.keras.layers.Layer):
  """Downscaling layer for music waveform encoder as in Jukebox: A generative model for music"""
  def __init__(self, filters=32, kernel_size=4, strides=2, num_res_layers=4):
    super(DownScaleJukebox, self).__init__()
    
    self.conv_layers = []
    self.conv_layers.append(Conv1DLayer(filters=filters, 
                              kernel_size=kernel_size, 
                              strides=strides, 
                              padding='same', 
                              dilation=1, 
                              use_transpose=False)
                           )
    self.conv_layers.append(
        ResConv1DStack(filters=filters, num_res_layers=num_res_layers, dilation_growth_rate=3, reverse_dilation=False)
    )
    
  def call(self, inputs, training=False):
    x = self.conv_layers[0](inputs, training)
    x = self.conv_layers[1](x, training)
    return x

class UpScaleJukebox(tf.keras.layers.Layer):
  """Upscaling layer for music waveform encoder as in Jukebox: A generative model for music"""
  def __init__(self, filters=32, kernel_size=4, strides=2, num_res_layers=4):
    super(UpScaleJukebox, self).__init__()
    
    self.conv_layers = []
    self.conv_layers.append(
        ResConv1DStack(filters=filters, num_res_layers=num_res_layers, dilation_growth_rate=3, reverse_dilation=True)
    )
    self.conv_layers.append(Conv1DLayer(filters=filters, 
                              kernel_size=kernel_size, 
                              strides=strides, 
                              padding='same', 
                              dilation=1, 
                              use_transpose=True)
                           )
    
  def call(self, inputs, training=False):
    x = self.conv_layers[0](inputs, training)
    x = self.conv_layers[1](x, training)
    return x

class LabquakeModelBase(tf.keras.Model):
  """Basic tf keras model class for labquake models"""
  def __init__(self, params_data, params_model):
    super(LabquakeModelBase, self).__init__()
    
    self.params_data = params_data
    self.params_model = params_model
    
    # Metrics
    self.loss_metric = tf.keras.metrics.Mean(name='loss')
    
  def call(self, inputs, training=False):
    pass
    
  def get_in_out_data(self, data):
    pass

  @property
  def metrics(self):
    return [
        self.loss_metric,
        ]
  
  def train_step(self, data):
    # Unpack data
    x, y = self.get_in_out_data(data)
    
    with tf.GradientTape() as tape:
      # Compute predictions
      y_pred, model_loss, _ = self.call(x, training=True)

      # Compute loss value
      total_loss = self.loss_fn(y[:,:,0], y_pred[...,0]) + model_loss + sum(self.losses)

    # Compute gradients
    gradients = tape.gradient(total_loss, self.trainable_variables)
    
    # Update weights
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    
    # Update metrics
    self.loss_metric.update_state(total_loss)
    
    # Log results
    return {
        "loss": self.loss_metric.result(),
        }
    
  def test_step(self, data):
    # Unpack data
    x, y = self.get_in_out_data(data)
    
    # Compute predictions
    y_pred, _, _ = self.call(x, training=False)
    total_loss = self.loss_fn(y[:,:,0], y_pred[...,0])
    
    # Update metrics
    self.loss_metric.update_state(total_loss)
    
    # Log results
    return {
        "loss": self.loss_metric.result(),
        }
  
class VectorQuantizerEMA(tf.keras.layers.Layer):
  """Vector Quantised layer of VQ-VAE model"""
  # Code adapted from
  # https://keras.io/examples/generative/vq_vae/
  # https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
  def __init__(self, num_embeddings, embedding_dim, beta=0.25, decay=0.99, epsilon=1e-5, **kwargs):
    super().__init__(**kwargs)
    self.embedding_dim = embedding_dim
    self.num_embeddings = num_embeddings
    self.beta = beta
    self.decay = decay
    self.epsilon = epsilon

    # Initialize the embeddings
    w_init = tf.random_uniform_initializer()
    self.embeddings = tf.Variable(
        initial_value=w_init(
            shape=(self.embedding_dim, self.num_embeddings), dtype=tf.float32
        ),
        trainable=False, 
        name=self.name+'_embeddings'
    )
    
    # Exponential moving average
    self.ema_cluster_size = tf.Variable(initial_value=tf.zeros([num_embeddings], dtype=tf.float32), trainable=False, dtype=tf.float32, name=self.name+'_ema_cluster_size')
    self.ema_dw = tf.Variable(initial_value=self.embeddings, trainable=False, dtype=tf.float32, name=self.name+'_ema_dw')
    self.ema_embeddings = tf.train.ExponentialMovingAverage(decay=decay, zero_debias=True)
    self.ema_embeddings.apply([self.ema_cluster_size, self.ema_dw])
    
  def call(self, inputs, training=False):
    # Reshape
    input_shape = tf.shape(inputs)
    flat_inputs = tf.reshape(inputs, [-1, self.embedding_dim])
    
    # Quantization indices
    encoding_indices = self.get_code_indices(flat_inputs, training)
    encodings = tf.one_hot(encoding_indices, self.num_embeddings, dtype=tf.float32)
    
    # Quantization
    quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
    quantized = tf.reshape(quantized, input_shape)
    
    # Loss
    commitment_loss = self.beta * tf.reduce_mean(
        (tf.stop_gradient(quantized) - inputs) ** 2
    )

    # Straight-through estimator
    quantized = inputs + tf.stop_gradient(quantized - inputs)
    
    # Update codebook with Exponential moving average
    if training:
      self.ema_cluster_size.assign(tf.reduce_sum(encodings, axis=0))
      
      dw = tf.matmul(flat_inputs, encodings, transpose_a=True)
      self.ema_dw.assign(dw)

      self.ema_embeddings.apply([self.ema_cluster_size, self.ema_dw])
      
      updated_ema_cluster_size = self.ema_embeddings.average(self.ema_cluster_size)
      updated_ema_dw = self.ema_embeddings.average(self.ema_dw)

      n = tf.reduce_sum(updated_ema_cluster_size)
      updated_ema_cluster_size = ((updated_ema_cluster_size + self.epsilon) /
                                  (n + self.num_embeddings * self.epsilon) * n)

      normalised_updated_ema_w = (
          updated_ema_dw / tf.reshape(updated_ema_cluster_size, [1, -1]))
      
      self.embeddings.assign(normalised_updated_ema_w)

    return quantized, commitment_loss, encoding_indices

  def get_code_indices(self, flat_inputs, training=False):
    # get latent vector's category number
    distances = (
        tf.reduce_sum(flat_inputs**2, 1, keepdims=True) -
        2 * tf.matmul(flat_inputs, self.embeddings) +
        tf.reduce_sum(self.embeddings**2, 0, keepdims=True))
    encoding_indices = tf.argmax(-distances, 1)
    return encoding_indices

class VectorQuantizer(tf.keras.layers.Layer):
  """Vector Quantised layer of VQ-VAE model, no vq update"""
  # Code adapted from
  # https://keras.io/examples/generative/vq_vae/
  # https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
  def __init__(self, num_embeddings, embedding_dim, beta=0.25, decay=0.99, epsilon=1e-5, **kwargs):
    super().__init__(**kwargs)
    self.embedding_dim = embedding_dim
    self.num_embeddings = num_embeddings
    self.beta = beta
    self.decay = decay
    self.epsilon = epsilon
    
    # Initialize the embeddings
    w_init = tf.random_uniform_initializer()
    self.embeddings = tf.Variable(
        initial_value=w_init(
            shape=(self.embedding_dim, self.num_embeddings), dtype=tf.float32
        ),
        trainable=False, 
        name=self.name+'_embeddings'
    )
    
  def call(self, inputs, training=False):
    # Reshape
    input_shape = tf.shape(inputs)
    flat_inputs = tf.reshape(inputs, [-1, self.embedding_dim])
    
    # Quantization indices
    encoding_indices = self.get_code_indices(flat_inputs, training)
    encodings = tf.one_hot(encoding_indices, self.num_embeddings, dtype=tf.float32)
    
    # Quantization
    quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
    quantized = tf.reshape(quantized, input_shape)
    
    # Loss
    commitment_loss = self.beta * tf.reduce_mean(
        (tf.stop_gradient(quantized) - inputs) ** 2
    )

    # Straight-through estimator
    quantized = inputs + tf.stop_gradient(quantized - inputs)
    
    return quantized, commitment_loss, encoding_indices

  def get_code_indices(self, flat_inputs, training=False):
    # get latent vector's category number
    distances = (
        tf.reduce_sum(flat_inputs**2, 1, keepdims=True) -
        2 * tf.matmul(flat_inputs, self.embeddings) +
        tf.reduce_sum(self.embeddings**2, 0, keepdims=True))
    encoding_indices = tf.argmax(-distances, 1)
    return encoding_indices

class MuModel(LabquakeModelBase):
  """Vector Quantised friction coefficient model"""
  def __init__(self, params_data, params_model):
    super(MuModel, self).__init__(params_data, params_model)
    
    # Loss
    self.loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    
    # Vector Quantised layer
    num_vq_codes      = params_model['num_vq_codes']
    embedding_dim     = params_model['embedding_dim']
    commitment_cost   = params_model['commitment_cost']
    ema_decay         = params_model['ema_decay']
    self.vq_layer = VectorQuantizerEMA(num_embeddings=num_vq_codes, embedding_dim=embedding_dim, beta=commitment_cost, decay=ema_decay)
    
  def call(self, inputs, training=False):
    x = inputs[:,:,0]
    quantized_z, vq_loss, encoding_indices = self.vq_layer(x, training)
    out = tf.expand_dims(quantized_z, axis=-1)
    return out, vq_loss, 0.0
    
  def get_in_out_data(self, data):
    # Unpack data
    in_data, out_data = data
    return in_data, out_data

# The code for basic transformer layers are from 
# https://www.tensorflow.org/text/tutorials/transformer
# Modified for KernelAttention

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  """sin cos positional encoding"""
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)

def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
  """Multi-Head Attention in Original Transformer"""
  def __init__(self,*, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
  """MLP layers in Transformer"""
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

class TransformerEncoderLayer(tf.keras.layers.Layer):
  """Transformer Encoder"""
  def __init__(self,*, d_model, num_heads, dff, rate=0.1):
    super(TransformerEncoderLayer, self).__init__()
    
    self.kernelatt = 0
    if self.kernelatt:
      self.mha = KernelAttention(
        num_heads=num_heads, 
        key_dim=d_model, 
        dropout=rate, 
        feature_transform='exp', 
        num_random_features=128, 
        seed=0, 
        redraw=True, 
        is_short_seq=False)
    else:
      self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask=None):
    
    if self.kernelatt:
      attn_output, _ = self.mha(query=x, value=x, training=training)
    else:
      attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2

class TransformerDecoderLayer(tf.keras.layers.Layer):
  """Transformer Decoder"""
  def __init__(self,*, d_model, num_heads, dff, rate=0.1):
    super(TransformerDecoderLayer, self).__init__()

    self.kernelatt = 0
    if self.kernelatt:
      self.mha2 = KernelAttention(
        num_heads=num_heads, 
        key_dim=d_model, 
        dropout=rate, 
        feature_transform='exp', 
        num_random_features=128, 
        seed=0, 
        redraw=True, 
        is_short_seq=False)
    else:
      self.mha2 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)
    
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training):
    # enc_output.shape == (batch_size, input_seq_len, d_model)
    out1 = x
    
    if self.kernelatt:
      attn2, attn_weights_block2 = self.mha2(query=out1, value=enc_output, training=training)
    else:
      attn2, attn_weights_block2 = self.mha2(
          enc_output, enc_output, out1, mask=None)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

    return out3, attn_weights_block2

class EncoderCED(tf.keras.layers.Layer):
  """Convolutional Encoder"""
  def __init__(self, use_act='relu'):
    super(EncoderCED, self).__init__()
    
    if 0:
      self.conv_layers = [
          DownScale(filters=8, kernel_size=4, strides=2, use_act=use_act),
          DownScale(filters=8, kernel_size=4, strides=2, use_act=use_act),
          DownScale(filters=16, kernel_size=4, strides=2, use_act=use_act),
          DownScale(filters=16, kernel_size=4, strides=2, use_act=use_act),
          DownScale(filters=32, kernel_size=4, strides=2, use_act=use_act),
          DownScale(filters=32, kernel_size=4, strides=2, use_act=use_act),
          DownScale(filters=64, kernel_size=4, strides=2, use_act=use_act),
      ]
    if 0:
      self.conv_layers = [
          DownScale(filters=8, kernel_size=11, strides=2, use_act=use_act),
          DownScale(filters=16, kernel_size=9, strides=2, use_act=use_act),
          DownScale(filters=16, kernel_size=9, strides=2, use_act=use_act),
          DownScale(filters=32, kernel_size=7, strides=2, use_act=use_act),
          DownScale(filters=32, kernel_size=5, strides=2, use_act=use_act),
          DownScale(filters=64, kernel_size=3, strides=2, use_act=use_act),
          DownScale(filters=64, kernel_size=2, strides=2, use_act=use_act),
          Conv1DLayer(filters=64, kernel_size=3, strides=1),
      ]
    if 1:
      self.conv_layers = [
          DownScale(filters=8, kernel_size=11, strides=4, use_act=use_act),
          DownScale(filters=16, kernel_size=9, strides=4, use_act=use_act),
          DownScale(filters=16, kernel_size=9, strides=4, use_act=use_act),
          DownScale(filters=32, kernel_size=7, strides=4, use_act=use_act),
          DownScale(filters=32, kernel_size=5, strides=2, use_act=use_act),
          DownScale(filters=64, kernel_size=3, strides=2, use_act=use_act),
          DownScale(filters=64, kernel_size=2, strides=2, use_act=use_act),
      ]
    if 0:
      num_downscale_layers = 7
      filters = 32
      kernel_size = 4
      strides = 2
      num_res_layers = 2
      d_model = 64
      self.conv_layers = []
      for ii in range(num_downscale_layers):
        self.conv_layers.append(
          DownScaleJukebox(filters=filters, kernel_size=kernel_size, strides=strides, num_res_layers=num_res_layers)
        )
      self.conv_layers.append(Conv1DLayer(filters=d_model, 
                                kernel_size=3, 
                                strides=1, 
                                padding='same', 
                                dilation=1, 
                                use_transpose=False)
                             )
    
  def call(self, inputs, training=False):
    x = inputs
    for ii, conv_layer in enumerate(self.conv_layers):
      x = conv_layer(x, training)
    return x

class LabquakeModel(LabquakeModelBase):
  """CED transformer model, inter-slip-events decoder"""
  def __init__(self, params_data, params_model):
    super(LabquakeModel, self).__init__(params_data, params_model)
    
    # Loss
    self.loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    
    # Mu VQ layer
    num_vq_codes      = params_model['num_vq_codes']
    embedding_dim     = params_model['embedding_dim']
    self.mu_vq_layer = VectorQuantizer(num_embeddings=num_vq_codes, embedding_dim=embedding_dim)
    self.mu_vq_layer.embeddings.assign(params_model['mu_embeddings'])
    
    # AE Encoder
    use_act = 'elu'
    self.use_act = use_act
    self.ae_encoder = EncoderCED(use_act=use_act)
    
    # Transformer
    num_layers = 1
    d_model = 64
    num_heads = 8
    dff = d_model*2
    dropout_rate = 0.1
    num_classes = params_model['num_vq_codes']
    
    self.dec_length = params_data['num_out_time_intvl']
    self.num_layers = num_layers
    self.d_model = d_model
    self.pos_enc = positional_encoding(1024, d_model)
    self.trsf_enc_layers = [TransformerEncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, rate=dropout_rate) for _ in range(num_layers)]
    self.trsf_dec_layers = [TransformerDecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, rate=dropout_rate) for _ in range(num_layers)]

    # Mu Decoder Layers
    if 0:
      self.mu_decoder = [
          tf.keras.layers.Conv1DTranspose(filters=32, kernel_size=3, strides=2, padding="same", activation=use_act),
          tf.keras.layers.Conv1DTranspose(filters=16, kernel_size=3, strides=2, padding="same", activation=use_act),
          tf.keras.layers.Conv1DTranspose(filters=8, kernel_size=3, strides=2, padding="same", activation=use_act),
          tf.keras.layers.Conv1D(filters=1, kernel_size=1, strides=1, padding="same", activation='linear'),
      ]
    if 1:
      self.mu_decoder = [
          tf.keras.layers.Conv1D(filters=32, kernel_size=1, strides=1, padding="same", activation=use_act),
          tf.keras.layers.Conv1D(filters=8, kernel_size=1, strides=1, padding="same", activation='linear'), 
      ]
    if 0:
      self.mu_decoder = [
          tf.keras.layers.Conv1D(filters=32, kernel_size=1, strides=1, padding="same", activation=use_act),
          tf.keras.layers.Conv1D(filters=16, kernel_size=1, strides=1, padding="same", activation=use_act),
          tf.keras.layers.Conv1D(filters=8, kernel_size=1, strides=1, padding="same", activation='linear'), 
      ]
  
  @tf.function
  def decode_mu(self, inputs, training=False):
    inputs_shape = tf.shape(inputs)
    x = inputs
    for ii, conv_layer in enumerate(self.mu_decoder):
      x = conv_layer(x)
    x, _, _ = self.mu_vq_layer(x, training=False)
    out = tf.reshape(x, [inputs_shape[0], -1, 1])
    return out
  
  @tf.function
  def call(self, inputs, training=False):
    inputs_shape = tf.shape(inputs)
    x = tf.reshape(inputs, [inputs_shape[0], -1, inputs_shape[-1]])
    
    # AE Encoder
    x = self.ae_encoder(x, training)
    
    # Transformer Encoder
    for i in range(self.num_layers):
      x = self.trsf_enc_layers[i](x, training, mask=None)
    enc_output = x
    
    # Transformer Decoder
    dec_input = tf.ones((inputs_shape[0], self.dec_length, self.d_model), dtype=tf.float32) * 0.0 + self.pos_enc[:,:self.dec_length,:]
    x = dec_input
    for i in range(self.num_layers):
      x, attention_weights = self.trsf_dec_layers[i](x, enc_output, training)
    
    # Mu Decoder
    out = self.decode_mu(x, training)
    
    return out, 0.0, attention_weights
    
  @tf.function
  def get_in_out_data(self, data):
    # Unpack data
    in_data  = data['ae']
    out_data = data['mu']
    out_data = self.get_target_vq_codes(out_data)
    return in_data, out_data
  
  @tf.function
  def get_target_vq_codes(self, y_data):
    """Get target vq codes for output mu from output mu signal"""
    y_data_shape = tf.shape(y_data)
    quantized_z, _, _ = self.mu_vq_layer(y_data[:,:,0], training=False)
    return quantized_z[...,tf.newaxis]

class LabquakeModel_MS(LabquakeModel):
  """CED transformer model, inter-slip-events decoder + major-slip-events decoder"""
  def __init__(self, params_data, params_model):
    super(LabquakeModel_MS, self).__init__(params_data, params_model)
    
    # Major Slip Decoder Layers
    if 0:
      self.slip_decoder = [
          tf.keras.layers.Conv1DTranspose(filters=32, kernel_size=3, strides=2, padding="same", activation=self.use_act),
          tf.keras.layers.Conv1DTranspose(filters=16, kernel_size=3, strides=2, padding="same", activation=self.use_act),
          tf.keras.layers.Conv1DTranspose(filters=8, kernel_size=3, strides=2, padding="same", activation=self.use_act),
          tf.keras.layers.Conv1D(filters=1, kernel_size=1, strides=1, padding="same", activation='linear'),
      ]
    if 1:
      self.slip_decoder = [
          tf.keras.layers.Conv1D(filters=32, kernel_size=1, strides=1, padding="same", activation=self.use_act),
          tf.keras.layers.Conv1D(filters=8, kernel_size=1, strides=1, padding="same", activation='linear'), 
      ]
    if 0:
      self.slip_decoder = [
          tf.keras.layers.Conv1D(filters=32, kernel_size=1, strides=1, padding="same", activation=self.use_act),
          tf.keras.layers.Conv1D(filters=16, kernel_size=1, strides=1, padding="same", activation=self.use_act),
          tf.keras.layers.Conv1D(filters=8, kernel_size=1, strides=1, padding="same", activation='linear'), 
      ]
  
  @tf.function
  def decode_slip(self, inputs, training=False):
    inputs_shape = tf.shape(inputs)
    x = inputs
    for ii, conv_layer in enumerate(self.slip_decoder):
      x = conv_layer(x)
    x, _, _ = self.mu_vq_layer(x, training=False)
    out = tf.reshape(x, [inputs_shape[0], -1, 1])
    return out
  
  @tf.function
  def call_slip(self, inputs, training=False):
    inputs_shape = tf.shape(inputs)
    x = tf.reshape(inputs, [inputs_shape[0], -1, inputs_shape[-1]])
    
    # AE Encoder
    x = self.ae_encoder(x, training)
    
    # Transformer Encoder
    for i in range(self.num_layers):
      x = self.trsf_enc_layers[i](x, training, mask=None)
    enc_output = x
    
    # Transformer Decoder
    dec_input = tf.ones((inputs_shape[0], self.dec_length, self.d_model), dtype=tf.float32) * 0.0 + self.pos_enc[:,:self.dec_length,:]
    x = dec_input
    for i in range(self.num_layers):
      x, attention_weights = self.trsf_dec_layers[i](x, enc_output, training)
    
    # Major Slip Decoder
    out = self.decode_slip(x, training)
    
    return out, 0.0, attention_weights
  
  def train_step(self, data):
    # Unpack data
    x, y = self.get_in_out_data(data)
    
    with tf.GradientTape() as tape:
      # Compute predictions
      y_pred, model_loss, _ = self.call_slip(x, training=True)

      # Compute loss value
      total_loss = self.loss_fn(y[:,:,0], y_pred[...,0]) + model_loss + sum(self.losses)

    # Compute gradients
    gradients = tape.gradient(total_loss, self.trainable_variables)
    
    # Update weights
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    
    # Update metrics
    self.loss_metric.update_state(total_loss)
    
    # Log results
    return {
        "loss": self.loss_metric.result(),
        }
    
  def test_step(self, data):
    # Unpack data
    x, y = self.get_in_out_data(data)
    
    # Compute predictions
    y_pred, _, _ = self.call_slip(x, training=False)
    total_loss = self.loss_fn(y[:,:,0], y_pred[...,0])
    
    # Update metrics
    self.loss_metric.update_state(total_loss)
    
    # Log results
    return {
        "loss": self.loss_metric.result(),
        }