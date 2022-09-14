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

import time
import numpy as np
import tensorflow as tf
from utils_data_windows import data_scaler_inverse_transform, getWindowIndices

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  # code from https://keras.io/examples/audio/transformer_asr/
  """Learning rate schedule linear warm up - linear decay"""
  def __init__(
      self,
      init_lr=0.00001,
      lr_after_warmup=0.001,
      final_lr=0.00001,
      warmup_epochs=20,
      decay_epochs=100,
      steps_per_epoch=100,
  ):
      super().__init__()
      self.init_lr = init_lr
      self.lr_after_warmup = lr_after_warmup
      self.final_lr = final_lr
      self.warmup_epochs = warmup_epochs
      self.decay_epochs = decay_epochs
      self.steps_per_epoch = steps_per_epoch

  def calculate_lr(self, epoch):
      """ linear warm up - linear decay """
      warmup_lr = (
          self.init_lr
          + ((self.lr_after_warmup - self.init_lr) / (self.warmup_epochs - 1)) * epoch
      )
      decay_lr = tf.math.maximum(
          self.final_lr,
          self.lr_after_warmup
          - (epoch - self.warmup_epochs)
          * (self.lr_after_warmup - self.final_lr)
          / (self.decay_epochs),
      )
      return tf.math.minimum(warmup_lr, decay_lr)

  def __call__(self, step):
      epoch = step // self.steps_per_epoch
      return self.calculate_lr(epoch)

def get_labquake_model(params_data, params_model, 
                       LabquakeModel, dataset, pre_weight=None):
  labquake_model = LabquakeModel(params_data, params_model)
  for ds in dataset.take(1):
    x, y = labquake_model.get_in_out_data(ds)
    _ = labquake_model(x, training=False)
  if params_model['verbose_model_train']:
    labquake_model.summary()
  if pre_weight != None:
    labquake_model.load_weights(pre_weight)
  return labquake_model

def train_labquake_model(params_data, params_model, 
                         LabquakeModel, 
                         train_dataset, valid_dataset, 
                         pre_weight=None):
  
  # Optimizer
  learning_rate = CustomSchedule(
      init_lr=0.00001,
      lr_after_warmup=params_model['learning_rate'],
      final_lr=0.00001,
      warmup_epochs=params_model['lr_warmup_epochs'], 
      decay_epochs=params_model['lr_decay_epochs'],
      steps_per_epoch=params_model['steps_per_epoch'],
  )
  optimizer = tf.keras.optimizers.Adam(learning_rate)
  
  # Earlystop callback
  early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=params_model['min_delta_stop'], patience=params_model['early_stop_wait'], 
                                                restore_best_weights=True, verbose=params_model['verbose_model_train'], mode='min')
  callbacks = [early_stop]

  # Create model
  strategy = tf.distribute.MirroredStrategy()
  with strategy.scope():
    labquake_model = get_labquake_model(params_data, params_model, 
                                        LabquakeModel, train_dataset, pre_weight)
    labquake_model.compile(optimizer=optimizer)
  
  # Training
  start_time = time.time()
  if params_model['verbose_model_train']:
    print('\n Model Training')
  
  history = labquake_model.fit(train_dataset, 
                               epochs=params_model['num_train_epochs'], 
                               validation_data=valid_dataset, 
                               callbacks=callbacks, 
                               verbose=params_model['verbose_model_train'])
  
  # Evaluate
  train_results = labquake_model.evaluate(train_dataset, verbose=params_model['verbose_model_train'])
  valid_results = labquake_model.evaluate(valid_dataset, verbose=params_model['verbose_model_train'])
  
  end_time = time.time()
  if params_model['verbose_model_train']:
    print('\n Model Training Time', end_time-start_time, 's')
  
  return labquake_model, history

def test_labquake_model(labquake_model, dataset, output_scaler):
  targets = []
  targets_time = []
  preds = []
  ii = 0
  for ds in dataset:
    x_data, _ = labquake_model.get_in_out_data(ds)
    y_data = ds['mu']
    
    y_pred, _, _ = labquake_model(x_data, training=False)
    
    targets.append(y_data)
    targets_time.append(ds['t_mu'])
    preds.append(y_pred)
    ii += 1
  
  targets = np.concatenate(targets)
  targets_time = np.concatenate(targets_time)
  preds = np.concatenate(preds)

  targets = data_scaler_inverse_transform(targets, output_scaler)
  preds = data_scaler_inverse_transform(preds, output_scaler)
  
  return targets, preds, targets_time

def test_labquake_model_detail(labquake_model, dataset, input_scaler, output_scaler):
  inputs = []
  inputs_time = []
  targets = []
  targets_time = []
  preds = []
  preds_slip = []
  attn = []
  
  ii = 0
  for ds in dataset:
    x_data, _ = labquake_model.get_in_out_data(ds)
    y_data = ds['mu']
    
    y_pred, _, attention_weights = labquake_model(x_data, training=False)
    y_pred_slip, _, attention_weights = labquake_model.call_slip(x_data, training=False)
    
    inputs.append(x_data[:,:,0,:])
    inputs_time.append(ds['t_ae'])
    targets.append(y_data)
    targets_time.append(ds['t_mu'])
    preds.append(y_pred)
    preds_slip.append(y_pred_slip)
    attn.append(attention_weights)
    
    ii += 1
  
  inputs = np.concatenate(inputs)
  inputs_time = np.concatenate(inputs_time)
  targets = np.concatenate(targets)
  targets_time = np.concatenate(targets_time)
  preds = np.concatenate(preds)
  preds_slip = np.concatenate(preds_slip)
  attn = np.concatenate(attn)
  
  inputs = data_scaler_inverse_transform(inputs, input_scaler)
  targets = data_scaler_inverse_transform(targets, output_scaler)
  preds = data_scaler_inverse_transform(preds, output_scaler)
  preds_slip = data_scaler_inverse_transform(preds_slip, output_scaler)
  
  return targets, preds, targets_time, preds_slip, inputs, inputs_time, attn