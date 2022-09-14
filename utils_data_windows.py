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
from sklearn import preprocessing
import h5py
import functools
import joblib

def readSeismicData(filenames):
  """read acoustic emission and friction data"""
  filename_seismic_data = filenames[0]
  filename_shear_data = filenames[1]

  # acoustic emission data
  with h5py.File(filename_seismic_data, 'r') as f:
    wf_ae_data = f.get('wf_ae')[:]
  wf_ae_data = np.abs(wf_ae_data)
  
  # friction data
  with h5py.File(filename_shear_data, 'r') as f:
    time_data = f.get('time')[:]
    mu_data = f.get('mu')[:]
  
  signalData = {}
  signalData['time'] = time_data
  signalData['ae'] = wf_ae_data
  signalData['mu'] = mu_data[...,np.newaxis]
  
  return signalData

def get_data_scaler(data, scaler_name):
  """minmax, meanstd, quantile scaler for signal data"""
  if scaler_name == 'minmax':
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
  elif scaler_name == 'standard':
    scaler = preprocessing.StandardScaler()
  elif scaler_name == 'qt_uniform':
    scaler = preprocessing.QuantileTransformer(output_distribution='uniform')
  elif scaler_name == 'qt_normal':
    scaler = preprocessing.QuantileTransformer(output_distribution='normal')
  elif scaler_name == 'robust':
    scaler = preprocessing.RobustScaler(unit_variance=True)
  scaler = scaler.fit(data.reshape([-1, data.shape[-1]]))
  return scaler

def data_scaler_transform(data, scaler):
  """scale data with scaler"""
  data_shape = data.shape
  data = scaler.transform(data.reshape([-1,data_shape[-1]]))
  data = data.reshape(data_shape)
  return data

def data_scaler_inverse_transform(data, scaler):
  """inverse scale data with scaler"""
  data_shape = data.shape
  data = scaler.inverse_transform(data.reshape([-1,data_shape[-1]]))
  data = data.reshape(data_shape)
  return data

def getWindowIndices(params_data, len_signal_data, start_index, 
                     in_time_intvl, out_time_intvl, in_out_time_diff, time_advance):
  """window indices for split acoustic emission and friction data"""
  InputInds = []
  OutputInds = []
  
  iStep = 0
  isStop = False
  while 1:
    # get indices of signals in input and output windows
    i_start_in = int(iStep*time_advance*params_data['sample_freq'])
    i_end_in = i_start_in + int(in_time_intvl*params_data['sample_freq'])

    i_start_out = i_start_in + int(in_out_time_diff*params_data['sample_freq'])
    i_end_out = i_start_out + int(out_time_intvl*params_data['sample_freq'])
    
    if i_end_out <= len_signal_data:
      InputInds.append([i_start_in, i_end_in])
      OutputInds.append([i_start_out, i_end_out])
    else:
      isStop = True
    
    iStep += 1
    if isStop:
      break
  
  InputInds = start_index + np.array(InputInds, dtype=int)
  OutputInds = start_index + np.array(OutputInds, dtype=int)
  
  return InputInds, OutputInds

def getWindowData(params_data, signal_data, inds_data):
  """window split acoustic emission and friction data"""
  win_data = []
  for index in inds_data:
    win_data_i = signal_data[index[0]:index[1]]
    win_data.append(np.expand_dims(win_data_i, axis=0))
  win_data = np.concatenate(win_data, axis=0)
  return win_data

def getWindowDataDict(params_data, input_signal, output_signal, time_signal):
  """window split acoustic emission and friction data"""
  # Data windows split
  InputInds, OutputInds = getWindowIndices(params_data, input_signal.shape[0], 0, 
                                           params_data['in_time_intvl'], params_data['out_time_intvl'], 
                                           params_data['in_out_time_diff'], params_data['time_advance'])
  
  # Window Data
  X_data = getWindowData(params_data, input_signal, InputInds)
  y_data = getWindowData(params_data, output_signal, OutputInds)
  X_data_time = getWindowData(params_data, time_signal, InputInds)
  y_data_time = getWindowData(params_data, time_signal, OutputInds)
  print('  Data Shape', X_data.shape, y_data.shape, X_data_time.shape, y_data_time.shape)
  
  data = {}
  data['X_data'] = X_data
  data['y_data'] = y_data
  data['X_data_time'] = X_data_time
  data['y_data_time'] = y_data_time
  
  return data

def LOAD_DATA(params_data):
  """function to load train/valid data"""
  # Read data
  filenames = params_data['filenames_train']
  signalData = readSeismicData(filenames)

  # Train/Valid data split
  numSignal = signalData['time'].shape[0]
  numTrain = int(numSignal*params_data['train_data_split'])
  numValid = numTrain + int(numSignal*params_data['valid_data_split'])

  time_train = signalData['time'][0:numTrain]
  input_signal_train = signalData['ae'][0:numTrain]
  output_signal_train = signalData['mu'][0:numTrain]

  time_valid = signalData['time'][numTrain:numValid]
  input_signal_valid = signalData['ae'][numTrain:numValid]
  output_signal_valid = signalData['mu'][numTrain:numValid]

  # Input/output scalers
  input_scaler = get_data_scaler(input_signal_train, params_data['in_scaler_mode'])
  output_scaler = get_data_scaler(output_signal_train, params_data['out_scaler_mode'])

  # Scaled data
  input_signal_train = data_scaler_transform(input_signal_train, input_scaler)
  output_signal_train = data_scaler_transform(output_signal_train, output_scaler)
  input_signal_valid = data_scaler_transform(input_signal_valid, input_scaler)
  output_signal_valid = data_scaler_transform(output_signal_valid, output_scaler)

  # Training Data
  print('\nTraining Data')
  train_data = getWindowDataDict(params_data, input_signal_train, output_signal_train, time_train)

  # Validation Data
  print('\nValidation Data')
  valid_data = getWindowDataDict(params_data, input_signal_valid, output_signal_valid, time_valid)

  return train_data, valid_data, input_scaler, output_scaler

def get_dataset(params_data, input_data, output_data, is_shuffle):
  """function to get tf datasets"""
  # Datasets
  ds = tf.data.Dataset.from_tensor_slices((input_data, output_data))
  if is_shuffle:
    ds = ds.shuffle(params_data['shuffle_buffer'], reshuffle_each_iteration=True)
  ds = ds.batch(params_data['batch_size'])
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))

def serialize_example(ae, mu, t_ae, t_mu):
  """function to serialize tfrecord examples"""
  feature = {
      't_ae': _float_feature(t_ae),
      't_mu': _float_feature(t_mu),
      'ae': _float_feature(ae),
      'mu': _float_feature(mu),
  }
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

def write_tfrecord_window_data(params_data, filename, data):
  """function to write tfrecord data"""
  fout = params_data['tfrecord_window_data'] + filename
  with tf.io.TFRecordWriter(fout) as tfrecord_writer:
    for ii in range(data['X_data'].shape[0]):
      if ii % 1000 == 0: print(ii)
      example_proto = serialize_example(data['X_data'][ii], data['y_data'][ii], data['X_data_time'][ii], data['y_data_time'][ii])
      tfrecord_writer.write(example_proto)

def parse_serialized_example(example_proto, params_data):
  """function to parse tfrecord data"""
  # parses a serialized tf.Example
  shape0 = int(params_data['in_time_intvl']*params_data['sample_freq']*params_data['num_unit_ae']*1)
  shape1 = int(params_data['in_time_intvl']*params_data['sample_freq'])
  shape2 = int(params_data['out_time_intvl']*params_data['sample_freq'])
  feature_description = {
    't_ae': tf.io.FixedLenFeature([shape1,], tf.float32),
    't_mu': tf.io.FixedLenFeature([shape2,], tf.float32),
    'ae': tf.io.FixedLenFeature([shape0,], tf.float32),
    'mu': tf.io.FixedLenFeature([shape2,], tf.float32),
  }
  parsed_features = tf.io.parse_single_example(example_proto, feature_description)
  
  parsed_features['ae'] = tf.reshape(parsed_features['ae'], [shape1, params_data['num_unit_ae'], 1])
  parsed_features['mu'] = tf.reshape(parsed_features['mu'], [shape2,1])
  
  return parsed_features

def read_tfrecord_dataset(params_data, filename, is_shuffle):
  """function to read tfrecord data"""
  autotune = tf.data.AUTOTUNE
  ds = tf.data.TFRecordDataset(params_data['tfrecord_window_data']+filename)
  ds = ds.map(functools.partial(parse_serialized_example, params_data=params_data), num_parallel_calls=autotune)
  if is_shuffle:
    ds = ds.shuffle(params_data['shuffle_buffer'], reshuffle_each_iteration=True)
  ds = ds.batch(params_data['batch_size'], num_parallel_calls=autotune)
  ds = ds.prefetch(autotune)
  return ds

def write_tfrecord_train_valid_data(params_data):
  """function to write train/valid tfrecord data"""
  # Read Train/Valid data
  start_time = time.time()
  print('Load Train/Valid Data')
  train_data, valid_data, input_scaler_train, output_scaler_train = LOAD_DATA(params_data)
  print('\nLoad Data Time', time.time()-start_time, 's')

  np.save(params_data['tfrecord_window_data']+'train_dataset_size.npy', np.array([train_data['X_data'].shape[0], valid_data['X_data'].shape[0]]))

  print('\nWrite Train Data')
  write_tfrecord_window_data(params_data, 'train_data.tfrecord', train_data)

  joblib.dump(input_scaler_train, params_data['tfrecord_window_data']+'input_scaler_train.save') 
  joblib.dump(output_scaler_train, params_data['tfrecord_window_data']+'output_scaler_train.save') 

  print('\nWrite Valid Data')
  write_tfrecord_window_data(params_data, 'valid_data.tfrecord', valid_data)

def write_tfrecord_train_valid_data_major_slips(params_data, train_fail_index, valid_fail_index):
  """function to write train/valid tfrecord major slips data"""
  # Read Train/Valid data
  print('\nLoad Train/Valid major slips Data')
  train_data, valid_data, input_scaler_train, output_scaler_train = LOAD_DATA(params_data)
  
  # Major slips
  _, mu_slip_index_train = get_mu_slip_loc_data_with_index(params_data, train_data['y_data'], train_fail_index)
  _, mu_slip_index_val = get_mu_slip_loc_data_with_index(params_data, valid_data['y_data'], valid_fail_index)

  data_index_train = []
  for ii in [0, -1, 0, -2, 0, -3, 0, -4, 0, -5, 0, -6, 0, -7, 0, -8, 0, -9]:
    for i_index in mu_slip_index_train+ii:
      data_index_train.append(i_index)
  data_index_train = np.array(data_index_train)
  
  data_index_val = []
  for ii in [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]:
    for i_index in mu_slip_index_val+ii:
      data_index_val.append(i_index)
  data_index_val = np.array(data_index_val)
  
  train_data['X_data'] = train_data['X_data'][data_index_train]
  train_data['y_data'] = train_data['y_data'][data_index_train]
  train_data['X_data_time'] = train_data['X_data_time'][data_index_train]
  train_data['y_data_time'] = train_data['y_data_time'][data_index_train]

  valid_data['X_data'] = valid_data['X_data'][data_index_val]
  valid_data['y_data'] = valid_data['y_data'][data_index_val]
  valid_data['X_data_time'] = valid_data['X_data_time'][data_index_val]
  valid_data['y_data_time'] = valid_data['y_data_time'][data_index_val]
  
  np.save(params_data['tfrecord_window_data']+'train_major_slips_dataset_size.npy', np.array([train_data['X_data'].shape[0], valid_data['X_data'].shape[0]]))

  print('\nWrite Train major slips Data')
  write_tfrecord_window_data(params_data, 'train_major_slips_data.tfrecord', train_data)
  
  print('\nWrite Valid major slips Data')
  write_tfrecord_window_data(params_data, 'valid_major_slips_data.tfrecord', valid_data)

def write_tfrecord_test_data(params_data, filenames_test):
  """function to write test tfrecord data"""
  # Read Test data
  start_time = time.time()
  print('\nLoad Test Data')
  signalData = readSeismicData(filenames_test)
  numSignal = signalData['time'].shape[0]

  # Input/output scalers
  numDataScaler = int(numSignal*1.0)
  input_scaler_test = get_data_scaler(signalData['ae'][:numDataScaler], params_data['in_scaler_mode'])
  output_scaler_test = get_data_scaler(signalData['mu'][:numDataScaler], params_data['out_scaler_mode'])

  # Scaled data
  input_signal_test = data_scaler_transform(signalData['ae'], input_scaler_test)
  output_signal_test = data_scaler_transform(signalData['mu'], output_scaler_test)

  # Test Data
  test_data = getWindowDataDict(params_data, input_signal_test, output_signal_test, signalData['time'])

  print('\nLoad Data Time', time.time()-start_time, 's')

  print('\nWrite Test Data')
  write_tfrecord_window_data(params_data, 'test_data.tfrecord', test_data)

  joblib.dump(input_scaler_test, params_data['tfrecord_window_data']+'input_scaler_test.save')
  joblib.dump(output_scaler_test, params_data['tfrecord_window_data']+'output_scaler_test.save')

def get_mu_slip_loc_data(params_data, mu_win_data):
  """function to get major slips locations from criteria on friction data"""
  mu_slip_loc_data = []
  mu_slip_index = []
  mu_slip_pos = []
  
  ii = 0
  for mu_win in mu_win_data:
    mu_win_range = np.max(mu_win) - np.min(mu_win)
    mu_slip_loc = np.zeros_like(mu_win)
    if mu_win_range > params_data['slip_range_throld']:
      mu_diff = mu_win[1:] - mu_win[:-1]
      mu_fail_index = np.where(mu_diff < params_data['slip_diff_throld'])[0][0] + 1
      start_ind = max(mu_fail_index-int(0.5*params_data['slip_seg_size']), 0)
      end_ind = min(mu_fail_index+int(0.5*params_data['slip_seg_size']), int(params_data['sample_freq']*params_data['out_time_intvl']))
      mu_slip_loc[start_ind:end_ind] = 1
      
      mu_slip_index.append(ii)
      
      mu_slip_pos.append(ii*int(params_data['sample_freq']*params_data['out_time_intvl']) + mu_fail_index)
      
    mu_slip_loc_data.append(mu_slip_loc)
    ii += 1
  
  mu_slip_loc_data = np.array(mu_slip_loc_data)
  mu_slip_index = np.array(mu_slip_index)
  mu_slip_pos = int(params_data['sample_freq']*params_data['in_out_time_diff']) + np.array(mu_slip_pos)
  
  return mu_slip_loc_data, mu_slip_index, mu_slip_pos

def get_mu_slip_loc_data_with_index(params_data, mu_data, fail_index):
  """function to get major slips locations from indices"""
  mu_slip_loc_data = []
  mu_slip_index = []
  iii = 0
  for ii in range(mu_data.shape[0]):
    mu_slip_loc = np.zeros_like(mu_data[ii])
    start_ind = int(params_data['sample_freq']*params_data['in_out_time_diff']) + ii*int(params_data['sample_freq']*params_data['out_time_intvl'])
    end_ind = start_ind + int(params_data['sample_freq']*params_data['out_time_intvl'])

    if iii < fail_index.shape[0]:
      if start_ind <= fail_index[iii]:
        if end_ind > fail_index[iii]:
          mu_fail_index = fail_index[iii] - start_ind
          start_ind = max(mu_fail_index-int(0.5*params_data['slip_seg_size']), 0)
          end_ind = min(mu_fail_index+int(0.5*params_data['slip_seg_size']), int(params_data['sample_freq']*params_data['out_time_intvl']))
          mu_slip_loc[start_ind:end_ind] = 1

          mu_slip_index.append(ii)
          
          iii += 1
    mu_slip_loc_data.append(mu_slip_loc)

  mu_slip_loc_data = np.array(mu_slip_loc_data)
  mu_slip_index = np.array(mu_slip_index)

  return mu_slip_loc_data, mu_slip_index