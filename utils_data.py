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

import os
import json
import h5py
import functools
import numpy as np
import tensorflow as tf
from scipy import stats
from sklearn import preprocessing
from scipy import signal

def get_t2f_data(t, mu, grad_thld, drop_thld, seg_len):
  # compute time to failure from friction coeffs
  mu_range = np.max(mu) - np.min(mu)
  mu_grad = np.gradient(mu)
  seg_half_len = int(seg_len/2)
  t2f = np.zeros_like(t)
  
  t_fail_index = np.where(mu_grad<=grad_thld)[0]
  inds = np.where(t_fail_index[1:]-t_fail_index[:-1] > 10)
  t_fail_index = np.concatenate([t_fail_index[inds], [t_fail_index[-1]]])
  
  t_fail_index_1 = []
  for ii in range(len(t_fail_index)):
    mu_seg = mu[(t_fail_index[ii]-seg_half_len):(t_fail_index[ii]+seg_half_len)]
    if (np.max(mu_seg) - np.min(mu_seg)) >= drop_thld*mu_range:
      t_fail_index_1.append(t_fail_index[ii]+(np.argmin(mu_seg)-seg_half_len))
  t_fail_index_1 = np.array(t_fail_index_1)
  
  for ii in range(len(t_fail_index_1)):
    time2fail = t[t_fail_index_1[ii]]-t
    time2fail[np.where(time2fail<=0.0)] = 0.0
    if ii >= 1:
      time2fail[:t_fail_index_1[ii-1]+1] = 0.0
    t2f += time2fail
  
  fail_time = t[t_fail_index_1]
  return t2f, fail_time


def get_ae_data_acfiles_v1(ACfile_DIR, acsettings, Syncfile, filenumber):
  # number of superframes per file
  numSFpfile = int(acsettings['numFrames'][0]/2)
  # number of WF per superframe and per channel
  numWFpSFpCH = int(acsettings['numAcqs'][0])
  # number of WF per file and per channel
  numWFpfilepCH = numSFpfile * numWFpSFpCH
  # number of channels
  numCHR = len(acsettings['channels2save'][0])
  # waveform length
  WFlength = int(acsettings['Nsamples'][0])
  # true number of points per file
  TruenumPointspfilepCH = numWFpfilepCH * WFlength

  # ajusted sampling period
  ts_adjusted = Syncfile['ts_adjusted'][0][0]
  ts = ts_adjusted * 1e-6
  # acoustic sampling rate
  fs = int(1/ts)

  # read files
  ACfilename = ACfile_DIR + 'WF_'+str(filenumber)+'.ac'
  ACdata = np.fromfile(ACfilename, dtype=np.int16)

  ACdata = ACdata.reshape([-1, numCHR, numSFpfile], order='F')
  ACdata = np.transpose(ACdata, [0, 2, 1])
  ACdata = ACdata.reshape([-1, numCHR], order='F')

  # time vector
  acTime = Syncfile['acTime'][0,:]
  acTime_dt = np.mean(acTime[1:]-acTime[:-1])/WFlength
  time_ae = np.zeros(WFlength*numWFpfilepCH, dtype=np.float64)
  time_ae_seg = np.zeros(numWFpfilepCH, dtype=np.float64)

  for jj in range(numWFpfilepCH):
    time_ae[jj*WFlength:(jj+1)*WFlength] = acTime[(filenumber-1)*numWFpfilepCH+jj] + np.arange(WFlength)*acTime_dt
    time_ae_seg[jj] = acTime[(filenumber-1)*numWFpfilepCH+jj]

  return ACdata, time_ae, time_ae_seg


def get_ae_data(DATA_FILE_NAME, AtWhichTimes):
  # read ae hdf5 file
  with h5py.File(DATA_FILE_NAME, "r") as f:
    WFlength = f['time_seg'].attrs['WFlength']
    numWFpfilepCH = f['time_seg'].attrs['numWFpfilepCH']
    TruenumPointspfilepCH = f['time_seg'].attrs['TruenumPointspfilepCH']
    
    time_seg = f['time_seg'][:]
    time_seg_start_ind = np.where((time_seg<AtWhichTimes[0]))[0][-1]
    time_seg_end_ind = np.where((time_seg>AtWhichTimes[-1]))[0][0]
    
    start_ind = time_seg_start_ind * WFlength
    end_ind = (time_seg_end_ind+1) * WFlength
    ae = f['ae'][start_ind:end_ind,:]
    time = f['time'][start_ind:end_ind]
    
    start_ind = np.where((time<AtWhichTimes[0]))[0][-1] + 1
    end_ind = np.where((time>AtWhichTimes[-1]))[0][0]
    ae = ae[start_ind:end_ind,:]
    time = time[start_ind:end_ind]
    
  return ae, time


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value.flatten()))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))

def serialize_example(wf_ae, time_window):
  feature = {
      't_win': _float_feature(time_window),
      'wf_ae': _int64_feature(wf_ae),
  }
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


def parse_serialized_example(example_proto, params_data):
  # parses a serialized tf.Example
  shape0 = int(params_data['ae_sample_freq']*1.0/params_data['sample_freq'])*params_data['numCHR']
  shape1 = int(params_data['ae_sample_freq']*1.0/params_data['sample_freq'])
  feature_description = {
    't_win': tf.io.FixedLenFeature([2,], tf.float32),
    'wf_ae': tf.io.FixedLenFeature([shape0,], tf.int64),
  }
  parsed_features = tf.io.parse_single_example(example_proto, feature_description)
  parsed_features['wf_ae'] = tf.reshape(parsed_features['wf_ae'], [shape1, params_data['numCHR']])
  return parsed_features


def read_tfrecord_dataset(params_data):
  TFRECORD_DATA_DIR = params_data['tfrecord_data_dir']
  filenames = sorted([os.path.join(TFRECORD_DATA_DIR, name) for name in os.listdir(TFRECORD_DATA_DIR) if os.path.isfile(os.path.join(TFRECORD_DATA_DIR, name))])
  
  autotune = tf.data.AUTOTUNE
  ds = tf.data.Dataset.from_tensor_slices((filenames))
  ds = tf.data.TFRecordDataset(ds)
  ds = ds.map(functools.partial(parse_serialized_example, params_data=params_data), num_parallel_calls=autotune)
  ds = ds.batch(1, drop_remainder=True, num_parallel_calls=autotune)
  ds = ds.prefetch(autotune)
  return ds