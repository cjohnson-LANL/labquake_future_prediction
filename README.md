# Labquake Future Prediction

## ABOUT
This repository contains Python codes accompanying the paper "Predicting future laboratory fault friction through deep learning transformer models", including Python Modules for data preparation and deep learning convolutional encoder-decoder and transformer models, Jupyter Notebooks for data preparation, step-by-step model training, and plotting prediction results from trained models.

## REQUIREMENTS
This code requires Python 3.7 or higher and Tensorflow 2.7 or higher.

## USING THIS CODE
a) Data preparation: run the Jupyter Notebooks (1) p4677_mech_data.ipynb, (2) p4581_mech_data.ipynb, (3) p4677_convert_acfiles_data_to_hdf5.ipynb, (4) p4677_convert_tfrecord_data.ipynb, (5) p4581_convert_acfiles_data_to_hdf5.ipynb, (6) p4581_convert_tfrecord_data.ipynb, (7) p4677_seismic_shear_data.ipynb, (8) p4581_seismic_shear_data.ipynb, (9) p4677_p4581_major_slip_positions.ipynb sequentially. These generate hdf5 datasets in the ./Data/ folder.

b) Model training: run the Jupyter Notebooks (1) Step1_mu_vq_latent_model.ipynb, (2) Step2_future_predict_model_inter-slip-events.ipynb, (3) Step3_future_predict_model_major-slip-events.ipynb sequentially. Modify the data parameter (num_in_time_intvl) to change the length of input AE windows and modify (num_out_time_intvl) to change the length of output friction windows. 

c) Plotting predictions: run the Jupyter Notebook Figure_3_4_Predict_Plot.ipynb for getting the models predictions for Figures 3, 4 of the paper.

## License
© 2022. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

This program is open source under the BSD-3 License. Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2.Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3.Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.