import os
import sys
from terrenas.terrenas import single_iteration_HDF5
import numpy as np

group = int(sys.argv[1])

seed_array = np.arange(10**6).reshape(10**3,10**3)

output_path = os.getenv('SCRATCH') + '/training_data/'
kwargs_observing = {}
kwargs_sample_redshifts = {'zd_min':0.5, 'zd_max':0.5,
                     'zs_min':1.5, 'zs_max':1.5}

kwargs_sample_macromodel = {'theta_E_low':0.8, 'theta_E_high':1.2,
                      'q_low':0.5, 'q_high':0.5,
                      'log10_shear_mag_low':np.log10(0.05), 'log10_shear_mag_high':np.log10(0.05),
                      'gamma_epl_low':1.9, 'gamma_epl_high':1.9,
                      'a4_mean':0.005, 'a4_variance':0.0}

kwargs_sample_source = {}
kwargs_sample_lens_light = {}
kwargs_sample_substructure = {'log_mlow': 6.0, 'log_mhigh': 10.0, 
                              'log10_sigma_sub_low': np.log10(0.05), 'log10_sigma_sub_high': np.log10(0.05), 
                              'LOS_norm_low': 1.0, 'LOS_norm_high': 1.0, 
                              'log10_rescale_mc_amp_min': 0.0, 'log10_rescale_mc_amp_max': 0.0, 
                              # turns off rescaling of MC relation
                              'rescale_mc_slope_min': 0.0, 'rescale_mc_slope_max': 0.0, 
                              # turns off rescaling of slope
                             }

numPix_kappa_map = 250

for data_index in range(1, 1001):
    seed = seed_array[group][data_index]
    single_iteration_HDF5(output_path, group, data_index, kwargs_observing, kwargs_sample_redshifts,
                         kwargs_sample_macromodel, kwargs_sample_source,
                         kwargs_sample_lens_light, kwargs_sample_substructure, save_image=True,
                         save_kappa_map=False, numPix_kappa_map=numPix_kappa_map, seed=seed)
