import os
import sys
from terrenas.terrenas import single_iteration_HDF5
import numpy as np
import pandas as pd

data_index = int(sys.argv[1])
seed_array = np.arange(10**6).reshape(10**3,10**3)
output_path = os.getenv('SCRATCH') + '/training_data/'
index_file = os.getenv('SCRATCH') + '/tarining_galaxies.csv'
kwargs_observing = {}
kwargs_sample_redshifts = {'zd_min':0.5, 'zd_max':0.5,
                     'zs_min':1.5, 'zs_max':1.5}

kwargs_sample_macromodel = {'theta_E_low':0.8, 'theta_E_high':1.2,
                      'q_low':0.75, 'q_high':0.75,
                      'log10_shear_mag_low':np.log10(0.05), 'log10_shear_mag_high':np.log10(0.05),
                      'gamma_epl_low':1.9, 'gamma_epl_high':2.1,
                      'a4_mean':0.0, 'a4_variance':0.0}

kwargs_sample_lens_light = {}
kwargs_sample_substructure = {'log_mlow': 6.0, 'log_mhigh': 10.0, 
                              'log10_sigma_sub_low': np.log10(0.05), 'log10_sigma_sub_high': np.log10(0.05), 
                              'LOS_norm_low': 0.5, 'LOS_norm_high': 2.0, 
                              'log10_rescale_mc_amp_min': 0.0, 'log10_rescale_mc_amp_max': 0.0, 
                              # turns off rescaling of MC relation
                              'rescale_mc_slope_min': 0.0, 'rescale_mc_slope_max': 0.0, 
                              # turns off rescaling of slope
                             }

numPix_kappa_map = 250

for group in range(1, 1001):
    seed = seed_array[group][data_index]
    index_array = pd.read_csv(index_file ,names=['catalog_i'])['catalog_i'].to_numpy()
    np.random.seed(seed)
    cosmos_source_index = index_array[np.random.randint(0, index_array.shape[0]-1)]
    kwargs_sample_source = {'cosmos_source_index': cosmos_source_index}

    single_iteration_HDF5(output_path, group, data_index, kwargs_observing, kwargs_sample_redshifts,
                         kwargs_sample_macromodel, kwargs_sample_source,
                         kwargs_sample_lens_light, kwargs_sample_substructure, save_image=True,
                         save_kappa_map=False, numPix_kappa_map=numPix_kappa_map, seed=seed)
