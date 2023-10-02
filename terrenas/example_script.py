import os
import sys
from terrenas.terrenas import single_iteration

seed = None # randomly initialize the seed
filename_index = int(sys.argv[1])
#output_path = os.getcwd() + '/correlation_function_test_1'
output_path = os.getenv('SCRATCH') + '/training_data/'
kwargs_observing = {}
kwargs_sample_redshifts = {}
kwargs_sample_macromodel = {}
kwargs_sample_source = {}
kwargs_sample_lens_light = {}
kwargs_sample_substructure = {'log_mlow': 6.0, 'log_mhigh': 10.0,
                              'log10_sigma_sub_low': -2.5, 'log10_sigma_sub_high': -0.5,
                              'LOS_norm_low': 0.5, 'LOS_norm_high': 2.0,
                              'log10_rescale_mc_amp_min': 0.0, 'log10_rescale_mc_amp_max': 0.0,
                              # turns off rescaling of MC relation
                              'rescale_mc_slope_min': 0.0, 'rescale_mc_slope_max': 0.0,
                              # turns off rescaling of slope
                             }
numPix_kappa_map = 250
N = 1000 # each CPU core performs the calculation N times.

for _ in range(0, N):
    single_iteration(output_path, filename_index, kwargs_observing, kwargs_sample_redshifts, kwargs_sample_macromodel, kwargs_sample_source,
          kwargs_sample_lens_light, kwargs_sample_substructure, save_correlation_function=True,
                 save_kappa_map=True, numPix_kappa_map=numPix_kappa_map, seed=seed)
