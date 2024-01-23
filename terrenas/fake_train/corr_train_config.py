import os

batch_size = 10
# The number of epochs to train for
n_epochs = 1
# The size of the images in the training set
img_size = (80,80,1)
# A random seed to us
random_seed = 2
# The list of learning parameters to use
learning_params = ['main_lens_parameter_theta_E', 'corr_func_parameter_As0',
            'corr_func_parameter_n0',
            'corr_func_parameter_As2',
            'corr_func_parameter_n2']
# Which parameters to consider flipping
flip_pairs = None
# Which terms to reweight
weight_terms = None
# The path to the fodler containing the npy images
# for training
npy_folders_train = [os.getenv('HOME') + '/terrenas/fake_train/training_data_group_%d/'%(i) for i in range(1,3)]
# The path to the tf_record for the training images
tfr_train_paths = [
    os.path.join(path,'data.tfrecord') for path in npy_folders_train]
# The path to the fodler containing the npy images
# for validation
npy_folder_val = npy_folders_train[0] #put everything into a single folder
# The path to the tf_record for the validation images
tfr_val_path = tfr_train_paths[0]
# The path to the training metadata
metadata_paths_train = [
    os.path.join(path,'metadata.csv') for path in npy_folders_train]
# The path to the validation metadata
metadata_path_val = metadata_paths_train[0]
# The path to the csv file to read from / write to for normalization
# of learning parameters.
input_norm_path = npy_folders_train[0] + 'norms.csv'
# The detector kwargs to use for on-the-fly noise generation
kwargs_detector = None
# Whether or not to normalize the images by the standard deviation
norm_images = True
# A string with which loss function to use.
loss_function = 'diag'
# A string specifying which model to use
model_type = 'xresnet34'
# A string specifying which optimizer to use
optimizer = 'Adam'
# Where to save the model weights
model_weights = (os.getenv('HOME') + '/xena-scratch/fake_model_corr.h5')
model_weights_init = None
# The learning rate for the model
learning_rate = 5e-3
# Whether or not to use random rotation of the input images
random_rotation = True
