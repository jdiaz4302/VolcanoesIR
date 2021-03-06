 



# High level summary:
# The goal of the script is to train one of many neural networks
# on previously process ASTER imagery.

# Input data:
# This input data is expected to be in the form of numpy arrays
# representing sequences of cropped thermal infrared images from
# 1999-2012. More specifically, this is anticipated to be temperature
# above the environmental background (kelvin/celsius) for volcanoes.
# This input data is gathered from the previously numbered scripts in
# the directory.

# Script arguments: (model selection and training data)
# Model selection:
# This training code has been designed with several different
# neural network architectures in mind. Namely,
#     1) "LSTM"
#     2 LSTM variants that handle irregular time intervals
#         * "Time-LSTM"
#         * "TimeAwareLSTM"
#     3) Convolutional LSTM (that allows images sequences)
#         * "ConvLSTM"
#     4) New/proposed models that allow image sequences that are
#        irregular through time
#         * Convolutional LSTM + Time-LSTM
#             - "ConvTimeLSTM"
#         * Convolutional LSTM + TimeAwareLSTM
#             - "ConvTimeAwareLSTM"
# The quoted keyword-names are the required input for this argument and will
# select which neural network is trained; this in turn affects preprocessing
# the data into different forms, the training batch size, etc...
#
# Training data:
# This training code has been designed with two sets of training data in mind.
# These two options are:
#     1) Single volcano; in this case, the argument should be the directory
#        name containing the volcano-specific data. All of these directories
#        are expected to be within the "data" directory which is located in
#        current working directory.
#     2) All volcanoes; in this case the argument should be "all".

# Hardware:
# This code was developed for Penn State's Advanced CyberInfrastructure's
# GPU instances which utilize dual NVIDIA Tesla K80 GPU cards resulting
# in 4 GPUs with 11 GB of RAM each


import argparse

parser = argparse.ArgumentParser(description='level2 training')
parser.add_argument('--model_selection', type=str, help='model')
parser.add_argument('--training_data_set', type=str, help='training_data_set')
parser.add_argument('--n_layers', type=int, help='n_layers')
parser.add_argument('--hidden_dim_ls', nargs='+', type=int, help='hidden_dim_ls')
args = parser.parse_args()

model_selection = args.model_selection
training_data_set = args.training_data_set
n_layers = args.n_layers
hidden_dim_ls = args.hidden_dim_ls


# Gathering function inputs
### model_selection = input("Which model do you select?")
print('Model selected:', model_selection)
assert(model_selection in ['Identity', 'AR', 'LSTM', 'TimeLSTM', 'TimeAwareLSTM', 'ConvLSTM', 'ConvTimeLSTM', 'ConvTimeAwareLSTM', 'ConvTimeLSTMUnet'])
### training_data_set = input("Which set of training data do you want to use?")
print('Training data set:', training_data_set)
### n_layers = int(input("Enter number of layers: ")) 
### hidden_dim_ls = []
# iterating till the range 
### for i in range(n_layers): 
### 	layer_dim = int(input()) 
### 	hidden_dim_ls.append(layer_dim)
print('Hidden layers:', hidden_dim_ls)


# Libraries and imports
import os
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils import data
if model_selection == 'LSTM':
	from models.LSTM import StackedLSTM as LSTM_Model
elif model_selection == 'Identity':
	from models.Identity import Identity2 as LSTM_Model
elif model_selection == 'AR':
	from models.AR import AR as LSTM_Model
elif model_selection == 'TimeLSTM':
	from models.TimeLSTM import StackedTimeLSTM as LSTM_Model
elif model_selection == 'TimeAwareLSTM':
	from models.TimeAwareLSTM import StackedTimeAwareLSTM as LSTM_Model
elif model_selection == 'ConvLSTM':
	from models.ConvLSTM import ConvLSTM as LSTM_Model
elif model_selection == 'ConvTimeLSTM':
	from models.ConvTimeLSTM2 import ConvTime_LSTM2 as LSTM_Model
elif model_selection == 'ConvTimeAwareLSTM':
	from models.ConvTimeAwareLSTM2 import ConvTimeAware_LSTM as LSTM_Model
elif model_selection == 'ConvTimeLSTMUnet':
    from models.ConvTimeLSTMUnet import ConvTime_LSTM2_Unet as LSTM_Model
from helper_fns.processing import scale_and_remove_na
from helper_fns.efcnt_data_AST08 import efficient_Dataset
from optimization import ssim
import numpy.ma as ma
from scipy.spatial import KDTree


# Setting seed for reproducibility
torch.manual_seed(0)


# Basic data import, step 0
print("Importing and formatting data")
volcanoes = os.listdir("data")
assert((training_data_set in volcanoes) or (training_data_set == "all"))


# Training parameters
# This needs to actually be variable, will do with later exploration
assert(hidden_dim_ls[0] in [2000, 1028, 128, 64])
if hidden_dim_ls[0] == 2000: # [2000, 2000, 1]
    assert(model_selection in ['AR', 'Identity', 'LSTM', 'TimeLSTM', 'TimeAwareLSTM'])
    # batch_size_dict = {'AR':84, 'Identity':84, 'LSTM':4, 'TimeLSTM':2, 'TimeAwareLSTM':2}
    batch_size_dict = {'AR':84, 'Identity':84, 'LSTM':24, 'TimeLSTM':15, 'TimeAwareLSTM':15}
    batch_size = batch_size_dict[model_selection]
if hidden_dim_ls[0] == 1028: # [1028, 512, 1]
    assert(model_selection in ['LSTM', 'TimeLSTM', 'TimeAwareLSTM'])
    # batch_size_dict = {'LSTM':12, 'TimeLSTM':6, 'TimeAwareLSTM':10}
    batch_size_dict = {'LSTM':36, 'TimeLSTM':18, 'TimeAwareLSTM':30}
    batch_size = batch_size_dict[model_selection]
if hidden_dim_ls[0] == 128: 
    if hidden_dim_ls[1] == 64: # [128, 64, 64, 1] for ConvLSTMs
        assert(model_selection in ['ConvLSTM', 'ConvTimeLSTM', 'ConvTimeAwareLSTM'])
        batch_size_dict = {'ConvLSTM':320, 'ConvTimeLSTM':216, 'ConvTimeAwareLSTM':320}
        batch_size = batch_size_dict[model_selection]
    else: # [128, 1] for LSTMs
        assert(model_selection in ['LSTM', 'TimeLSTM', 'TimeAwareLSTM'])
        # batch_size_dict = {'LSTM':128, 'TimeLSTM':80, 'TimeAwareLSTM':114}
        batch_size_dict = {'LSTM':360, 'TimeLSTM':224, 'TimeAwareLSTM':320}
        batch_size = batch_size_dict[model_selection]
if hidden_dim_ls[0] == 64: # [64, 64, 1]; [64, 128, 64, 1] for Unet
    assert(model_selection in ['ConvLSTM', 'ConvTimeLSTM', 'ConvTimeAwareLSTM', 'ConvTimeLSTMUnet'])
    # batch_size_dict = {'ConvLSTM':108, 'ConvTimeLSTM':76, 'ConvTimeAwareLSTM':124, 'ConvTimeLSTMUnet':36}
    batch_size_dict = {'ConvLSTM':320, 'ConvTimeLSTM':216, 'ConvTimeAwareLSTM':352, 'ConvTimeLSTMUnet':80}
    batch_size = batch_size_dict[model_selection]
lag_dict = {"all":6, "ErtaAle":9, "Kilauea":10, "Masaya":3, "Nyamuragira":3, "Nyiragongo":3, "Pacaya":4, "Puuoo":8}

num_input_scenes = lag_dict[training_data_set]
train_percent = 0.70
out_samp_perc = 0.15


# Removing possible directory clutter
try:
	volcanoes.remove(".ipynb_checkpoints")
except ValueError as e:
	do = 'nothing'

count = 0
vol_cutoff_indices = []
vol_cutoff_indices_valid = []
vol_cutoff_indices_test = []
vol_name_ls = []
for vol in os.listdir('data'):
	### Basic data import ###
	numpy_data_location = "AST_08_data/" + vol + "/background_subtracted"
	table_data_location = "data/" + vol + "/good_df.csv"
	volcano_scenes = np.load(numpy_data_location, allow_pickle = True)
	tabular_metadata = pd.read_csv(table_data_location)
	### Taking care of the unretrievable scenes for Erebus ###
    ### This should always be done when referencing processed data (AST_08_data) with metadata (data) ###
	if vol == 'Erebus':
		tabular_metadata['acquisition_datetimes'] = [file.split('_')[2] for file in tabular_metadata['nighttime_volcano_files']]
		tabular_metadata = tabular_metadata[tabular_metadata.acquisition_datetimes != '00304102013143221'] 
		tabular_metadata = tabular_metadata[tabular_metadata.acquisition_datetimes != '00308162018142109']
		tabular_metadata = tabular_metadata.reset_index()
	### Separate model inputs and outputs
	# Determine number in each partition
	train_n = int(np.floor((len(volcano_scenes) - num_input_scenes)*train_percent))
	out_n = int(np.floor((len(volcano_scenes) - num_input_scenes)*out_samp_perc))
	# For every data partition
	# Array for the prior scenes
	#   "train_n - 1" is to remove the first scene that wont have an associated TimeAwareLSTM time interval
	x_scenes_train = ma.zeros([train_n - 1, num_input_scenes, volcano_scenes.shape[1], volcano_scenes.shape[2]])
	x_scenes_valid = ma.zeros([out_n, num_input_scenes, volcano_scenes.shape[1], volcano_scenes.shape[2]])
	x_scenes_test = ma.zeros([out_n, num_input_scenes, volcano_scenes.shape[1], volcano_scenes.shape[2]])
	# Array for the time differences between scenes
	time_differences_train = np.ones(x_scenes_train.shape)
	time_differences_valid = np.ones(x_scenes_valid.shape)
	time_differences_test = np.ones(x_scenes_test.shape)
	# Array for the target scenes
	y_scenes_train = ma.zeros([train_n - 1, 1, volcano_scenes.shape[1], volcano_scenes.shape[2]])
	y_scenes_valid = ma.zeros([out_n, 1, volcano_scenes.shape[1], volcano_scenes.shape[2]])
	y_scenes_test = ma.zeros([out_n, 1, volcano_scenes.shape[1], volcano_scenes.shape[2]])
	# Formatting the string dates as datetime objects
	formatted_dates = [datetime.strptime(date, '%Y-%m-%d') for date in tabular_metadata['dates']]
	# For all observations - acknowledging that the first (n-1) wont have n prior observations
	#     Also, the first data point wont have a TimeAwareLSTM time value, so it is omitted
	for i in range(num_input_scenes + 1, x_scenes_train.shape[0] + x_scenes_valid.shape[0] + x_scenes_test.shape[0] + num_input_scenes+1):
		if i < (train_n + num_input_scenes):
			# Store the image data
			x_scenes_train[i - num_input_scenes - 1, :, :, :] = volcano_scenes[(i - num_input_scenes):i, :, :]
			y_scenes_train[i - num_input_scenes - 1, 0, :, :] = volcano_scenes[i, :, :]
			# Compute the time differences and store
			# Time LSTM uses forward-time interval
			if model_selection in ['TimeLSTM', 'ConvTimeLSTM']:
				dates_i_plus_1 = formatted_dates[(i - num_input_scenes + 1):(i + 1)]
				dates_i = formatted_dates[(i - num_input_scenes):i]
				for j in range(len(dates_i_plus_1)):
					time_differences_train[i - num_input_scenes - 1, j] = (dates_i_plus_1[j] - dates_i[j]).days
			# While TimeAwareLSTM uses backwards-time interval
			else:
				dates_i = formatted_dates[(i - num_input_scenes):i]
				dates_i_minus_1 = formatted_dates[(i - num_input_scenes - 1):(i - 1)]
				for j in range(len(dates_i)):
					time_differences_train[i - num_input_scenes - 1, j] = (dates_i[j] - dates_i_minus_1[j]).days
		elif i < (train_n + out_n + num_input_scenes):
			# Store the image data
			x_scenes_valid[i - train_n - num_input_scenes, :, :, :] = volcano_scenes[(i - num_input_scenes):i, :, :]
			y_scenes_valid[i - train_n - num_input_scenes, 0, :, :] = volcano_scenes[i, :, :]
			# Compute the time differences and store
			# Time LSTM uses forward-time interval
			if model_selection in ['TimeLSTM', 'ConvTimeLSTM']:
				dates_i_plus_1 = formatted_dates[(i - num_input_scenes + 1):(i + 1)]
				dates_i = formatted_dates[(i - num_input_scenes):i]
				for j in range(len(dates_i_plus_1)):
					time_differences_valid[i - num_input_scenes - train_n - 1, j] = (dates_i_plus_1[j] - dates_i[j]).days
			# While TimeAwareLSTM uses backwards-time interval
			else:
				dates_i = formatted_dates[(i - num_input_scenes):i]
				dates_i_minus_1 = formatted_dates[(i - num_input_scenes - 1):(i - 1)]
				for j in range(len(dates_i)):
					time_differences_valid[i - num_input_scenes - train_n - 1, j] = (dates_i[j] - dates_i_minus_1[j]).days
		else:
			# Store the image data
			x_scenes_test[i - train_n - out_n - num_input_scenes, :, :, :] = volcano_scenes[(i - num_input_scenes):i, :, :]
			y_scenes_test[i - train_n - out_n - num_input_scenes, 0, :, :] = volcano_scenes[i, :, :]
			# Compute the time differences and store
			# Time LSTM uses forward-time interval
			if model_selection in ['TimeLSTM', 'ConvTimeLSTM']:
				dates_i_plus_1 = formatted_dates[(i - num_input_scenes + 1):(i + 1)]
				dates_i = formatted_dates[(i - num_input_scenes):i]
				for j in range(len(dates_i_plus_1)):
					time_differences_test[i - num_input_scenes - train_n - out_n - 1, j] = (dates_i_plus_1[j] - dates_i[j]).days
			# While TimeAwareLSTM uses backwards-time interval
			else:
				dates_i = formatted_dates[(i - num_input_scenes):i]
				dates_i_minus_1 = formatted_dates[(i - num_input_scenes - 1):(i - 1)]
				for j in range(len(dates_i)):
					time_differences_test[i - num_input_scenes - train_n - out_n - 1, j] = (dates_i[j] - dates_i_minus_1[j]).days
	if count == 0:
		x_train = x_scenes_train
		t_train = time_differences_train
		y_train = y_scenes_train
		x_valid = x_scenes_valid
		t_valid = time_differences_valid
		y_valid = y_scenes_valid
		x_test = x_scenes_test
		t_test = time_differences_test
		y_test = y_scenes_test
	else:
		x_train = ma.append(x_train, x_scenes_train, axis = 0)
		t_train = np.append(t_train, time_differences_train, axis = 0)
		y_train = ma.append(y_train, y_scenes_train, axis = 0)
		x_valid = ma.append(x_valid, x_scenes_valid, axis = 0)
		t_valid = np.append(t_valid, time_differences_valid, axis = 0)
		y_valid = ma.append(y_valid, y_scenes_valid, axis = 0)
		x_test = ma.append(x_test, x_scenes_test, axis = 0)
		t_test = np.append(t_test, time_differences_test, axis = 0)
		y_test = ma.append(y_test, y_scenes_test, axis = 0)
	count += 1
	vol_cutoff_indices.append(y_train.shape[0])
	vol_cutoff_indices_valid.append(y_valid.shape[0])
	vol_cutoff_indices_test.append(y_test.shape[0])
	vol_name_ls.append(vol)
	print('\timported ' + str(x_scenes_train.shape[0]) + ' training scenes from ' + vol)
	print('\t\timported ' + str(x_scenes_valid.shape[0]) + ' validation scenes from ' + vol)
	print('\t\timported ' + str(x_scenes_test.shape[0]) + ' test scenes from ' + vol)


print("Processing data")
# Gap fill missing values with previous ones or nearest neighbor
for i in range(len(x_train)):
	for j in range(x_train.shape[1]):
		# Identifying missing values
		ma = np.ma.masked_invalid(x_train[i, j, :, :])
		# If the mask found NA values
		if not np.all(x_train[i, j, :, :].mask == False):
			# Using previous value to fill
			if j == 0:
				# Unless there's no previous, then nearest neighbor interpolate
				if i == 0 or i in vol_cutoff_indices:
					print('\t\tNo prior scene; nearest neighbor interpolation')
					scene = x_train[i, j, :, :]
					# Explicitly retrieving good and bad locations from the mask
					# note that these differ from (x, y) which earlier defined bad pixel locations
					X, Y = np.mgrid[0:scene.shape[0], 0:scene.shape[1]]
					xygood = np.array((X[~scene.mask], Y[~scene.mask])).T
					xybad = np.array((X[scene.mask], Y[scene.mask])).T
					# Performing the nearest neighbor gap-filling
					x_train[i, j, :, :][ma.mask == True] = scene[~scene.mask][KDTree(xygood).query(xybad)[1]]
				else:
					x_train[i, j, :, :][ma.mask == True] = x_train[i-1, j, :, :][ma.mask == True]
					t_train[i, j, :, :][ma.mask == True] = t_train[i, j, :, :][ma.mask == True] + t_train[i-1, j, :, :][ma.mask == True]
			else:
				x_train[i, j, :, :][ma.mask == True] = x_train[i, j-1, :, :][ma.mask == True]
				t_train[i, j, :, :][ma.mask == True] = t_train[i, j, :, :][ma.mask == True] + t_train[i, j-1, :, :][ma.mask == True]
for i in range(len(x_valid)):
	for j in range(x_valid.shape[1]):
		# Identifying missing values
		ma = np.ma.masked_invalid(x_valid[i, j, :, :])
		# If the mask found NA values
		if not np.all(ma.mask == False):
			# Using previous value to fill
			if j == 0:
				# Unless there's no previous, then nearest neighbor interpolate
				if i == 0 or i in vol_cutoff_indices_valid:
					scene = x_valid[i, j, :, :]
					# Explicitly retrieving good and bad locations from the mask
					# note that these differ from (x, y) which earlier defined bad pixel locations
					X, Y = np.mgrid[0:scene.shape[0], 0:scene.shape[1]]
					xygood = np.array((X[~scene.mask], Y[~scene.mask])).T
					xybad = np.array((X[scene.mask], Y[scene.mask])).T
					# Performing the nearest neighbor gap-filling
					x_valid[i, j, :, :][ma.mask == True] = scene[~scene.mask][KDTree(xygood).query(xybad)[1]]
				else:
					x_valid[i, j, :, :][ma.mask == True] = x_valid[i-1, j, :, :][ma.mask == True]
					t_valid[i, j, :, :][ma.mask == True] = t_valid[i, j, :, :][ma.mask == True] + t_valid[i-1, j, :, :][ma.mask == True]
			else:
				x_valid[i, j, :, :][ma.mask == True] = x_valid[i, j-1, :, :][ma.mask == True]
				t_valid[i, j, :, :][ma.mask == True] = t_valid[i, j, :, :][ma.mask == True] + t_valid[i, j-1, :, :][ma.mask == True]
for i in range(len(x_test)):
	for j in range(x_test.shape[1]):
		# Identifying missing values
		ma = np.ma.masked_invalid(x_test[i, j, :, :])
		# If the mask found NA values
		if not np.all(ma.mask == False):
			# Using previous value to fill
			if j == 0:
				# Unless there's no previous, then nearest neighbor interpolate
				if i == 0 or i in vol_cutoff_indices_test:
					scene = x_test[i, j, :, :]
					# Explicitly retrieving good and bad locations from the mask
					# note that these differ from (x, y) which earlier defined bad pixel locations
					X, Y = np.mgrid[0:scene.shape[0], 0:scene.shape[1]]
					xygood = np.array((X[~scene.mask], Y[~scene.mask])).T
					xybad = np.array((X[scene.mask], Y[scene.mask])).T
					# Performing the nearest neighbor gap-filling
					x_test[i, j, :, :][ma.mask == True] = scene[~scene.mask][KDTree(xygood).query(xybad)[1]]
				else:
					x_test[i, j, :, :][ma.mask == True] = x_test[i-1, j, :, :][ma.mask == True]
					t_test[i, j, :, :][ma.mask == True] = t_test[i, j, :, :][ma.mask == True] + t_test[i-1, j, :, :][ma.mask == True]
			else:
				x_test[i, j, :, :][ma.mask == True] = x_test[i, j-1, :, :][ma.mask == True]
				t_test[i, j, :, :][ma.mask == True] = t_test[i, j, :, :][ma.mask == True] + t_test[i, j-1, :, :][ma.mask == True]
for i in range(len(y_train)):
	ma = np.ma.masked_invalid(y_train[i, :, :])
	# If the mask found NA values
	if not np.all(ma.mask == False):
		if i == 0 or i in vol_cutoff_indices: 
			y_train[i, :, :][ma.mask == True] = x_train[i, [-1], :, :][ma.mask == True]
		else:
			y_train[i, :, :][ma.mask == True] = y_train[i-1, :, :, :][ma.mask == True]
for i in range(len(y_valid)):
	ma = np.ma.masked_invalid(y_valid[i, :, :, :])
	# If the mask found NA values
	if not np.all(ma.mask == False):
		if i == 0 or i in vol_cutoff_indices_valid:
			y_valid[i, :, :, :][ma.mask == True] = x_valid[i, [-1], :, :][ma.mask == True]
		else:
			y_valid[i, :, :, :][ma.mask == True] = y_valid[i-1, :, :, :][ma.mask == True]
for i in range(len(y_test)):
	ma = np.ma.masked_invalid(y_test[i, :, :, :])
	# If the mask found NA values
	if not np.all(ma.mask == False):
		if i == 0 or i in vol_cutoff_indices_test:
			y_test[i, :, :, :][ma.mask == True] = x_test[i, [-1], :, :][ma.mask == True]
		else:
			y_test[i, :, :, :][ma.mask == True] = y_test[i-1, :, :, :][ma.mask == True]


# Scale 0-1 using min and max from training set
# Also attempts to find NAs and replace with 0s, but those shouldnt exist anymore
stored_parameters = np.zeros([2, 3])
x_train, stored_parameters = scale_and_remove_na(x_train, stored_parameters, 0)
x_valid = (x_valid - stored_parameters[0, 0]) / (stored_parameters[1, 0] - stored_parameters[0, 0])
x_test = (x_test - stored_parameters[0, 0]) / (stored_parameters[1, 0] - stored_parameters[0, 0])
t_train, stored_parameters = scale_and_remove_na(t_train, stored_parameters, 1)
t_valid = (t_valid - stored_parameters[0, 1]) / (stored_parameters[1, 1] - stored_parameters[0, 1])
t_test = (t_test - stored_parameters[0, 1]) / (stored_parameters[1, 1] - stored_parameters[0, 1])
y_train, stored_parameters = scale_and_remove_na(y_train, stored_parameters, 2)
y_valid = (y_valid - stored_parameters[0, 2]) / (stored_parameters[1, 2] - stored_parameters[0, 2])
y_test = (y_test - stored_parameters[0, 2]) / (stored_parameters[1, 2] - stored_parameters[0, 2])
np.save("outputs/transformation_parameters.npy", stored_parameters)


# Convert to torch tensors
x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
x_valid = torch.from_numpy(x_valid).type(torch.FloatTensor)
t_train = torch.from_numpy(t_train).type(torch.FloatTensor)
t_test = torch.from_numpy(t_test).type(torch.FloatTensor)
t_valid = torch.from_numpy(t_valid).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.FloatTensor)
y_valid = torch.from_numpy(y_valid).type(torch.FloatTensor)


# Defining model parameters
# Picking one of the like-sequence tensors within the list to set parameters
print("Setting up methods")
channels = 1
height = x_train.shape[2]
width = x_train.shape[3]
lstm_model = LSTM_Model(input_dim=channels,hidden_dim=hidden_dim_ls,GPU=True,input_size=(height,width),num_layers=n_layers)
# Print number of model parameters
total_params = sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)
print('\tTotal number of model parameters:', total_params)



# Passing to GPU
if model_selection != 'AR':
	lstm_model.cuda()


# Defining data sets and loaders for parallelization option
if training_data_set == 'all':
	training_set = efficient_Dataset(data_indices=range(y_train.shape[0]), x = x_train, t=t_train, y = y_train)
else:
	print('\tNote: imported all data, but only using training data for', training_data_set)
	vol_ID = vol_name_ls.index(training_data_set)
	index_max = vol_cutoff_indices[vol_ID]
	if vol_ID == 0:
		index_min = 0
	else:
		index_min = vol_cutoff_indices[vol_ID - 1]
	curr_data_indices = range(index_min, index_max)
	training_set = efficient_Dataset(data_indices=curr_data_indices, x = x_train, t=t_train, y = y_train)
validation_set = efficient_Dataset(data_indices=range(y_valid.shape[0]), x = x_valid, t = t_valid, y = y_valid)
train_loader = torch.utils.data.DataLoader(dataset = training_set, batch_size = batch_size, shuffle = True)
validation_loader = torch.utils.data.DataLoader(dataset = validation_set, batch_size = batch_size, shuffle = True)


# Determining compute options (GPU? Parallel?)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if model_selection != 'AR':
	lstm_model = torch.nn.DataParallel(lstm_model)


# Setting optimization methods
loss = torch.nn.MSELoss()
# Defining many regularization strengths to iterate over to combat overfitting
l2_regularization_strengths = [0.0001, 0.001, 0.01, 0.1, 1]
for penalization in l2_regularization_strengths:
	optimizer = torch.optim.Adam(lstm_model.parameters(), weight_decay = penalization)
	# Training loop
	print("Beginning training for regularization strength:", penalization)
	loss_list = []
	#epochs = int(np.ceil((7*10**5) / x_train.shape[0]))
	epochs = 100
	loop_begin_time = datetime.now()
	for i in range(epochs):
		# Marking the beginning time of epoch
		begin_time = datetime.now()
		for data in train_loader:
			
			# data loader
			batch_x, batch_t, batch_y = data
			batch_x.unsqueeze_(2) # code was originally wrote for 5-band level-1 data
			batch_t.unsqueeze_(2) # this unsqueeze is just allowing the explicit acknowledgement
			batch_y.unsqueeze_(2) # that the level-2 data is 1-band deep
			
			# reshaping data if needed for non-spatial LSTMs
			if model_selection in ['AR', 'Identity', 'LSTM', 'TimeLSTM', 'TimeAwareLSTM']:
				batch_x = batch_x.permute(0, 3, 4, 1, 2)
				x_sh = batch_x.shape
				batch_x = batch_x.reshape(x_sh[0]*x_sh[1]*x_sh[2], x_sh[3], x_sh[4])
			# Only further processing time in a time-conscious, non-spatial LSTM
			if model_selection in ['TimeLSTM', 'TimeAwareLSTM']:
				batch_t = batch_t.permute(0, 3, 4, 1, 2)
				t_sh = batch_t.shape
				batch_t = batch_t.reshape(t_sh[0]*t_sh[1]*t_sh[2], t_sh[3], t_sh[4])
				# This next line is fragile to the assumption that
				# bands have the same sampling time difference
			if model_selection in ['TimeLSTM', 'TimeAwareLSTM', 'ConvTimeAwareLSTM']:
				batch_t = batch_t[:,:,[0]]
			
			# move to GPU
			if model_selection != 'AR':
				batch_x = batch_x.to(device)
				# Only move time tensors to GPU if time-conscious LSTM
				if model_selection not in ['AR', 'Identity', 'LSTM', 'ConvLSTM']:
					batch_t = batch_t.to(device)
				batch_y = batch_y.to(device)
			
			# Run the model, determining forward pass based on model selected
			if model_selection in ['AR', 'Identity', 'LSTM', 'ConvLSTM']:
				batch_y_hat = lstm_model(batch_x)
			elif model_selection in ['TimeAwareLSTM', 'ConvTimeAwareLSTM']:
				batch_y_hat = lstm_model(batch_x, batch_t)
			elif model_selection in ['TimeLSTM', 'ConvTimeLSTM', 'ConvTimeLSTMUnet']:
				batch_y_hat = lstm_model(batch_x, batch_x, batch_t)
			
			# Extracting the target prediction based on model output
			if model_selection not in ['AR', 'Identity']:
				batch_y_hat = batch_y_hat[0][0]
			if model_selection in ['AR', 'Identity', 'LSTM', 'TimeLSTM', 'TimeAwareLSTM']:
				batch_y_hat = batch_y_hat.reshape(x_sh)
				batch_y_hat = batch_y_hat.permute(0, 3, 4, 1, 2)
			batch_y_hat = batch_y_hat[:, [-1], :, :, :]
			
			# calculate and store the loss
			batch_loss = loss(batch_y, batch_y_hat)
			loss_list.append(batch_loss.item())
			
			# update parameters
			if model_selection != 'Identity':
				optimizer.zero_grad()
				batch_loss.backward()
				optimizer.step()
			
		# Marking the end time and computing time difference, also printing epoch information
		end_time = datetime.now()
		time_diff = (end_time - loop_begin_time).total_seconds()
		# With l2 regularization iterations, too many prints; only printing every 10 epochs
		if (i + 1) % 10 == 0:
			print('\tEpoch:', i, '\n\t\tMost recent batch loss:', batch_loss.item(), '\n\t\t' + str(time_diff) + ' seconds elapsed')
	loop_end_time = datetime.now()
	loop_time_diff = (loop_end_time - loop_begin_time).total_seconds()
	print('\tTotal training-loop time:', loop_time_diff)


	# Converting loss values into array and saving
	loss_array = np.asarray(loss_list)
	np.save("outputs/loss_over_iterations" + str(penalization) + ".npy", loss_array)


	# Trying to free GPU memory
	del batch_x
	del batch_t
	del batch_y
	del batch_y_hat
	del batch_loss
	torch.cuda.empty_cache()


	print("Beginning evaluation")


	# Determine train set performance by volcano
	with torch.no_grad():
		count = 0
		for i in range(len(y_train)):
			batch_x = x_train[[i], :, :, :].unsqueeze(2)
			batch_t = t_train[[i], :, :, :].unsqueeze(2)
			batch_y = y_train[[i], :, :, :].unsqueeze(2)
			
			# reshaping data if needed for non-spatial LSTMs
			if model_selection in ['AR', 'Identity', 'LSTM', 'TimeLSTM', 'TimeAwareLSTM']:
				batch_x = batch_x.permute(0, 3, 4, 1, 2)
				x_sh = batch_x.shape
				batch_x = batch_x.reshape(x_sh[0]*x_sh[1]*x_sh[2], x_sh[3], x_sh[4])
			# Only further processing time in a time-conscious, non-spatial LSTM
			if model_selection in ['TimeLSTM', 'TimeAwareLSTM']:
				batch_t = batch_t.permute(0, 3, 4, 1, 2)
				t_sh = batch_t.shape
				batch_t = batch_t.reshape(t_sh[0]*t_sh[1]*t_sh[2], t_sh[3], t_sh[4])
				# This next line is fragile to the assumption that
				# bands have the same sampling time difference
			if model_selection in ['TimeLSTM', 'TimeAwareLSTM', 'ConvTimeAwareLSTM']:
				batch_t = batch_t[:,:,[0]]
				
			# move to GPU
			if model_selection != 'AR':
				batch_x = batch_x.to(device)
				# Only move time tensors to GPU if time-conscious LSTM
				if model_selection not in ['AR', 'Identity', 'LSTM', 'ConvLSTM']:
					batch_t = batch_t.to(device)
				batch_y = batch_y.to(device)
				
			# Run the model, determining forward pass based on model selected
			if model_selection in ['AR', 'Identity', 'LSTM', 'ConvLSTM']:
				batch_y_hat = lstm_model(batch_x)
			elif model_selection in ['TimeAwareLSTM', 'ConvTimeAwareLSTM']:
				batch_y_hat = lstm_model(batch_x, batch_t)
			elif model_selection in ['TimeLSTM', 'ConvTimeLSTM', 'ConvTimeLSTMUnet']:
				batch_y_hat = lstm_model(batch_x, batch_x, batch_t)
			
			# Extracting the target prediction based on model output
			if model_selection not in ['AR', 'Identity']:
				batch_y_hat = batch_y_hat[0][0]
			if model_selection in ['AR', 'Identity', 'LSTM', 'TimeLSTM', 'TimeAwareLSTM']:
				batch_y_hat = batch_y_hat.reshape(x_sh)
				batch_y_hat = batch_y_hat.permute(0, 3, 4, 1, 2)
			batch_y_hat = batch_y_hat[:, [-1], :, :, :]
			
			# Moving data off GPU now that model has ran
			batch_y = batch_y.cpu()
			batch_y_hat = batch_y_hat.cpu()
			
			# Transformating the data to temperature values
			train_y_min = torch.tensor(stored_parameters[0, 0])
			train_y_max = torch.tensor(stored_parameters[1, 0])
			batch_y = (batch_y * (train_y_max - train_y_min)) + train_y_min
			batch_y_hat = (batch_y_hat * (train_y_max - train_y_min)) + train_y_min
			
			# Storing all temperature-valued truths and predictions for
			# one root mean squared error calculation to get error in
			# terms of temperature
			if count == 0:
				cpu_y_temps = batch_y
				cpu_y_hat_temps = batch_y_hat
			else:
				cpu_y_temps = torch.cat([cpu_y_temps, batch_y], dim = 0)
				cpu_y_hat_temps = torch.cat([cpu_y_hat_temps, batch_y_hat], dim = 0)
			count = count + 1
		# calculate and store the loss
		train_set_loss = loss(cpu_y_hat_temps, cpu_y_temps)
		train_set_loss = torch.sqrt(train_set_loss)
		train_set_loss = train_set_loss.item()
		# print traination set loss
		print("\tTraining set loss:", train_set_loss)
		# Saving the predictions and corresponding truths
		np.save("outputs/train_prediction" + str(penalization) + ".npy", cpu_y_hat_temps.numpy())
		np.save("outputs/train_truth" + str(penalization) + ".npy", cpu_y_temps.numpy())
		
		vol_ID = 0
		for vol in vol_name_ls:
			if vol_ID == 0:
				index_min = 0
			else:
				index_min = vol_cutoff_indices[vol_ID - 1]
			index_max = vol_cutoff_indices[vol_ID]
			pred_vol = cpu_y_hat_temps[index_min:index_max, :, :, :, :]
			true_vol = cpu_y_temps[index_min:index_max, :, :, :, :]
			vol_loss = loss(pred_vol, true_vol)
			vol_loss = torch.sqrt(vol_loss)
			vol_loss = vol_loss.item()
			print('\t\tTraining set loss for', vol, ':', vol_loss)
			vol_ID += 1

	# Saving the train set loss
	np.save("outputs/final_train_loss" + str(penalization) + ".npy", np.asarray(train_set_loss))


	# Determine validation set performance by volcano
	with torch.no_grad():
		count = 0
		for i in range(len(y_valid)):
			batch_x = x_valid[[i], :, :, :].unsqueeze(2)
			batch_t = t_valid[[i], :, :, :].unsqueeze(2)
			batch_y = y_valid[[i], :, :, :].unsqueeze(2)
			
			# reshaping data if needed for non-spatial LSTMs
			if model_selection in ['AR', 'Identity', 'LSTM', 'TimeLSTM', 'TimeAwareLSTM']:
				batch_x = batch_x.permute(0, 3, 4, 1, 2)
				x_sh = batch_x.shape
				batch_x = batch_x.reshape(x_sh[0]*x_sh[1]*x_sh[2], x_sh[3], x_sh[4])
			# Only further processing time in a time-conscious, non-spatial LSTM
			if model_selection in ['TimeLSTM', 'TimeAwareLSTM']:
				batch_t = batch_t.permute(0, 3, 4, 1, 2)
				t_sh = batch_t.shape
				batch_t = batch_t.reshape(t_sh[0]*t_sh[1]*t_sh[2], t_sh[3], t_sh[4])
				# This next line is fragile to the assumption that
				# bands have the same sampling time difference
			if model_selection in ['TimeLSTM', 'TimeAwareLSTM', 'ConvTimeAwareLSTM']:
				batch_t = batch_t[:,:,[0]]
				
			# move to GPU
			if model_selection != 'AR':
				batch_x = batch_x.to(device)
				# Only move time tensors to GPU if time-conscious LSTM
				if model_selection not in ['AR', 'Identity', 'LSTM', 'ConvLSTM']:
					batch_t = batch_t.to(device)
				batch_y = batch_y.to(device)
				
			# Run the model, determining forward pass based on model selected
			if model_selection in ['AR', 'Identity', 'LSTM', 'ConvLSTM']:
				batch_y_hat = lstm_model(batch_x)
			elif model_selection in ['TimeAwareLSTM', 'ConvTimeAwareLSTM']:
				batch_y_hat = lstm_model(batch_x, batch_t)
			elif model_selection in ['TimeLSTM', 'ConvTimeLSTM', 'ConvTimeLSTMUnet']:
				batch_y_hat = lstm_model(batch_x, batch_x, batch_t)
			
			# Extracting the target prediction based on model output
			if model_selection not in ['AR', 'Identity']:
				batch_y_hat = batch_y_hat[0][0]
			if model_selection in ['AR', 'Identity', 'LSTM', 'TimeLSTM', 'TimeAwareLSTM']:
				batch_y_hat = batch_y_hat.reshape(x_sh)
				batch_y_hat = batch_y_hat.permute(0, 3, 4, 1, 2)
			batch_y_hat = batch_y_hat[:, [-1], :, :, :]
			
			# Moving data off GPU now that model has ran
			batch_y = batch_y.cpu()
			batch_y_hat = batch_y_hat.cpu()
			
			# Transformating the data to temperature values
			train_y_min = torch.tensor(stored_parameters[0, 0])
			train_y_max = torch.tensor(stored_parameters[1, 0])
			batch_y = (batch_y * (train_y_max - train_y_min)) + train_y_min
			batch_y_hat = (batch_y_hat * (train_y_max - train_y_min)) + train_y_min
			
			# Storing all temperature-valued truths and predictions for
			# one root mean squared error calculation to get error in
			# terms of temperature
			if count == 0:
				cpu_y_temps = batch_y
				cpu_y_hat_temps = batch_y_hat
			else:
				cpu_y_temps = torch.cat([cpu_y_temps, batch_y], dim = 0)
				cpu_y_hat_temps = torch.cat([cpu_y_hat_temps, batch_y_hat], dim = 0)
			count = count + 1
		# calculate and store the loss
		valid_set_loss = loss(cpu_y_hat_temps, cpu_y_temps)
		valid_set_loss = torch.sqrt(valid_set_loss)
		valid_set_loss = valid_set_loss.item()
		# print validation set loss
		print("\tValidation set loss:", valid_set_loss)
		# Saving the predictions and corresponding truths
		np.save("outputs/valid_prediction" + str(penalization) + ".npy", cpu_y_hat_temps.numpy())
		np.save("outputs/valid_truth" + str(penalization) + ".npy", cpu_y_temps.numpy())
		
		vol_ID = 0
		for vol in vol_name_ls:
			if vol_ID == 0:
				index_min = 0
			else:
				index_min = vol_cutoff_indices_valid[vol_ID - 1]
			index_max = vol_cutoff_indices_valid[vol_ID]
			pred_vol = cpu_y_hat_temps[index_min:index_max, :, :, :, :]
			true_vol = cpu_y_temps[index_min:index_max, :, :, :, :]
			vol_loss = loss(pred_vol, true_vol)
			vol_loss = torch.sqrt(vol_loss)
			vol_loss = vol_loss.item()
			print('\t\tValidation set loss for', vol, ':', vol_loss)
			vol_ID += 1

	# Saving the validation set loss
	np.save("outputs/final_valid_loss" + str(penalization) + ".npy", np.asarray(valid_set_loss))
	# Saving the model
	torch.save(lstm_model, "outputs/model" + str(penalization) + ".pt")
	
	# Print breakpoint for human reading between regularization strengths
	print('\n')
