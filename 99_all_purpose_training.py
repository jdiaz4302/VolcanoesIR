 



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
#         * "Time-Aware LSTM"
#     3) Convolutional LSTM (that allows images sequences)
#         * "ConvLSTM"
#     4) New/proposed models that allow image sequences that are
#        irregular through time
#         * Convolutional LSTM + Time-LSTM
#             - "ConvTimeLSTM"
#         * Convolutional LSTM + Time-Aware LSTM
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


# Gathering function inputs
model_selection = input("Which model do you select?")
assert(model_selection in ['LSTM', 'Time-LSTM', 'Time-Aware LSTM', 'ConvLSTM', 'ConvTimeLSTM', 'ConvTimeAwareLSTM'])
training_data_set = input("Which set of training data do you want to use?")


# Basic data import, step 0
print("Importing and formatting data")
volcanoes = os.listdir("data")
assert((training_data_set in volcanoes) or (training_data_set == "all"))


# Training parameters
# This needs to actually be variable, will do with later exploration
batch_size_dict = {'LSTM':4, 'Time-LSTM':4, 'Time-Aware LSTM':4, 'ConvLSTM':4, 'ConvTimeLSTM':4, 'ConvTimeAwareLSTM':4}
print("W A R N I N G: Further exploratory work needded for variable batch size")
lag_dict = {"all":6, "ErtaAle":9, "Kilauea":10, "Masaya":3, "Nyamuragira":3, "Nyiragongo":3, "Pacaya":4, "Puuoo":8}
batch_size = batch_size_dict[model_selection]
num_input_scenes = lag_dict[training_data_set]
train_percent = 0.70
out_samp_perc = 0.15


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
	from torch.nn import LSTM as LSTM_Model
elif model_selection == 'TimeLSTM':
	from models.TimeLSTM import StackedTimeLSTM as LSTM_Model
elif model_selection == 'Time-Aware LSTM':
	from models.TimeAwareLSTM import StackedTimeAwareLSTM as LSTM_Model
elif model_selection == 'ConvLSTM':
	from models.ConvLSTM import ConvLSTM as LSTM_Model
elif model_selection == 'ConvTimeLSTM':
	from models.ConvTimeLSTM2 import ConvTime_LSTM2 as LSTM_Model
from helper_fns.processing import scale_and_remove_na
from helper_fns.efcnt_data import efficient_Dataset


# Removing possible directory clutter
try:
	volcanoes.remove(".ipynb_checkpoints")
except ValueError as e:
	do = 'nothing'

count = 0
for vol in volcanoes:
	### Basic data import ###
	numpy_data_location = "data/" + vol + "/numpy_data_cube.npy"
	table_data_location = "data/" + vol + "/good_df.csv"
	volcano_scenes = np.load(numpy_data_location)
	tabular_metadata = pd.read_csv(table_data_location)
	### Separate model inputs and outputs
	# Determine number in each partition
	train_n = int(np.floor((len(volcano_scenes) - num_input_scenes)*train_percent))
	out_n = int(np.floor((len(volcano_scenes) - num_input_scenes)*out_samp_perc))
	# For every data partition
	# Array for the prior scenes
	x_scenes_train = np.zeros([train_n, num_input_scenes, volcano_scenes.shape[1], volcano_scenes.shape[2], volcano_scenes.shape[3]])
	x_scenes_valid = np.zeros([out_n, num_input_scenes, volcano_scenes.shape[1], volcano_scenes.shape[2], volcano_scenes.shape[3]])
	x_scenes_test = np.zeros([out_n, num_input_scenes, volcano_scenes.shape[1], volcano_scenes.shape[2], volcano_scenes.shape[3]])
	# Array for the time differences between scenes
	time_differences_train = np.ones(x_scenes_train.shape)
	time_differences_valid = np.ones(x_scenes_valid.shape)
	time_differences_test = np.ones(x_scenes_test.shape)
	# Array for the target scenes
	y_scenes_train = np.zeros([train_n, 1, volcano_scenes.shape[1], volcano_scenes.shape[2], volcano_scenes.shape[3]])
	y_scenes_valid = np.zeros([out_n, 1, volcano_scenes.shape[1], volcano_scenes.shape[2], volcano_scenes.shape[3]])
	y_scenes_test = np.zeros([out_n, 1, volcano_scenes.shape[1], volcano_scenes.shape[2], volcano_scenes.shape[3]])
	# Array for the prior max temperature above the background
	x_temperatures_train = np.zeros([train_n, num_input_scenes])
	x_temperatures_valid = np.zeros([out_n, num_input_scenes])
	x_temperatures_test = np.zeros([out_n, num_input_scenes])
	# Array for the target max temperature above the background
	y_temperatures_train = np.zeros([train_n])
	y_temperatures_valid = np.zeros([out_n])
	y_temperatures_test = np.zeros([out_n])
	# Formatting the string dates as datetime objects
	formatted_dates = [datetime.strptime(date, '%Y-%m-%d') for date in tabular_metadata['dates']]
	# For all observations - acknowledging that the first (n-1) wont have n prior observations
	for i in range(num_input_scenes, x_scenes_train.shape[0] + x_scenes_valid.shape[0] + x_scenes_test.shape[0] + 10):
		if i < (train_n + num_input_scenes):
			# Store the image data
			x_scenes_train[i - num_input_scenes, :, :, :, :] = volcano_scenes[(i - num_input_scenes):i, :, :, :]
			y_scenes_train[i - num_input_scenes, 0, :, :, :] = volcano_scenes[i, :, :, :]
			# Store the max temperature scalars
			x_temperatures_train[i - num_input_scenes, :] = tabular_metadata['T_above_back'].values[(i - num_input_scenes):i]
			y_temperatures_train[i - num_input_scenes] = tabular_metadata['T_above_back'].values[i]
			# Compute the time differences and store
			dates_i_plus_1 = formatted_dates[(i - num_input_scenes + 1):(i + 1)]
			dates_i = formatted_dates[(i - num_input_scenes):i]
			for j in range(len(dates_i_plus_1)):
				time_differences_train[i - num_input_scenes, j] = (dates_i_plus_1[j] - dates_i[j]).days
		elif i < (train_n + out_n + num_input_scenes):
			# Store the image data
			x_scenes_valid[i - train_n - num_input_scenes, :, :, :, :] = volcano_scenes[(i - num_input_scenes):i, :, :, :]
			y_scenes_valid[i - train_n - num_input_scenes, 0, :, :, :] = volcano_scenes[i, :, :, :]
			# Store the max temperature scalars
			x_temperatures_valid[i - train_n - num_input_scenes, :] = tabular_metadata['T_above_back'].values[(i - num_input_scenes):i]
			y_temperatures_valid[i - train_n - num_input_scenes] = tabular_metadata['T_above_back'].values[i]
			# Compute the time differences and store
			dates_i_plus_1 = formatted_dates[(i - num_input_scenes + 1):(i + 1)]
			dates_i = formatted_dates[(i - num_input_scenes):i]
			for j in range(len(dates_i_plus_1)):
				time_differences_valid[i - train_n - num_input_scenes, j] = (dates_i_plus_1[j] - dates_i[j]).days
		else:
			# Store the image data
			x_scenes_test[i - train_n - out_n - num_input_scenes, :, :, :, :] = volcano_scenes[(i - num_input_scenes):i, :, :, :]
			y_scenes_test[i - train_n - out_n - num_input_scenes, 0, :, :, :] = volcano_scenes[i, :, :, :]
			# Store the max temperature scalars
			x_temperatures_test[i - train_n - out_n - num_input_scenes, :] = tabular_metadata['T_above_back'].values[(i - num_input_scenes):i]
			y_temperatures_test[i - train_n - out_n - num_input_scenes] = tabular_metadata['T_above_back'].values[i]
			# Compute the time differences and store
			dates_i_plus_1 = formatted_dates[(i - num_input_scenes + 1):(i + 1)]
			dates_i = formatted_dates[(i - num_input_scenes):i]
			for j in range(len(dates_i_plus_1)):
				time_differences_test[i - train_n - out_n - num_input_scenes, j] = (dates_i_plus_1[j] - dates_i[j]).days
	if ((vol == "all") and (count == 0)) or (vol == training_data_set):
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
		x_train = np.append(x_train, x_scenes_train, axis = 0)
		t_train = np.append(t_train, time_differences_train, axis = 0)
		y_train = np.append(y_train, y_scenes_train, axis = 0)
		x_valid = np.append(x_valid, x_scenes_valid, axis = 0)
		t_valid = np.append(t_valid, time_differences_valid, axis = 0)
		y_valid = np.append(y_valid, y_scenes_valid, axis = 0)
		x_test = np.append(x_test, x_scenes_test, axis = 0)
		t_test = np.append(t_test, time_differences_test, axis = 0)
		y_test = np.append(y_test, y_scenes_train, axis = 0)
	count += 1
	print('\t\timported ' + str(x_scenes_train.shape[0]) + ' training scenes from ' + vol)


# Scale 0-1, replace NAs with scaled 0s
print("Processing data")
stored_parameters = np.zeros([2, 9])
x_train, stored_parameters = scale_and_remove_na(x_train, stored_parameters, 0)
x_valid, stored_parameters = scale_and_remove_na(x_valid, stored_parameters, 1)
x_test, stored_parameters = scale_and_remove_na(x_test, stored_parameters, 2)
t_train, stored_parameters = scale_and_remove_na(t_train, stored_parameters, 3)
t_valid, stored_parameters = scale_and_remove_na(t_valid, stored_parameters, 4)
t_test, stored_parameters = scale_and_remove_na(t_test, stored_parameters, 5)
y_train, stored_parameters = scale_and_remove_na(y_train, stored_parameters, 6)
y_valid, stored_parameters = scale_and_remove_na(y_valid, stored_parameters, 7)
y_test, stored_parameters = scale_and_remove_na(y_test, stored_parameters, 8)
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
channels = x_train.shape[2]
height = x_train.shape[3]
width = x_train.shape[4]
lstm_model = LSTM_Model(input_sz = channels, layer_sizes = [128, 64, 64, channels], GPU = True)


# Passing to GPU
lstm_model.cuda()


# Setting optimization methods
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters())


# Defining data sets and loaders for parallelization option
training_set = efficient_Dataset(data_indices=range(y_train.shape[0]), x = x_train, t=t_train, y = y_train)
validation_set = efficient_Dataset(data_indices=range(y_valid.shape[0]), x = x_valid, t = t_valid, y = y_valid)
train_loader = torch.utils.data.DataLoader(dataset = training_set, batch_size = batch_size, shuffle = True)
validation_loader = torch.utils.data.DataLoader(dataset = validation_set, batch_size = batch_size, shuffle = True)


# Determining compute options (GPU? Parallel?)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lstm_model = torch.nn.DataParallel(lstm_model)


# Training loop
print("Beginning training")
loss_list = []
#epochs = int(np.ceil((7*10**5) / x_train.shape[0]))
epochs = 2
for i in range(epochs):
	# Marking the beginning time of epoch
	begin_time = datetime.now()
	for data in train_loader:
		
		# data loader
		batch_x, batch_t, batch_y = data
		
		# reshaping data if needed for non-spatial LSTMs
		if model_selection in ['LSTM', 'Time-LSTM', 'Time-Aware LSTM']:
			x_sh = batch_x.shape
			batch_x = batch_x.view(x_sh[0]*x_sh[3]*x_sh[4], x_sh[1], x_sh[2])
			# We wont reshape y, instead y_hat to fit y
			y_sh = batch_y.shape
		# Only further processing time in a time-conscious, non-spatial LSTM
		if model_selection in ['Time-LSTM', 'Time-Aware LSTM']:
			t_sh = batch_t.shape
			batch_t = batch_t.view(t_sh[0]*t_sh[3]*t_sh[4], t_sh[1], t_sh[2])
			# This next line is fragile to the assumption that
			# bands have the same sampling time difference
			batch_t = batch_t[:,:,0:1]
		
		# move to GPU
		batch_x = batch_x.to(device)
		# Only move time tensors to GPU if time-conscious LSTM
		if model_selection not in ['LSTM', 'ConvLSTM']:
			batch_t = batch_t.to(device)
		batch_y = batch_y.to(device)
		
		# Run the model, determining forward pass based on model selected
		if model_selection in ['LSTM', 'ConvLSTM']:
			batch_y_hat = lstm_model(batch_x)
		elif model_selection in ['Time-LSTM', 'Time-Aware LSTM']:
			batch_y_hat = lstm_model(batch_x, batch_t)
		elif model_selection == 'ConvTimeLSTM':
			batch_y_hat = lstm_model(batch_x, batch_x, batch_t)
		
		# Extracting the target prediction based on model output
		if model_selection in ['LSTM', 'Time-LSTM', 'Time-Aware LSTM']:
			batch_y_hat = batch_y_hat[0]
			batch_y_hat = batch_y_hat.view(x_sh)
		else:
			batch_y_hat = batch_y_hat[0][0]
		batch_y_hat = batch_y_hat[:, -2:-1, :, :, :]
		
		# calculate and store the loss
		batch_loss = loss(batch_y, batch_y_hat)
		loss_list.append(batch_loss.item())
		
		# update parameters
		optimizer.zero_grad()
		batch_loss.backward()
		optimizer.step()
		
	# Marking the end time and computing difference, also printing epoch information
	end_time = datetime.now()
	time_diff = (end_time - begin_time).total_seconds()
	print('Epoch: ', i, '\n\tMost recent batch loss: ', batch_loss.item(), '\n\t' + str(time_diff) + ' seconds elapsed')


# Saving the last training batch for reference
np.save("outputs/train_prediction.npy", batch_y_hat.cpu().data.numpy())
np.save("outputs/train_truth.npy", batch_y.cpu().data.numpy())


# Converting loss values into array and saving
loss_array = np.asarray(loss_list)
np.save('outputs/loss_over_iterations.npy', loss_array)


# Trying to free GPU memory
del batch_x
del batch_t
del batch_y
del batch_y_hat
del batch_loss
torch.cuda.empty_cache()


# Getting the loss value for the whole training set
# torch.no_grad allows some efficiency by not tracking
# values for optimization
with torch.no_grad():
	count = 0
	for data in train_loader:
		
		# data loader
		batch_x, batch_t, batch_y = data
		
		# reshaping data if needed for non-spatial LSTMs
		if model_selection in ['LSTM', 'Time-LSTM', 'Time-Aware LSTM']:
			x_sh = batch_x.shape
			batch_x = batch_x.view(x_sh[0]*x_sh[3]*x_sh[4], x_sh[1], x_sh[2])
			# We wont reshape y, instead y_hat to fit y
			y_sh = batch_y.shape
		# Only further processing time in a time-conscious, non-spatial LSTM
		if model_selection in ['Time-LSTM', 'Time-Aware LSTM']:
			t_sh = batch_t.shape
			batch_t = batch_t.view(t_sh[0]*t_sh[3]*t_sh[4], t_sh[1], t_sh[2])
			# This next line is fragile to the assumption that
			# bands have the same sampling time difference
			batch_t = batch_t[:,:,0:1]
		
		# move to GPU
		batch_x = batch_x.to(device)
		# Only move time tensors to GPU if time-conscious LSTM
		if model_selection not in ['LSTM', 'ConvLSTM']:
			batch_t = batch_t.to(device)
		batch_y = batch_y.to(device)
		
		# Run the model, determining forward pass based on model selected
		if model_selection in ['LSTM', 'ConvLSTM']:
			batch_y_hat = lstm_model(batch_x)
		elif model_selection in ['Time-LSTM', 'Time-Aware LSTM']:
			batch_y_hat = lstm_model(batch_x, batch_t)
		elif model_selection == 'ConvTimeLSTM':
			batch_y_hat = lstm_model(batch_x, batch_x, batch_t)
		
		# Extracting the target prediction based on model output
		if model_selection in ['LSTM', 'Time-LSTM', 'Time-Aware LSTM']:
			batch_y_hat = batch_y_hat[0]
			batch_y_hat = batch_y_hat.view(x_sh)
		else:
			batch_y_hat = batch_y_hat[0][0]
		batch_y_hat = batch_y_hat[:, -2:-1, :, :, :]
		
		# Moving data off GPU now that model has ran
		batch_y = batch_y.cpu()
		batch_y_hat = batch_y_hat.cpu()
		
		# Transformating the data to temperature values
		train_y_min = torch.from_numpy(stored_parameters[0, 0])
		train_y_max = torch.from_numpy(stored_parameters[1, 0])
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


# Saving the training set loss
np.save('outputs/final_train_loss.npy', np.asarray(train_set_loss))


# Getting the loss value for the whole validation set
with torch.no_grad():
	count = 0
	for data in validation_loader:
		
		# data loader
		batch_x, batch_t, batch_y = data
		
		# reshaping data if needed for non-spatial LSTMs
		if model_selection in ['LSTM', 'Time-LSTM', 'Time-Aware LSTM']:
			x_sh = batch_x.shape
			batch_x = batch_x.view(x_sh[0]*x_sh[3]*x_sh[4], x_sh[1], x_sh[2])
			# We wont reshape y, instead y_hat to fit y
			y_sh = batch_y.shape
		# Only further processing time in a time-conscious, non-spatial LSTM
		if model_selection in ['Time-LSTM', 'Time-Aware LSTM']:
			t_sh = batch_t.shape
			batch_t = batch_t.view(t_sh[0]*t_sh[3]*t_sh[4], t_sh[1], t_sh[2])
			# This next line is fragile to the assumption that
			# bands have the same sampling time difference
			batch_t = batch_t[:,:,0:1]
		
		# move to GPU
		batch_x = batch_x.to(device)
		# Only move time tensors to GPU if time-conscious LSTM
		if model_selection not in ['LSTM', 'ConvLSTM']:
			batch_t = batch_t.to(device)
		batch_y = batch_y.to(device)
		
		# Run the model, determining forward pass based on model selected
		if model_selection in ['LSTM', 'ConvLSTM']
			batch_y_hat = lstm_model(batch_x)
		elif model_selection in ['Time-LSTM', 'Time-Aware LSTM']:
			batch_y_hat = lstm_model(batch_x, batch_t)
		elif model_selection == 'ConvTimeLSTM':
			batch_y_hat = lstm_model(batch_x, batch_x, batch_t)
		
		# Extracting the target prediction based on model output
		if model_selection in ['LSTM', 'Time-LSTM', 'Time-Aware LSTM']:
			batch_y_hat = batch_y_hat[0]
			batch_y_hat = batch_y_hat.view(x_sh)
		else:
			batch_y_hat = batch_y_hat[0][0]
		batch_y_hat = batch_y_hat[:, -2:-1, :, :, :]
		
		# Moving data off GPU now that model has ran
		batch_y = batch_y.cpu()
		batch_y_hat = batch_y_hat.cpu()
		
		# Transformating the data to temperature values
		valid_y_min = torch.from_numpy(stored_parameters[0, 1])
		valid_y_max = torch.from_numpy(stored_parameters[1, 1])
		batch_y = (batch_y * (valid_y_max - valid_y_min)) + valid_y_min
		batch_y_hat = (batch_y_hat * (valid_y_max - valid_y_min)) + valid_y_min
		
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


# Saving the validation set loss
np.save('outputs/final_valid_loss.npy', np.asarray(valid_set_loss))


# Generate some validation predictions
with torch.no_grad():
	for i in range(25):
		batch_x, batch_t, batch_y = next(iter(validation_loader))
		if model_selection in ['LSTM', 'Time-LSTM', 'Time-Aware LSTM']:
			x_sh = batch_x.shape
			batch_x = batch_x.view(x_sh[0]*x_sh[3]*x_sh[4], x_sh[1], x_sh[2])
			y_sh = batch_y.shape
		if model_selection in ['Time-LSTM', 'Time-Aware LSTM']:
			t_sh = batch_t.shape
			batch_t = batch_t.view(t_sh[0]*t_sh[3]*t_sh[4], t_sh[1], t_sh[2])
			batch_t = batch_t[:,:,0:1]
		batch_x = batch_x.to(device)
		if model_selection not in ['LSTM', 'ConvLSTM']:
			batch_t = batch_t.to(device)
		batch_y = batch_y.to(device)
		if model_selection in ['LSTM', 'ConvLSTM']:
			batch_y_hat = lstm_model(batch_x)
		elif model_selection in ['Time-LSTM', 'Time-Aware LSTM']:
			batch_y_hat = lstm_model(batch_x, batch_t)
		elif model_selection == 'ConvTimeLSTM':
			batch_y_hat = lstm_model(batch_x, batch_x, batch_t)
		if model_selection in ['LSTM', 'Time-LSTM', 'Time-Aware LSTM']:
			batch_y_hat = batch_y_hat[0]
			batch_y_hat = batch_y_hat.view(x_sh)
		else:
			batch_y_hat = batch_y_hat[0][0]
		batch_y_hat = batch_y_hat[:, -2:-1, :, :, :]
		batch_y_hat = batch_y_hat.cpu().data.numpy()
		np.save("outputs/valid_prediction_" + str(i) + ".npy", batch_y_hat)
		np.save("outputs/valid_truth_" + str(i) + ".npy", batch_y)