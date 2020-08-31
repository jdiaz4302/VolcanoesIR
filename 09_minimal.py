



# Training parameters
batch_size = 8


# Libraries and imports
import os
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils import data
from models.ConvTimeLSTM2 import ConvTime_LSTM2
from helper_fns.processing import scale_and_remove_na
from helper_fns.efcnt_data import efficient_Dataset


# Required input
num_input_scenes = 10
train_percent = 0.70
out_samp_perc = 0.15 # validation and testing


# Basic data import
print("Importing and formatting data")
volcanoes = os.listdir("data")

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
x_train = scale_and_remove_na(x_train)
x_valid = scale_and_remove_na(x_valid)
x_test = scale_and_remove_na(x_test)
t_train = scale_and_remove_na(t_train)
t_valid = scale_and_remove_na(t_valid)
t_test = scale_and_remove_na(t_test)
y_train = scale_and_remove_na(y_train)
y_valid = scale_and_remove_na(y_valid)
y_test = scale_and_remove_na(y_test)


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
conv_time_lstm = ConvTime_LSTM2(input_size = (height, width), input_dim = channels, hidden_dim = [128, 64, 64, 5], kernel_size = (5, 5), num_layers = 4, batch_first = True, bias = True, return_all_layers = False, GPU = True)


# Passing to GPU
conv_time_lstm.cuda()


# Setting optimization methods
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(conv_time_lstm.parameters())


# Defining data sets and loaders for parallelization option
training_set = efficient_Dataset(data_indices=range(y_train.shape[0]), x = x_train, t=t_train, y = y_train)
validation_set = efficient_Dataset(data_indices=range(y_valid.shape[0]), x = x_valid, t = t_valid, y = y_valid)
train_loader = torch.utils.data.DataLoader(dataset = training_set, batch_size = batch_size, shuffle = True)
validation_loader = torch.utils.data.DataLoader(dataset = validation_set, batch_size = batch_size, shuffle = True)


# Determining compute options (GPU? Parallel?)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
conv_time_lstm = torch.nn.DataParallel(conv_time_lstm)


# Training loop
print("Beginning training")
loss_list = []
#epochs = int(np.ceil((7*10**5) / x_train.shape[0]))
epochs = 100
for i in range(epochs):
	# Marking the beginning time of epoch
	begin_time = datetime.now()
	for data in train_loader:
		
		# data loader
		batch_x, batch_t, batch_y = data
		
		# move to GPU
		batch_x = batch_x.to(device)
		batch_t = batch_t.to(device)
		batch_y = batch_y.to(device)
		
		# run model and get the prediction
				# one batch_x for hidden transform, one for preserve
		batch_y_hat = conv_time_lstm(batch_x, batch_x, batch_t)
		batch_y_hat = batch_y_hat[0][0][:, -2:-1, :, :, :]
		
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


# Converting loss values into array and saving
loss_array = np.asarray(loss_list)
np.save('outputs/loss_over_iterations.npy', loss_array)


# Generate validation predictions
for i in range(25):
	rand_x, rand_t, rand_y = next(iter(validation_loader))
	rand_y = rand_y.cpu().data.numpy()
	rand_y_hat = conv_time_lstm(rand_x.to(device), rand_x.to(device), rand_t.to(device))[0][0][:, -2:-1, :, :, :]
	rand_y_hat = rand_y_hat.cpu().data.numpy()
	np.save("outputs/valid_prediction_" + str(i) + ".npy", rand_y_hat)
	np.save("outputs/valid_truth_" + str(i) + ".npy", rand_y)

