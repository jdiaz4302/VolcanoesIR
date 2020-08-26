



# Training parameter
batch_size = 16


# Libraries and imports
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils import data
from models.ConvTimeLSTM1 import ConvTime_LSTM1
from helper_fns.processing import scale_and_remove_na


# Required input
print("Importing and formatting data")
vol = "ErtaAle"
num_input_scenes = 10
train_percent = 0.70
out_samp_perc = 0.15 # validation and testing


# Basic data import
numpy_data_location = "data/" + vol + "/numpy_data_cube.npy"
table_data_location = "data/" + vol + "/good_df.csv"
volcano_scenes = np.load(numpy_data_location)
tabular_metadata = pd.read_csv(table_data_location)


# Separate model inputs (previous $n$ scenes, time differences) and outputs (subsequent scene)
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


# Scale temperatures between 0 and 1. If temperature is missing, assigned a scaled value of 0
print("Processing data")
x_scenes_train = scale_and_remove_na(x_scenes_train)
x_scenes_train = scale_and_remove_na(x_scenes_train)
x_scenes_train = scale_and_remove_na(x_scenes_train)
time_differences_train = scale_and_remove_na(time_differences_train)
time_differences_train = scale_and_remove_na(time_differences_train)
time_differences_train = scale_and_remove_na(time_differences_train)
y_scenes_train = scale_and_remove_na(y_scenes_train)
y_scenes_train = scale_and_remove_na(y_scenes_train)
y_scenes_train = scale_and_remove_na(y_scenes_train)


# Passing to pytorch and formatting
x_scenes_train = torch.from_numpy(x_scenes_train).type(torch.FloatTensor)
x_scenes_test = torch.from_numpy(x_scenes_test).type(torch.FloatTensor)
x_scenes_valid = torch.from_numpy(x_scenes_valid).type(torch.FloatTensor)
time_differences_train = torch.from_numpy(time_differences_train).type(torch.FloatTensor)
time_differences_test = torch.from_numpy(time_differences_test).type(torch.FloatTensor)
time_differences_valid = torch.from_numpy(time_differences_valid).type(torch.FloatTensor)
y_scenes_train = torch.from_numpy(y_scenes_train).type(torch.FloatTensor)
y_scenes_test = torch.from_numpy(y_scenes_test).type(torch.FloatTensor)
y_scenes_valid = torch.from_numpy(y_scenes_valid).type(torch.FloatTensor)


# Defining model parameters
# Picking one of the like-sequence tensors within the list to set parameters
print("Beginning training")
channels = x_scenes_train.shape[2]
height = x_scenes_train.shape[3]
width = x_scenes_train.shape[4]
conv_time_lstm = ConvTime_LSTM1(input_size = (height, width), input_dim = channels, hidden_dim = [128, 64, 64, 1], kernel_size = (5, 5), num_layers = 4, batch_first = True, bias = True, return_all_layers = False, GPU = False)


# Passing to GPU
conv_time_lstm.cuda()


# Setting optimization methods
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(conv_time_lstm.parameters())


# Defining data set and data loaders for parallelization
class train_Dataset(data.Dataset):
	'Characterizes a dataset for PyTorch'
	def __init__(self, data_indices):
		'Initialization'
		self.data_indices = data_indices
	
	def __len__(self):
		'Denotes the total number of samples'
		return len(self.data_indices)
	
	def __getitem__(self, index):
		'Generates one sample of data'
		# Select sample
		IDs = self.data_indices[index]

		# Load data and get label
		curr_x = x_scenes_train[IDs, :, :, :, :]
		curr_t = time_differences_train[IDs, :, :, :, :]
		curr_y = y_scenes_train[IDs, :, :, :, :]

		#return X, y
		return(curr_x, curr_t, curr_y)
	
class validation_Dataset(data.Dataset):
	'Characterizes a dataset for PyTorch'
	def __init__(self, data_indices):
		'Initialization'
		self.data_indices = data_indices
	
	def __len__(self):
		'Denotes the total number of samples'
		return len(self.data_indices)
	
	def __getitem__(self, index):
		'Generates one sample of data'
		# Select sample
		IDs = self.data_indices[index]

		# Load data and get label
		curr_x = x_scenes_valid[IDs, :, :, :, :]
		curr_t = time_differences_valid[IDs, :, :, :, :]
		curr_y = y_scenes_valid[IDs, :, :, :, :]

		#return X, y
		return(curr_x, curr_t, curr_y)
training_set = train_Dataset(data_indices=range(y_scenes_train.shape[0]))
validation_set = validation_Dataset(data_indices=range(y_scenes_valid.shape[0]))
train_loader = torch.utils.data.DataLoader(dataset = training_set, batch_size = batch_size, shuffle = True)
validation_loader = torch.utils.data.DataLoader(dataset = validation_set, batch_size = batch_size, shuffle = True)


# ## Retrieving available computing devices and using parallel GPUs if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
conv_time_lstm = torch.nn.DataParallel(conv_time_lstm)


# Training loop
print("Beginning training")
loss_list = []
epochs = int(np.ceil((7*10**5) / x_scenes_train.shape[0]))
for i in range(epochs):
	for data in train_loader:
		
		# data loader
		batch_x, batch_t, batch_y = data
		
		# move to GPU
		batch_x = batch_x.to(device)
		batch_t = batch_t.to(device)
		batch_y = batch_y.to(device)
		
		# run model and get the prediction
		batch_y_hat = conv_time_lstm(batch_x,
									 batch_t)
		batch_y_hat = batch_y_hat[0][0][:, -2:-1, :, :, :]
		
		# calculate and store the loss
		batch_loss = loss(batch_y, batch_y_hat)
		loss_list.append(batch_loss.item())
		
		# update parameters
		optimizer.zero_grad()
		batch_loss.backward()
		optimizer.step()
		
		print('Epoch: ', i, '\n\tBatch loss: ', batch_loss.item(), '\n')
		
	print('Epoch: ', i, '\n\tBatch loss: ', batch_loss.item(), '\n')


# In[ ]:




