



# Assumptions:
#   1. That x_train/valid, t_train/valid, and y_train/valid
#      are defined in the environment


# Dependencies
from torch.utils import data


# Defining data set/loading classes for efficient
# PyTorch computing
# ...for the training set
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
		curr_x = x_train[IDs, :, :, :, :]
		curr_t = t_train[IDs, :, :, :, :]
		curr_y = y_train[IDs, :, :, :, :]
		#return X, y
		return(curr_x, curr_t, curr_y)


# ...for the validation set
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
		curr_x = x_valid[IDs, :, :, :, :]
		curr_t = t_valid[IDs, :, :, :, :]
		curr_y = y_valid[IDs, :, :, :, :]
		#return X, y
		return(curr_x, curr_t, curr_y)