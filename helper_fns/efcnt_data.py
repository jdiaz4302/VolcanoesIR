



# Dependencies
from torch.utils import data


# Defining data set/loading classes for efficient
# PyTorch computing
class efficient_Dataset(data.Dataset):
	'Characterizes a dataset for PyTorch'
	def __init__(self, data_indices, x, t, y):
		'Initialization'
		self.data_indices = data_indices
                self.x = x
                self.t = t
                self.y = y
	def __len__(self):
		'Denotes the total number of samples'
		return len(self.data_indices)
	def __getitem__(self, index):
		'Generates one sample of data'
		# Select sample
		IDs = self.data_indices[index]
		# Load data and get label
		curr_x = x[IDs, :, :, :, :]
		curr_t = t[IDs, :, :, :, :]
		curr_y = y[IDs, :, :, :, :]
		#return X, y
        	return(curr_x, curr_t, curr_y)
