
import numpy as np
import os, time
import h5py
import sys
import argparse

from utils.data_handling import WCH5Dataset
from utils.plot_utils import get_plot_array, event_displays, plot_pmt_var

import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--skip_tutorial", help="Run with this option first to get a tutorial style script",
                    action="store_true")
args = parser.parse_args()
if args.skip_tutorial:
    print("Skipping tutorial style")

filepath = "/fast_scratch/TRISEP_data/NUPRISM.h5"

try:
    f=h5py.File(filepath,"r")
except FileNotFoundError:
    print("File not Found!")
    quit()

# `keys()` will give us all the hdf5 datasets stored in the file


print(f"Opened {filepath}")
print(f"Keys in the data: {f.keys()}")

if not args.skip_tutorial:

    input("Press Enter to continue...")  # Waits for user input

# Let's look at the shapes of the data:

print(f"Number of events: {f['labels'].shape}")

print(f"Shape of the barrel-only data: {f['event_data'].shape}")

# We have 900k simulated scattering events here! labels are 0, 1, 2 for $\gamma$,$e$ and $\mu$ respectively. 
# The 'event_data' contains only the barrel portion of the tank which has been 'unrolled'. 
# The first dimension (900k) enumerates over the events, the second two dimensions (16,40) enumerate over the row and column in the module 'grid'. 
# Finally last two dimensions enumerate over the PMT within the module (again there are 19 in each mPMT module) 
# first 19 entries correspond to charge collected on a given PMT and last 19 correspond to the time.
# 
# 

# Note that the object returned by the subscript looks like an array - we can subscript it and even do fancy indexing:

#f['event_data'][[42,1984],:,:,:]

#  In fact the object is not a numpy array -it is a hdf5 `Dataset` object - the data itself lives on disk until we request it

print(f"The data type is: {type(f['event_data'])}")

# The size of the dataset will make it difficult to load all at once into memory on many systems

print("Size of the bulk of the data is {:.1f} GB".format( (f['event_data'].size * 4 / (1024**3)) ))

# One important feature of the dataset it is uncompressed and contiguous or 'unchunked':

print("dataset chunks: {} compression: {}".format(f['event_data'].chunks,f['event_data'].compression))

# The dataset has been prepared as contiguous and uncompressed so that we are not obliged to load it all in memory but we can access it very fast. 
# BUT it will take more spave on disk. In the next section we will see an example of how to deal with datasets with these sizes.

# ## Pytorch Dataset object

# Let's import and create a Dataset object - you are welcome to look at the [source](utils/data_handling.py)

# First we need to include the sources in the python search path

currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 


# The class derives from the torch Dataset object. The two heavy lifters are the __init__ function and __getitem__ function.
# Let's look at what init does:  
#        
# ```python
#         self.f=h5py.File(path,'r')
#         hdf5_event_data = self.f["event_data"]
#         hdf5_labels=self.f["labels"]
#         hdf5_energies=self.f["energies"]
# 
#         event_data_shape = hdf5_event_data.shape
#         event_data_offset = hdf5_event_data.id.get_offset()
#         event_data_dtype = hdf5_event_data.dtype         
# ```
#       
# -here we opened the file and got the offsets, shapes, and data types of the datasets. 
# Why are we doing this? This is because the hdf5 file is uncompressed and the datasets within are contiguous - this allows us to do memory mapping of the file. 
# This is important with large datasets like this - where we are most like not going to be able to load everything into memory. 
# With memory map we can only load what we need, when we need it
# 

# The memory map itself happens here:
# 
# ```python
#         self.event_data = np.memmap(path, mode='r', 
#                                     shape=event_data_shape, 
#                                     offset=event_data_offset, 
#                                     dtype=event_data_dtype) 
# ```
# We will just load the labels and energies into memory - this is only several MB
# ```python
#         self.labels = np.array(hdf5_labels)
#         self.energies = np.array(hdf5_energies)
# ```

# The rest of __init__ function computes indices for training, validation and testing subsets.
# 
# For Machine Learning applications splitting the dataset into three subsets: training, validation and testing is a common practice. 
# We will be using the training dataset to 'learn' the parameters of the model. 
# Validation set is used to determine if the model is 'generalizing' well (i.e. learing underlying patterns in the data) or 'overfitting' (basically memorizing our training dataset). 
# Validation set is normally also use to perform multiple trials with different models or model and training parameters (this is called hyperparameter tuning). 
# Because hyperparameter tuning may also induce bias we quote model performance on the testing dataset.
# 
# The split into the three subsets is based on a random (but consistent) shuffle of events. Why may we want to access the records in the file in randomized order? 
# This really depends on how the dataset was created - for instance here we just concatenated a bunch of simulation files in order - so the examples are in blocks:

#f['labels'][0::1000]

# We will definitelly want to avoid showing the model only photons then only electrons and then only muons - hence the index 
# shuffling:
# ```python
#     np.random.shuffle(indices)
# 
#     self.train_indices = indices[:-n_val-n_test]
#     self.val_indices = indices[-n_test-n_val:-n_test]
#     self.test_indices = indices[-n_test:]
# ````

# Finally we have the __getitem__ method - this provides functionality for the subscript [] operator. Only here we actually load the event_data that was requested:  
# ```python
#         return np.array(self.event_data[index,:]),  self.labels[index], self.energies[index][0]       
# ```
# -we return a tuple with three elements - first is the event 'image', second is the label, and third is the 'true' energy of the generated particle. If you look at the code you will notice that there is also a provision for providing a transform. This is very useful if you want to do data augmentation on-the-fly. E.g. we could flip the images to 'populate' the dataset to reflect the variability we expect in the dataset we will want to apply the model. We could also use it to pre-process the data on-the fly as well.
# 

# finally we need the len method - this just needs to return how many exmples we have in the dataset

# Ok let's instantiate the dataset and ask it for a few examples:

dset=WCH5Dataset("/fast_scratch/TRISEP_data/NUPRISM.h5",val_split=0.1,test_split=0.1)


print(f"Length of dataset object: {len(dset)}")


# Let's get some random event and label from the training dataset:

event, label, energy=dset[dset.train_indices[1984]]

print("Label {} and energy: {} (MeV) ".format(label,energy))

#Make some event displays, save them to disk
event_displays(event, label, plot_path='plots/data_exploration/')

# Always try to learn as much as possible about the dataset before throwing ML at it. Let's quickly histogram the charges. 
# We won't load the full dataset but taking few thousand should be fine (since we have 12k PMTs)

data_to_plot=dset[dset.train_indices[0:2000]]

data_to_plot_events=data_to_plot[0]
data_to_plot_labels=data_to_plot[1]
data_to_plot_energies=data_to_plot[2]

# Let's plot the charge distributions
charge_pmt_idx_min = 0
charge_pmt_idx_max = 10

charge_photon = data_to_plot_events[np.where(data_to_plot_labels==0)][:,:,:,charge_pmt_idx_min:charge_pmt_idx_max].flatten()
charge_electron = data_to_plot_events[np.where(data_to_plot_labels==1)][:,:,:,charge_pmt_idx_min:charge_pmt_idx_max].flatten()
charge_muon = data_to_plot_events[np.where(data_to_plot_labels==2)][:,:,:,charge_pmt_idx_min:charge_pmt_idx_max].flatten()

charge_data_to_plot = [charge_photon, charge_electron, charge_muon]
charge_labels_to_plot = ['Photon', 'Electron', 'Muon']
charge_colors_to_plot = ['blue', 'red', 'green']

charge_bins=np.linspace(0.0,20.0,51)



plot_pmt_var(charge_data_to_plot, charge_labels_to_plot, charge_colors_to_plot, bins = charge_bins, xlabel = 'PMT Energy (photo electrons)', plot_path='plots/data_exploration/all_mpmt_charge.png', do_log=True)

# Now plot the time histogram -notice anything strange?
# Hint charge is indices 0-19 (one for each mPMT), then time is the indices after that...



# Let's also plot the total energy in the event and also the true particle energy

e_sum_gamma=np.sum(data_to_plot_events[np.where(data_to_plot_labels==0)][:,:,:,0:19],axis=(1,2,3))/1000.0
e_sum_e=np.sum(data_to_plot_events[np.where(data_to_plot_labels==1)][:,:,:,0:19],axis=(1,2,3))/1000.0
e_sum_mu=np.sum(data_to_plot_events[np.where(data_to_plot_labels==2)][:,:,:,0:19],axis=(1,2,3))/1000.0

sum_charge_data_to_plot = [e_sum_gamma, e_sum_e, e_sum_mu]
sum_charge_labels_to_plot = ['Photon', 'Electron', 'Muon']
sum_charge_colors_to_plot = ['blue', 'red', 'green']

max_bin=max(np.amax(e_sum_gamma),np.amax(e_sum_e),np.amax(e_sum_mu))
sum_charge_bins=np.linspace(0.0,max_bin,20)

plot_pmt_var(sum_charge_data_to_plot, sum_charge_labels_to_plot, sum_charge_colors_to_plot, bins = sum_charge_bins,xlabel = 'PMT Energy Sum (photo electrons)', plot_path='plots/data_exploration/sum_mpmt_charge.png', do_log=False)
# Muons seem to look very different - but is that expected?

#Are there other plots you could make? Either event-level or PMT-level?







