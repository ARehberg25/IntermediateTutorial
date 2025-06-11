import sys
import os, time
import argparse

import h5py
import numpy as np

#add to path so that we can grab functions from other directories
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


from utils.data_handling import WCH5Dataset
from utils.plot_utils import get_plot_array, event_displays, plot_pmt_var

currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

# Here is how you would iterate over several batches:

def loop_over_set(loader,loop_limit=3):

    # Let's measure time that takes in each loop
    trecord = np.zeros([loop_limit],dtype=np.float32)
    t = time.time()
    for iteration, batch in enumerate(loader):

        data,labels,energies = batch

        # Print out some content info
        print('Iteration',iteration,'... time:',time.time()-t,'[s]')
        print('    Labels:',labels)

        trecord[iteration] = time.time() - t
        t = time.time()

        # break when reaching the loop limit
        if (iteration+1) == loop_limit:
            break
    return trecord

# ## DataLoader objects and streaming the data

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--skip_tutorial", help="Run with this option first to get a tutorial style script",
                    action="store_true")
args = parser.parse_args()
if args.skip_tutorial:
    print("Skipping tutorial style")

filepath = "/fast_scratch/TRISEP_data/NUPRISM.h5"

#Load up the file and convert it to WCH5Dataset like in the data_exploration script 
try:
    f=h5py.File(filepath,"r")
except FileNotFoundError:
    print("File not Found!")
    quit()

dset=WCH5Dataset(filepath,val_split=0.1,test_split=0.1)

# Now lets create DataLoader objects - one for each training, validation and testing set - each DataLoader uses the same dataset -that way we keep only one open file (this is the 'standard' pytorch way)
# 
# These objects will give us the capability to loop over the data repeatedly in batches. 
# The basic model for training the model is to process a batch of events and update model parameters based on results. 
# We then keep processing batches until some stopping condition is satisfied.

train_iter=DataLoader(dset,batch_size=64,shuffle=False,sampler=SubsetRandomSampler(dset.train_indices),num_workers=2)
val_iter=DataLoader(dset,batch_size=64,shuffle=False,sampler=SubsetRandomSampler(dset.val_indices),num_workers=2)
test_iter=DataLoader(dset,batch_size=64,shuffle=False,sampler=SubsetRandomSampler(dset.test_indices),num_workers=2)

if not args.skip_tutorial:

    print(f"Opened {filepath}")
    print(f"Made a training iterator of type: {type(train_iter)}")
    input("Press Enter to continue...")  # Waits for user input

# You see the parameters - like batch_size and sampler - the sampler uses the indices we computed for the training, validation and testing set 
# - if you use a sampler shuffle has to be False. On each iteration the DataLoader object will ask the dataset for a bunch of indices 
# (calling the __getitem__ function we coded earlier) and then collate the data into a batch tensor. 
# The collating can also be customized by providing collate_fn - but for now we will leave it with a default behavior. Did you notice the `num_workers` argument? 
# if >0 this will enable multiprocessing - several processes will be reading examples 
# (if supplied applying the augmentation transformation) and putting the data on queue that would be than 'consumed' by your training/evaluation process. 
# Your 'instance' has 4 CPUs for the job so we will use that. We are beating on the same storage with all threads 
# - so if we aren't doing much preprocessing it doesn't make sense to make this too high.


# Convince yourself that the `data` and `labels` are already tensors - which we could plug into our future model - Let's iterate over first 40 batches:

num_iterations=40
trecord = loop_over_set(train_iter, num_iterations)


if not args.skip_tutorial:

    print(f"Time record of iterations: {trecord}")
    print(f"We just looped through {num_iterations} iterations of the training iterator")
    print("Did you notice that roughly every 2nd iteration the time it takes to give a batch is huge? Why? Hint: look at how we defined train_iter")
    input("Press Enter to continue...")  # Waits for user input

# - do you notice that roughly every 2nd iteration the time it takes to give a batch is huge? Why?
# Hint: look at how we defined train_iter

# To remove the bottleneck due to data access. In general we want to avoid:
#    * Network transfers during training
#    * Spinning media if possible
# If data is small enough - load into memory


next_batch=next(iter(train_iter))

if not args.skip_tutorial:

    print(f"Next batch labels: {next_batch[1]}")
    print(f"Does next batch required gradient computation: {next_batch[0].requires_grad}. If we want to train, this would have to be set to true in order to do a backwards pass.")
    input("Press Enter to continue...")  # Waits for user input



# What if we want to pre-process the data? We can do that directly in the loop over the train iterator
# Can you modify loop_over_set function so that time is centered around 0 with a standard deviation of 1?
# Input data being arounds this range can help converge faster
# Can you make plots using some of the plotting utitilies in data_exploration showing the difference in time?
# Are there any other pre-processing steps you could think of to do with our data?