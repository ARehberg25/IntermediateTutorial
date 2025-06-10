# Hyper Kamiokande TRISEP Machine Learning hands-on session

## Introduction
This repository holds the scripts and classes for the Machine Learning hands-on session at 2025 TRISEP Summer School. We will explore the application of Convolutional Neural Networks to the problem of particle identification in Water Cherenkov Detector.
It is advisable to fork this repository by clicking on a button above in top right corner of the page before proceeding.

## Acknowledgements
I borrowed code liberally from [code and tutorials](https://github.com/WatChMaL) developed by [Kazu Terao](https://github.com/drinkingkazu) and code by [Julian Ding](https://github.com/search?q=user%3Ajulianzding) and [Abhishek Kajal](https://github.com/search?q=user%3Aabhishekabhishek). Big thanks also to the [Water Cherenkov Machine Learning](https://github.com/WatChMaL) collaboration for lending their data - particularly [Nick Prouse](https://github.com/nickwp) for actually running the simulations and to Julian for 'massaging' the data.
Thanks to Wojtek Fedorko for providing this code and for assistance.

## Setting up on triumf-ml1

To access, from a terminal
```
ssh -Y username@triumf-ml1.phas.ubc.ca
```

First time setup, if you forked fill out your GitHub username as [your-username], if not you can use felix-cormier as the username instead:
```
mkdir hk_ml_trisep_tutorial
cd hk_ml_trisep_tutorial
git clone https://github.com/[your-username]/HK_ML_tutorial.git
cd HK_ML_tutorial
```

Every time, once you login, and if you've used the same directory names as above, you'll have to do
```
cd hk_ml_trisep_tutorial/HK_ML_tutorial
source setup_environment.sh
```

In general we would suggest using the VSCode IDE over ssh to go through this tutorial. It also makes it much easier to view plots that you make.


 # Project overview and data visualization and streaming tutorial
 

 ## Project Overview
 When going through a water detector, such as Super Kamiokande, neutrinos have a small chance to interact with a water molecule. This will often (but not always!) produce the neutrino's corresponding lepton.
 Due to the Cherenkov effect, the lepton will produce a ring of light, which can be used for both classification and regression.
 In this project we will tackle the task of classification of neutrino type ($\nu_e$ or $\nu_\mu$) or rather the charged leptons resulting from the nuclear scatter ($e$ and  $\mu$) as well as an irreducible background from neutral current $\gamma$ production. The dataset comes from simulated Water Cherenkov detector originally called NuPRISM, now called the Intermediate Water Cherenkov Detector (IWCD), which is part of the complex Hyper Kamiokande Detector currently under construction in Japan. The detector has a cylindrical geometry and can be lowered and raised in a shaft to sample different energy distribution of incoming neutrinos! ![NUPRISM](img/NUPRISM_diag.png)

 The cylinder wall or 'barrel' and end-caps are lined with 'multi-PMT' or 'mPMT' modules arranged in a rectangular grid. Each mPMT is a dome with 19 PMTs arranged in two rings and one at the center:![mPMT](img/mPMT.png)

 Here is an event display where the barrel was 'unrolled':
 ![eventdisp](img/ev_disp.png) 
 
 You can clearly see a Cherenkov ring appearing
 The 'brightness' corresponds to charge collected by each PMT. Each PMT also tells us the arrival time of the signal.


## Notebook order in the tutorial
The sequence of the tutorial is:
  1. Data Exploration and Streaming
  1. Training with different architectures
  1. Monitoring training and analyzing outputs
  
### Data Exploration and Streaming

The first script we will run will be in the _scripts_ directory, called _data\_exploration.py_.
From your repo directory you can run 
```
python scripts/data_exploration.py
```

This will run this script tutorial-style, going through different lines print lines. We'll also go through the lines here. If you want to skip the tutorial style, you can add the option _-s_ to skip the prompts and print out everything at once.

First, we will open the [.h5 file](https://github.com/felix-cormier/HK_ML_tutorial/blob/trisep_dev/scripts/data_exploration.py#L24)
```python
    f=h5py.File(filepath,"r")
```
.h5 files are very performant when reading from disk, and so are widely used in ML.
The code will open the file and print out the different keys, which are variables labelled with a name.

Next, we'll look at the number of events. We [print out the shape of the 'labels' variable](https://github.com/felix-cormier/HK_ML_tutorial/blob/trisep_dev/scripts/data_exploration.py#L43)

```python
print(f"Number of events: {f['labels'].shape}")
```

Next, investigate the shape of the PMT variables, which is the detector variables we will eventually use to train our networks. We [print out the shape of the 'event_data' variable](https://github.com/felix-cormier/HK_ML_tutorial/blob/trisep_dev/scripts/data_exploration.py#L49)


```python
print(f"Shape of the barrel-only data: {f['event_data'].shape}")
```

We have 900k simulated scattering events here! labels are 0, 1, 2 for $\gamma$,electrons and muons respectively. 
The 'event_data' contains only the barrel portion of the tank which has been 'unrolled'. 
The first dimension (900k) enumerates over the events, the second two dimensions (16,40) enumerate over the row and column in the module 'grid'. 
Finally last two dimensions enumerate over the PMT within the module (again there are 19 in each mPMT module) 
Note: the first 19 entries correspond to charge collected on a given
 mPMT and last 19 correspond to the time.


We can now take a look at some details about our data.
Note that the object returned by the subscript looks like an array - we can subscript it and even do fancy indexing:

#f['event_data'][[42,1984],:,:,:]

 In fact the object is not a numpy array -it is a hdf5 `Dataset` object - the data itself lives on disk until we request it

We [print out the type of the 'event_data' variable](https://github.com/felix-cormier/HK_ML_tutorial/blob/trisep_dev/scripts/data_exploration.py#L69)

```python
print(f"The data type is: {type(f['event_data'])}")
```

Next, we see the size of the dataset will make it difficult to load all at once into memory on many systems by [checkign the size of the PMT data](https://github.com/felix-cormier/HK_ML_tutorial/blob/trisep_dev/scripts/data_exploration.py#L77)

```python
print("Size of the bulk of the data is {:.1f} GB".format( (f['event_data'].size * 4 / (1024**3)) ))
```

One important feature of the dataset it is uncompressed and contiguous or 'unchunked', as we can see [when we look at the chunks and compression](https://github.com/felix-cormier/HK_ML_tutorial/blob/trisep_dev/scripts/data_exploration.py#L85)
```python
print("dataset chunks: {} compression: {}".format(f['event_data'].chunks,f['event_data'].compression))
```

The dataset has been prepared as contiguous and uncompressed so that we are not obliged to load it all in memory but we can access it very fast. 
BUT it will take more spave on disk. In the next section we will see an example of how to deal with datasets with these sizes.

#### Using Pytorch Dataset object

Let's import and create a Dataset object - you are welcome to look at the [source](utils/data_handling.py) code in _utils/data\_handling.py_.
We've added a more detailed walk-through of data handling in the [comments](https://github.com/felix-cormier/HK_ML_tutorial/blob/trisep_dev/scripts/data_exploration.py#L102-164) of [data_exploration](scripts/data_exploration.py) that you can read at your leisure. Since this tutorial is more about networks than data handling, we leave it as optional.

The [data handling](utils/data_handling.py) utility has a class named _WCH5Dataset_ that we now [instantiate](https://github.com/felix-cormier/HK_ML_tutorial/blob/trisep_dev/scripts/data_exploration.py#L170-171), it should have the same length as the file we were looking at earlier
```python
dset=WCH5Dataset("/fast_scratch/TRISEP_data/NUPRISM.h5",val_split=0.1,test_split=0.1)
print(f"Length of dataset object: {len(dset)}")
```

Next, let's [get some random event](https://github.com/felix-cormier/HK_ML_tutorial/blob/trisep_dev/scripts/data_exploration.py#L179-182) and label from the training dataset, and plots an event display of this single event using [a plotting script](utils/plot_utils.py)
```python
event, label, energy=dset[dset.train_indices[1984]]
print("Label {} and energy: {} (MeV) ".format(label,energy))
#Make some event displays, save them to disk
event_displays(event, label, plot_path='plots/data_exploration/')
```

The default place to store the plots you made are in _plots/data\_exploration_, so you can check them there. There should be 3 different event display plots, do you understand what they all show?

#### Making plots of your data

Always try to learn as much as possible about the dataset before throwing ML at it. Let's quickly histogram the charges. 
We won't load the full dataset but taking few thousand should be fine (since we have 12k PMTs).
We'll [prepare the dataset](https://github.com/felix-cormier/HK_ML_tutorial/blob/trisep_dev/scripts/data_exploration.py#L193-211), then finally use a plotting function (plot_pmt_var)[https://github.com/felix-cormier/HK_ML_tutorial/blob/trisep_dev/scripts/data_exploration.py#L214]

```python
#Using plotting functions
plot_pmt_var(charge_data_to_plot, charge_labels_to_plot, charge_colors_to_plot, bins = charge_bins, xlabel = 'PMT Energy (photo electrons)', plot_path='plots/data_exploration/all_mpmt_charge.png', do_log=True)
```

**Can you do the same for time variables, rather than charge?** You can work directly in the script (and use option _-s_ to skip tutorial mode). Hint: charge is indices 0-19 (one for each mPMT), then time is the indices after that...

Let's also plot the total energy in the event and also the true particle energy. Again we'll [prepare the dataset](https://github.com/felix-cormier/HK_ML_tutorial/blob/trisep_dev/scripts/data_exploration.py#L228-237), then finally use a plotting function (plot_pmt_var)[https://github.com/felix-cormier/HK_ML_tutorial/blob/trisep_dev/scripts/data_exploration.py#L239]

```python
plot_pmt_var(sum_charge_data_to_plot, sum_charge_labels_to_plot, sum_charge_colors_to_plot, bins = sum_charge_bins,xlabel = 'PMT Energy Sum (photo electrons)', plot_path='plots/data_exploration/sum_mpmt_charge.png', do_log=False)
```
Muons seem to look very different - but is that expected?

This is the end of the data exploration script. **Are there other plots you could make? Either event-level or PMT-level?** You can work by running the script with the _-s_ option to skip the tutorial step-by-step instructions.

### Data Iteration

This is a short script to walk you through how to use PyTorch DataLoader objects to iterate through the data. It is not necessary for the rest of the tutorial, but might be good practice if you wanted to try your hand at any pre-processing while running training scripts.

To the run the script do
```
python scripts/data_iteration.py
```

Here we load DataLoader from PyTorch

```python
from torch.utils.data import DataLoader
```

then [define train, validation and testing DataLoaders from our dataset](https://github.com/felix-cormier/HK_ML_tutorial/blob/trisep_dev/scripts/data_iteration.py#L73-75)

```python
train_iter=DataLoader(dset,batch_size=64,shuffle=False,sampler=SubsetRandomSampler(dset.train_indices),num_workers=2)
val_iter=DataLoader(dset,batch_size=64,shuffle=False,sampler=SubsetRandomSampler(dset.val_indices),num_workers=2)
test_iter=DataLoader(dset,batch_size=64,shuffle=False,sampler=SubsetRandomSampler(dset.test_indices),num_workers=2)
```

You see the parameters - like batch_size and sampler - the sampler uses the indices we computed for the training, validation and testing set - if you use a sampler shuffle has to be False. On each iteration the DataLoader object will ask the dataset for a bunch of indices (calling the __getitem__ function we coded earlier) and then collate the data into a batch tensor. The collating can also be customized by providing collate_fn - but for now we will leave it with a default behavior. Did you notice the `num_workers` argument? if >0 this will enable multiprocessing - several processes will be reading examples (if supplied applying the augmentation transformation) and putting the data on queue that would be than 'consumed' by your training/evaluation process.Your 'instance' has 2 CPUs for the job so we will use that. We are beating on the same storage with all threads - so if we aren't doing much preprocessing it doesn't make sense to make this too high.

Now convince yourself that the `data` and `labels` are already tensors - which we could plug into our future model - [let's iterate over first 40 batches:](https://github.com/felix-cormier/HK_ML_tutorial/blob/trisep_dev/scripts/data_iteration.py#L95-96)

```python
num_iterations=40
trecord = loop_over_set(train_iter, num_iterations)
```

This calls the _loop\_over\_set.py_ function defined at the top of _data\_iteration.py_. This loop is technically all we would need to train, as we load a batch of training data from memory, we could train over it.

By the way, do you notice that roughly every 2nd iteration the time it takes to give a batch is huge? Why?


What if we want to pre-process the data? We can do that directly in the loop over the train iterator
**Can you modify loop_over_set function so that time is centered around 0 with a standard deviation of 1?**
Input data being arounds this range can help converge faster.
**Can you make plots using some of the plotting utitilies in data_exploration showing the difference in time?**
**Are there any other pre-processing steps you could think of to do with our data?**


### Training

First we'll look at a Multi-Layer Perceptron (MLP). An MLP is a very basic neutral network, with a set of fully-connected layers connecting input (PMT information) to output (class prediction).