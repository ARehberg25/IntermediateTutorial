# Science Week Machine Learning hands-on session

## Introduction
This repository holds the notebooks and code for the Machine Learning hands-on session at 2024 HK Canada Meeting, WatChMaL tutorial. We will explore the application of Convolutional Neural Networks to the problem of particle identification in Water Cherenkov Detector.
Before proceeding please fork this repository by clicking on a button above in top right corner of the page.

## Acknowledgements
I borrowed code liberally from [code and tutorials](https://github.com/WatChMaL) developed by [Kazu Terao](https://github.com/drinkingkazu) and code by [Julian Ding](https://github.com/search?q=user%3Ajulianzding) and [Abhishek Kajal](https://github.com/search?q=user%3Aabhishekabhishek). Big thanks also to the [Water Cherenkov Machine Learning](https://github.com/WatChMaL) collaboration for lending their data - particularly [Nick Prouse](https://github.com/nickwp) for actually running the simulations and to Julian for 'massaging' the data.
Thanks to Wojtek Fedorko for providing this code and for assistance.

## Setting up on Cedar

This tutorial is meant to work on cedar, the compute canada cluster. If you don't have a compute canada account, I suggest pairing up with someone who does. Some instructions for a local version of this tutorial will follow this section. To login to Cedar:

```
ssh [username]@cedar.computecanada.ca
```

Navigate to the directory where you want to save the tutorial.

```
mkdir -p ~/watchmal/tutorials/
cd  ~/watchmal/tutorials/
```

Then either clone the git repo (as below), or fork your own and clone it.

```
git clone https://github.com/felix-cormier/Science_Week_ML_tutorial.git
```

To setup environments (I suggest doing this a bit before the tutorial as it may take 15-20 minutes to load and install packages).

```
cd Science_Week_ML_tutorial/
env -i HOME=$HOME bash -l
source env_setup.sh
```

If all goes well your environment should be set up.

## Running the notebook

Navigate to: https://jupyterhub.cedar.alliancecan.ca/user/fcormier/lab/workspaces/auto-b
Login via your compute canada username and password (and dual authentication!).
When choosing your notebook, opt for 2 CPUs, 6300 Mb memory, and over 1 hour of time.

It might take a few seconds to minutes for the job to launch.
Once you are inside jupyterlab, navigate to your notebook (Data\_Exploration\_And\_Streaming.ipnyb first). On the top right, click on the Kernel that should say something like 'Python 3.10', and select the "Kernel HKCA Python 3.x Kernel".

You are ready to navigate the notebook

## Notebook order in the tutorial
The sequence of the tutorial is:
  1. `Data_Exploration_And_Streaming.ipynb`
  1. `MLP_CNN.ipynb`
  1. `Training diagnostics and performance metrics.ipynb`
The notebook `Training monitor.ipynb` is meant to display some live diagnostics during network training process and can be run anytime in parallel.

