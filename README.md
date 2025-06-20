# Hyper Kamiokande TRISEP Machine Learning hands-on session

## Introduction
This repository holds the scripts and classes for the Hyper Kamiokande Machine Learning hands-on session at 2025 TRISEP Summer School. We will explore the application of different Neural Networks to the problem of particle identification in Water Cherenkov Detector.
It is advisable to fork this repository by clicking on a button above in top right corner of the page before proceeding.

## Acknowledgements
I borrowed code liberally from [code and tutorials](https://github.com/WatChMaL) developed by [Kazu Terao](https://github.com/drinkingkazu) and code by [Julian Ding](https://github.com/search?q=user%3Ajulianzding) and [Abhishek Kajal](https://github.com/search?q=user%3Aabhishekabhishek). Big thanks also to the [Water Cherenkov Machine Learning](https://github.com/WatChMaL) collaboration for lending their data - particularly [Nick Prouse](https://github.com/nickwp) for actually running the simulations and to Julian for 'massaging' the data.
Thanks to Wojtek Fedorko for providing this code and for assistance.

## Setting up on triumf-ml1 or triumf-ml2

To access triumf-ml1 or triumf-ml2, follow the instructions to [log in to the TRIUMF ML server and launch the container as described here](https://github.com/TRISEP-2025-ML-tutorials/Intro-notebooks/blob/main/SETTING_UP.md)

You should fork the HyperK ML repo in [the TRISEP 2025 ML Repo](https://github.com/TRISEP-2025-ML-tutorials/IntermediateTutorial)

First time setup, if you forked fill out your GitHub username as [your-username]
```
mkdir hk_ml_trisep_tutorial
cd hk_ml_trisep_tutorial
git clone https://github.com/[your-username]/IntermediateTutorial.git
cd IntermediateTutorial
mkdir -p plots/data_exploration
mkdir -p plots/training
mkdir -p plots/analysis
mkdir diagrams
mkdir model_state_dumps
```

Every time, once you login, and if you've used the same directory names as above, you'll have to do
```
cd hk_ml_trisep_tutorial/HK_ML_tutorial
source source.me
```

In general we would suggest using e.g. [VSCode IDE](https://code.visualstudio.com) over ssh to go through this tutorial. It also makes it much easier to view plots that you make.


 # Overview and tutorial order
 

 ## Project Overview
 When going through a water detector, such as Super Kamiokande, neutrinos have a small chance to interact with a water molecule. This will often (but not always!) produce the neutrino's corresponding lepton.
 Due to the Cherenkov effect, the lepton will produce a ring of light, which can be used for both classification and regression.
 In this project we will tackle the task of classification of neutrino type ($\nu_e$ or $\nu_\mu$) or rather the charged leptons resulting from the nuclear scatter ($e$ and  $\mu$) as well as an irreducible background from neutral current $\gamma$ production. The dataset comes from simulated Water Cherenkov detector originally called NuPRISM, now called the Intermediate Water Cherenkov Detector (IWCD), which is part of the complex Hyper Kamiokande Detector currently under construction in Japan. The detector has a cylindrical geometry and can be lowered and raised in a shaft to sample different energy distribution of incoming neutrinos! ![NUPRISM](img/NUPRISM_diag.png)

 The cylinder wall or 'barrel' and end-caps are lined with 'multi-PMT' or 'mPMT' modules arranged in a rectangular grid. Each mPMT is a dome with 19 PMTs arranged in two rings and one at the center:![mPMT](img/mPMT.png)

 Here is an event display where the barrel was 'unrolled':
 ![eventdisp](img/ev_disp.png) 
 
 You can clearly see a Cherenkov ring appearing
 The 'brightness' corresponds to charge collected by each PMT. Each PMT also tells us the arrival time of the signal.


## Order of the tutorial
The sequence of the tutorial is:
  1. [Data Exploration and Iteration](tutorial/exploration_iteration.md)
  2. [Training Multi-Layer Perceptrons](tutorial/training.md)
  3. [Monitoring training and analyzing outputs](tutorial/evaluation.md)
  4. [Training with different architectures](tutorial/training_cnn_resnet.md)
  