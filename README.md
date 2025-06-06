# HK Canada Machine Learning hands-on session

## Introduction
This repository holds the notebooks and code for the Machine Learning hands-on session at 2024 HK Canada Meeting, WatChMaL tutorial. We will explore the application of Convolutional Neural Networks to the problem of particle identification in Water Cherenkov Detector.
Before proceeding please fork this repository by clicking on a button above in top right corner of the page.

## Acknowledgements
I borrowed code liberally from [code and tutorials](https://github.com/WatChMaL) developed by [Kazu Terao](https://github.com/drinkingkazu) and code by [Julian Ding](https://github.com/search?q=user%3Ajulianzding) and [Abhishek Kajal](https://github.com/search?q=user%3Aabhishekabhishek). Big thanks also to the [Water Cherenkov Machine Learning](https://github.com/WatChMaL) collaboration for lending their data - particularly [Nick Prouse](https://github.com/nickwp) for actually running the simulations and to Julian for 'massaging' the data.
Thanks to Wojtek Fedorko for providing this code and for assistance.

## Setting up on triumf-ml1


 %% [markdown]
 # Project overview and data visualization and streaming tutorial
 

 %% [markdown]
 ## Project Overview
 I will assume everybody here is roughly familiar with physics of neutrinos and Water Cherenkov detectors.
 In this project we will tackle the task of classification of neutrino type ($\nu_e$ or $\nu_\mu$) or rather the charged leptons resulting from the nuclear scatter ($e$ and  $\mu$) as well as an irreducible background from neutral current $\gamma$ production. The dataset comes from simulated Water Cherenkov detector called NuPRISM. NuPRISM is a proposed 'intermediate' detector for the Hyper-Kamiokande project. The detector has a cylindrical geometry and can be lowered and raised in a shaft to sample different energy distribution of incoming neutrinos! ![NUPRISM](../img/NUPRISM_diag.png)

 %% [markdown]
 The cylinder wall or 'barrel' and end-caps are lined with 'multi-PMT' or 'mPMT' modules arranged in a rectangular grid. Each mPMT is a dome with 19 PMTs arranged in two rings and one at the center:![mPMT](../img/mPMT.png)

 %% [markdown]
 Here is an event display where the barrel was 'unrolled':
 ![eventdisp](../img/ev_disp.png) - you can clearly see a Cherenkov ring appearing
 The 'brightness' corresponds to charge collected by each PMT. Each PMT also tells us the arrival time of the signal.

 %% [markdown]
 In this part of the we will take a look at the data and how to organize streaming it in batches so that we can feed it to our neural model

## Notebook order in the tutorial
The sequence of the tutorial is:
  1. `Data_Exploration_And_Streaming.ipynb`
  1. `MLP_CNN.ipynb`
  1. `Training diagnostics and performance metrics.ipynb`
The notebook `Training monitor.ipynb` is meant to display some live diagnostics during network training process and can be run anytime in parallel.

