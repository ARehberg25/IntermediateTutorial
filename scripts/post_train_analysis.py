# # In this tutorial we'll see what to watch for during and after the network training

# ## Monitoring training health

# First let's look at at the training history - we want to display the training set loss and vlidation set loss as a function of 'iteration' or the batch number seen by our model

import os,sys
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import numpy as np

from utils import plot_utils

from utils.data_handling import WCH5Dataset
from utils.data_utils import rotate_chan
from utils.engine import Engine

from models.simpleCNN import SimpleCNN
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

# You will need to change the location of where these files are

# %%
loc="model_state_dumps/20250606_141919/"

# %%
plot_utils.disp_learn_hist(loc, show=False, losslim=1)


# Why so bumpy?
#    - there is 'noise' associated with batch-to-batch variation BUT
#    - The main effect is actually the real bumpiness in the loss landscape being traversed during learning
#    
# To learn anything we need to smooth out the plot - for instance by using moving average

plot_utils.disp_learn_hist_smoothed(loc,window_train=200,window_val=1)

# This actually looks pretty good - we get initially a very quick learning and then a plateau. 
# Both training and validation loss is still decreasing slightly and tracking - which means we could probably kept on training


# ## Evaluating model performance in classification task

# Now let's go back to the full dataset and load the model trained last time on the full dataset

class CONFIG:
    pass
config=CONFIG()
config.batch_size_test =512
config.batch_size_train = 256
config.batch_size_val = 512
config.lr=0.01
config.device = 'gpu'
config.gpu_number=5
config.num_workers_train=6
config.num_workers_val=1
config.num_workers_test=1
config.checkpoint=False
config.dump_path = 'model_state_dumps'

dset=WCH5Dataset("/fast_scratch/TRISEP_data/NUPRISM.h5",val_split=0.1,test_split=0.1,transform=rotate_chan)

model_resnet=resnet152(num_input_channels=38,num_classes=3)
engine=Engine(model_resnet,dset,config)


# %%
engine.restore_state("model_state_dumps/20250606_141919/ResNetBEST.pth")

engine.dirpath="model_state_dumps/20250606_141919/"

#engine.validate()

# %% [markdown]# ### Examination of classifier output

# %% [markdown]
# Plot the classifier softmax output for various classes and outputs

# %%
labels_val=np.load(engine.dirpath + "labels.npy")
predictions_val=np.load(engine.dirpath + "predictions.npy")
softmax_out_val=np.load(engine.dirpath + "softmax.npy")

print(softmax_out_val)

# %%
from matplotlib import pyplot as plt
def plot_resp(labels,softmax_out):
    fig1, ax1 = plt.subplots(figsize=(12,8),facecolor="w")
    ax1.tick_params(axis="both", labelsize=20)
    softmax_out_val_gamma_Pe=softmax_out_val[labels_val==0][:,1]
    softmax_out_val_e_Pe=softmax_out_val[labels_val==1][:,1]
    
    bins=np.linspace(0.0,1.0,51)
    values, bins, patches = ax1.hist(softmax_out_val_gamma_Pe, bins=bins, 
                                    label= 'gamma', color='blue', density=True,
                                    alpha=0.3)
    
    values, bins, patches = ax1.hist(softmax_out_val_e_Pe, bins=bins, 
                                    label= 'electron', color='red', density=True,
                                    alpha=0.3)
    ax1.legend(prop={'size': 16})
    ax1.set_xlabel('$P(e)$',fontweight='bold',fontsize=24,color='black')
    
    fig2, ax2 = plt.subplots(figsize=(12,8),facecolor="w")
    ax2.tick_params(axis="both", labelsize=20)
    softmax_out_val_e_Pmu=softmax_out_val[labels_val==1][:,2]
    softmax_out_val_mu_Pmu=softmax_out_val[labels_val==2][:,2]
    
    values, bins, patches = ax2.hist(softmax_out_val_mu_Pmu, bins=bins, 
                                    label= 'muon', color='green', density=True,
                                    alpha=0.3)
    
    values, bins, patches = ax2.hist(softmax_out_val_e_Pmu, bins=bins, 
                                    label= 'electron', color='red', density=True,
                                    alpha=0.3, log=True)
    ax2.legend(prop={'size': 16})
    ax2.set_xlabel('$P(\mu)$',fontweight='bold',fontsize=24,color='black')
    
    
    
    plt.savefig("plots/analysis/test.png")
    plt.clf()

# %%
plot_resp(labels_val,softmax_out_val)

# %% [markdown]
# ### The confusion matrix

# %%
plot_utils.plot_confusion_matrix(labels_val, predictions_val, ['$\gamma$','$e$','$\mu$'],output_path = 'plots/analysis/confusion_matrix.png')

# %% [markdown]
# ### Receiver Operating Characteristic
#    - We will plot ROC treating electrons as 'signal' and photons as 'background', but we have also muons - which is also a signal, so we have more reasonable possibilities of plotting ROC curves - can you name advantages and disadvantages?

# %%
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
labels_val_e_gamma=labels_val[np.where( (labels_val==0) | (labels_val==1))]
softmax_out_val_e_gamma=softmax_out_val[np.where( (labels_val==0) | (labels_val==1))][:,1]
fpr,tpr,thr=roc_curve(labels_val_e_gamma,softmax_out_val_e_gamma)
roc_AUC=auc(fpr,tpr)
fig1, ax1 = plt.subplots(figsize=(12,8),facecolor="w")
ax1.tick_params(axis="both", labelsize=20)
ax1.plot(fpr,tpr,label=r'$e$ VS $\gamma$ ROC, AUC={:.3f}'.format(roc_AUC))
ax1.set_xlabel('FPR',fontweight='bold',fontsize=24,color='black')
ax1.set_ylabel('TPR',fontweight='bold',fontsize=24,color='black')
ax1.legend(loc="lower right",prop={'size': 16})

rejection=1.0/(fpr+1e-10)

fig2, ax2 = plt.subplots(figsize=(12,8),facecolor="w")
ax2.tick_params(axis="both", labelsize=20)
plt.yscale('log')
plt.ylim(1.0,1.0e3)
#plt.grid(b=True, which='major', color='gray', linestyle='-')
#plt.grid(b=True, which='minor', color='gray', linestyle='--')
ax2.plot(tpr, rejection, label=r'$e$ VS $\gamma$ ROC, AUC={:.3f}'.format(roc_AUC))
ax2.set_xlabel('efficiency',fontweight='bold',fontsize=24,color='black')
ax2.set_ylabel('Rejection',fontweight='bold',fontsize=24,color='black')
ax2.legend(loc="upper right",prop={'size': 16})

plt.savefig("plots/analysis/roc_rej.png")

fig2, ax2 = plt.subplots(figsize=(12,8),facecolor="w")
ax2.tick_params(axis="both", labelsize=20)
#plt.yscale('log')
#plt.ylim(1.0,1)
#plt.grid(b=True, which='major', color='gray', linestyle='-')
#plt.grid(b=True, which='minor', color='gray', linestyle='--')
ax2.plot(tpr, tpr/np.sqrt(fpr), label=r'$e$ VS $\gamma$ ROC, AUC={:.3f}'.format(roc_AUC))
ax2.set_xlabel('efficiency',fontweight='bold',fontsize=24,color='black')
ax2.set_ylabel('~significance gain',fontweight='bold',fontsize=24,color='black')
ax2.legend(loc="upper right",prop={'size': 16})

plt.savefig("plots/analysis/roc_sig.png")
