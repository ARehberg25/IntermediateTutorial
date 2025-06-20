# # In this tutorial we'll see what to watch for during and after the network training

# ## Monitoring training health

# First let's look at at the training history - we want to display the training set loss and vlidation set loss as a function of 'iteration' or the batch number seen by our model

import os,sys
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import numpy as np
import argparse

from utils import plot_utils

from utils.data_handling import WCH5Dataset
from utils.data_utils import rotate_chan
from utils.engine import Engine

from models.simpleCNN import SimpleCNN
from models.simpleMLP import SimpleMLP
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

# You will need to change the location of where these files are

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--monitor_only", help="Only run loss/accuracy monitoring, without model evaluation",
                    action="store_true")
parser.add_argument("-s", "--skip_evaluation", help="If you've already run evaluation, and just want to make plots",
                    action="store_true")
parser.add_argument("-l", "--location", help="Model location, usually model_state_dumps/time-and-date")
args = parser.parse_args()

loc=args.location

plot_utils.disp_learn_hist(loc, show=False, losslim=1, output_name="plots/training/log_training_cnn.png")


# Why so bumpy?
#    - there is 'noise' associated with batch-to-batch variation BUT
#    - The main effect is actually the real bumpiness in the loss landscape being traversed during learning
#    
# To learn anything we need to smooth out the plot - for instance by using moving average

plot_utils.disp_learn_hist_smoothed(loc,window_train=200,window_val=1, output_name="plots/training/log_training_cnn.png")

# This actually looks pretty good - we get initially a very quick learning and then a plateau. 
# Both training and validation loss is still decreasing slightly and tracking - which means we could probably kept on training


# ## Evaluating model performance in classification task

# Now let's go back to the full dataset and load the model trained last time on the full dataset


if not args.monitor_only:
    class CONFIG:
        pass
    config=CONFIG()
    config.batch_size_test =512
    config.batch_size_train = 256
    config.batch_size_val = 512
    config.lr=0.01
    config.kernel_size=2
    config.device = 'gpu'
    config.gpu_number=5
    config.num_workers_train=6
    config.num_workers_val=1
    config.num_workers_test=1
    config.checkpoint=False
    config.dump_path = 'model_state_dumps'

    #Choose which dataset to use depending on network
    #MLP
    dset=WCH5Dataset("/fast_scratch_1/TRISEP_data/NUPRISM.h5",val_split=0.1,test_split=0.1)
    #CNN or ResNet
    #dset=WCH5Dataset("/fast_scratch_1/TRISEP_data/NUPRISM.h5",val_split=0.1,test_split=0.1,transform=rotate_chan)

    #Modify depending on architecture
    model_MLP=SimpleMLP(num_classes=3)
    engine=Engine(model_MLP,dset,config, eval_only=True)


    #Will need to be modified for CNN, ResNet, or other networks
    engine.restore_state(loc+"SimpleMLPBEST.pth")

    engine.dirpath=loc

    if not args.skip_evaluation:
        engine.validate()

    ## Examination of classifier output

    # Plot the classifier softmax output for various classes and outputs

    labels_val=np.load(engine.dirpath + "labels.npy")
    predictions_val=np.load(engine.dirpath + "predictions.npy")
    softmax_out_val=np.load(engine.dirpath + "softmax.npy")


    plot_utils.plot_resp(labels_val, softmax_out_val, output_name = 'plots/analysis/softmax_mlp.png')

    # ### The confusion matrix
    plot_utils.plot_confusion_matrix(labels_val, predictions_val, ['gamma','electron','muon'],output_name = 'plots/analysis/confusion_matrix_mlp.png')

    # ### Receiver Operating Characteristic
    #    - We will plot ROC treating electrons as 'signal' and photons as 'background', 
    # but we have also muons - which is also a signal, so we have more reasonable possibilities of plotting ROC curves - 
    # can you name advantages and disadvantages?
    plot_utils.plot_roc_curves(labels_val, softmax_out_val, output_path = 'plots/analysis/mlp_')
