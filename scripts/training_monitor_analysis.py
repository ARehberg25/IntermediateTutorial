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
parser.add_argument("-l_mlp", "--location_mlp", help="MLP Model location, usually model_state_dumps/time-and-date")
parser.add_argument("-l_cnn", "--location_cnn", help="CNN Model location, usually model_state_dumps/time-and-date")
args = parser.parse_args()

loc_mlp=args.location_mlp
loc_cnn=args.location_cnn

plot_utils.disp_learn_hist(loc_mlp, show=False, losslim=1, output_name="../plots/training/log_training_mlp.png")
plot_utils.disp_learn_hist(loc_cnn, show=False, losslim=1, output_name="../plots/training/log_training_cnn.png")


# Why so bumpy?
#    - there is 'noise' associated with batch-to-batch variation BUT
#    - The main effect is actually the real bumpiness in the loss landscape being traversed during learning
#    
# To learn anything we need to smooth out the plot - for instance by using moving average

plot_utils.disp_learn_hist_smoothed(loc_mlp,window_train=200,window_val=1, output_name="../plots/training/log_training_mlp.png")
plot_utils.disp_learn_hist_smoothed(loc_cnn,window_train=200,window_val=1, output_name="../plots/training/log_training_cnn.png")

# This actually looks pretty good - we get initially a very quick learning and then a plateau. 
# Both training and validation loss is still decreasing slightly and tracking - which means we could probably kept on training


# ## Evaluating model performance in classification task

# Now let's go back to the full dataset and load the model trained last time on the full dataset


if not args.monitor_only:
    class CONFIG_MLP:
        pass
    config_mlp=CONFIG_MLP()
    config_mlp.batch_size_test =512
    config_mlp.batch_size_train = 256
    config_mlp.batch_size_val = 512
    config_mlp.lr=0.01
    config_mlp.kernel_size=2
    config_mlp.device = 'gpu'
    config_mlp.gpu_number=5
    config_mlp.num_workers_train=6
    config_mlp.num_workers_val=1
    config_mlp.num_workers_test=1
    config_mlp.checkpoint=False
    config_mlp.dump_path = 'model_state_dumps'

    class CONFIG_CNN:
        pass
    config_cnn=CONFIG_CNN()
    config_cnn.batch_size_test =512
    config_cnn.batch_size_train = 256
    config_cnn.batch_size_val = 512
    config_cnn.lr=0.01
    config_cnn.kernel_size=2
    config_cnn.device = 'gpu'
    config_cnn.num_workers_train=6
    config_cnn.num_workers_val=1
    config_cnn.num_workers_test=1
    config_cnn.checkpoint=False
    config_cnn.dump_path = 'model_state_dumps'

    #Choose which dataset to use depending on network
    #MLP
    dset_mlp=WCH5Dataset("/fast_scratch_1/TRISEP_data/NUPRISM.h5",val_split=0.1,test_split=0.1)
    #CNN or ResNet
    dset_cnn=WCH5Dataset("/fast_scratch_1/TRISEP_data/NUPRISM.h5",val_split=0.1,test_split=0.1,transform=rotate_chan)

    #Modify depending on architecture
    model_MLP=SimpleMLP(num_classes=3)
    engine_mlp=Engine(model_MLP,dset_mlp,config_mlp, eval_only=True)

    model_CNN=SimpleCNN(config_cnn, num_input_channels=38,num_classes=3)
    engine_cnn=Engine(model_CNN,dset_cnn,config_cnn, eval_only=True)


    #Will need to be modified for CNN, ResNet, or other networks
    engine_mlp.restore_state(loc_mlp+"/SimpleMLPBEST.pth")
    engine_cnn.restore_state(loc_cnn+"/SimpleCNNBEST.pth")  

    engine_mlp.dirpath=loc_mlp
    engine_cnn.dirpath=loc_cnn

    if not args.skip_evaluation:
        engine_mlp.validate()
        engine_cnn.validate()

    ## Examination of classifier output

    # Plot the classifier softmax output for various classes and outputs

    labels_val_mlp=np.load(engine_mlp.dirpath + "labels.npy")
    predictions_val_mlp=np.load(engine_mlp.dirpath + "predictions.npy")
    softmax_out_val_mlp=np.load(engine_mlp.dirpath + "softmax.npy")

    labels_val_cnn=np.load(engine_cnn.dirpath + "labels.npy")
    predictions_val_cnn=np.load(engine_cnn.dirpath + "predictions.npy")
    softmax_out_val_cnn=np.load(engine_cnn.dirpath + "softmax.npy")


    plot_utils.plot_resp(labels_val_mlp, softmax_out_val_mlp, output_name = '../plots/analysis/softmax_mlp.png')
    plot_utils.plot_resp(labels_val_cnn, softmax_out_val_cnn, output_name = '../plots/analysis/softmax_cnn.png')


    # ### The confusion matrix
    plot_utils.plot_confusion_matrix(labels_val_mlp, predictions_val_mlp, ['gamma','electron','muon'],output_name = '../plots/analysis/confusion_matrix_mlp.png')
    plot_utils.plot_confusion_matrix(labels_val_cnn, predictions_val_cnn, ['gamma','electron','muon'],output_name = '../plots/analysis/confusion_matrix_cnn.png')


    # ### Receiver Operating Characteristic
    #    - We will plot ROC treating electrons as 'signal' and photons as 'background', 
    # but we have also muons - which is also a signal, so we have more reasonable possibilities of plotting ROC curves - 
    # can you name advantages and disadvantages?
    plot_utils.plot_roc_curves(labels_val_mlp, softmax_out_val_mlp, output_path = '../plots/analysis/mlp_')
    plot_utils.plot_roc_curves(labels_val_cnn, softmax_out_val_cnn, output_path = '../plots/analysis/cnn_')

