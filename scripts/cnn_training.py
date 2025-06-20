import os,sys

import h5py
import numpy as np

#add to path so that we can grab functions from other directories
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from utils.data_handling import WCH5Dataset
from utils.engine import Engine
from utils.data_utils import rotate_chan

from models.simpleCNN import SimpleCNN

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--skip_tutorial", help="Run with this option just to train the MLP",
                    action="store_true")
args = parser.parse_args()
class CONFIG:
    pass
config=CONFIG()
config.batch_size_test =512
config.batch_size_train = 256
config.batch_size_val = 512
config.lr=0.01
config.kernel_size=2
config.device = 'gpu'
config.num_workers_train=6
config.num_workers_val=1
config.num_workers_test=1
config.checkpoint=False
config.dump_path = 'model_state_dumps'


model_CNN=SimpleCNN(config, num_input_channels=38,num_classes=3)



dset=WCH5Dataset("/fast_scratch_1/TRISEP_data/NUPRISM.h5",val_split=0.1,test_split=0.1,transform=rotate_chan)

engine=Engine(model_CNN,dset,config)

if not args.skip_tutorial:
    for name, param in model_CNN.named_parameters():
        print("name of a parameter: {}, type: {}, parameter requires a gradient?: {}".
            format(name, type(param),param.requires_grad))

engine.train(epochs=1,report_interval=100,valid_interval=200)