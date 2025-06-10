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

from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

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

model_resnet=#What comes here?
engine=#What comes here?

for name, param in model_resnet.named_parameters():
    print("name of a parameter: {}, type: {}, parameter requires a gradient?: {}".
          format(name, type(param),param.requires_grad))

# Train
engine.train(epochs=5,report_interval=100,valid_interval=200)