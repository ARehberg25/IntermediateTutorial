
import os,sys

import h5py
import numpy as np

from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import CrossEntropyLoss

#add to path so that we can grab functions from other directories
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from utils.data_handling import WCH5Dataset
from utils.engine import Engine

from models.simpleMLP import SimpleMLPSEQ
from models.simpleMLP import SimpleMLP

import argparse

from torchviz import make_dot




# ## Building a simple fully connected network (a Multi-Layer Perceptron)

# Let's set up the paths and make a dataset again:

# Now Let's make our model. We'll talk about 
#   - model parameters
#   - inputs and the forward method
#   - Modules containing modules
#   - Sequential Module  

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--skip_tutorial", help="Run with this option just to train the MLP",
                    action="store_true")
args = parser.parse_args()
if args.skip_tutorial:
    print("Running just the MLP training")

if not args.skip_tutorial:

    model_MLP=SimpleMLP(num_classes=3)

    # Let's look at the parameters:

    for name, param in model_MLP.named_parameters():
        print("name of a parameter: {}, type: {}, parameter requires a gradient?: {}".
            format(name, type(param),param.requires_grad))

    # As we can see by default the parameters have `requires_grad` set - i.e. we will be able to obtain gradient of the loss function with respect to these parameters.

    # Let's quickly look at the [source](https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear) for the linear module

    # The parameters descend from the `Tensor` class. When `Parameter` object is instantiated as a member of a `Module` object class the parameter is added to `Module`s list of parameters automatically. 
    # This list and values are captured in the 'state dictionary' of a module:

    model_MLP.state_dict()

    # Did you notice that the values are not 0? This is actually by design - by default that initialization follows an accepted scheme - but many strategies are possible

    # Now let's look at sequential version

    model_MLPSEQ=SimpleMLPSEQ(num_classes=3)

    for name, param in model_MLPSEQ.named_parameters():
        print("name of a parameter: {}, type: {}, parameter requires a gradient?: {}".
            format(name, type(param),param.requires_grad))

    print(model_MLPSEQ.state_dict())

    # As we can see the parameters look similar but have different names

    # ## Training a model

    # First let's make a dataset object

    # %%
    try:
        f=h5py.File("/fast_scratch/TRISEP_data/NUPRISM.h5","r")
    except FileNotFoundError:
        print("File not Found!")
        quit()

    dset=WCH5Dataset("/fast_scratch/TRISEP_data/NUPRISM.h5",reduced_dataset_size=100000,val_split=0.1,test_split=0.1)

    # Let's make a dataloader and grab a first batch


    train_dldr=DataLoader(dset,
                        batch_size=32,
                        shuffle=False,
                        sampler=SubsetRandomSampler(dset.train_indices))
    train_iter=iter(train_dldr)

    batch0=next(train_iter)

    data=batch0[0]
    labels=batch0[1]

    # Now compute the model output on the data

    model_out=model_MLP(data)



    print(labels)

    print(model_out)

    # Now we have model's predictions and we above got 'true' labels from the dataset, so we can now compute the loss - CrossEntropyLoss is the apropropriate one to use here. 
    # We will use `CrossEntropyLoss` from `torch.nn` - btw it is also a `Module`. First create it:

    loss_module=CrossEntropyLoss()

    # Now evaluate the loss. 

    loss_tensor=loss_module(model_out,labels)

    print(loss_tensor)

    # This was a 'forward pass'. We should now have a computational graph available - let's plot it for kicks...

    make_dot(model_out, params=dict(list(model_MLP.named_parameters()))).render("diagrams/mlp_torchviz", format="png")

    # Before we calculate the gradients - let's check what they are now...

    for name, param in model_MLP.named_parameters():
        print("name of a parameter: {}, gradient: {}".
            format(name, param.grad))

    # No wonder - let's calculate them

    loss_tensor.backward()

    for name, param in model_MLP.named_parameters():
        print("name of a parameter: {}, gradient: {}".
            format(name, param.grad))

    # All we have to do now is subtract the gradient of a given parameter from the parameter tensor itself and do it for all parameters of the model 
    # - that should decrease the loss. Normally the gradient is multiplied by a learning rate parameter $\lambda$ so we don't go too far in the loss landscape

    lr=0.0001
    for param in model_MLP.parameters():
        param.data.add_(-lr*param.grad.data)

    # call to backward **accumulates** gradients - so we also need to zero the gradient tensors if we want to keep going

    for param in model_MLP.parameters():
        param.grad.data.zero_()

    # There is a much simpler way of doing this - we can use the pytorch [optim](https://pytorch.org/docs/stable/optim.html) classes. i
    # This allows us to easily use more advanced optimization options (like momentum or adaptive optimizers like [Adam](https://arxiv.org/abs/1412.6980)):

    optimizer = optim.SGD(model_MLP.parameters(), lr=0.0001)

    # Lets get a new batch of events

    batch1=next(train_iter)

    data=batch1[0]
    labels=batch1[1]

    # 
    print(labels)

    model_out=model_MLP(data)
    loss_tensor=loss_module(model_out,labels)
    loss_tensor.backward()
    optimizer.step()

# We could just put the code above in a loop and be done with it, but the usual practice would be to wrap this functionality in a training object. 
# Here we'll use the [engine](/edit/utils/engine.py) class. Let's examine it. We'll talk about:
#   1. Implementation of the training loop
#   2. Evaluation on validation set and training and test modes.
#   3. Turning evaluation of gradients on and off.
#   4. Saving and retrieving the model and optimizer state.


# Let's first create a configuration object -we'll use this to set up our training engine

else:

    model_MLP=SimpleMLP(num_classes=3)

    try:
        f=h5py.File("/fast_scratch/TRISEP_data/NUPRISM.h5","r")
    except FileNotFoundError:
        print("File not Found!")
        quit()

    dset=WCH5Dataset("/fast_scratch/TRISEP_data/NUPRISM.h5",reduced_dataset_size=100000,val_split=0.1,test_split=0.1)

    class CONFIG:
        pass
    config=CONFIG()
    config.batch_size_test =512
    config.batch_size_train = 64
    config.batch_size_val = 512
    config.lr=0.00001
    config.device = 'gpu'
    config.gpu_number=5
    config.num_workers_train=6
    config.num_workers_val=1
    config.num_workers_test=1
    config.checkpoint=False
    config.dump_path = 'model_state_dumps'

    engine=Engine(model_MLP,dset,config)

    print(vars(config))

    engine.train(epochs=5,report_interval=100,valid_interval=200)





