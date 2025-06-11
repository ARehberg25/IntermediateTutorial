## Training

First we'll look at a Multi-Layer Perceptron (MLP). An MLP is a very basic neutral network, with a set of fully-connected layers connecting input (PMT information) to output (class prediction).
We'll use the script _scripts/mlp\_training.py_ to do this. By running this script and following along here, we'll manually go through training through one batch of our input data.

This script load the class [SimpleMLP](models/simpleMLP.py), which I encourage you to peruse. [We load this class](https://github.com/felix-cormier/HK_ML_tutorial/blob/trisep_dev/scripts/mlp_training.py#L49)

```python
model_MLP=SimpleMLP(num_classes=3)
```

The code will then print out the names of parameters, and whether they require a gradient to be computed (for a backwards pass):

```python
    for name, param in model_MLP.named_parameters():
        print("name of a parameter: {}, type: {}, parameter requires a gradient?: {}".
            format(name, type(param),param.requires_grad))
```

**Can you link all of these parameters to those defined in the [SimpleMLP](models/simpleMLP.py) class?**

As we can see by default the parameters have `requires_grad` set - i.e. we will be able to obtain gradient of the loss function with respect to these parameters.

Let's quickly look at the [source](https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear) for the linear module

The parameters descend from the `Tensor` class. When `Parameter` object is instantiated as a member of a `Module` object class the parameter is added to `Module`s list of parameters automatically. 

This list and values are captured in the 'state dictionary' of a module, which is essentially all the weights and biases of the model. This can just be [accessed by](https://github.com/felix-cormier/HK_ML_tutorial/blob/trisep_dev/scripts/mlp_training.py#L67):
```python
model_MLP.state_dict()
```

Did you notice that the values are not 0? This is actually by design - by default that initialization follows an accepted scheme - but many strategies are possible


### Training a Multi-layer Perceptron



Let's [load a dataset using our WCH5Dataset class, and use a dataloader](https://github.com/felix-cormier/HK_ML_tutorial/blob/trisep_dev/scripts/mlp_training.py#L96-103)
```python
    dset=WCH5Dataset("/fast_scratch/TRISEP_data/NUPRISM.h5",reduced_dataset_size=100000,val_split=0.1,test_split=0.1)

    # Let's make a dataloader and grab a first batch
    train_dldr=DataLoader(dset,
                        batch_size=32,
                        shuffle=False,
                        sampler=SubsetRandomSampler(dset.train_indices))
    train_iter=iter(train_dldr)
```

Then we can [grab the data and labels from the 1st batch of the iterator we just made](https://github.com/felix-cormier/HK_ML_tutorial/blob/trisep_dev/scripts/mlp_training.py#L104-113):
```python
#Set up iterator over training set
train_iter=iter(train_dldr)
#Get the first batch
batch0=next(train_iter)
#Get PMT data from batch
data=batch0[0]
#Get labels from batch
labels=batch0[1]
# Now compute the model output on the data
model_out=model_MLP(data)
```

Now we have model's predictions and we above got 'true' labels from the dataset, so we can now compute the loss - CrossEntropyLoss is the apropropriate one to use here. 
We will use `CrossEntropyLoss` from `torch.nn` - btw it is also a `Module`. [First create it, then evaluate](https://github.com/felix-cormier/HK_ML_tutorial/blob/trisep_dev/scripts/mlp_training.py#L123-126):
```python
#Define the kind of loss
loss_module=CrossEntropyLoss()
# Now evaluate the loss. 
loss_tensor=loss_module(model_out,labels)
```

Now that we've defined a model, we can [save a diagram of it to remember our architecture](https://github.com/felix-cormier/HK_ML_tutorial/blob/trisep_dev/scripts/mlp_training.py#L135). **This can also be used for different architectures later on.**

Before we calculate the gradients - let's [check what they are now](https://github.com/felix-cormier/HK_ML_tutorial/blob/trisep_dev/scripts/mlp_training.py#L141-143):
```python
#Print out each parameter's gradient
for name, param in model_MLP.named_parameters():
    print("name of a parameter: {}, gradient: {}".
        format(name, param.grad))
```

This doesn't look right! Don't forget we have to [actually do the backwards pass](https://github.com/felix-cormier/HK_ML_tutorial/blob/trisep_dev/scripts/mlp_training.py#L148)
```python
#Compute gradients by doing backward pass
loss_tensor.backward()
#Print out each parameter's gradient
for name, param in model_MLP.named_parameters():
    print("name of a parameter: {}, gradient: {}".
        format(name, param.grad))
```

All we have to do now is [subtract the gradient of a given parameter from the parameter tensor itself and do it for all parameters of the model](https://github.com/felix-cormier/HK_ML_tutorial/blob/trisep_dev/scripts/mlp_training.py#L159-163) - that should decrease the loss. Normally the gradient is multiplied by a learning rate parameter $\lambda$ so we don't go too far in the loss landscape
```python
#Define learning rate
lr=0.0001
#Multiply each parameter by the learning rate times graduent 
for param in model_MLP.parameters():
  param.data.add_(-lr*param.grad.data)
# call to backward **accumulates** gradients - so we also need to zero the gradient tensors if we want to keep going
for param in model_MLP.parameters():
  param.grad.data.zero_()
```
We've now succesfully updated our model parameters by doing a backward pass! 

There is a much simpler way of doing this - we can use the pytorch [optim](https://pytorch.org/docs/stable/optim.html) classes. i
This allows us to easily use more advanced optimization options (like momentum or adaptive optimizers like [Adam](https://arxiv.org/abs/1412.6980)), [no loops over parameters needed](https://github.com/felix-cormier/HK_ML_tutorial/blob/trisep_dev/scripts/mlp_training.py#L172-185):
```python
#Define Optimizer as SGD
optimizer = optim.SGD(model_MLP.parameters(), lr=0.0001)
# Lets get a new batch of events
batch1=next(train_iter)
data=batch1[0]
labels=batch1[1]
#Forward pass
model_out=model_MLP(data)
#Compute loss
loss_tensor=loss_module(model_out,labels)
#Backward pass
loss_tensor.backward()
#Update parameters
optimizer.step()
```

We could just put the code above in a loop and be done with it, but the usual practice would be to wrap this functionality in a training object. 
Here we'll use the [engine](utils/engine.py) class. Let's examine it. These are the general steps the engine abstracts:
   1. Implementation of the training loop
   2. Evaluation on validation set and training and test modes.
   3. Turning evaluation of gradients on and off.
   4. Saving and retrieving the model and optimizer state.

To run an MLP training session, simply run _scripts/mlp\_training.py -s_; the _-s_ option skips past tutorial to a training with the engine. Especially note the [configuration parameters](../scripts/mlp_training.py#L209-222). Some of these settings can be changed - feel free to play around with them! For the settings that affect the training, they are called **hyperparameters**, and finding optimal ones is called **hyperparameter tuning**. The _dump\_path_ config 

```python
config.dump_path = 'model_state_dumps'
```

Will be where the model is saved - a directory with current date-time will be made for every new training. 
The _gpu\_number_ parameter decides the GPU where the training occurs:
```python
config.gpu_number=5
```
On triumf-ml1 there are 8 GPUS (0-7). You can check which ares are in use by, on the command line, doing the command
```
nvidia-smi
```
Try to change this config so that you use an un-utilized GPU. Coordinate with other teams so that you each use separate GPUs! This is not so important for a simple MLP, but for more complex models, every bit of free memory can help.

Most of the configuration parameters should be related to something we've done in this tutorial already, so try to think of what they all mean!
