## Training Other Network Architectures

We've built classes for two more architectures, a Convolutional Neural Network (CNN) and Residual Netowrk (ResNet).

### CNN

We've pre-built a [CNN script](../scripts/cnn_training.py). The content is very similar to what we had at the end of the MLP training tutorial, except now we use the [simpleCNN](../models/simpleCNN.py) class. I would encourage checking out the simpleCNN class and understanding the multiple layers. We again use a config object to pass some training variables and hyper-parameters; are there new ones introduced since the MLP? Can you think of any others you could pass to the simpleCNN class?

As usual, you can run the script with the following command from the repo
```
python scripts/cnn_training.py
```

The first time, it will print out some parameters, similar to previously. Simply add the option _-s_ to skip that.

You can now run evaluation on this trained network. You'll have to modify the evaluation script we ran previously to accept the CNN rather than MLP. Some of the work is done already (but commented out), but the rest we leave up to you.

### ResNet

ResNet ([paper](https://arxiv.org/abs/1512.03385), [blog explanation](https://viso.ai/deep-learning/resnet-residual-neural-network/#:~:text=What%20is%20ResNet?,with%2050%20neural%20network%20layers.) is a residual neural network. At its heart is a series of convolutional layers, but with the added complexity of residual connections. These residual connections pass information through the layers by essentially skipping through them. This helps convergence by minimizing vanishing gradients.

We've started building a [ResNet script](../scripts/resnet_training.py), but it is incomplete. Can you use your knowledge of the previous tutorials to complete it and train a residual network on IWCD data? Notice that the _import_ statement for ResNet has multiple different network classes. Can you test whether using a larger ResNet makes the network better at classifying IWCD events?

As usual, you can run the script with the following command from the repo
```
python scripts/resnet_training.py
```

The first time, it will print out some parameters, similar to previously. Simply add the option _-s_ to skip that.

### Next Steps

You can now run evaluation on this trained network. You'll have to modify the evaluation script we ran previously to accept the CNN rather than MLP. Some of the work is done already (but commented out), but the rest we leave up to you.

From here on out the tutorial ends. You can use the tools given to investigate different ways to train networks to classify IWCD data. 

Here are some proposed ways to go deeper in this tutorial:

- Compare the base performance of the three architectures, training for more epochs until the monitored performance starts to be stable
- Modify hyperparameters: both those explicitly stated in config, and any others you can find in the individual network code
- Add pre-processing to the input charge and/or time to make it easier for the network to learn
- Add more in-depth evaluation. For example, does your classifier have a momentum dependence?
