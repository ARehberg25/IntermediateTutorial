## Model Monitoring and evaluation

### Monitoring

We will use the script in _scripts/training\_monitor\_analysis.py_ to both monitor the loss and accuracy and evaluate the performance of the trained model. This script has possiblities of input arguments, namely asking to only make monitoring plots (useful while still training) and the location of the logs and model to evaluate.

To see what to use for these input arguments, simply do:

```
python scripts/training_monitor_analysis.py --help
```
This should output _-m_ as monitor-only mode, and _-l_ for the location. If you want to do a full monitor and evaluation of model at _model\_state\_dumps/time-date/_, you would do:
```
python scripts/training_monitor_analysis.py -l model_state_dumps/time-date
```

The monitoring part uses functions from [plot utils](utils/plot_utils.py). [It first runs](../scripts/training_monitor_analysis.py#L34) 
```python
plot_utils.disp_learn_hist(loc, show=False, losslim=1, output_name="plots/training/log_training.png")
```

To make a plot which shows the evolution of loss and accuracy per iteration. 
Why so bumpy?
    - there is 'noise' associated with batch-to-batch variation BUT
The main effect is actually the real bumpiness in the loss landscape being traversed during learning
    
 To learn anything we need to smooth out the plot - for instance by [using moving average](../scripts/training_monitor_analysis.py#L43) 
```python
plot_utils.disp_learn_hist_smoothed(loc,window_train=200,window_val=1, output_name="plots/training/log_training.png")
```

What does it look like? The shape can help you decide if you're undertrained, converged, or overtained. What would each look like?


### Evaluation

The [following lines](../scripts/training_monitor_analysis.py#L71-82)  setting up the engine for validation are set up for a simpleMLP, they'll need to be modified for a CNN or ResNet, which are the more advanced network you will encounter later in the tutorial:
```python
#Choose which dataset to use depending on network
#MLP
dset=WCH5Dataset("/fast_scratch_1/TRISEP_data/NUPRISM.h5",val_split=0.1,test_split=0.1)
#CNN or ResNet
#dset=WCH5Dataset("/fast_scratch_1/TRISEP_data/NUPRISM.h5",val_split=0.1,test_split=0.1,transform=rotate_chan)
#Will need to change model if using CNN or ResNet
#Set up model
model_MLP=SimpleMLP(num_classes=3)
engine=Engine(model_MLP,dset,config)
#Will need to be modified for CNN, ResNet, or other networks
#Load model
engine.restore_state(loc+"SimpleMLPBEST.pth")
```

After this the engine will run validation
```python
engine.validate()
```

Which outputs some _.npy_ files that we will [then load](../scripts/training_monitor_analysis.py#L92-94):
```python
labels_val=np.load(engine.dirpath + "labels.npy")
predictions_val=np.load(engine.dirpath + "predictions.npy")
softmax_out_val=np.load(engine.dirpath + "softmax.npy")
```

We can then [plot the softmax output](../scripts/training_monitor_analysis.py#L97) (0 to 1 number, for each class, 1 is most confident that it is that class):
```python
plot_utils.plot_resp(labels_val, softmax_out_val, output_name = 'plots/analysis/softmax.png')
```

The [confusion matrix](../scripts/training_monitor_analysis.py#L100)
```python
plot_utils.plot_confusion_matrix(labels_val, predictions_val, ['$\gamma$','$e$','$\mu$'],output_name = 'plots/analysis/confusion_matrix.png')
```
can you figure out what the matrix is telling you?

And [finally the ROC curve](../scripts/training_monitor_analysis.py#L106), specifically for electrons as signal and gammas as background.
```python
plot_utils.plot_roc_curves(labels_val, softmax_out_val, output_path = 'plots/analysis/roc.png')
```
ROC stands for Receiver-Operating curve, which is fancy machine learning jargon. Essentially we are plotting the true positive rate on the x-axis versus the false positive rate on the y-axis. We also make a second ROC curve, labelled **rejection**, where the y-axis is replaced with background rejection, which is simply the reciprocal of false positive rate. This is helpful to visualize small differences, and gives an intuitive understanding (rejection is basically how many backgrounds get rejected for every one that gets classified as signal)

Next let's learn how to [train more advanced models](../tutorial/training_cnn_resnet.md).

