import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

POS_MAP = [(8,4), #0
           (7,2), #1
           (6,0), #2
           (4,0), #3
           (2,0), #4
           (1,1), #5
           (0,4), #6
           (1,6), #7
           (2,8), #8
           (4,8), #9
           (6,8), #10
           (7,6), #11
           # Inner ring
           (6,4), #12
           (5,2), #13
           (3,2), #14
           (2,4), #15
           (3,6), #16
           (5,6), #17
           (4,4)] #18

PADDING = 1

def event_displays(event, label, plot_path = 'plots/data_exploration/'):
    """Generates event display 3 ways

    Args:
        event (_type_): PMT data from one event
        label (_type_): class label for the event
    """

    # We are going to plot only the PMT charge for the 'center' PMT in mPMT modules - i believe this is at channel 18
    fig, ax = plt.subplots(figsize=(16,8),facecolor='w')
    plt.imshow(event[:,:,18],cmap='jet',origin='lower')
    ax.set_title('Event Data, center PMT',fontsize=20,fontweight='bold')
    print('class is {}'.format(label))
    plt.savefig(plot_path+"/pmt_center_eventDisplay_charge.png")
    plt.clf()

    # We can also display the sum charge in the PMT

    fig, ax = plt.subplots(figsize=(16,8),facecolor='w')
    plt.imshow(np.sum(event[:,:,0:19],axis=-1),cmap='jet',origin='lower')
    ax.set_title('Event Data, charge sum in mPMT',fontsize=20,fontweight='bold')
    plt.savefig(plot_path+"/pmt_sum_eventDisplay_charge.png")
    plt.clf()

    # Let's plot this in a slightly nicer way - this is not 100% eact - we will put each PMT on a grid and display that



    fig, ax = plt.subplots(figsize=(16,8),facecolor='w')
    cmap = plt.cm.viridis
    cmap.set_bad(color='black')
    a=get_plot_array(event[:,:,0:19])
#a = np.ma.masked_where(a < 0.05, a)
    plt.imshow(a,
           origin="upper",
           cmap=cmap)
           #norm=matplotlib.colors.LogNorm(vmax=np.amax(event),
           #                               clip=True))
    ax.set_title('Event Data, charge in mPMT',fontsize=20,fontweight='bold')
    plt.savefig(plot_path+"/mpmt_eventDisplay_charge.png")
    plt.clf()

def plot_pmt_var(data, labels, colors, bins , xlabel = 'X', plot_path = 'plots/data_exploration/test_plot.png', do_log=False):
    """Plots PMT data from min_idx to max_idx, divided into

    Args:
        data_to_plot_events (_type_): _description_
        data_to_plot_labels (_type_): _description_
        pmt_min_idx (int, optional): _description_. Defaults to 0.
        pmt_max_idx (int, optional): _description_. Defaults to 19.
    """
    fig, ax = plt.subplots(figsize=(12,8),facecolor="w")

    ax.tick_params(axis="both", labelsize=20)

    values, bins, patches = ax.hist(data,
                                bins=bins, 
                                label= labels, color=colors, linestyle='--', linewidth=2,
                                log=do_log,
                                histtype='step')
    ax.set_xlabel(xlabel,fontweight='bold',fontsize=24,color='black')

    ax.legend(prop={'size': 16})

    plt.savefig(plot_path)
    plt.clf()


def get_plot_array(event_data):
    
    # Assertions on the shape of the data and the number of input channels
    assert(len(event_data.shape) == 3 and event_data.shape[2] == 19)
    
    # Extract the number of rows and columns from the event data
    rows = event_data.shape[0]
    cols = event_data.shape[1]
    
    # Make empty output pixel grid
    output = np.zeros(((10+PADDING)*rows, (10+PADDING)*cols))
    
    i, j = 0, 0
    
    for row in range(rows):
        j = 0
        for col in range(cols):
            pmts = event_data[row, col]
            tile(output, (i, j), pmts)
            j += 10 + PADDING
        i += 10 + PADDING
        
    return output
            
def tile(canvas, ul, pmts):
    
    # First, create 10x10 grid representing single mpmt
    mpmt = np.zeros((10, 10))
    for i, val in enumerate(pmts):
        mpmt[POS_MAP[i][0]][POS_MAP[i][1]] = val

    # Then, place grid on appropriate position on canvas
    for row in range(10):
        for col in range(10):
            canvas[row+ul[0]][col+ul[1]] = mpmt[row][col]
            


def disp_learn_hist(location,losslim=None,show=True, output_name="plots/training/training_log.png"):
    train_log=location+'/log_train.csv'
    val_log=location+'/log_val.csv'

    train_log_csv = pd.read_csv(train_log)
    val_log_csv  = pd.read_csv(val_log)

    fig, ax1 = plt.subplots(figsize=(12,8),facecolor='w')
    line11 = ax1.plot(train_log_csv.epoch, train_log_csv.loss, linewidth=2, label='Train loss', color='b', alpha=0.3)
    line12 = ax1.plot(val_log_csv.epoch, val_log_csv.loss, marker='o', markersize=3, linestyle='', label='Validation loss', color='blue')
    if losslim is not None:
        ax1.set_ylim(0.,losslim)
    
    ax2 = ax1.twinx()
    line21 = ax2.plot(train_log_csv.epoch, train_log_csv.accuracy, linewidth=2, label='Train accuracy', color='r', alpha=0.3)
    line22 = ax2.plot(val_log_csv.epoch, val_log_csv.accuracy, marker='o', markersize=3, linestyle='', label='Validation accuracy', color='red')

    ax1.set_xlabel('Epoch',fontweight='bold',fontsize=24,color='black')
    ax1.tick_params('x',colors='black',labelsize=18)
    ax1.set_ylabel('Loss', fontsize=24, fontweight='bold',color='b')
    ax1.tick_params('y',colors='b',labelsize=18)
    
    
    ax2.set_ylabel('Accuracy', fontsize=24, fontweight='bold',color='r')
    ax2.tick_params('y',colors='r',labelsize=18)
    ax2.set_ylim(0.,1.05)

    # added these four lines
    lines  = line11 + line12 + line21 + line22
    labels = [l.get_label() for l in lines]
    leg    = ax2.legend(lines, labels, fontsize=16, loc='best', numpoints=1)
    leg_frame = leg.get_frame()
    leg_frame.set_facecolor('white')

    if show:
        plt.grid()
        plt.show()
        return
    
    plt.savefig(output_name)

    return fig
    
def disp_learn_hist_smoothed(location, losslim=None, window_train=400,window_val=40,show=False, output_name="plots/training/training_log_smoothed.png"):
    train_log=location+'/log_train.csv'
    val_log=location+'/log_val.csv'
    
    train_log_csv = pd.read_csv(train_log)
    val_log_csv  = pd.read_csv(val_log)

    epoch_train    = moving_average(np.array(train_log_csv.epoch),window_train)
    accuracy_train = moving_average(np.array(train_log_csv.accuracy),window_train)
    loss_train     = moving_average(np.array(train_log_csv.loss),window_train)
    
    epoch_val    = moving_average(np.array(val_log_csv.epoch),window_val)
    accuracy_val = moving_average(np.array(val_log_csv.accuracy),window_val)
    loss_val     = moving_average(np.array(val_log_csv.loss),window_val)

    epoch_val_uns    = np.array(val_log_csv.epoch)
    accuracy_val_uns = np.array(val_log_csv.accuracy)
    loss_val_uns     = np.array(val_log_csv.loss)
    saved_best=np.array(val_log_csv.saved_best)
    stored_indices=np.where(saved_best>1.0e-3)
    epoch_val_st=epoch_val_uns[stored_indices]
    accuracy_val_st=accuracy_val_uns[stored_indices]
    loss_val_st=loss_val_uns[stored_indices]

    fig, ax1 = plt.subplots(figsize=(12,8),facecolor='w')
    line11 = ax1.plot(epoch_train, loss_train, linewidth=2, label='Average training loss', color='b', alpha=0.3)
    line12 = ax1.plot(epoch_val, loss_val, label='Average validation loss', color='blue')
    line13 = ax1.scatter(epoch_val_st, loss_val_st, label='BEST validation loss',
                         facecolors='none', edgecolors='blue',marker='o')
    
    ax1.set_xlabel('Epoch',fontweight='bold',fontsize=24,color='black')
    ax1.tick_params('x',colors='black',labelsize=18)
    ax1.set_ylabel('Loss', fontsize=24, fontweight='bold',color='b')
    ax1.tick_params('y',colors='b',labelsize=18)
    if losslim is not None:
        ax1.set_ylim(0.,losslim)
    
    ax2 = ax1.twinx()
    line21 = ax2.plot(epoch_train, accuracy_train, linewidth=2, label='Average training accuracy', color='r', alpha=0.3)
    line22 = ax2.plot(epoch_val, accuracy_val, label='Average validation accuracy', color='red')
    line23 = ax2.scatter(epoch_val_st, accuracy_val_st, label='BEST accuracy',
                         facecolors='none', edgecolors='red',marker='o')
    
    
    ax2.set_ylabel('Accuracy', fontsize=24, fontweight='bold',color='r')
    ax2.tick_params('y',colors='r',labelsize=18)
    ax2.set_ylim(0.,1.0)
    
    # added these four lines
    lines  = line11+ line12+ [line13]+ line21+ line22+ [line23]
    #lines_sctr=[line13,line23]
    #lines=lines_plt+lines_sctr

    labels = [l.get_label() for l in lines]
    
    leg    = ax2.legend(lines, labels, fontsize=16, loc=5, numpoints=1)
    leg_frame = leg.get_frame()
    leg_frame.set_facecolor('white')

    if show:
        plt.grid()
        plt.show()
        return

    plt.savefig(output_name)

    return fig


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# Function to plot a confusion matrix
def plot_confusion_matrix(labels, predictions, class_names, output_name = 'plots/analysis/confusion_matrix.png'):
    
    """
    plot_confusion_matrix(labels, predictions, class_names)
    
    Purpose : Plot the confusion matrix for a given energy interval
    
    Args: labels              ... 1D array of true label value, the length = sample size
          predictions         ... 1D array of predictions, the length = sample size
          class_names         ... 1D array of string label for classification targets, the length = number of categories
       
 
    """
    
  
    
    
    fig, ax = plt.subplots(figsize=(12,8),facecolor='w')
    num_labels = len(class_names)
    max_value = np.max([np.max(np.unique(labels)),np.max(np.unique(labels))])
    assert max_value < num_labels
    mat,_,_,im = ax.hist2d(predictions, labels,
                           bins=(num_labels,num_labels),
                           range=((-0.5,num_labels-0.5),(-0.5,num_labels-0.5)),cmap=plt.cm.Blues)

    # Normalize the confusion matrix
    mat = mat.astype("float") / mat.sum(axis=0)[:, np.newaxis]

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=20) 
        
    ax.set_xticks(np.arange(num_labels))
    ax.set_yticks(np.arange(num_labels))
    ax.set_xticklabels(class_names,fontsize=20)
    ax.set_yticklabels(class_names,fontsize=20)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.set_xlabel('Prediction',fontsize=20)
    ax.set_ylabel('True Label',fontsize=20)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(i,j, r"${0:0.3f}$".format(mat[i,j]),
                    ha="center", va="center", fontsize=20,
                    color="white" if mat[i,j] > (0.5*mat.max()) else "black")
    fig.tight_layout()
    plt.title("Confusion matrix", fontsize=20) 
   
    plt.savefig(output_name)
    plt.clf()

def plot_resp(labels,softmax_out,output_name="plots/analysis/softmax.png"):

    fig1, ax1 = plt.subplots(figsize=(12,8),facecolor="w")
    ax1.tick_params(axis="both", labelsize=20)
    softmax_out_val_gamma_Pe=softmax_out[labels==0][:,1]
    softmax_out_val_e_Pe=softmax_out[labels==1][:,1]
    
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
    softmax_out_val_e_Pmu=softmax_out[labels==1][:,2]
    softmax_out_val_mu_Pmu=softmax_out[labels==2][:,2]
    
    values, bins, patches = ax2.hist(softmax_out_val_mu_Pmu, bins=bins, 
                                    label= 'muon', color='green', density=True,
                                    alpha=0.3)
    
    values, bins, patches = ax2.hist(softmax_out_val_e_Pmu, bins=bins, 
                                    label= 'electron', color='red', density=True,
                                    alpha=0.3, log=True)
    ax2.legend(prop={'size': 16})
    ax2.set_xlabel('$P(\mu)$',fontweight='bold',fontsize=24,color='black')
    
    
    
    plt.savefig(output_name)
    plt.clf()


def plot_roc_curves(labels_val, softmax_out_val, output_path = 'plots/analysis/'):
    labels_val_e_gamma=labels_val[np.where( (labels_val==0) | (labels_val==1))]
    softmax_out_val_e_gamma=softmax_out_val[np.where( (labels_val==0) | (labels_val==1))][:,1]
    fpr,tpr,thr=roc_curve(labels_val_e_gamma,softmax_out_val_e_gamma)
    roc_AUC=auc(fpr,tpr)
    fig1, ax1 = plt.subplots(figsize=(12,8),facecolor="w")
    ax1.tick_params(axis="both", labelsize=20)
    ax1.plot(fpr,tpr,label=r'e VS gamma ROC, AUC={:.3f}'.format(roc_AUC))
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
    ax2.plot(tpr, rejection, label=r'e VS gamma ROC, AUC={:.3f}'.format(roc_AUC))
    ax2.set_xlabel('efficiency',fontweight='bold',fontsize=24,color='black')
    ax2.set_ylabel('Rejection',fontweight='bold',fontsize=24,color='black')
    ax2.legend(loc="upper right",prop={'size': 16})

    plt.savefig(output_path+"roc_rej.png")
    plt.clf()

    fig2, ax2 = plt.subplots(figsize=(12,8),facecolor="w")
    ax2.tick_params(axis="both", labelsize=20)
    #plt.yscale('log')
    #plt.ylim(1.0,1)
    #plt.grid(b=True, which='major', color='gray', linestyle='-')
    #plt.grid(b=True, which='minor', color='gray', linestyle='--')
    ax2.plot(tpr, tpr/np.sqrt(fpr), label=r'e VS gamma ROC, AUC={:.3f}'.format(roc_AUC))
    ax2.set_xlabel('efficiency',fontweight='bold',fontsize=24,color='black')
    ax2.set_ylabel('~significance gain',fontweight='bold',fontsize=24,color='black')
    ax2.legend(loc="upper right",prop={'size': 16})

    plt.savefig(output_path+"roc_sig.png")
    plt.clf()

