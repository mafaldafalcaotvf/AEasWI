import pandas
import numpy as np
import keras.backend as K

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns

# x -> DataFrame
# genesList -> list
def selectProtGenes(x, genesList):
    selected = pandas.DataFrame([])
    features = []
    for feature in x.columns.values:
        f, _ = feature.split('.')
        if(f in genesList):
            features += [feature]
            selected.insert(len(features)-1, feature, x.loc[:, feature])
    return selected

# X -> DataFrame
def removeConstantValues(X):
    return X.loc[:,X.apply(pandas.Series.nunique) != 1]

# Custom Metrics

INTERESTING_CLASS_ID = 2

def single_class_accuracy(y_true, y_pred):
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    accuracy_mask = K.cast(K.equal(class_id_preds, INTERESTING_CLASS_ID), 'int32')
    class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
    class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
    return class_acc


# Plots

def correlationMatrix(df, colNames, filename):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('THCA Feature Correlation')
    ax1.set_xticklabels(colNames,fontsize=6)
    ax1.set_yticklabels(colNames,fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.show()
    fig.savefig('images/'+filename, bbox_inches='tight')

def plotFail(scores, features, filename):
    plt.rcParams["figure.figsize"] = 5,2
    f, ax = plt.subplots() #figsize=100

    all_data = [[np.mean(e) for e in scores]]
    
    ax.imshow(all_data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks([y for y in range(len(all_data))])
    ax.set_xticklabels(features, fontsize=5)
    ax.set_yticks([])
    #ax.set_xlim(extent[0], extent[1])


    # add x-tick labels
    #plt.setp(ax, xticks=[y for y in range(len(all_data))])
    #plt.xticks(fontsize=6, rotation='vertical')
    #plt.tight_layout()
    plt.show()
    f.savefig('images/'+filename+'.pdf')
    print('[INFO] Image Saved!')

def plot(scores, features, filename):
    #plt.rcParams["figure.figsize"] = 5,2
    f, ax = plt.subplots() #figsize=100

    data = [[np.mean(e) for e in scores]]

    cmap = sns.diverging_palette(10, 220, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(data, cmap=cmap, cbar=True, vmin=min(data[0]), vmax=max(data[0]), yticklabels=[], xticklabels=features, linewidths=.5, ax=ax)
    plt.tight_layout()
    f.savefig('images/'+filename+'.pdf')
    print('[INFO] Image Saved!')
