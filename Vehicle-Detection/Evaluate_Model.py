# -*- coding: utf-8 -*-
"""
Created on Mon May 14 14:07:19 2018

@author: Vamshidhar P
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 11:25:50 2018

@author: Vamshidhar P
"""
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score

def evaluate_model(predictions, labels, text, pred_proba, cold_encode = False, ROC_Curve  = True):     
    """
    Prints the confusion matrix and accuracy of the model.
    """
    def plot_confusion_matrix(cm, text, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues
                              ):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
       
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]            
            
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.title(text)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
    
    #Convert test labels from one hot 
    def one_cold_encode(labels):
        test_actual_labels = []
        for i in range(len(labels)):
            if labels[i][0] == 1:
                test_actual_labels.append(0)
            else:
                test_actual_labels.append(1)
        return test_actual_labels
            
        
    if cold_encode:
        test_actual_labels = one_cold_encode(labels)
    else:        
        test_actual_labels = labels
        
        
    # Plot ROC Curve
    if ROC_Curve:
        fpr, tpr, threshold = roc_curve(y_true = test_actual_labels, y_score = pred_proba)
        roc_auc = auc(x = fpr, y = tpr)
            
        label = text + " ROC Curve"
        plt.figure()
        plt.plot(fpr, tpr, color = 'green', linestyle = '-', label =  '%s (auc = %0.2f)' % (label, roc_auc))            
        plt.legend(loc = 'lower_righht')
        plt.plot([0,1],[0,1], linestyle = '--', color= 'gray', linewidth = 2)
        plt.xlim([-0.1,1.1])
        plt.ylim([-0.1,1.1])
        plt.grid()
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()      
    
    cmatrix = confusion_matrix(test_actual_labels, predictions)
    
    accuracy = (cmatrix[0][0] + cmatrix[1][1])/len(predictions)
    precision = cmatrix[0][0] / (cmatrix[0][0] + cmatrix[1][0])
    recall = cmatrix[0][0] / (cmatrix[0][0] + cmatrix[0][1])
    sensitivity = recall
    specificity = cmatrix[1][1] / (cmatrix[1][1] + cmatrix[1][0])

    text = text

    plt.figure()
    plot_confusion_matrix(cmatrix, text, classes=np.array(['NOT CAR','CAR']),title='Confusion matrix')
    
    print (text + " Accuracy: %f" % (accuracy))
    print (text + " Precision: %f" % (precision))
    print (text + " Recall: %f" % (recall))
    print (text + " Sensitivity: %f" % (sensitivity))
    print (text + " Specificity: %f" % (specificity))