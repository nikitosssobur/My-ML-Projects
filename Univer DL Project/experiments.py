import numpy as np
from data_processing import Datasets
from models import get_cnn_model, get_rnn_model, get_combined_model
from train import train
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc, precision_recall_curve, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt


'''
Creating instance of the dataset class for loading data and preprocessing.
Creating datasets in ndarray format using Datasets class interface.
'''
datasets = Datasets("data/mitbih_train.csv", "data/mitbih_test.csv", 
                    "data/ptbdb_abnormal.csv",  "data/ptbdb_normal.csv")

x_train1, y_train1, x_test1, y_test1 = datasets.get_mitbih_data()

x_train2, y_train2, x_test2, y_test2 = datasets.get_ptbdb_data()


print('Classes distribution (MIT BIH):', datasets.calculate_labels_distribution(datasets.mitbih_train_data))
print(x_train1.shape, y_train1.shape)
print(x_test1.shape, y_test1.shape)
print(x_train2.shape, y_train2.shape)
print(x_test2.shape, y_test2.shape)



def test_model(model, x_test, y_test, dataset_name):
    '''
    Function for testing models on the test data
    '''
    pred_test = model.predict(x_test)
    if dataset_name == 'mitbih': 
        pred_test = np.argmax(pred_test, axis = -1)
    else:
        #pred_test = np.where(pred_test >= 0.5, 1, 0)
        pred_test = (pred_test > 0.5).astype(np.int8)

    f1 = f1_score(y_test, pred_test, average = "macro")

    print("Test f1 score : %s "% f1)

    acc = accuracy_score(y_test, pred_test)

    print("Test accuracy score : %s "% acc)


    if dataset_name == 'ptbdb':
        auroc = roc_auc_score(y_test, pred_test)

        print("Test AUROC score : %s "% auroc)

        auprc = average_precision_score(y_test, pred_test)

        print("Test average_precision_score AUPRC score : %s "% auprc)

    #build_curves(pred_test, y_test)



def build_curves(y_pred, y_test):
    '''
    Function for building ROC and PR curves
    '''
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)

    fig, axs = plt.subplots(1, 2, figsize = (12, 6)) #plt.subplots(1, 2)
    axs[0].plot(fpr, tpr, color = 'darkorange', lw = 2, label = f'ROC curve (area = {roc_auc:.2f})')
    axs[0].plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    axs[0].set_title('Receiver Operating Characteristic')
    axs[0].legend(loc = "lower right")
    
    
    axs[1].plot(recall, precision, color = 'darkorange', lw = 2, label = f'PR curve (area = {pr_auc:.2f})')
    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')
    axs[1].set_title('Precision-Recall curve') 
    axs[1].legend(loc = "lower left")
    plt.show()

    
'''
cnn_model = get_cnn_model(classes_num = 5, input_size = 187)

train(cnn_model, "new_cnn_mitbih.keras", 700, x_train1, y_train1)

test_model(cnn_model, x_test1, y_test1, 'mitbih')
'''

'''
cnn_model2 = get_cnn_model(classes_num = 1, input_size = 187)

train(cnn_model2, "new_cnn_ptbdb.keras", 700, x_train2, y_train2)

test_model(cnn_model2, x_test2, y_test2, 'ptbdb')
'''


'''
rnn_model = get_rnn_model(classes_num = 5, input_size = 187)

train(rnn_model, "rnn_mitbih.keras", 700, x_train1, y_train1)

test_model(rnn_model, x_test1, y_test1, 'mitbih')
'''


rnn_model2 = get_rnn_model(classes_num = 1, input_size = 187)

train(rnn_model2, "rnn_ptbdb.keras", 700, x_train2, y_train2)

test_model(rnn_model2, x_test2, y_test2, 'ptbdb')


'''
comb_model = get_combined_model(5, 187)

train(comb_model, "comb_model_mitbih.keras", 700, x_train1, y_train1)

test_model(comb_model, x_test1, y_test1, 'mitbih')
'''

'''
comb_model2 = get_combined_model(1, 187)

train(comb_model2, "comb_model_ptbdb.keras", 700, x_train2, y_train2)

test_model(comb_model2, x_test2, y_test2, 'ptbdb')
'''