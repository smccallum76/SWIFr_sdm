"""
This script is specific to my thesis and is used to generate visualizations that will be used in the Results section.
Therefore, this script can be largely ignored.

"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, roc_auc_score

'''
-----------------------------------------------------------------------------------------------------------------------
Extract data from the SQLite DB for visualizations 
-----------------------------------------------------------------------------------------------------------------------
'''
states = 2  # change from 2, 3, or 4 states
do_cm = 'yes'
do_roc = 'yes'

path = 'C:/Users/scott/PycharmProjects/SWIFr_sdm/SWIFr_HMM/hmm/output_db/'
conn = sqlite3.connect(path + 'hmm_predictions.db')
# table names that contain the stochastic backtrace, viterbi, and gamms
table_xpehh = f'sbt_prediction_xpehh_{states}class'
table_ihs = f'sbt_prediction_ihs_afr_std_{states}class'
table_fst = f'sbt_prediction_fst_{states}class'

sql_xpehh = (f"""
       SELECT *
        FROM {table_xpehh}
       """)

sql_ihs = (f"""
       SELECT *
        FROM {table_ihs}
       """)

sql_fst = (f"""
       SELECT *
        FROM {table_fst}
       """)


# collect a list of the unique simulations
xpehh = pd.read_sql(sql_xpehh, conn)
ihs = pd.read_sql(sql_ihs, conn)
fst = pd.read_sql(sql_fst, conn)

# drop nans prior to analysis
xpehh = xpehh[xpehh['xpehh'] != -998.0].reset_index(drop=True)
xpehh_classes = list(xpehh['label'].unique())
ihs = ihs[ihs['ihs_afr_std'] != -998.0].reset_index(drop=True)
ihs_classes = list(ihs['label'].unique())
fst = fst[fst['fst'] != -998.0].reset_index(drop=True)
fst_classes = list(fst['label'].unique())

"""
---------------------------------------------------------------------------------------------------
Confusion Matrix - 2, 3 or 4 states
---------------------------------------------------------------------------------------------------
"""

if do_cm == 'yes':
    """ ------------- XPEHH ------------- """
    
    fig, axs = plt.subplots(1,2, figsize=(12, 6))
    cm1 = confusion_matrix(xpehh['label'], xpehh['viterbi_class_xpehh'], labels=xpehh_classes, normalize='true')
    cm2 = confusion_matrix(xpehh['label'], xpehh['viterbi_class_xpehh'], labels=xpehh_classes)
    ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=xpehh_classes).plot(
        include_values=True, ax=axs[0], cmap='cividis')
    ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=xpehh_classes).plot(
        include_values=True, ax=axs[1], cmap='cividis')
    axs[0].set_title('HMM Classification using XP-EHH - Normalized')
    axs[1].set_title('HMM Classification using XP-EHH - Counts')

    fig.tight_layout()
    plt.savefig(f'plots_thesis/cm_hmm_{states}class_xpehh.svg', bbox_inches='tight')
    plt.show()

    """ ------------- iHS ------------- """
    fig, axs = plt.subplots(1,2, figsize=(12, 6))
    ihs_classes = list(ihs['label'].unique())
    cm3 = confusion_matrix(ihs['label'], ihs['viterbi_class_ihs_afr_std'], labels=ihs_classes, normalize='true')
    cm4 = confusion_matrix(ihs['label'], ihs['viterbi_class_ihs_afr_std'], labels=ihs_classes)
    ConfusionMatrixDisplay(confusion_matrix=cm3, display_labels=ihs_classes).plot(
        include_values=True, ax=axs[0], cmap='cividis')
    ConfusionMatrixDisplay(confusion_matrix=cm4, display_labels=ihs_classes).plot(
        include_values=True, ax=axs[1], cmap='cividis')
    axs[0].set_title('HMM Classification using iHS - Normalized')
    axs[1].set_title('HMM Classification using iHS - Counts')

    fig.tight_layout()
    plt.savefig(f'plots_thesis/cm_hmm_{states}class_ihs.svg', bbox_inches='tight')
    plt.show()

    """ ------------- fst ------------- """
    fig, axs = plt.subplots(1,2, figsize=(12, 6))
    fst_classes = list(fst['label'].unique())
    cm5 = confusion_matrix(fst['label'], fst['viterbi_class_fst'], labels=fst_classes, normalize='true')
    cm6 = confusion_matrix(fst['label'], fst['viterbi_class_fst'], labels=fst_classes)
    ConfusionMatrixDisplay(confusion_matrix=cm5, display_labels=fst_classes).plot(
        include_values=True, ax=axs[0], cmap='cividis')
    ConfusionMatrixDisplay(confusion_matrix=cm6, display_labels=fst_classes).plot(
        include_values=True, ax=axs[1], cmap='cividis')
    axs[0].set_title('HMM Classification using fst - Normalized')
    axs[1].set_title('HMM Classification using fst - Counts')

    fig.tight_layout()
    plt.savefig(f'plots_thesis/cm_hmm_{states}class_fst.svg', bbox_inches='tight')
    plt.show()

""" 
---------------------------------------------------------------------------------------------------
ROC - need to make a function [macro averaging] - using average of all probs
---------------------------------------------------------------------------------------------------
"""

if do_roc == 'yes':
    # Unique classes
    colors = ['magenta', 'dodgerblue', 'darkviolet', 'blue']  # enough for four classes

    """------------------ XP-EHH ------------------ """
    # dummy var for HMM
    y_onehot_test = pd.get_dummies(xpehh['label'], dtype=int)
    # initialize figure
    plt.figure(figsize=(9, 7))
    for i, j in enumerate(xpehh_classes):  # HMM ROC Curve Loop
        fpr, tpr, thresh = roc_curve(y_onehot_test.loc[:, j], xpehh[f"P({j}_xpehh)"], pos_label=1)
        auc = roc_auc_score(y_onehot_test.loc[:, j], xpehh[f"P({j}_xpehh)"])
        plt.plot(fpr, tpr, color=colors[i], label=f'HMM {j} vs Rest (AUC) = ' + str(round(auc, 2)))

    # plot the chance curve
    plt.plot(np.linspace(0, 1, 50 ), np.linspace(0, 1, 50), color='black',
             linestyle='dashed', label='Chance Level (AUC) = 0.50')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'One-vs-Rest ROC curves [XP-EHH]')
    plt.legend()
    plt.savefig(f'plots_thesis/roc_hmm_{states}class_xpehh.svg', bbox_inches='tight')
    plt.show()

    """------------------ iHS ------------------ """
    # dummy var for HMM
    y_onehot_test = pd.get_dummies(ihs['label'], dtype=int)
    # initialize figure
    plt.figure(figsize=(9, 7))
    for i, j in enumerate(ihs_classes):  # HMM ROC Curve Loop
        fpr, tpr, thresh = roc_curve(y_onehot_test.loc[:, j], ihs[f"P({j}_ihs_afr_std)"], pos_label=1)
        auc = roc_auc_score(y_onehot_test.loc[:, j], ihs[f"P({j}_ihs_afr_std)"])
        plt.plot(fpr, tpr, color=colors[i], label=f'HMM {j} vs Rest (AUC) = ' + str(round(auc, 2)))

    # plot the chance curve
    plt.plot(np.linspace(0, 1, 50 ), np.linspace(0, 1, 50), color='black',
             linestyle='dashed', label='Chance Level (AUC) = 0.50')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'One-vs-Rest ROC curves [iHS]')
    plt.legend()
    plt.savefig(f'plots_thesis/roc_hmm_{states}class_ihs.svg', bbox_inches='tight')
    plt.show()

    """------------------ fst ------------------ """
    # dummy var for HMM
    y_onehot_test = pd.get_dummies(fst['label'], dtype=int)
    # initialize figure
    plt.figure(figsize=(9, 7))
    for i, j in enumerate(fst_classes):  # HMM ROC Curve Loop
        fpr, tpr, thresh = roc_curve(y_onehot_test.loc[:, j], fst[f"P({j}_fst)"], pos_label=1)
        auc = roc_auc_score(y_onehot_test.loc[:, j], fst[f"P({j}_fst)"])
        plt.plot(fpr, tpr, color=colors[i], label=f'HMM {j} vs Rest (AUC) = ' + str(round(auc, 2)))

    # plot the chance curve
    plt.plot(np.linspace(0, 1, 50 ), np.linspace(0, 1, 50), color='black',
             linestyle='dashed', label='Chance Level (AUC) = 0.50')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'One-vs-Rest ROC curves [fst]')
    plt.legend()
    plt.savefig(f'plots_thesis/roc_hmm_{states}class_fst.svg', bbox_inches='tight')
    plt.show()





