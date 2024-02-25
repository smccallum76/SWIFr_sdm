import hmm_funcs as hmm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, roc_auc_score

""" 
---------------------------------------------------------------------------------------------------
Path to data
---------------------------------------------------------------------------------------------------
"""
stat = 'ihs_afr_std'  # fst is problematic
state_count = 3  # 3 states implies neutral, sweep, and link; 2 states implies neutral and sweep
cut_point = 0  # set to zero if all data is to be used

if state_count == 2:
    gmm_path = '../../swifr_pkg/test_data/simulations_4_swifr_2class/'
    data_path = '../../swifr_pkg/test_data/simulations_4_swifr_test_2class/test/test'
    swifr_path = '../../swifr_pkg/test_data/simulations_4_swifr_test_2class/test/test_classified'
elif state_count == 3:
    gmm_path = '../../swifr_pkg/test_data/simulations_4_swifr/'
    data_path = '../../swifr_pkg/test_data/simulations_4_swifr_test/test/test'
    swifr_path = '../../swifr_pkg/test_data/simulations_4_swifr_test/test/test_classified'
""" 
---------------------------------------------------------------------------------------------------
Load the GMM params from SWIFr_train
---------------------------------------------------------------------------------------------------
"""
gmm_params = hmm.hmm_init_params(gmm_path)
gmm_params = gmm_params[gmm_params['stat'] == stat].reset_index(drop=True)  # limit to one stat for now

""" 
---------------------------------------------------------------------------------------------------
Load the test data and the data classified by SWIFr GMM
---------------------------------------------------------------------------------------------------
"""
# this will need to be a 'get' function, but keep it external for now
data_orig = hmm.hmm_get_data(data_path)
# import swifr data that has been classified using gmm
swfr_classified = hmm.hmm_get_data(swifr_path)

""" 
---------------------------------------------------------------------------------------------------
Labels to Numbers -- This is temporary and used to simplify link_left and link_right to link
---------------------------------------------------------------------------------------------------
"""
# convert the row labels from a string to a numeric value
conditions = [
            data_orig['label'] == 'neutral',  # 0
            data_orig['label'] == 'link_left',  # 2
            data_orig['label'] == 'link_right',  # 2
            data_orig['label'] == 'sweep'  # 1
            ]
if state_count == 3:
    choices = [0, 2, 2, 1]  # 3 classes
elif state_count == 2:
    choices = [0, 0, 0, 1]  # 2 classes

data_orig['label_num'] = np.select(conditions, choices, default=-998)

# repeat the above, but for the swfr_classified (could tech use the same since the indexing is the same)
conditions = [
            swfr_classified['label'] == 'neutral',  # 0
            swfr_classified['label'] == 'link_left',  # 2
            swfr_classified['label'] == 'link_right',  # 2
            swfr_classified['label'] == 'sweep'  # 1
            ]
if state_count == 3:
    choices = [0, 2, 2, 1]  # 3 classes
elif state_count == 2:
    choices = [0, 0, 0, 1]  # 2 classes
swfr_classified['label_num'] = np.select(conditions, choices, default=-998)

""" 
---------------------------------------------------------------------------------------------------
Initiate Pi and the transition matrix
---------------------------------------------------------------------------------------------------
"""
# for now I will define the pi vector using the label_num, but this is only b/c the data labels are not a match
# with the hmm classes (e.g., link_left and link_right are being defined as link)
pi, class_check = hmm.hmm_define_pi(data_orig, 'label_num')
A_trans = hmm.hmm_define_trans(data_orig, 'label_num')

""" 
---------------------------------------------------------------------------------------------------
Cutting the data to a smaller frame for dev purposes
---------------------------------------------------------------------------------------------------
"""
# for dev, just use stat at a time
data = data_orig[stat][data_orig[stat] != -998].reset_index(drop=True)
true_labels = data_orig[data_orig[stat] != -998].reset_index(drop=True)
if cut_point > 0:
    data = data.iloc[0:cut_point]
    true_labels = true_labels.iloc[0:cut_point]

""" 
---------------------------------------------------------------------------------------------------
HMM - Forward, Backward, Gamma, and Viterbi
---------------------------------------------------------------------------------------------------
"""
# the lines below represent a run block for a single stat. This would be repeated for all stats
fwd_ll_new, alpha_new = hmm.hmm_forward(gmm_params, data, A_trans, pi, stat=stat)
bwd_ll_new, beta_new = hmm.hmm_backward(gmm_params, data, A_trans, pi, stat=stat)
z, gamma = hmm.hmm_gamma(alpha=alpha_new, beta=beta_new, n=len(data))
# pi = hmm.hmm_update_pi(z)
# A_trans = hmm.hmm_update_trans(z)
v_path = hmm.hmm_viterbi(gmm_params, data, a=A_trans, pi_viterbi=pi, stat=stat)
pi = np.exp(pi)
# add the predicted viterbi path to the true labels for comparison (need to also add gamma/prob for each class)
true_labels['pred_class'] = v_path

""" 
---------------------------------------------------------------------------------------------------
Histogram - need to make a function
---------------------------------------------------------------------------------------------------
"""
plt.figure(figsize=(12, 6))
plt.hist(data, density=True, bins=75, color='dodgerblue', alpha=0.6, label=stat)
plt.legend(loc='upper left')
plt.xlabel('values')
plt.ylabel('density')
plt.title(stat + ' Distribution')
plt.show()

""" 
---------------------------------------------------------------------------------------------------
Confusion Matrix - need to make a function
---------------------------------------------------------------------------------------------------
"""
path_actual = true_labels['label_num']
path_pred = v_path

target_names = gmm_params['class'].unique()
# target_names = ['Neutral', 'Sweep']
fig, axs = plt.subplots(1,2, figsize=(14, 6))
cm1 = confusion_matrix(path_actual, path_pred, normalize='true')
cm2 = confusion_matrix(path_actual, path_pred)

axs[0].set_title(stat + ' Normalized')
axs[1].set_title(stat + ' Counts')
ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=target_names).plot(
    include_values=True, ax=axs[0], cmap='cividis')
ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=target_names).plot(
    include_values=True, ax=axs[1], cmap='cividis')
fig.tight_layout()
plt.show()

""" 
---------------------------------------------------------------------------------------------------
ROC - need to make a function [macro averaging]
---------------------------------------------------------------------------------------------------
"""
# Unique classes
classes = gmm_params['class'].unique()
colors = ['magenta', 'dodgerblue', 'darkviolet', 'blue']  # enough for four classes

# dummy var for HMM
y_onehot_test = pd.get_dummies(true_labels['label_num'], dtype=int)
# dummy var for SWIFr
y_onehot_test2 = pd.get_dummies(swfr_classified['label_num'], dtype=int)

# initialize figure
plt.figure(figsize=(9, 7))
for i in range(len(classes)):  # HMM ROC Curve Loop
    fpr, tpr, thresh = roc_curve(y_onehot_test.loc[:, i], np.transpose(gamma[i]), pos_label=1)
    auc = roc_auc_score(y_onehot_test.loc[:, i], np.transpose(gamma[i]))
    plt.plot(fpr, tpr, color=colors[i], label=f'HMM {classes[i]} vs Rest (AUC) = ' + str(round(auc, 2)))

for i in range(len(classes)):  # SWIFr ROC Curve Loop
    swfr_name = 'P(' + classes[i] + ')'
    fpr, tpr, thresh = roc_curve(y_onehot_test2.loc[:, i], swfr_classified[swfr_name], pos_label=1)
    auc = roc_auc_score(y_onehot_test2.loc[:, i], swfr_classified[swfr_name])
    plt.plot(fpr, tpr, linestyle='dashed', color=colors[i], label=f'SWIFr {classes[i]} vs Rest (AUC) = ' + str(round(auc, 2)))

# plot the chance curve
plt.plot(np.linspace(0, 1, 50 ), np.linspace(0, 1, 50), color='black',
         linestyle='dashed', label='Chance Level (AUC) = 0.50')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Macro-Averaging, One-vs-Rest ROC curves [{stat.upper()}]')
plt.legend()
plt.show()

""" 
---------------------------------------------------------------------------------------------------
ROC - need to make a function [micro averaging] - just used ravel to flatten the arrays
---------------------------------------------------------------------------------------------------
"""
# micro averaging the HMM data
plt.figure(figsize=(9, 7))
# flatten the data
cols = y_onehot_test.columns.tolist()
onehot_test_flat = y_onehot_test[cols].to_numpy().flatten(order='F')
fpr, tpr, thresh = roc_curve(onehot_test_flat, gamma.ravel(order='C'), pos_label=1)
auc = roc_auc_score(onehot_test_flat, gamma.ravel(order='C'))
plt.plot(fpr, tpr, color=colors[0], label='HMM One vs Rest (AUC) = ' + str(round(auc, 2)))

# micro averaging the swifr data
# flatten the swifr data by neutral, sweep, and link
swfr_names = ['P(' + i + ')' for i in classes]
swfr_flat = swfr_classified[swfr_names].to_numpy().flatten(order='F')
cols = y_onehot_test2.columns.tolist()
onehot_test_flat2 = y_onehot_test2[cols].to_numpy().flatten(order='F')

fpr1, tpr1, thresh1 = roc_curve(onehot_test_flat2, swfr_flat, pos_label=1)
auc1 = roc_auc_score(onehot_test_flat2, swfr_flat)
plt.plot(fpr1, tpr1, linestyle='dashed', color=colors[0],
         label='SWIFr One vs Rest (AUC) = ' + str(round(auc1, 2)))
# chance curve
plt.plot(np.linspace(0, 1, 50 ), np.linspace(0, 1, 50), color='black',
         linestyle='dashed', label='Chance Level (AUC) = 0.50')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Micro-Averaging, One-vs-Rest ROC curves [{stat.upper()}]')
plt.legend()
plt.show()