import hmm_funcs as hmm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, roc_auc_score

""" 
---------------------------------------------------------------------------------------------------
Path to data
---------------------------------------------------------------------------------------------------
"""
state_count = 4  # 3 states implies neutral, sweep, and link; 2 states implies neutral and sweep
cut_point = 100000  # set to zero if all data is to be used

if state_count == 2:
    gmm_path = '../../swifr_pkg/test_data/simulations_4_swifr_2class/'
    data_path = '../../swifr_pkg/test_data/simulations_4_swifr_test_2class/test/test'
    swifr_path = '../../swifr_pkg/test_data/simulations_4_swifr_test_2class/test/test_classified'
elif state_count == 3:
    gmm_path = '../../swifr_pkg/test_data/simulations_4_swifr/'
    data_path = '../../swifr_pkg/test_data/simulations_4_swifr_test/test/test'
    swifr_path = '../../swifr_pkg/test_data/simulations_4_swifr_test/test/test_classified'
elif state_count == 4:
    gmm_path = '../../swifr_pkg/test_data/simulations_4_swifr_4class/'
    data_path = '../../swifr_pkg/test_data/simulations_4_swifr_test_4class/test/test'
    swifr_path = '../../swifr_pkg/test_data/simulations_4_swifr_test_4class/test/test_classified'
""" 
---------------------------------------------------------------------------------------------------
Load the GMM params from SWIFr_train
---------------------------------------------------------------------------------------------------
"""
gmm_params = hmm.hmm_init_params(gmm_path)
classes = gmm_params['class'].unique()
stats = gmm_params['stat'].unique()
stats = ['ihs_afr_std']  # overwriting the stats field for dev purposes

""" 
---------------------------------------------------------------------------------------------------
Load the test data and the data classified by SWIFr GMM
---------------------------------------------------------------------------------------------------
"""
# this will need to be a 'get' function, but keep it external for now
data_orig = hmm.hmm_get_data(data_path).reset_index(drop=False)
data_orig = data_orig.rename(columns={'index': 'idx_key'})  # this will be used for later merge
# include only rows with all stats not nan
# data_orig = data_orig[(data_orig['xpehh'] != -998) &
#                         (data_orig['ihs_afr_std'] != -998) &
#                         (data_orig['fst'] != -998)].reset_index(drop=False)
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
            data_orig['label'] == 'link_right',  # 3
            data_orig['label'] == 'sweep'  # 1
            ]
if state_count == 3:
    choices = [0, 2, 2, 1]  # 3 classes
elif state_count == 2:
    choices = [0, 0, 0, 1]  # 2 classes
elif state_count == 4:
    # choices = [0, 2, 3, 1]  # 2 classes
    choices = [0, 1, 2, 3]  # 2 classes

data_orig['label_num'] = np.select(conditions, choices, default=-998)

# repeat the above, but for the swfr_classified (could tech use the same since the indexing is the same)
conditions = [
            swfr_classified['label'] == 'neutral',  # 0
            swfr_classified['label'] == 'link_left',  # 2
            swfr_classified['label'] == 'link_right',  # 3
            swfr_classified['label'] == 'sweep'  # 1
            ]
if state_count == 3:
    choices = [0, 2, 2, 1]  # 3 classes
elif state_count == 2:
    choices = [0, 0, 0, 1]  # 2 classes
elif state_count == 4:
    # choices = [0, 2, 3, 1]  # 2 classes
    choices = [0, 1, 2, 3]  # 2 classes

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

for stat in stats:
    print(f'Running HMM on {stat}')
    # note that dropping nans one stat per loop implies results in a different set of data
    # for each sat (since each has its own set of nans). However, this can help overall b/c one
    # stat can bridge another stats "nan gap" thereby providing more continuous data.
    data = data_orig[stat][data_orig[stat] != -998].reset_index(drop=False)  # drop nans first
    data = data.rename(columns={'index': 'idx_key'})  # rename the index column to use for later joins
    if cut_point > 0:  # cut the data to a smaller frame to run faster (dev only)
        data = data.iloc[0:cut_point]

    """ 
    ---------------------------------------------------------------------------------------------------
    HMM - Forward, Backward, Gamma, and Viterbi
    ---------------------------------------------------------------------------------------------------
    """
    # the lines below represent a run block for a single stat. This would be repeated for all stats
    gmm_params_ = gmm_params[gmm_params['stat'] == stat].reset_index(drop=True)  # limit gmm params to current stat
    fwd_ll_new, alpha_new = hmm.hmm_forward(gmm_params_, data[stat], A_trans, pi, stat=stat)
    bwd_ll_new, beta_new = hmm.hmm_backward(gmm_params_, data[stat], A_trans, pi, stat=stat)
    z, gamma = hmm.hmm_gamma(alpha=alpha_new, beta=beta_new, n=len(data))
    # pi = hmm.hmm_update_pi(z)
    # A_trans = hmm.hmm_update_trans(z)
    v_path = hmm.hmm_viterbi(gmm_params_, data[stat], a=A_trans, pi_viterbi=pi, stat=stat)
    # need to figure out why viterbi updates the Pi and A_trans matrix outside of this function, but for now
    # return the values to the exp of the log.
    pi = np.exp(pi)
    A_trans = np.exp(A_trans)
    # add the predicted viterbi path to the true labels for comparison (need to also add gamma/prob for each class)
    data[f'pred_class_{stat}'] = v_path
    # add the gamma values as probabilities for each stat and class
    temp_cols = []
    for g in range(len(gamma)):
        temp_cols.append(f'P({classes[g]}_{stat})')
        data[f'P({classes[g]}_{stat})'] = gamma[g]

    data_cols = ['idx_key', f'pred_class_{stat}'] + temp_cols
    data_orig = pd.merge(data_orig, data[data_cols], on='idx_key', how='left')

# drop nans where all stats are null (we can't make any prediction here without imputation)
# nans will be determined using the 'pred_class_<stat>' columns. The actual stat columns could be used
cols = data_orig.filter(regex=("pred_class_*")).columns.tolist()
data_noNans = data_orig.dropna(subset=cols, how='all').reset_index(drop=True)

""" 
---------------------------------------------------------------------------------------------------
Plot -- Path and stat comparison [flashlight plot]
---------------------------------------------------------------------------------------------------
"""
xs = np.arange(0, len(data_noNans), 1)
cmap = mpl.colormaps['viridis']
legend_colors = cmap(np.linspace(0, 1, len(pi)))

fig = plt.figure(figsize=(16, 7))
gs = fig.add_gridspec(3, hspace=0)
axs = gs.subplots(sharex=True, sharey=False)
fig.suptitle(f'Actual Path, Predicted Path, and {stats[0]}')
axs[0].plot(xs, data_noNans['label_num'], color='black')
axs[0].scatter(xs, data_noNans['label_num'], c=data_noNans['label_num'],
               cmap='viridis', edgecolor='none', s=30)

axs[1].plot(xs, data_noNans[f'pred_class_{stats[0]}'], color='black')
axs[1].scatter(xs, data_noNans[f'pred_class_{stats[0]}'], c=data_noNans[f'pred_class_{stats[0]}'],
               cmap='viridis', edgecolor='none', s=30)

axs[2].scatter(xs, data_noNans[f'{stats[0]}'], c=data_noNans[f'{stats[0]}'], cmap='viridis', edgecolor='none', s=3)

axs[0].set(ylabel='Actual State')
axs[1].set(ylabel='Predicted State')
axs[2].set(ylabel=f'Value {stats[0]}')
# Hide x labels and tick labels for all but bottom plot.
for ax in axs:
    ax.label_outer()

legend_elements = [Line2D([0], [0], marker='o', color='w', label='0: Neutral',
                          markerfacecolor=legend_colors[0], markersize=15),
                   Line2D([0], [0], marker='o', color='w', label='1: Link_Left',
                          markerfacecolor=legend_colors[1], markersize=15),
                   Line2D([0], [0], marker='o', color='w', label='2: Link_Right',
                          markerfacecolor=legend_colors[2], markersize=15),
                   Line2D([0], [0], marker='o', color='w', label='3: Sweep',
                          markerfacecolor=legend_colors[3], markersize=15)]

axs[0].legend(handles=legend_elements, loc='upper left')
axs[1].legend(handles=legend_elements, loc='upper left')
plt.show()

""" 
---------------------------------------------------------------------------------------------------
Confusion Matrix - need to make a function
---------------------------------------------------------------------------------------------------
"""
# path_actual = data_noNans['label_num']
# path_pred = data_noNans[f'pred_class_{stats[0]}']  # for dev purposes only
#
# fig, axs = plt.subplots(1,2, figsize=(14, 6))
# cm1 = confusion_matrix(path_actual, path_pred, normalize='true')
# cm2 = confusion_matrix(path_actual, path_pred)
#
# axs[0].set_title(f'Path using {stats[0]} - Normalized')
# axs[1].set_title(f'Path using {stats[0]} - Counts')
# ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=classes).plot(
#     include_values=True, ax=axs[0], cmap='cividis')
# ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=classes).plot(
#     include_values=True, ax=axs[1], cmap='cividis')
# fig.tight_layout()
# plt.show()



# """
# ---------------------------------------------------------------------------------------------------
# ROC - need to make a function [macro averaging] - using average of all probs
# ---------------------------------------------------------------------------------------------------
# """
# # Unique classes
# colors = ['magenta', 'dodgerblue', 'darkviolet', 'blue']  # enough for four classes
#
# # dummy var for HMM
# y_onehot_test = pd.get_dummies(data_noNans['label_num'], dtype=int)
# # dummy var for SWIFr
# y_onehot_test2 = pd.get_dummies(swfr_classified['label_num'], dtype=int)
#
# # initialize figure
# plt.figure(figsize=(9, 7))
# for i in range(len(classes)):  # HMM ROC Curve Loop
#     fpr, tpr, thresh = roc_curve(y_onehot_test.loc[:, i], data_noNans[f"P({classes[i]}_All_avg)"], pos_label=1)
#     auc = roc_auc_score(y_onehot_test.loc[:, i], data_noNans[f"P({classes[i]}_All_avg)"])
#     plt.plot(fpr, tpr, color=colors[i], label=f'HMM {classes[i]} vs Rest (AUC) = ' + str(round(auc, 2)))
#
# for i in range(len(classes)):  # SWIFr ROC Curve Loop
#     swfr_name = 'P(' + classes[i] + ')'
#     fpr, tpr, thresh = roc_curve(y_onehot_test2.loc[:, i], swfr_classified[swfr_name], pos_label=1)
#     auc = roc_auc_score(y_onehot_test2.loc[:, i], swfr_classified[swfr_name])
#     plt.plot(fpr, tpr, linestyle='dashed', color=colors[i], label=f'SWIFr {classes[i]} vs Rest (AUC) = ' + str(round(auc, 2)))
#
# # plot the chance curve
# plt.plot(np.linspace(0, 1, 50 ), np.linspace(0, 1, 50), color='black',
#          linestyle='dashed', label='Chance Level (AUC) = 0.50')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title(f'Macro-Averaging, One-vs-Rest ROC curves [All Stats - AVG]')
# plt.legend()
# plt.show()
#
# """
# ---------------------------------------------------------------------------------------------------
# ROC - need to make a function [macro averaging] - using max of all probs
# ---------------------------------------------------------------------------------------------------
# """
# # Unique classes
# colors = ['magenta', 'dodgerblue', 'darkviolet', 'blue']  # enough for four classes
#
# # dummy var for HMM
# y_onehot_test = pd.get_dummies(data_noNans['label_num'], dtype=int)
# # dummy var for SWIFr
# y_onehot_test2 = pd.get_dummies(swfr_classified['label_num'], dtype=int)
#
# # initialize figure
# plt.figure(figsize=(9, 7))
# for i in range(len(classes)):  # HMM ROC Curve Loop
#     fpr, tpr, thresh = roc_curve(y_onehot_test.loc[:, i], data_noNans[f"P({classes[i]}_All_max)"], pos_label=1)
#     auc = roc_auc_score(y_onehot_test.loc[:, i], data_noNans[f"P({classes[i]}_All_max)"])
#     plt.plot(fpr, tpr, color=colors[i], label=f'HMM {classes[i]} vs Rest (AUC) = ' + str(round(auc, 2)))
#
# for i in range(len(classes)):  # SWIFr ROC Curve Loop
#     swfr_name = 'P(' + classes[i] + ')'
#     fpr, tpr, thresh = roc_curve(y_onehot_test2.loc[:, i], swfr_classified[swfr_name], pos_label=1)
#     auc = roc_auc_score(y_onehot_test2.loc[:, i], swfr_classified[swfr_name])
#     plt.plot(fpr, tpr, linestyle='dashed', color=colors[i], label=f'SWIFr {classes[i]} vs Rest (AUC) = ' + str(round(auc, 2)))
#
# # plot the chance curve
# plt.plot(np.linspace(0, 1, 50 ), np.linspace(0, 1, 50), color='black',
#          linestyle='dashed', label='Chance Level (AUC) = 0.50')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title(f'Macro-Averaging, One-vs-Rest ROC curves [All Stats - MAX]')
# plt.legend()
# plt.show()
