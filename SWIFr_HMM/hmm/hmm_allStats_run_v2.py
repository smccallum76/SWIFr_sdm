import hmm_funcs as hmm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
from tqdm import tqdm
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
stoch_sims = 5

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
stats = ['xpehh']  # overwriting the stats field for dev purposes

if stats[0] == 'ihs_afr_std':
    swifr_path_1stat = '../../swifr_pkg/test_data/simulations_4_swifr_test_4class_ihs/test/test_classified'
elif stats[0] == 'xpehh':
    swifr_path_1stat = '../../swifr_pkg/test_data/simulations_4_swifr_test_4class_xpehh/test/test_classified'
elif stats[0] == 'fst':
    swifr_path_1stat = '../../swifr_pkg/test_data/simulations_4_swifr_test_4class_fst/test/test_classified'

""" 
---------------------------------------------------------------------------------------------------
Load the test data and the data classified by SWIFr GMM
---------------------------------------------------------------------------------------------------
"""
# this will need to be a 'get' function, but keep it external for now
data_orig = hmm.hmm_get_data(data_path).reset_index(drop=False)
data_orig = data_orig.rename(columns={'index': 'idx_key'})  # this will be used for later merge
swfr_classified = hmm.hmm_get_data(swifr_path)
'''
SWIFr scenarios below were trained and run on only one stat at a time. This was done for direct comparison
with the HMM trials using only one stat at a time.
'''
swfr_classified_1stat = hmm.hmm_get_data(swifr_path_1stat)
swfr_classified_1stat = swfr_classified_1stat[swfr_classified_1stat[f'{stats[0]}'] != -998].reset_index(drop=True)


# convert swifr probabilities into a classification code
swifr_cols = [
                'P(neutral)',
                'P(link_left)',
                'P(link_right)',
                'P(sweep)'
            ]
# find the largest prob
swfr_classified_1stat['swfr_class'] = swfr_classified_1stat[swifr_cols].idxmax(axis='columns')
# create a new class column and then replace the strings with numbers
swfr_classified_1stat['swfr_class_num'] = swfr_classified_1stat['swfr_class']
swfr_classified_1stat['swfr_class_num'] = swfr_classified_1stat['swfr_class_num'].replace(swifr_cols, [0, 1, 2, 3])

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
    # choices = [0, 1, 2, 3]  # 4
    choices = [0, 2, 3, 1]

data_orig['label_num'] = np.select(conditions, choices, default=-998)

# repeat the above, but for the swfr_classified (could tech use the same since the indexing is the same)
conditions = [
            swfr_classified['label'] == 'neutral',  # 0
            swfr_classified['label'] == 'link_left',  # 1
            swfr_classified['label'] == 'link_right',  # 2
            swfr_classified['label'] == 'sweep'  # 3
            ]
if state_count == 3:
    choices = [0, 2, 2, 1]  # 3 classes
elif state_count == 2:
    choices = [0, 0, 0, 1]  # 2 classes
elif state_count == 4:
    # choices = [0, 1, 2, 3]  # 2 classes
    choices = [0, 2, 3, 1]

swfr_classified['label_num'] = np.select(conditions, choices, default=-998)

""" 
---------------------------------------------------------------------------------------------------
Initiate Pi and the transition matrix
---------------------------------------------------------------------------------------------------
"""
# for now I will define the pi vector using the label_num
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
        swfr_classified_1stat = swfr_classified_1stat.iloc[0:cut_point]

    """ 
    ---------------------------------------------------------------------------------------------------
    HMM - Forward, Backward, Gamma, and Viterbi
    ---------------------------------------------------------------------------------------------------
    """
    # the lines below represent a run block for a single stat. This would be repeated for all stats
    gmm_params_ = gmm_params[gmm_params['stat'] == stat].reset_index(drop=True)  # limit gmm params to current stat
    # fwd_ll_new, alpha_new = hmm.hmm_forward(gmm_params_, data[stat], A_trans, pi, stat=stat)
    delta = hmm.hmm_forward_2(gmm_params, data[stat], np.copy(A_trans), np.copy(pi), stat=stat)
    # bwd_ll_new, beta_new = hmm.hmm_backward(gmm_params_, data[stat], A_trans, pi, stat=stat)
    # z, gamma = hmm.hmm_gamma(alpha=alpha_new, beta=beta_new, n=len(data))
    v_path, v_delta = hmm.hmm_viterbi(gmm_params_, data[stat], a=np.copy(A_trans), pi_viterbi=np.copy(pi), stat=stat)

    # add the predicted viterbi path to the true labels for comparison (need to also add gamma/prob for each class)
    data[f'viterbi_class_{stat}'] = v_path
    # add the gamma values as probabilities for each stat and class
    temp_cols = []
    # for g in range(len(gamma)):
    #     temp_cols.append(f'P({classes[g]}_{stat})')
    #     data[f'P({classes[g]}_{stat})'] = gamma[g]

    data_cols = ['idx_key', f'viterbi_class_{stat}'] + temp_cols
    data_orig = pd.merge(data_orig, data[data_cols], on='idx_key', how='left')
# drop nans where all stats are null (we can't make any prediction here without imputation)
# nans will be determined using the 'pred_class_<stat>' columns. The actual stat columns could be used
viterbi_cols = data_orig.filter(regex=("viterbi_class_*")).columns.tolist()
viterbi_noNans = data_orig.dropna(subset=viterbi_cols, how='all').reset_index(drop=True)
"""
Stochastic Backtrace loop
"""

for i in tqdm(range(stoch_sims)):
    if i == 0:
        sb_path = hmm.stochastic_backtrace(gmm_params, np.copy(A_trans), np.copy(delta))
    else:
        temp = hmm.stochastic_backtrace(gmm_params, np.copy(A_trans), np.copy(delta))
        sb_path = np.vstack((sb_path, temp))

sb_path_df = pd.DataFrame(sb_path.T, columns=['sb_' + str(i) for i in range(stoch_sims)])
sb_path_df['idx_key'] = data['idx_key']
sb_path_df = pd.merge(sb_path_df, data_orig, on='idx_key', how='left')
# save stochastic backtrace path
sb_path_df.to_csv(f'output/stochastic_bt_{stoch_sims}_sims_{stats[0]}.csv', index=False)
""" 
---------------------------------------------------------------------------------------------------
Signal Stack
Count of all non-neutral class predictions (i.e., link_left, link_right, OR sweep).
The assumption non-neutral events are more likely when all the stats indicate a non-neutral event:
    - 0 = all stats are nan or neutral
    - 1 = all stats indicate a non-neutral event (full fold)
    - 0<x<1 = Some fraction of the stats indicate a non-neutral event
---------------------------------------------------------------------------------------------------
"""
viterbi_noNans['viterbi_class_nonNeut'] = np.nansum(viterbi_noNans[viterbi_cols] > 0, axis=1) / len(stats)

""" 
---------------------------------------------------------------------------------------------------
Plot -- Path and stat comparison [flashlight plot]
---------------------------------------------------------------------------------------------------
"""
xs = np.arange(0, len(viterbi_noNans), 1)
xs2 = np.arange(0, len(swfr_classified_1stat), 1)
cmap = mpl.colormaps['viridis']
legend_colors = cmap(np.linspace(0, 1, len(pi)))

fig = plt.figure(figsize=(18, 7))
gs = fig.add_gridspec(5, hspace=0)
axs = gs.subplots(sharex=True, sharey=False)
fig.suptitle(f'Actual Path, Predicted Path, and {stats[0]}')
axs[0].plot(xs, viterbi_noNans['label_num'], color='black')
axs[0].scatter(xs, viterbi_noNans['label_num'], c=viterbi_noNans['label_num'],
               cmap='viridis', edgecolor='none', s=30)

# axs[1].plot(xs, data_noNans[f'viterbi_class_nonNeut'], color='black')
# axs[1].scatter(xs, data_noNans[f'viterbi_class_nonNeut'], c=data_noNans[f'viterbi_class_nonNeut'],
#                cmap='viridis', edgecolor='none', s=30)

axs[1].plot(xs, viterbi_noNans[f'viterbi_class_{stat}'], color='black')
axs[1].scatter(xs, viterbi_noNans[f'viterbi_class_{stat}'], c=viterbi_noNans[f'viterbi_class_{stat}'],
               cmap='viridis', edgecolor='none', s=30)
for i in range(len(sb_path[:,0])):
    axs[2].plot(xs, sb_path[i, :], color='lightgrey')
# axs[2].scatter(xs, sb_paths, c=sb_paths,cmap='viridis', edgecolor='none', s=30)

axs[3].plot(xs2, swfr_classified_1stat['swfr_class_num'], color='black')
axs[3].scatter(xs2, swfr_classified_1stat['swfr_class_num'], c=swfr_classified_1stat['swfr_class_num'],
               cmap='viridis', edgecolor='none', s=30)

axs[4].scatter(xs, viterbi_noNans[f'{stats[0]}'], c=viterbi_noNans[f'{stats[0]}'], cmap='viridis', edgecolor='none', s=3)

axs[0].set(ylabel='Actual State')
axs[1].set(ylabel='Viterbi Pred')
axs[2].set(ylabel='BackTrc Pred')
axs[3].set(ylabel='SWIFr Pred')
axs[4].set(ylabel=f'Value {stats[0]}')
# Hide x labels and tick labels for all but bottom plot.
for ax in axs:
    ax.label_outer()

legend_elements = [Line2D([0], [0], marker='o', color='w', label='0: Neutral',
                          markerfacecolor=legend_colors[0], markersize=15),
                   Line2D([0], [0], marker='o', color='w', label='1: Sweep',
                          markerfacecolor=legend_colors[1], markersize=15),
                   Line2D([0], [0], marker='o', color='w', label='2: Link_Left',
                          markerfacecolor=legend_colors[2], markersize=15),
                   Line2D([0], [0], marker='o', color='w', label='3: Link_Right',
                          markerfacecolor=legend_colors[3], markersize=15)]

axs[0].legend(handles=legend_elements, loc='upper left')
# axs[1].legend(handles=legend_elements, loc='upper left')
plt.show()

""" 
---------------------------------------------------------------------------------------------------
Density Signal Stack
- This is unfinished and not sure if it is worth the effort
- Probably need to think of a cleaner way to combine the stats
---------------------------------------------------------------------------------------------------
"""
signal = viterbi_noNans[['idx_key', 'snp_position'] + viterbi_cols + ['label_num', 'viterbi_class_nonNeut']]
signal = signal.sort_values(by='snp_position', ascending=True)
signal['snp_seconds'] = pd.to_datetime(signal['snp_position'], unit='s')
signal['v_path_count'] = np.nansum(signal[viterbi_cols] > 0, axis=1)
signal['v_path_density'] = signal['v_path_count'].rolling(150, center=True).sum()
""" 
---------------------------------------------------------------------------------------------------
Plot -- Path and stat comparison [flashlight plot]
---------------------------------------------------------------------------------------------------
"""
xs = np.arange(0, len(signal), 1)
cmap = mpl.colormaps['viridis']
legend_colors = cmap(np.linspace(0, 1, len(pi)))

fig = plt.figure(figsize=(16, 7))
gs = fig.add_gridspec(2, hspace=0)
axs = gs.subplots(sharex=True, sharey=False)
fig.suptitle(f'Actual Path, Predicted Path, and {stats[0]}')
axs[0].plot(xs, signal['label_num'], color='black')
axs[0].scatter(xs, signal['label_num'], c=viterbi_noNans['label_num'],
               cmap='viridis', edgecolor='none', s=30)

axs[1].plot(xs, signal['v_path_density'], color='black')
axs[1].scatter(xs, signal['v_path_density'], c=signal[f'viterbi_class_nonNeut'],
               cmap='viridis', edgecolor='none', s=30)

axs[0].set(ylabel='Actual State')
axs[1].set(ylabel='Predicted State')
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
# axs[1].legend(handles=legend_elements, loc='upper left')
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

