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
cut_point = 30000  # set to zero if all data is to be used

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
classes = gmm_params['class'].unique()
stats = gmm_params['stat'].unique()
# gmm_params = gmm_params[gmm_params['stat'] == stat].reset_index(drop=True)  # limit to one stat for now

""" 
---------------------------------------------------------------------------------------------------
Load the test data and the data classified by SWIFr GMM
---------------------------------------------------------------------------------------------------
"""
# this will need to be a 'get' function, but keep it external for now
data_orig = hmm.hmm_get_data(data_path).reset_index(drop=False)
data_orig = data_orig.rename(columns={'index': 'idx_key'})  # this will be used for later merge
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
for stat in stats:
    print(f'Running HMM on {stat}')
    # for dev, just use stat at a time
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
    pi = np.exp(pi)
    # add the predicted viterbi path to the true labels for comparison (need to also add gamma/prob for each class)
    data[f'pred_class_{stat}'] = v_path
    # add the gamma values as probabilities for each stat and class
    temp_cols = []
    for g in range(len(gamma)):
        temp_cols.append(f'P({stat}_{classes[g]})')
        data[f'P({stat}_{classes[g]})'] = gamma[g]

    data_cols = ['idx_key', f'pred_class_{stat}'] + temp_cols
    data_orig = pd.merge(data_orig, data[data_cols], on='idx_key', how='left')

print('done')