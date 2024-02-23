import hmm_funcs as hmm
import numpy as np
import pandas as pd
from sklearn import mixture
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay

""" Path to data and params from GMM """
stat = 'ihs_afr_std'  # fst is problematic
# swifr_path = '../../swifr_pkg/test_data/simulations_4_swifr_2class/'
# data_path = '../../swifr_pkg/test_data/simulations_4_swifr_test_2class/test/test'
swifr_path = '../../swifr_pkg/test_data/simulations_4_swifr/'
data_path = '../../swifr_pkg/test_data/simulations_4_swifr_test/test/test'
gmm_params = hmm.hmm_init_params(swifr_path)
gmm_params = gmm_params[gmm_params['stat'] == stat].reset_index(drop=True)  # limit to one stat for now

""" Path to data and data load (external for now) """
# this will need to be a 'get' function, but keep it external for now
data_orig = pd.read_table(data_path, sep='\t')
# data_labels = pd.DataFrame(data_orig['label'], columns=['label'])
# convert the row labels from a string to a numeric value
conditions = [
            data_orig['label'] == 'neutral',  # 0
            data_orig['label'] == 'link_left',  # 2
            data_orig['label'] == 'link_right',  # 2
            data_orig['label'] == 'sweep'  # 1
            ]
choices = [0, 2, 2, 1]  # 3 classes
# choices = [0, 0, 0, 1]  # 2 classes
data_orig['label_num'] = np.select(conditions, choices, default=-998)
# for now I will define the pi vector using the label_num, but this is only b/c the data labels are not a match
# with the hmm classes (e.g., link_left and link_right are being defined as link)
pi, class_check = hmm.hmm_define_pi(data_orig, 'label_num')
A_trans = hmm.hmm_define_trans(data_orig, 'label_num')

# the transition matrix below results in an identification of 7/9 of the sweep events for ihs_afr_std (3 classes).
# need to retain this matrix as it will be important in the grid search.
# a_list = [0.998, 1e-4, 0.002-1e-4,   1e-4, 1e-4, 1-2*1e-4,   0.01, 0.09, 0.9]
# A_trans = hmm.hmm_init_trans(a_list=a_list)

# for dev, just use stat at a time
data = data_orig[stat][data_orig[stat] != -998].reset_index(drop=True)
true_labels = data_orig[data_orig[stat] != -998].reset_index(drop=True)
cut_point = 60000
data = data.iloc[0:cut_point]
true_labels = true_labels.iloc[0:cut_point]


# the lines below represent a run block for a single stat. This would be repeated for all stats
fwd_ll_new, alpha_new = hmm.hmm_forward(gmm_params, data, A_trans, pi, stat=stat)
bwd_ll_new, beta_new = hmm.hmm_backward(gmm_params, data, A_trans, pi, stat=stat)
z, gamma = hmm.hmm_gamma(alpha=alpha_new, beta=beta_new, n=len(data))
v_path = hmm.hmm_viterbi(gmm_params, data, a=A_trans, pi_viterbi=pi, stat=stat)
pi = np.exp(pi)
print("Pi: ", pi)
print("Pi sum: ", np.sum(pi))
print("A: ", A_trans)
print("A sum: ", np.sum(A_trans, axis=1))


true_labels['pred_class'] = v_path
class_num = str(len(pi))
true_labels.to_csv('output/' + class_num + '_classes_' + stat + '_predictions.csv', index=False)

sweeps = true_labels[true_labels['label'] == 'sweep']
links = true_labels[true_labels['label_num'] == 2]

plt.figure(figsize=(10, 10))
plt.hist(data, density=True, bins=75, color='black', alpha=0.2, label='Data')
plt.legend(loc='upper left')
plt.xlabel('values')
plt.ylabel('density')
plt.show()

''' CONFUSION MATRIX '''
path_actual = true_labels['label_num']
path_pred = v_path

cm = confusion_matrix(path_actual, path_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Neutral', 'Sweep', 'link'])
disp.plot()
plt.show()