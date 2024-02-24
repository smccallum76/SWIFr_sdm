import hmm_funcs as hmm
import numpy as np
import pandas as pd
from sklearn import mixture
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay, roc_curve, roc_auc_score

""" Path to data and params from GMM """
stat = 'xpehh'  # fst is problematic
# swifr_path = '../../swifr_pkg/test_data/simulations_4_swifr_2class/'
# data_path = '../../swifr_pkg/test_data/simulations_4_swifr_test_2class/test/test'
swifr_path = '../../swifr_pkg/test_data/simulations_4_swifr/'
data_path = '../../swifr_pkg/test_data/simulations_4_swifr_test/test/test'
gmm_params = hmm.hmm_init_params(swifr_path)
gmm_params = gmm_params[gmm_params['stat'] == stat].reset_index(drop=True)  # limit to one stat for now

""" Path to data and data load (external for now) """
# this will need to be a 'get' function, but keep it external for now
data_orig = pd.read_table(data_path, sep='\t')
# import swifr data that has been classified using gmm
swfr_classified = pd.read_table(data_path + '_classified', sep='\t')
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

conditions = [
            swfr_classified['label'] == 'neutral',  # 0
            swfr_classified['label'] == 'link_left',  # 2
            swfr_classified['label'] == 'link_right',  # 2
            swfr_classified['label'] == 'sweep'  # 1
            ]
choices = [0, 2, 2, 1]  # 3 classes
# choices = [0, 0, 0, 1]  # 2 classes
swfr_classified['label_num'] = np.select(conditions, choices, default=-998)
# for now I will define the pi vector using the label_num, but this is only b/c the data labels are not a match
# with the hmm classes (e.g., link_left and link_right are being defined as link)
pi, class_check = hmm.hmm_define_pi(data_orig, 'label_num')
A_trans = hmm.hmm_define_trans(data_orig, 'label_num')

# for dev, just use stat at a time
data = data_orig[stat][data_orig[stat] != -998].reset_index(drop=True)
true_labels = data_orig[data_orig[stat] != -998].reset_index(drop=True)
cut_point = 60000
data = data.iloc[0:cut_point]
true_labels = true_labels.iloc[0:cut_point]

for i in range(5):
    # the lines below represent a run block for a single stat. This would be repeated for all stats
    fwd_ll_new, alpha_new = hmm.hmm_forward(gmm_params, data, A_trans, pi, stat=stat)
    bwd_ll_new, beta_new = hmm.hmm_backward(gmm_params, data, A_trans, pi, stat=stat)
    z, gamma = hmm.hmm_gamma(alpha=alpha_new, beta=beta_new, n=len(data))
    pi = hmm.hmm_update_pi(z)
    A_trans = hmm.hmm_update_trans(z)
    print("Pi: ", pi)
    print("Pi sum: ", np.sum(pi))
    print("A: ", A_trans)
    print("A sum: ", np.sum(A_trans, axis=1))


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

''' Histogram '''
plt.figure(figsize=(12, 6))
plt.hist(data, density=True, bins=75, color='black', alpha=0.2, label=stat)
plt.legend(loc='upper left')
plt.xlabel('values')
plt.ylabel('density')
plt.title(stat + ' Distribution')
plt.show()

''' CONFUSION MATRIX '''

path_actual = true_labels['label_num']
path_pred = v_path

target_names = ['Neutral', 'Sweep', 'link']
# target_names = ['Neutral', 'Sweep']
fig, axs = plt.subplots(1,2, figsize=(14, 6))
cm1 = confusion_matrix(path_actual, path_pred, normalize='true')
cm2 = confusion_matrix(path_actual, path_pred)

axs[0].set_title(stat + ' Normalized')
axs[1].set_title(stat + ' Counts')
ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=target_names).plot(
    include_values=True, ax=axs[0])
ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=target_names).plot(
    include_values=True, ax=axs[1])
fig.tight_layout()
plt.show()

''' ROC '''
label_binarizer = LabelBinarizer().fit(true_labels['label_num'])
# y_onehot_test will be 3 columns of dummy vars with a 1 in the true instance
# y_onehot_test are the true classes, which we will compare with the gamma values
y_onehot_test = label_binarizer.transform(true_labels['label_num'])

label_binarizer2 = LabelBinarizer().fit(swfr_classified['label_num'])
y_onehot_test2 = label_binarizer.transform(swfr_classified['label_num'])

# roc curve calculates the roc values for a specific comparison

fpr0, tpr0, thresh0 = roc_curve(y_onehot_test[:, 0], np.transpose(gamma[0]), pos_label=1)
fpr1, tpr1, thresh1 = roc_curve(y_onehot_test[:, 1], np.transpose(gamma[1]), pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(y_onehot_test[:, 2], np.transpose(gamma[2]), pos_label=1)
auc0 = roc_auc_score(y_onehot_test[:, 0], np.transpose(gamma[0]))
auc1 = roc_auc_score(y_onehot_test[:, 1], np.transpose(gamma[1]))
auc2 = roc_auc_score(y_onehot_test[:, 2], np.transpose(gamma[2]))

fpr3, tpr3, thresh3 = roc_curve(y_onehot_test2[:, 0], swfr_classified['P(neutral)'], pos_label=1)
fpr4, tpr4, thresh4 = roc_curve(y_onehot_test2[:, 1], swfr_classified['P(sweep)'], pos_label=1)
fpr5, tpr5, thresh5 = roc_curve(y_onehot_test2[:, 2], swfr_classified['P(link)'], pos_label=1)
auc3 = roc_auc_score(y_onehot_test2[:, 0], swfr_classified['P(neutral)'])
auc4 = roc_auc_score(y_onehot_test2[:, 1], swfr_classified['P(sweep)'])
auc5 = roc_auc_score(y_onehot_test2[:, 2], swfr_classified['P(link)'])

plt.figure()
plt.plot(fpr0, tpr0, color='magenta',  label='HMM Neutral vs Rest (AUC) = ' + str(round(auc0, 2)))
plt.plot(fpr1, tpr1, color='dodgerblue',  label='HMM Sweep vs Rest (AUC) = ' + str(round(auc1, 2)))
plt.plot(fpr2, tpr2, color='darkviolet',  label='HMM Link vs Rest (AUC) = ' + str(round(auc2, 2)))

plt.plot(fpr3, tpr3, linestyle='dashed', color='magenta',  label='SWIFr Neutral vs Rest (AUC) = ' + str(round(auc3, 2)))
plt.plot(fpr4, tpr4, linestyle='dashed', color='dodgerblue',  label='SWIFr Sweep vs Rest (AUC) = ' + str(round(auc4, 2)))
plt.plot(fpr5, tpr5, linestyle='dashed', color='darkviolet',  label='SWIFr Link vs Rest (AUC) = ' + str(round(auc5, 2)))

plt.plot(np.linspace(0, 1, 50 ), np.linspace(0, 1, 50), color='black',
         linestyle='dashed', label='Chance Level (AUC) = 0.50')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'One-vs-Rest ROC curves [{stat}]')
plt.legend()
plt.show()

print('done')
