import hmm_funcs as hmm
import numpy as np
import pandas as pd
from tqdm import tqdm
from sqlalchemy import create_engine

"""
The purpose of this script is to run the HMM and save the output_db to a database. This database will then be used to
generate visualizations (primarily for the purposes of supporting the Results section of my thesis). 
"""

update_db = input("Would you like to update the DB, 'yes' or 'no': ")
#update_db = 'yes'
db_name = 'hmm_predictions.db'
hmm_table_name = 'hmm_predictions'
sbt_table_name = 'sbt_prediction_'  # stochastic backtrace (sbt)
db_path = 'C:/Users/scott/PycharmProjects/SWIFr_sdm/SWIFr_HMM/hmm/output_db/'

if update_db == 'yes':
    engine = create_engine('sqlite:///' + db_path + db_name, echo=False)

""" 
---------------------------------------------------------------------------------------------------
Path to data
    - gmm_path --> path to the GMMs learned during SWIFr train
    - test_path --> path to the test data that is to be evaluated by HMM (unseen data)
    - train_path --> path to the training data was used by SWIFr (but has been concatenated). The
    train_path data is used ONLY to define the pi and A_trans values. 
---------------------------------------------------------------------------------------------------
"""
state_count = 2  # 3 states implies neutral, sweep, and link; 2 states implies neutral and sweep
cut_point = 0  # set to zero if all data is to be used
stoch_sims = 100

if state_count == 2:
    gmm_path = '../../swifr_pkg/test_data/thesis_2class/'
    test_path = '../../swifr_pkg/test_data/thesis_2class_test/test/test'
    train_path = '../../../population_simulations/SWIFr_data_prep/simulations_4_swifr/2_class/train/train'

elif state_count == 3:
    gmm_path = '../../swifr_pkg/test_data/thesis_3class/'
    test_path = '../../swifr_pkg/test_data/thesis_3class_test/test/test'
    train_path = '../../../population_simulations/SWIFr_data_prep/simulations_4_swifr/3_class/train/train'

elif state_count == 4:
    gmm_path = '../../swifr_pkg/test_data/thesis_4class/'
    test_path = '../../swifr_pkg/test_data/thesis_4class_test/test/test'
    train_path = '../../../population_simulations/SWIFr_data_prep/simulations_4_swifr/4_class/train/train'


""" 
---------------------------------------------------------------------------------------------------
Load the GMM params from SWIFr_train
---------------------------------------------------------------------------------------------------
"""
gmm_params = hmm.hmm_init_params(gmm_path)
classes = gmm_params['class'].unique()
stats = gmm_params['stat'].unique()


""" 
---------------------------------------------------------------------------------------------------
Load the test data and the data classified by SWIFr GMM
---------------------------------------------------------------------------------------------------
"""
# get test data to be evaluated by hmm
test_orig = hmm.hmm_get_data(test_path).reset_index(drop=False)
test_orig = test_orig.rename(columns={'index': 'idx_key'})  # this will be used for later merge
# get the training data to be used ONLY to define the pi and A_trans values
train_orig = hmm.hmm_get_data(train_path).reset_index(drop=False)
train_orig = train_orig.rename(columns={'index': 'idx_key'})

"""
---------------------------------------------------------------------------------------------------
Create new column using ints for labels (i.e., replacing, neutral, sweep, etc.)
---------------------------------------------------------------------------------------------------
"""
# data_orig is the test data
test_orig['label_num'] = test_orig['label']
train_orig['label_num'] = train_orig['label']

for c in range(len(classes)):
    print("class: ", classes[c])
    test_orig['label_num'] = test_orig['label_num'].replace(to_replace=classes[c], value=c)
    train_orig['label_num'] = train_orig['label_num'].replace(to_replace=classes[c], value=c)

""" 
---------------------------------------------------------------------------------------------------
Initiate Pi and the transition matrix
---------------------------------------------------------------------------------------------------
"""
# for now, I will define the pi vector using the label_num
pi, class_check = hmm.hmm_define_pi(train_orig, 'label_num')
A_trans = hmm.hmm_define_trans(train_orig, 'label_num')
print('Pi: ', pi)
print('A (transition matrix): \n', A_trans)
# training data is no longer needed, delete to free space.
del[train_orig]

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
    test = test_orig[stat][test_orig[stat] != -998].reset_index(drop=False)  # drop nans first
    test = test.rename(columns={'index': 'idx_key'})  # rename the index column to use for later joins
    if cut_point > 0:  # cut the data to a smaller frame to run faster (dev only)
        test = test.iloc[0:cut_point]

    """ 
    ---------------------------------------------------------------------------------------------------
    HMM - Forward, Backward, Gamma, and Viterbi
    ---------------------------------------------------------------------------------------------------
    """
    # the lines below represent a run block for a single stat. This would be repeated for all stats
    gmm_params_ = gmm_params[gmm_params['stat'] == stat].reset_index(drop=True)  # limit gmm params to current stat
    # fwd_ll_new, alpha_new = hmm.hmm_forward(gmm_params_, data[stat], A_trans, pi, stat=stat)
    delta = hmm.hmm_forward_2(gmm_params, test[stat], np.copy(A_trans), np.copy(pi), stat=stat)
    _, beta_new = hmm.hmm_backward(gmm_params_, test[stat], A_trans, pi, stat=stat)
    _, gamma = hmm.hmm_gamma(alpha=delta, beta=beta_new, n=len(test))
    v_path, _, v_delta = hmm.hmm_viterbi(gmm_params_, test[stat], a=np.copy(A_trans), pi_viterbi=np.copy(pi), stat=stat)
    # add the predicted viterbi path to the true labels for comparison (need to also add gamma/prob for each class)
    test[f'viterbi_class_{stat}'] = v_path
    # add the gamma values as probabilities for each stat and class
    temp_cols = []
    for g in range(len(gamma)):
        temp_cols.append(f'P({classes[g]}_{stat})')
        test[f'P({classes[g]}_{stat})'] = gamma[g]

    data_cols = ['idx_key', f'viterbi_class_{stat}'] + temp_cols
    test_orig = pd.merge(test_orig, test[data_cols], on='idx_key', how='left')
# save the data
if update_db == 'yes':
    test_orig.to_sql(f"{hmm_table_name}_{state_count}class", con=engine, if_exists='replace')
"""
Stochastic Backtrace loop
"""
for stat in stats:
    print(f'Running stochastic backtrace on {stat}')
    test = test_orig[stat][test_orig[stat] != -998].reset_index(drop=False)  # drop nans first
    test = test.rename(columns={'index': 'idx_key'})  # rename the index column to use for later joins
    if cut_point > 0:  # cut the data to a smaller frame to run faster (dev only)
        test = test.iloc[0:cut_point]

    delta = hmm.hmm_forward_2(gmm_params, test[stat], np.copy(A_trans), np.copy(pi), stat=stat)
    for i in tqdm(range(stoch_sims)):
        if i == 0:
            _, sb_path = hmm.stochastic_backtrace(gmm_params, np.copy(A_trans), np.copy(delta))
        else:
            _, temp = hmm.stochastic_backtrace(gmm_params, np.copy(A_trans), np.copy(delta))
            sb_path = np.vstack((sb_path, temp))

    sb_path_df = pd.DataFrame(sb_path.T, columns=['sb_' + str(i) for i in range(stoch_sims)])
    sb_path_df['idx_key'] = test['idx_key']
    sb_path_df = pd.merge(sb_path_df, test_orig, on='idx_key', how='right')
    # save stochastic backtrace path
    if update_db == 'yes':
        sb_path_df.to_sql(f"{sbt_table_name}{stat}_{state_count}class", con= engine, if_exists='replace')

