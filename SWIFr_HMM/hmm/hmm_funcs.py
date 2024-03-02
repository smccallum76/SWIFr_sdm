"""
First pass at building a Hidden Markov Model using the simple simulations data. This code will use portions of the
code from Stat Modeling Final.  The objective of this code is to identify the neutral and sweep events with min
amount of errors
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
import pickle
import os

def hmm_get_data(path, sep='\t'):
    data = pd.read_table(path, sep=sep)
    return data

def hmm_init_params(path):
    """
    Function to collect the classes, stats, and gmm names and return them as a df
    """
    # information below is a collection of lists that identifies the classes, stats, and names of the 1D gmm params
    gmm_param_list = hmm_get_swifr_param_names(path)
    class_list = hmm_get_swifr_classes(path)
    stat_list = hmm_get_swifr_stats(path)

    df = pd.DataFrame(columns=['stat', 'class', 'gmm_name', 'gmm_path'])
    count=0
    for c in class_list:
        for s in stat_list:
            df.loc[count, 'stat'] = s
            df.loc[count, 'class'] = c
            string = s + '_' + c
            for g in gmm_param_list:
                if string in g:
                    df.loc[count, 'gmm_name'] = g  # name of gmm
                    df.loc[count, 'gmm_path'] = path + 'AODE_params/' + g
                    # df.loc[count, 'gmm'] = pickle.load(open(path + 'AODE_params/' + g, 'rb'))  # actual gmm
            count += 1
    return df

def hmm_define_pi(data, label):
    """
    Function to calculate the fraction of each class that is present. The labels can be numeric or strings. The
    class label list is returned to ensure that the pi order matches the class order.
    """
    classes = np.sort(data[label].unique())
    pi_define = []
    class_label = []
    for c in classes:
        pi_define.append(len(data[data[label] == c]) / len(data))
        class_label.append(c)

    return pi_define, class_label

def hmm_define_trans(data, label):
    """
    This method uses a colum shift coupled with an element wise addition to update the transition matrix. If there
    exist 3 classes then the A matrix is 3x3. Therefore, a total of 9 shifts and addition operations are needed.
    The general workflow is as follows:
    1. Set up a loop to iterate through the transition matrix [i, j].
    2. The initial iteration will be at i=j or position 0,0.
    3. This results in the z[0] column first being shifted and then being added.
    4. The summation results in a value of 2 whenever the current state and the next state are the same (sunny to sunny)
    5. Counting the number of 2's in the column represents the number of transitions from one state to the next.
    6. At the next iteration of the loop i=0 and j=1, we are performing the same operation, but now the count of 2's
    represents the number of times we transitioned from state i=0 to state j=1.
    """

    # going to need a set of dummy columns, one column for each class. this will act as the z vectors below.
    # dummies returns a column for each class with a 1 if true and 0 otherwise
    z = pd.get_dummies(data[label], dtype=int)

    classes = z.columns
    a = np.empty(shape=(len(classes), len(classes)))
    for i in classes:
        for j in classes:
            # temp = z[i][0:-1] + z[j][1::]
            temp = z[i] + z[j].shift(periods=-1)
            class_count = len(temp[temp == 2])
            # the conditional below prevents zero entries in the transition matrix (which are problematic)
            # if class_count < 1:
            #     class_count = 1
            a[i, j] = class_count

    # normalize the 'a' matrix based on the total count of occurrences in each state
    a_sum = np.sum(a, axis=1)  # axis 1 is correct, we want to sum from left to right across the columns
    a = a / a_sum.reshape((len(a_sum), 1))

    return a
def hmm_init_trans(a_list):
    """
    Reshape a list of transition values into a square matrix
    :param a_list:
    List of transition values.
    Position 0 is transition from 0 to 0
    Position 1 is transition from 0 to 1
    Position 3 is transition from 1 to 0
    Position 4 is transition from 1 to 1...
    :return:
    The square matrix of transition values (A).
    """
    shape = (int(np.sqrt(len(a_list))), int(np.sqrt(len(a_list))))
    a = np.array(a_list)
    a = np.reshape(a, newshape=shape, order='C')
    return a

def hmm_norm_pdf(x, mu, sd):
    """
    Calculate the probability density based on a normal distribution
    :param x:
    Value to be evaluated
    :param mu:
    Mean of the distribution
    :param sd:
    Standard deviation of the distribution
    :return:
    Probability density of the value x
    """
    p = stats.norm.pdf(x=x,  loc=mu, scale=sd)
    return p

def hmm_forward(gmm_params, data, A_trans, pi, stat='xpehh'):
    # some initializations and settings
    n = len(data)
    A_new = A_trans

    classes = gmm_params['class'].unique()
    # create a collection of lists that can be used to store intermediate values
    bx = {}
    log_alpha = {}
    m_alpha = {}  # holds the max (log) alpha for each iteration
    exp_sum = {}
    for x in range(len(classes)):
        bx[x] = []
        log_alpha[x] = []
        m_alpha[x] = []
        exp_sum[x] = []

    """ build the matrix of pdf values for each class [may want to make this its own function] """
    for c in range(len(classes)):
        # extract the path to the correct gmm based on the stat and the class
        gmm_current_path = gmm_params['gmm_path'][(gmm_params['stat'] == stat) &
                                                  (gmm_params['class'] == classes[c])].reset_index(drop=True)
        # load the correct gmm based on the correct path
        gmm = hmm_pickle_load(gmm_current_path.loc[0])  # required to ensure that the gmm model is extracted from the df
        # extract the components, mean, covariances, and weights for the current gmm
        components = gmm.n_components
        mu = gmm.means_
        cov = gmm.covariances_
        wgt = gmm.weights_
        for k in range(components):
            if k == 0:  # initialize the bx vector
                bx_temp = hmm_norm_pdf(x=data, mu=mu[k], sd=cov[k]**0.5) * wgt[k]
            else:
                bx_temp = np.append(bx_temp, hmm_norm_pdf(x=data, mu=mu[k], sd=cov[k]**0.5) * wgt[k], axis=0)
        bx[c].append(np.sum(bx_temp, axis=0).tolist())
    # the bx matrix is the matrix of pdf's for each stat and class. It is organized in the same manner as the
    # gmm_params df. Therefore, each row of bx_matrix pertains to the same specific stat and class as that of the
    # same row in gmm_params.

    """ Kick off values adjusted for the priors """
    # adjust for the prior probability of the state. The initial log_alpha value is nothing more than a kick off
    # value. Basically, what's the probability of being in any one of the x states/classes. The initial value
    # will be the same for all classes
    # temp = []
    for c in range(len(classes)):
        # bx fixed at the first column b/c this is all that is needed to initiate
        log_alpha[c].append(np.log(bx[c][0][0]) + np.log(pi[c]))

    """ Begin cycling through each sample and each class """
    for t in range(1, n):  # cycle through all the samples
        for ci in range(len(classes)):
            """ determine max alpha for a given class """
            m_alpha_temp = []
            for cj in range(len(classes)):
                # Alpha for i=1, there will need to be j classes (neutral, link, sweep, ...)
                if (A_new[cj, ci] == 0) or (A_new[cj, ci] == -np.inf):  # catching div by zero errors
                    log_A_new = -np.inf
                else:
                    log_A_new = np.log(A_new[cj, ci])
                # m_alpha_temp.append(log_alpha[cj][t - 1] + np.log(A_new[cj, ci]))
                m_alpha_temp.append(log_alpha[cj][t - 1] + log_A_new)

            m_alpha[ci].append(max(m_alpha_temp))  # this may be able to be flushed after each iteration (it's needed only as calc)

            """ determine the sum of the exponential for a given class """
            exp_sum_temp = []
            for cj in range(len(classes)):
                # note, m_alpha is t-1 b/c the first m_alpha entry was done at t=1 (so it is offset)
                if (A_new[cj, ci] == 0) or (A_new[cj, ci] == -np.inf):  # catching div by zero errors
                    log_A_new = -np.inf
                else:
                    log_A_new = np.log(A_new[cj, ci])
                # exp_sum_temp.append(np.exp(log_alpha[cj][t - 1] + np.log(A_new[cj, ci]) - m_alpha[ci][t - 1]))
                exp_sum_temp.append(np.exp(log_alpha[cj][t - 1] + log_A_new - m_alpha[ci][t-1]))
            exp_sum[ci].append(sum(exp_sum_temp))

            """ finally, update log alpha """
            b = np.log(bx[ci][0][t])  # the [0] is b/c there is always only 1 list per class
            m = m_alpha[ci][t-1]  # t-1 b/c the max alpha is initially updated at t=1
            e = exp_sum[ci][t-1]  # t-1 b/c the max alpha is initially updated at t=1
            log_alpha[ci].append(b + m + np.log(e))

    # max value for log-likelihood, forward algorithm
    temp = []
    for c in range(len(classes)):
        temp.append(log_alpha[c][n-1])
    m_alpha_ll = max(temp)
    # Forward algorithm log-likelihood
    temp = []
    for c in range(len(classes)):
        temp.append(np.exp(log_alpha[c][n-1] - m_alpha_ll))
    temp = np.log(sum(temp))
    fwd_ll = m_alpha_ll + temp

    return fwd_ll, log_alpha

def hmm_backward(gmm_params, data, A_trans, pi, stat='xpehh'):
    # the 'stat' variable is not ideal, but will work as a placeholder that represents the specific data column that
    # should be used.

    # some initializations and settings
    n = len(data)
    A_new = A_trans

    # Probability density values for the data using distribution 1

    classes = gmm_params['class'].unique()
    # create a collection of lists that can be used to store intermediate values
    bx = {}
    log_beta = {}
    # m_beta = {}  # holds the max (log) alpha for each iteration
    # exp_sum = {}
    for x in range(len(classes)):
        bx[x] = []
        log_beta[x] = []
        # m_beta[x] = []
        # exp_sum[x] = []

    """ build the matrix of pdf values for each class [may want to make this its own function] """
    for c in range(len(classes)):
        # extract the path to the correct gmm based on the stat and the class
        gmm_current_path = gmm_params['gmm_path'][(gmm_params['stat'] == stat) &
                                                  (gmm_params['class'] == classes[c])].reset_index(drop=True)
        # load the correct gmm based on the correct path
        gmm = hmm_pickle_load(gmm_current_path.loc[0])  # required to ensure that the gmm model is extracted from the df
        # extract the components, mean, covariances, and weights for the current gmm
        components = gmm.n_components
        mu = gmm.means_
        cov = gmm.covariances_
        wgt = gmm.weights_
        for k in range(components):
            if k == 0:  # initialize the bx vector
                bx_temp = hmm_norm_pdf(x=data, mu=mu[k], sd=cov[k]**0.5) * wgt[k]
            else:
                bx_temp = np.append(bx_temp, hmm_norm_pdf(x=data, mu=mu[k], sd=cov[k]**0.5) * wgt[k], axis=0)
        bx[c].append(np.sum(bx_temp, axis=0).tolist())
    # the bx matrix is the matrix of pdf's for each stat and class. It is organized in the same manner as the
    # gmm_params df. Therefore, each row of bx_matrix pertains to the same specific stat and class as that of the
    # same row in gmm_params.

    """ Kick off values adjusted for the priors """
    # adjust for the prior probability of the state. The initial log_alpha value is nothing more than a kick off
    # value. Basically, what's the probability of being in any one of the x states/classes. The initial value
    # will be the same for all classes
    for c in range(len(classes)):
        # log is not needed here b/c the log_beta at location n would be log of 1, which is zero
        log_beta[c].append([0] * n)

    """ Begin cycling through each sample and each class """
    for t in reversed(range(0, (n-1))):  # cycle through all the samples
        for ci in range(len(classes)):
            """ determine max alpha for a given class """
            m_beta_temp = []
            for cj in range(len(classes)):
                # beta for i=n, there will need to be j classes (neutral, link, sweep, ...)
                if (A_new[cj, ci] == 0) or (A_new[cj, ci] == -np.inf):  # catching div by zero errors
                    log_A_new = -np.inf
                else:
                    log_A_new = np.log(A_new[cj, ci])
                # m_beta_temp.append(log_beta[cj][0][t + 1] + np.log(A_new[ci, cj]) + np.log(bx[cj][0][t + 1]))
                m_beta_temp.append(log_beta[cj][0][t + 1] + log_A_new + np.log(bx[cj][0][t + 1]))
            # m_beta[ci].append(max(m_beta_temp))  # this may be able to be flushed after each iteration (it's needed only as calc)
            m_beta = max(m_beta_temp)
            """ determine the sum of the exponential for a given class """
            exp_sum_temp = []
            for cj in range(len(classes)):
                if (A_new[ci, cj] == 0) or (A_new[ci, cj] == -np.inf):  # catching div by zero errors
                    log_A_new = -np.inf
                else:
                    log_A_new = np.log(A_new[ci, cj])
                # exp_sum_temp.append(np.exp(log_beta[cj][0][t + 1] + np.log(A_new[ci, cj]) + np.log(bx[cj][0][t + 1])
                #                            - m_beta))
                exp_sum_temp.append(np.exp(log_beta[cj][0][t + 1] + log_A_new + np.log(bx[cj][0][t + 1])
                                           - m_beta))
            exp_sum = sum(exp_sum_temp)

            """ finally, update log alpha """
            # b = np.log(bx[ci][0][t])  # the [0] is b/c there is always only 1 list per class
            m = m_beta  # t-1 b/c the max alpha is initially updated at t=1
            # e = exp_sum[ci][t+1]  # t-1 b/c the max alpha is initially updated at t=1
            e = exp_sum
            log_beta[ci][0][t] = m + np.log(e)

    # max value for log-likelihood, forward algorithm
    temp = []
    for c in range(len(classes)):
        temp.append(log_beta[c][0][0] + np.log(pi[c]) + np.log(bx[c][0][0]))
    m_beta_ll = max(temp)

    # Forward algorithm log-likelihood
    temp = []
    for c in range(len(classes)):
        temp.append(np.exp(log_beta[c][0][0] + np.log(pi[c]) + np.log(bx[c][0][0]) - m_beta_ll))
    temp = np.log(sum(temp))
    bwd_ll = m_beta_ll + temp

    return bwd_ll, log_beta

def hmm_gamma(alpha, beta, n):
    # log gamma z's
    # np.maximum returns an array that contains the max value of both comparisons, and then np.max returns the max
    # of the np.maximum array.
    """
    log_gamma = a + b - m - log(sum(exp(a + b - m)))
    To Do:
    - This is currently written as though there are only two states, but it needs to be generalized to handle
    multiple states. For instance, if there were three states, it might look like this:
    m_gamma = np.max(np.maximum(alpha[0] + beta[0], alpha[1] + beta[1], alpha[2) + beta[2))
    log_gamma1 = alpha[0] + beta[0] - m_gamma - np.log(np.exp(alpha[0] + beta[0] - m_gamma) + np.exp(alpha[1] + beta[1] - m_gamma + np.exp(alpha[2] + beta[2] - m_gamma))
    log_gamma2 = alpha[1] + beta[1] - m_gamma - np.log(np.exp(alpha[0] + beta[0] - m_gamma) + np.exp(alpha[1] + beta[1] - m_gamma + np.exp(alpha[2] + beta[2] - m_gamma))
    gamma1 = np.exp(log_gamma1)
    gamma2 = np.exp(log_gamma2)
    gamma3 = 1 - (gamma1 + gamma2)
    """
    """ Build a np arrays to hold the gamma values """
    # The number of classes will always be equal to the length of alpha or beta (alpha is used)
    class_count = len(alpha)
    gamma = np.empty(shape=(class_count, n))
    z = np.empty(shape=(class_count, n))

    """ Temp list to hold the sum of each alpha and beta vector and then define the max gamma value"""
    temp = []
    for c in range(class_count):
        temp.append(np.add(alpha[c], beta[c]))
    m_gamma = np.max(temp)

    """ Sum of the exponential component of the gamma function"""
    temp = []
    for c in range(class_count):
        temp.append(np.exp(np.subtract(np.add(alpha[c], beta[c]), m_gamma)))
    exp_sum = np.log(np.sum(temp, axis=0))

    """ Final gamma sum using all the components, one gamma vector for each class """
    for c in range(class_count):
        gamma[c] = np.exp(np.subtract(np.subtract(np.add(alpha[c], beta[c]), m_gamma), exp_sum))

    """ Assign class assignment from random draw and comparison with gamma """
    # this loop could be problematic in that it is possible for z[class1] and z[class2] could both be assigned
    # a value of 1 if z_draw happens to be less than both z[class1] and z[class2].
    z_draw = np.random.uniform(low=0, high=1, size=n)
    z_sum = np.zeros(n)  # z_sum is used to ensure that only one class is assigned a value of 1
    for c in range(class_count-1):
        z[c][z_draw <= gamma[c]] = 1
        # z_sum should never exceed 1, but if it does then set the z[c] values the occurrence to 0. Technically this
        # set up will favor the first class to be assigned a 1, but
        z_sum = z_sum + z[c]
        z[c][z_sum > 1] = 0
    # the last class if assigned a value of 1 where all other classes are zero
    z[class_count-1] = np.where(np.sum(z, axis=0) == 0, 1, 0)
    # it is still possible, due to the random draw, for all classes to be assigned a value of zero. z_check is used
    # to identify this scenario and randomly pick a class to assign a value of 1
    z_check = np.sum(np.sum(z, axis=0))

    return z, gamma

def hmm_update_pi(z):

    class_count = len(z)
    pi_update = []
    for c in range(class_count):
        z_count = np.sum(z[c])
        check = len(z[c])
        if z_count < 1:
            z_count = 1
        pi_update.append(z_count / len(z[c]))
    return pi_update


def hmm_update_trans(z):
    """
    This method uses a colum shift coupled with an element wise addition to update the transition matrix. If there
    exist 3 classes then the A matrix is 3x3. Therefore, a total of 9 shifts and addition operations are needed.
    The general workflow is as follows:
    1. Set up a loop to iterate through the transition matrix [i, j].
    2. The initial iteration will be at i=j or position 0,0.
    3. This results in the z[0] column first being shifted and then being added.
    4. The summation results in a value of 2 whenever the current state and the next state are the same (sunny to sunny)
    5. Counting the number of 2's in the column represents the number of transitions from one state to the next.
    6. At the next iteration of the loop i=0 and j=1, we are performing the same operation, but now the count of 2's
    represents the number of times we transitioned from state i=0 to state j=1.
    """
    class_count = len(z)
    a = np.empty(shape=(class_count, class_count))
    for i in range(class_count):
        for j in range(class_count):
            temp = z[i][0:-1] + z[j][1::]
            trans_count = len([k for k in temp if k == 2])
            if trans_count < 1:
                trans_count = 1
            a[i, j] = trans_count

    # normalize the 'a' matrix based on the total count of occurrences in each state
    a_sum = np.sum(a, axis=1) # axis 1 is correct, we want to sum from left to right across the columns
    a = a / a_sum.reshape((len(a_sum), 1))
    # replace any zero values with an extremely small number to prevent div by zero wanting
    a[a == 0] = 1e-4
    # a[a == np.nan] = 1e-4
    return a

def hmm_viterbi(gmm_params, data, a, pi_viterbi, stat='xpehh'):
    # some initializations and settings
    n = len(data)
    classes = gmm_params['class'].unique()
    p = np.empty(shape=(len(classes), n))
    bx = np.empty(shape=(len(classes), n))

    """ build the matrix of pdf values for each class [may want to make this its own function - DO THIS] """
    for c in range(len(classes)):
        # extract the path to the correct gmm based on the stat and the class
        gmm_current_path = gmm_params['gmm_path'][(gmm_params['stat'] == stat) &
                                                  (gmm_params['class'] == classes[c])].reset_index(drop=True)
        # load the correct gmm based on the correct path
        gmm = hmm_pickle_load(gmm_current_path.loc[0])  # required to ensure that the gmm model is extracted from the df
        # extract the components, mean, covariances, and weights for the current gmm
        components = gmm.n_components
        mu = gmm.means_
        cov = gmm.covariances_
        wgt = gmm.weights_
        for k in range(components):
            if k == 0:  # initialize the bx vector
                bx_temp = hmm_norm_pdf(x=data, mu=mu[k], sd=cov[k] ** 0.5) * wgt[k]
            else:
                bx_temp = np.append(bx_temp, hmm_norm_pdf(x=data, mu=mu[k], sd=cov[k] ** 0.5) * wgt[k], axis=0)
        bx[c] = np.sum(bx_temp, axis=0)

    # take the log of all the Viterbi required properties
    for c in range(len(classes)):
        bx[c] = np.log(bx[c])
        pi_viterbi[c] = np.log(pi_viterbi[c])

    # a = np.log(a)
    # loop to catch div by zero warnings
    for ci in range(len(classes)):
        for cj in range(len(classes)):
            if (a[ci, cj] == 0) or (a[ci, cj] == -np.inf):
                a[ci, cj] = -np.inf
            else:
                a[ci, cj] = np.log(a[ci, cj])

    # initiate the sequence for each class at the 0th index
    for c in range(len(classes)):
        p[c, 0] = pi_viterbi[c] + bx[c][0]

    for t in range(1, n):
        for ci in range(len(classes)):
            temp = []
            for cj in range(len(classes)):
                # temp hold the values intermediate values that evaluate the probability of being in the current
                # class when considering all possible transitions to this class.
                temp.append(p[cj][t - 1] + a[cj, ci])
            p[ci, t] = bx[ci][t] + np.max(temp)
    # return the index of the max value in each row which corresponds the classes in the same order as 'classes'
    # note that in the event of a tie the first index is returned. However, in a real situation a tie would be extremely
    # unlikely.
    path = np.argmax(p, axis=0)

    return path

def hmm_get_swifr_classes(path):
    # function to return the classes used in SWIFr (e.g., neutral, sweep, etc.)
    c_list = pd.read_table(path + 'classes.txt', header=None)
    c_list = list(c_list.iloc[:, 0])
    return c_list

def hmm_get_swifr_stats(path):
    # function to copy the list of statistics that SWIFr used (e.g., fst, ihs, etc.)
    s_list = pd.read_table(path + 'component_stats.txt', header=None)
    s_list = list(s_list.iloc[:, 0])
    return s_list

def hmm_get_swifr_param_names(path):
    # function to copy the pickled gmm parameters that were defined using SWIFr
    sub_folder = 'AODE_params'  # folder that is created by SWIFr to stash pickled gmm
    files = os.listdir(path + sub_folder)  # list of all files in the AODE_params folder
    # HMM only needs the 1D params (no joint relationships are needed)
    file_list = [f for f in files if "_1D_GMMparams" in f]
    return file_list

def hmm_pickle_load(path):
    x = pickle.load(open(path, 'rb'))  # actual gmm
    return x