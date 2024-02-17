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
from sklearn import mixture
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay

def hmm_get_data(path):
    """
    Import data with the expectation of two distributions
    :param path:
    path where the data is stored
    :return:
    Data as a pandas dataframe
    """
    x = pd.read_csv(path)
    x = np.array(x)

    # delete the transforms below, this was for testing with known data set that had weird formatting. Then
    # uncomment the x statement above
    # x = pd.read_csv(path, delimiter=',', header=None)
    # x = x.iloc[:, 0:4]
    # x = np.array(x)
    # x = x.flatten(order='C')
    # x = x.reshape((len(x), 1))
    return x
def hmm_init_params(mu_list, sd_list, pi_list, state_list):
    """
    Initialization parameters for HMM
    :param mu_list:
    List of mean values for each normal distribution
    :param sd_list:
    List of the standard deviations for each normal distribution
    :param pi_list:
    List of the pi fractions for each class (percentage of each class in data set)
    :return:
    Dataframe containing the mu, sd, and pi values provided
    """
    init = pd.DataFrame(columns=['mu', 'sd', 'pi', 'state'])
    init['mu'] = mu_list
    init['sd'] = sd_list
    init['pi'] = pi_list
    init['state'] = state_list
    return init

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
def hmm_forward(data, A_trans, pi):
    # some initializations and settings
    n = len(data)
    A_new = A_trans

    # this needs to be scrapped and replaced with the gmm data and init_df
    # mu_n = params['mu'][params['state'] == 'neutral'].tolist()
    # mu_p = params['mu'][params['state'] == 'sweep'].tolist()
    # sd_n = params['sd'][params['state'] == 'neutral'].tolist()
    # sd_p = params['sd'][params['state'] == 'sweep'].tolist()
    # pi_n = params['pi'][params['state'] == 'neutral'].tolist()[0]
    # pi_p = params['pi'][params['state'] == 'sweep'].tolist()[0]

    mu_n = [-1.21758821, -0.75719649]
    mu_p = [0.39927606]
    sd_n = [0.12109503**0.5, 0.08834092**0.5]  # needs to be sqr root
    sd_p = [0.37233887**0.5]
    wgt_n = [0.42728606, 0.57271394]
    wgt_p = [1]
    pi_n = pi[0]
    pi_p = pi[1]



    # Probability density values for the data using distribution 1
    """
    [see sync notes 10/24/2023]
    I did make a mistake with the bx1 and bx2 calculation. What I should have done is calculate
    the pdf for each of the neutral humps (modes) and then multiply those pdf by the weight of each hump. This is not
    an urgent problem to fix b/c we will get these weights from the GMM anyway, but probably worth adding on my own.
    See day 12 notes from class, but the idea is below. basically if hump-1 has weight of 0.25 and hump-2 has weight
    0.75 then the pdf would be [wgt1 * pdf1 + wgt2 * pdf2]. This can be done using g.weights_
    """
    bx1_temp = hmm_norm_pdf(x=data, mu=mu_n[0], sd=sd_n[0]) * wgt_n[0]
    for i in range(1, len(mu_n)):
        temp1 = hmm_norm_pdf(x=data, mu=mu_n[i], sd=sd_n[i]) * wgt_n[i]
        bx1_temp = np.append(bx1_temp, temp1, axis=0)
    bx1_temp = bx1_temp.reshape((len(data), len(mu_n)), order='F')
    bx1 = np.sum(bx1_temp, axis=1)
    # I think that once the weights are included, the np.max function will be replaced with np.sum
    # bx1 = np.sum(bx1_temp, axis=1)
    alpha1 = np.array(np.log(bx1[0] * pi_n)).reshape((1,))
    # bx1 = hmm_norm_pdf(x=data, mu=params.loc[0, 'mu'], sd=params.loc[0, 'sd'])  # old code for 2 distributions
    # alpha1 = np.array(np.log(bx1[0] * params.loc[0, 'pi']))  # old code for 2 distributions

    # Probability density values for the data using distribution 2
    bx2_temp = hmm_norm_pdf(x=data, mu=mu_p[0], sd=sd_p[0]) * wgt_p[0]
    for i in range(1, len(mu_p)):
        temp1 = hmm_norm_pdf(x=data, mu=mu_n[i], sd=sd_n[i]) * wgt_p[i]
        bx2_temp = np.append(bx2_temp, temp1, axis=0)
    bx2_temp = bx2_temp.reshape((len(data), len(mu_p)), order='F')
    bx2 = np.sum(bx2_temp, axis=1)
    alpha2 = np.array(np.log(bx2[0] * pi_p)).reshape((1,))
    # bx2 = hmm_norm_pdf(x=data, mu=params.loc[1, 'mu'], sd=params.loc[1, 'sd'])  # old code for 2 distributions
    # alpha2 = np.array(np.log(bx2[0] * params.loc[1, 'pi']))  # old code for 2 distributions

    # Initial m values (slightly modified from R code)
    m1_alpha = np.array(max(alpha1, alpha2))
    m2_alpha = np.array(max(alpha1, alpha2))

    for t in range(1, n):
        # Alpha for i=1, there will need to be j classes (neutral, link, sweep, ...)
        m1_alpha_j1 = (alpha1[t - 1]) + np.log(A_new[0, 0])  # m when j=0 and i=0
        m1_alpha_j2 = (alpha2[t - 1]) + np.log(A_new[1, 0])  # m when j=1 and i=0
        m1_alpha = np.append(m1_alpha, max(m1_alpha_j1, m1_alpha_j2))  # max of m1_j1 and m1_j2
        # calculation for alpha when i=1
        alpha1 = np.append(alpha1, np.log(bx1[t]) + m1_alpha[t] + np.log(np.exp(m1_alpha_j1 - m1_alpha[t]) +
                                                                         np.exp(m1_alpha_j2 - m1_alpha[t])))

        # Alpha for i=2, there will need to be j classes (neutral, link, sweep, ...)
        m2_alpha_j1 = (alpha1[t - 1]) + np.log(A_new[0, 1])  # m when j=1 and i=2
        m2_alpha_j2 = (alpha2[t - 1]) + np.log(A_new[1, 1])  # m when j=2 and i=2
        m2_alpha = np.append(m2_alpha, max(m2_alpha_j1, m2_alpha_j2))  # max of m2_j1 and m2_j2
        # calculation of alpha when i=2
        alpha2 = np.append(alpha2, np.log(bx2[t]) + m2_alpha[t] + np.log(np.exp(m2_alpha_j1 - m2_alpha[t]) +
                                                                         np.exp(m2_alpha_j2 - m2_alpha[t])))

    # max value for log-likelihood, forward algorithm
    m_alpha_ll = max(alpha1[n-1], alpha2[n-1])
    # Forward algorithm log-likelihood
    fwd_ll = m_alpha_ll + np.log(np.exp(alpha1[n-1] - m_alpha_ll) + np.exp(alpha2[n-1] - m_alpha_ll))
    # package the alpha vectors into a list
    alpha = [alpha1, alpha2]

    return fwd_ll, alpha


def hmm_forward_new(gmm_params, data, A_trans, pi, stat='xpehh'):
    # the 'stat' variable is not ideal, but will work as a placeholder that represents the specific data column that
    # should be used.

    # some initializations and settings
    n = len(data)
    A_new = A_trans

    # Probability density values for the data using distribution 1

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
                # what about a temp m1_alpha to hold the values...I just need the max
                m_alpha_temp.append(log_alpha[cj][t - 1] + np.log(A_new[cj, ci]))
            m_alpha[ci].append(max(m_alpha_temp))  # this may be able to be flushed after each iteration (it's needed only as calc)

            """ determine the sum of the exponential for a given class """
            exp_sum_temp = []
            for cj in range(len(classes)):
                # note, m_alpha is t-1 b/c the first m_alpha entry was done at t=1 (so it is offset)
                exp_sum_temp.append(np.exp(log_alpha[cj][t - 1] + np.log(A_new[cj, ci]) - m_alpha[ci][t-1]))
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


def hmm_backward(data, A_trans, pi):
    # some initializations and settings
    n = len(data)
    A_new = A_trans
    # mu_n = params['mu'][params['state'] == 'neutral'].tolist()
    # mu_p = params['mu'][params['state'] == 'sweep'].tolist()
    # sd_n = params['sd'][params['state'] == 'neutral'].tolist()
    # sd_p = params['sd'][params['state'] == 'sweep'].tolist()
    # pi_n = params['pi'][params['state'] == 'neutral'].tolist()[0]
    # pi_p = params['pi'][params['state'] == 'sweep'].tolist()[0]

    mu_n = [-1.21758821, -0.75719649]
    mu_p = [0.39927606]
    sd_n = [0.12109503**0.5, 0.08834092**0.5]  # needs to be sqr root
    sd_p = [0.37233887**0.5]
    wgt_n = [0.42728606, 0.57271394]
    wgt_p = [1]
    pi_n = pi[0]
    pi_p = pi[1]

    # Initial values for beta1 and beta2 at t=n
    # Probability density values for the data using distribution 1

    bx1_temp = hmm_norm_pdf(x=data, mu=mu_n[0], sd=sd_n[0]) * wgt_n[0]
    for i in range(1, len(mu_n)):
        temp1 = hmm_norm_pdf(x=data, mu=mu_n[i], sd=sd_n[i]) * wgt_n[i]
        bx1_temp = np.append(bx1_temp, temp1, axis=0)
    bx1_temp = bx1_temp.reshape((len(data), len(mu_n)), order='F')
    bx1 = np.sum(bx1_temp, axis=1)


    # Probability density values for the data using distribution 2
    bx2_temp = hmm_norm_pdf(x=data, mu=mu_p[0], sd=sd_p[0]) * wgt_p[0]
    for i in range(1, len(mu_p)):
        temp1 = hmm_norm_pdf(x=data, mu=mu_n[i], sd=sd_n[i]) * wgt_p[i]
        bx2_temp = np.append(bx2_temp, temp1, axis=0)
    bx2_temp = bx2_temp.reshape((len(data), len(mu_p)), order='F')
    bx2 = np.sum(bx2_temp, axis=1)

    beta1 = np.zeros(n)
    beta1[n-1] = (np.log(1))
    beta2 = np.zeros(n)
    beta2[n-1] = (np.log(1))

    for t in reversed(range(0, (n-1))):  # recall that n-2 is actually the second to last position and n-1 is the last position
      # beta for i=1
      m1_beta_j1 = (beta1[t+1] + np.log(A_new[0, 0]) + np.log(bx1[t+1]))  # m when j=0 and i=0
      m1_beta_j2 = (beta2[t+1] + np.log(A_new[0, 1]) + np.log(bx2[t+1]))  # m when j=1 and i=0
      m1_beta = max(m1_beta_j1, m1_beta_j2)
      beta1[t] = m1_beta + np.log(np.exp(m1_beta_j1 - m1_beta) + np.exp(m1_beta_j2 - m1_beta))

      # beta for i=2
      m2_beta_j1 = (beta1[t+1] + np.log(A_new[1, 0]) + np.log(bx1[t+1]))  # m when j=0 and i=1
      m2_beta_j2 = (beta2[t+1] + np.log(A_new[1, 1]) + np.log(bx2[t+1]))  # m when j=1 and i=1
      m2_beta = max(m2_beta_j1, m2_beta_j2)
      beta2[t] = m2_beta + np.log(np.exp(m2_beta_j1 - m2_beta) + np.exp(m2_beta_j2 - m2_beta))

    # first and second parts of m value for log-likelihood backward algorithm
    m_beta_ll1 = (beta1[0] + np.log(pi_n) + np.log(bx1[0]))
    m_beta_ll2 = (beta2[0] + np.log(pi_p) + np.log(bx2[0]))

    # m value for log likelihood, backward algorithm
    m_beta_ll = max(m_beta_ll1, m_beta_ll2)
    # Backward algorithm log likelihood
    bwd_ll = m_beta_ll + np.log(np.exp(m_beta_ll1 - m_beta_ll) + np.exp(m_beta_ll2 - m_beta_ll))
    # package the beta vectors into a list
    beta = [beta1, beta2]

    return bwd_ll, beta

def hmm_backward_new(gmm_params, data, A_trans, pi, stat='xpehh'):
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
                m_beta_temp.append(log_beta[cj][0][t + 1] + np.log(A_new[ci, cj]) + np.log(bx[cj][0][t + 1]))
            # m_beta[ci].append(max(m_beta_temp))  # this may be able to be flushed after each iteration (it's needed only as calc)
            m_beta = max(m_beta_temp)
            """ determine the sum of the exponential for a given class """
            exp_sum_temp = []
            for cj in range(len(classes)):
                exp_sum_temp.append(np.exp(log_beta[cj][0][t + 1] + np.log(A_new[ci, cj]) + np.log(bx[cj][0][t + 1])
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
    m_gamma = np.max(np.maximum(alpha[0] + beta[0], alpha[1] + beta[1]))
    log_gamma1 = alpha[0] + beta[0] - m_gamma - np.log(np.exp(alpha[0] + beta[0] - m_gamma) + np.exp(alpha[1] + beta[1] - m_gamma))
    gamma1 = np.exp(log_gamma1)

    z = np.zeros(n)  # n is the length of the data
    z_draw = np.random.uniform(low=0, high=1, size=n)
    z[z_draw <= gamma1] = 1
    return z, gamma1
def hmm_gamma_new(alpha, beta, n):
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
    """ Build a dictionary to hold the gamma values """
    # The number of classes will always be equal to the length of alpha or beta (alpha is used)
    class_count = len(alpha)
    gamma = {}
    z = {}
    for x in range(class_count):
        gamma[x] = []
        z[x] = np.zeros(n)

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
    for c in range(class_count):
        z[c][z_draw <= gamma[c][0]] = 1
        # z_sum should never exceed 1, but if it does then set the z[c] values the occurrence to 0. Technically this
        # set up will favor the first class to be assigned a 1, but
        z_sum = z_sum + z[c]
        z[c][z_sum > 1] = 0

    return z, gamma

def hmm_update_pi(z, gamma):
    pi1 = len(z[z == 1]) / len(z)
    # pi2 = 1 - gamma[0]
    pi2 = 1 - pi1
    return pi1, pi2

def hmm_update_pi_new(z):

    class_count = len(z)
    pi_update = []
    for c in range(class_count):
        pi_update.append(len(z[c][z[c] == 1]) / len(z[c]))

    return pi_update

def hmm_update_trans(z):
    # indicator function for the transition matrix
    z1_stay = 0
    z1_arrive = 0
    for i in range(0, len(z) - 1):
        if (z[i] == 1) and (z[i + 1] == 1):
            z1_stay += 1
        if (z[i] == 0) and (z[i + 1] == 1):
            z1_arrive += 1
    # update of transition matrix using the indicator function
    A_trans[0, 0] = z1_stay / sum(z)
    A_trans[0, 1] = 1 - A_trans[0, 0]
    A_trans[1, 0] = z1_arrive / len(z[z == 0])
    A_trans[1, 1] = 1 - A_trans[1, 0]
    return A_trans

def hmm_update_trans_new(z):

    class_count = len(z)
    A = np.empty(shape=(class_count, class_count))

    for i in range(class_count):
        for j in range(class_count):
            temp = z[i][0:-1] + z[j][1::]
            A[i, j] = len([k for k in temp if k == 2])

    A = A / len(z[0])

    return A

def hmm_viterbi(params, data, A_trans):
    # some initializations and settings
    n = len(data)
    mu_n = params['mu'][params['state'] == 'neutral'].tolist()
    mu_p = params['mu'][params['state'] == 'sweep'].tolist()
    sd_n = params['sd'][params['state'] == 'neutral'].tolist()
    sd_p = params['sd'][params['state'] == 'sweep'].tolist()
    pi_n = params['pi'][params['state'] == 'neutral'].tolist()[0]
    pi_p = params['pi'][params['state'] == 'sweep'].tolist()[0]

    # Probability density values for the data using distribution 1
    bx1_temp = hmm_norm_pdf(x=data, mu=mu_n[0], sd=sd_n[0])
    for i in range(1, len(mu_n)):
        bx1_temp = np.append(bx1_temp, hmm_norm_pdf(x=data, mu=mu_n[i], sd=sd_n[i]), axis=1)
    bx1 = np.max(bx1_temp, axis=1)

    # Probability density values for the data using distribution 2
    bx2_temp = hmm_norm_pdf(x=data, mu=mu_p[0], sd=sd_p[0])
    for i in range(1, len(mu_p)):
        bx2_temp = np.append(bx2_temp, hmm_norm_pdf(x=data, mu=mu_p[i], sd=sd_p[i]), axis=1)
    bx2 = np.max(bx2_temp, axis=1)

    p_n = []  # list to hold the log probs of being in state H
    p_p = []  # list to hold the log probs of being in state L
    # take the log of all the Viterbi required properties
    bx1 = np.log(bx1)
    bx2 = np.log(bx2)
    A_trans = np.log(A_trans)
    pi_n = np.log(pi_n)
    pi_p = np.log(pi_p)

    for i in range(n):
        if i == 0:
            # The first element (0th position)
            p_n.append(pi_n + bx1[i])
            p_p.append(pi_p + bx2[i])
        else:
            # The second element
            # H[datat[i]] is going to be equal to the probability of data[i] given state neutral (so, bx1)
            p_n.append(bx1[i] + np.maximum(p_n[i - 1] + A_trans[0, 0], p_p[i - 1] + A_trans[1, 0]))
            p_p.append(bx2[i] + np.maximum(p_p[i - 1] + A_trans[1, 1], p_n[i - 1] + A_trans[0, 1]))

    p_zip = np.array([p_p, p_n])
    # return the index of the max value in each column which corresponds to states 0, 1, ...
    # note that in the event of a tie the first index is returned. However, in a real situation a tie would be extremely
    # unlikely.
    path = np.argmax(p_zip, axis=0)

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

def hmm_init_params2(path):
    """
    Function to collect the classes, stats, and gmm names and return them as a df
    """
    # information below is a collection of lists that identifies the classes, stats, and names of the 1D gmm params
    gmm_param_list = hmm_get_swifr_param_names(swifr_path)
    class_list = hmm_get_swifr_classes(swifr_path)
    stat_list = hmm_get_swifr_stats(swifr_path)

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

""" Path to data and params from GMM """
swifr_path = '../../swifr_pkg/test_data/simulations_4_swifr_2class/'
data_path = '../../swifr_pkg/test_data/simulations_4_swifr_test_2class/test/test'
gmm_params = hmm_init_params2(swifr_path)
gmm_params = gmm_params[gmm_params['stat'] == 'xpehh'].reset_index(drop=True)  # limit to one stat for now

# s_means = gmm_params.loc[0, 'gmm']

""" Path to data and data load (external for now) """
# this will need to be a 'get' function, but keep it external for now
data_orig = pd.read_table('../../swifr_pkg/test_data/simulations_4_swifr_test_2class/test/test', sep='\t')
# for dev, just use xpehh
data = data_orig['xpehh'][data_orig['xpehh'] != -998].reset_index(drop=True)
data = data.iloc[0:30000]
# data = np.array(data['xpehh']).reshape((len(data), 1))  # not sure if I need to convert to numpy, don't if not needed

# s_means = g_sweep.means_
# s_cov = g_sweep.covariances_
# s_wgt = g_sweep.weights_


# initialize the transition matrix, hard coded for now, but will need to  adjust to calculate from the data
# ensure (for now) that the order of transitions matches the
a_list = [0.999, 0.001, 1, 0]  # transition matrix in a00, a01, a10, a11 format
A_trans = hmm_init_trans(a_list=a_list)
pi = [0.9999, 0.0001]  # state probabilities for neutral and sweep

# bwd_ll_old, beta_old = hmm_backward(data, A_trans, pi)
# fwd_ll_old, alpha_old = hmm_forward(data, A_trans, pi)

fwd_ll_new, alpha_new = hmm_forward_new(gmm_params, data, A_trans, pi)
bwd_ll_new, beta_new = hmm_backward_new(gmm_params, data, A_trans, pi)
# z_old, gamma_old = hmm_gamma(alpha=alpha_old, beta=beta_old, n=len(data))
z, gamma = hmm_gamma_new(alpha=alpha_new, beta=beta_new, n=len(data))
pi = hmm_update_pi_new(z)
A_trans = hmm_update_trans_new(z)


# print(fwd_ll_old)
# print(bwd_ll_old)

print(bwd_ll_new)
print(fwd_ll_new)
print('done')
"""
Go here for a python implementation of HMM. This can be used to compare output and timing of my method vs 
another:
https://hmmlearn.readthedocs.io/en/stable/tutorial.html
"""

'''
The setup below is kind of problematic, but can be worked out later. For now, just ensure that the mu list contains
every mean for all of the neutral and sweep events (same for the sd list). This allows for multiple mu entries for
(say) a neutral signature that might be characterized by multiple modes (same for sweeps).  However, you must be sure
that the 'state_list' contains the correct labels for the mu and sd entries (order matters).
'''
# mu and sd values pulled directly from the simple sim generator assuming the params would be known
# mu_n = [1, 4]  # mean for neutral state normal distribution
# sd_n = [1, 1]  # sd for neutral state normal distribution
# mu_p = [7, 10]  # mean for positive state normal distribution
# sd_p = [1, 1]  # sd for positive state normal distribution
# state_n = ['neutral' for i in range(len(mu_n))]
# state_p = ['sweep' for i in range(len(mu_p))]
# pi_n = [pi_temp for i in range(len(mu_n))]
# pi_p = [1-pi_temp for i in range(len(mu_p))]

# mu_list = mu_n + mu_p  # mean of each normal distribution
# sd_list = sd_n + sd_p  # std deviation of each normal distribution
# pi_list = pi_n + pi_p  # fraction of neutral and sweep events
# state_list = state_n + state_p  # clas labels for each mu and sd entry
# ''' Parameter setup ends here...needs work'''
# a_list = [0.95, 0.05, 0.2, 0.8]  # transition matrix in a00, a01, a10, a11 format
#
# params = hmm_init_params(mu_list=mu_list, sd_list=sd_list, pi_list=pi_list, state_list=state_list)
# A_trans = hmm_init_trans(a_list=a_list)
#
# fwd, alpha = hmm_forward(params=params, data=data, A_trans=A_trans)
# bwd, beta = hmm_backward(params=params, data=data, A_trans=A_trans)
# z, gamma = hmm_gamma(alpha=alpha, beta=beta, n=len(data))
# pi1_new, pi2_new = hmm_update_pi(z, gamma)
# A_trans = hmm_update_trans(z)
# # find the most likely path using the Viterbi algorithm
# path_viterbi = hmm_viterbi(params=params, data=data, A_trans=A_trans)
#
# ''' PLOT THE DATA '''
# path_pred = z
# print(A_trans)
# print(params)
# print('pi1 after update: ', round(pi1_new, 3), ' | pi2 after update: ', round(pi2_new, 3))
#
# plt.figure(figsize=(10, 10))
# plt.hist(data, density=True, bins=75, color='black', alpha=0.2, label='Data')
# plt.legend(loc='upper left')
# plt.xlabel('values')
# plt.ylabel('density')
# plt.show()
#
# ''' CONFUSION MATRIX '''
# cm = confusion_matrix(path_actual, path_pred)
# disp = ConfusionMatrixDisplay(cm, display_labels=['Sweep', 'Neutral'])
# disp.plot()
# plt.show()
#
# ''' CONFUSION MATRIX '''
# cm = confusion_matrix(path_actual, path_viterbi)
# disp = ConfusionMatrixDisplay(cm, display_labels=['Sweep', 'Neutral'])
# disp.plot()
# plt.show()

"""
Notes and next steps:
- The script does a nice job of identifying the params of the two normal distributions
- I will need to think about how to add flexibility for multiple distributions
- I will need to add burn-in and lag
- I need to review the steps as was done for SWIFr (create a graphical review of the mechanics)
- I will likely need to convert from stochastic EM to Baum-Welch
- I will need to think about how to combine more than one column of data (e.g., this script simulates Fst, but I need
to combine Fst, XP-EHH, DDAF, iHS). 

"""





