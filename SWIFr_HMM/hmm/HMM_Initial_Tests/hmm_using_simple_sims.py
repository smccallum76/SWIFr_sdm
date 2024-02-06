"""
First pass at building a Hidden Markov Model using the simple simulations data. This code will use portions of the
code from Stat Modeling Final.  The objective of this code is to identify the neutral and sweep events with min
amount of errors
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
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
    # x = pd.read_csv(path)
    # x = np.array(x)

    # delete the tranforms below, this was for testing with known data set that had weird formatting. Then
    # uncomment the x statement above
    x = pd.read_csv(path, delimiter=',', header=None)
    x = x.iloc[:, 0:4]
    x = np.array(x)
    x = x.flatten(order='C')
    x = x.reshape((len(x), 1))
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
def hmm_forward(params, data, A_trans):
    # some initializations and settings
    n = len(data)
    A_new = A_trans

    # Probability density values for the data using distribution 1
    bx1 = hmm_norm_pdf(x=data, mu=params.loc[0, 'mu'], sd=params.loc[0, 'sd'])
    alpha1 = np.array(np.log(bx1[0] * params.loc[0, 'pi']))

    # Probability density values for the data using distribution 2
    bx2 = hmm_norm_pdf(x=data, mu=params.loc[1, 'mu'], sd=params.loc[1, 'sd'])
    alpha2 = np.array(np.log(bx2[0] * params.loc[1, 'pi']))

    # Initial m values (slightly modified from R code)
    m1_alpha = np.array(max(alpha1, alpha2))
    m2_alpha = np.array(max(alpha1, alpha2))

    for t in range(1, n):
        # Alpha for i=1
        m1_alpha_j1 = (alpha1[t - 1]) + np.log(A_new[0, 0])  # m when j=0 and i=0
        m1_alpha_j2 = (alpha2[t - 1]) + np.log(A_new[1, 0])  # m when j=1 and i=0
        m1_alpha = np.append(m1_alpha, max(m1_alpha_j1, m1_alpha_j2))  # max of m1_j1 and m1_j2
        # calculation for alpha when i=1
        alpha1 = np.append(alpha1, np.log(bx1[t]) + m1_alpha[t] + np.log(np.exp(m1_alpha_j1 - m1_alpha[t]) + np.exp(m1_alpha_j2 - m1_alpha[t])))

        # Alpha for i=2
        m2_alpha_j1 = (alpha1[t - 1]) + np.log(A_new[0, 1])  # m when j=1 and i=2
        m2_alpha_j2 = (alpha2[t - 1]) + np.log(A_new[1, 1])  # m when j=2 and i=2
        m2_alpha = np.append(m2_alpha, max(m2_alpha_j1, m2_alpha_j2))  # max of m2_j1 and m2_j2
        # calculation of alpha when i=2
        alpha2 = np.append(alpha2, np.log(bx2[t]) + m2_alpha[t] + np.log(np.exp(m2_alpha_j1 - m2_alpha[t]) + np.exp(m2_alpha_j2 - m2_alpha[t])))

    # max value for log-likelihood, forward algorithm
    m_alpha_ll = max(alpha1[n-1], alpha2[n-1])
    # Forward algorithm log-likelihood
    fwd_ll = m_alpha_ll + np.log(np.exp(alpha1[n-1] - m_alpha_ll) + np.exp(alpha2[n-1] - m_alpha_ll))
    # package the alpha vectors into a list
    alpha = [alpha1, alpha2]

    return fwd_ll, alpha

def hmm_backward(params, data, A_trans):
    # some initializations and settings
    n = len(data)
    A_new = A_trans

    # Initial values for beta1 and beta2 at t=n
    # Probability density values for the data using distribution 1
    bx1 = hmm_norm_pdf(x=data, mu=params.loc[0, 'mu'], sd=params.loc[0, 'sd'])
    # Probability density values for the data using distribution 2
    bx2 = hmm_norm_pdf(x=data, mu=params.loc[1, 'mu'], sd=params.loc[1, 'sd'])

    beta1 = np.zeros(n)
    beta1[n-1] = (np.log(1))
    beta2 = np.zeros(n)
    beta2[n-1] = (np.log(1))

    for t in reversed(range(0, (n-1))):  # recall that n-2 is actually the second to last position and n-1 is the last position
      # beta for i=1
      m1_beta_j1 = (beta1[t+1] + np.log(A_new[0,0]) + np.log(bx1[t+1]))[0]  # m when j=0 and i=0
      m1_beta_j2 = (beta2[t+1] + np.log(A_new[0,1]) + np.log(bx2[t+1]))[0]  # m when j=1 and i=0
      m1_beta = max(m1_beta_j1, m1_beta_j2)
      beta1[t] = m1_beta + np.log(np.exp(m1_beta_j1 - m1_beta) + np.exp(m1_beta_j2 - m1_beta))

      # beta for i=2
      m2_beta_j1 = (beta1[t+1] + np.log(A_new[1, 0]) + np.log(bx1[t+1]))[0]  # m when j=0 and i=1
      m2_beta_j2 = (beta2[t+1] + np.log(A_new[1, 1]) + np.log(bx2[t+1]))[0] # m when j=1 and i=1
      m2_beta = max(m2_beta_j1, m2_beta_j2)
      beta2[t] = m2_beta + np.log(np.exp(m2_beta_j1 - m2_beta) + np.exp(m2_beta_j2 - m2_beta))

    # first and second parts of m value for log-likelihood backward algorithm
    m_beta_ll1 = (beta1[0] + np.log(params.loc[0, 'pi']) + np.log(bx1[0]))[0]
    m_beta_ll2 = (beta2[0] + np.log(params.loc[1, 'pi']) + np.log(bx2[0]))[0]
    # m value for log likelihood, backward algorithm
    m_beta_ll = max(m_beta_ll1, m_beta_ll2)
    # Backward algorithm log likelihood
    bwd_ll = m_beta_ll + np.log(np.exp(m_beta_ll1 - m_beta_ll) + np.exp(m_beta_ll2 - m_beta_ll))
    # package the beta vectors into a list
    beta = [beta1, beta2]

    return bwd_ll, beta

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

def hmm_update_mu(z):
    # This will need to be generalized to not expect only two distributions
    mu1 = np.sum(data[z == 1]) / np.sum(z)
    mu2 = np.sum(data[z == 0]) / (len(z) - np.sum(z))
    return mu1, mu2

def hmm_update_sigma(z):
    sigma1 = np.sum((data[z == 1] - mu1_new) ** 2) / np.sum(z)
    sigma2 = np.sum((data[z == 0] - mu2_new) ** 2) / (len(z) - np.sum(z))
    return np.sqrt(sigma1), np.sqrt(sigma2)

def hmm_update_pi(z, gamma):
    pi1 = len(z[z == 1]) / len(z)
    # pi2 = 1 - gamma[0]
    pi2 = 1 - pi1
    return pi1, pi2

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

# input the data
path = '../output/HMM_data_final.txt'
# data = hmm_get_data(path)
data = pd.read_csv('../output/simple_sims.csv')
pi_temp = len(data[data['label'] == 'N']) / len(data) # directly calculate the fraction of neutral events in use in pi_list
path_actual = data['state'] # the actual sequence of 0's and 1's contained in the data ("truth")
data = np.array(data['value']).reshape((len(data), 1))

# initialize some params
'''
The setup below is kind of problematic, but can be worked out later. For now, just ensure that the mu list contains
every mean for all of the neutral and sweep events (same for the sd list). This allows for multiple mu entries for 
(say) a neutral signature that might be characterized by multiple modes (same for sweeps).  However, you must be sure 
that the 'state_list' contains the correct labels for the mu and sd entries (order matters). 
'''
determine_params = 'no'  # if we assume the distribution params are given then 'no', else type 'yes'
mu_list = [3.0, 8.0]  # mean of each normal distribution
sd_list = [2.0, 2.0]  # std deviation of each normal distribution
pi_list = [pi_temp, 1-pi_temp]  # fraction of neutral and sweep events
state_list = ['neutral', 'sweep']  # clas labels for each mu and sd entry
a_list = [0.95, 0.05, 0.2, 0.8]  # transition matrix in a00, a01, a10, a11 format

params = hmm_init_params(mu_list=mu_list, sd_list=sd_list, pi_list=pi_list, state_list=state_list)
A_trans = hmm_init_trans(a_list=a_list)

# this will get looped for some set of iterations
num_iter = 1  # set to 1 unless determine_params ==  'yes'

for i in range(num_iter):
    if i % 10 == 0:
        print(i)
    fwd, alpha = hmm_forward(params=params, data=data, A_trans=A_trans)
    bwd, beta = hmm_backward(params=params, data=data, A_trans=A_trans)
    z, gamma = hmm_gamma(alpha=alpha, beta=beta, n=len(data))
    mu1_new, mu2_new = hmm_update_mu(z)
    sigma1_new, sigma2_new = hmm_update_sigma(z)
    pi1_new, pi2_new = hmm_update_pi(z, gamma)
    A_trans = hmm_update_trans(z)
    if determine_params == 'yes':
        # update the params df with the updated values above
        params.loc[0, 'mu'] = mu1_new
        params.loc[1, 'mu'] = mu2_new
        params.loc[0, 'sd'] = sigma1_new
        params.loc[1, 'sd'] = sigma2_new
        params.loc[0, 'pi'] = pi1_new
        params.loc[1, 'pi'] = pi2_new
    else:
        params = params

path_pred = z
print(A_trans)
print(params)

''' PLOT THE DATA '''
x_axis = np.linspace(np.floor(np.min(data)), np.ceil(np.max(data)), num=100)
dist1 = stats.norm.pdf(x_axis, loc=params.loc[0, 'mu'], scale=params.loc[0, 'sd']) * params.loc[0, 'pi']
dist2 = stats.norm.pdf(x_axis, loc=params.loc[1, 'mu'], scale=params.loc[1, 'sd']) * params.loc[1, 'pi']

plt.figure(figsize=(10, 10))
plt.hist(data, density=True, bins=75, color='black', alpha=0.2, label='Data')
plt.plot(x_axis, dist1, color='red', label='Dist1')
plt.plot(x_axis, dist2, color='blue', label='Dist2')
plt.plot(x_axis, dist1 + dist2, color='purple', label='Dist1 + Dist2')
plt.legend(loc='upper left')
plt.xlabel('values')
plt.ylabel('density')
plt.show()

''' CONFUSION MATRIX '''
cm = confusion_matrix(path_actual, path_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Sweep', 'Neutral'])
disp.plot()
plt.show()

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





