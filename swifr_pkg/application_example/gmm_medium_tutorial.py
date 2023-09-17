"""
This code is copied from the Medium article linked below. It has been included to better understand the GMM as this
model is critical to the SWIFr model.
https://towardsdatascience.com/gaussian-mixture-model-clearly-explained-115010f7d4cf
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_pdf(mu, sigma, label, alpha=0.5, linestyle='k--', density=True, color='green'):
    """
    Plot 1-D data and its PDF curve.

    Parameters
    ----------
    X : array-like, shape (n_samples,)
        The input data.
    """
    # Compute the mean and standard deviation of the data
    # Plot the data
    X = norm.rvs(mu, sigma, size=1000)
    plt.hist(X, bins=50, density=density, alpha=alpha, label=label, color=color)
    # Plot the PDF
    x = np.linspace(X.min(), X.max(), 1000)
    y = norm.pdf(x, mu, sigma)
    plt.plot(x, y, linestyle)

def random_init(n_components):
    """
    Initialize means, weights, and variance randomly and plot the visualization
    """
    pi = np.ones((n_components)) / n_components
    means = np.random.choice(X, n_components) # random sample from 1-d array
    variances = np.random.random_sample(size=n_components) # random values from uni distribution n_components long

    plot_pdf(means[0], variances[0], 'Random Init 01', color='green')
    plot_pdf(means[1], variances[1], 'Random Init 02', color='blue')
    plot_pdf(means[2], variances[2], 'Random Init 03', color='red')
    plt.legend()
    plt.show()

    return means, variances, pi

def step_expectation(X, n_components, means, variances):
    """
    E Step
    X: array-like , shape (n_samples,), the data
    n_components : int, the number of clusters
    means: array like, shape (n_components) the means of each mixture component
    variances: array like, shape (n_components) the variances of each mixture

    returns
    weights: array-like, shape (n_components, n_samples)
    """

    weights = np.zeros((n_components, len(X)))
    for j in range(n_components):
        weights[j, :] = norm(loc=means[j], scale=np.sqrt(variances[j])).pdf(X)
    return weights

def step_maximization(X, weights, means, variances, n_components, pi):
    """
    M Step
    Params
    X: the data (n_samples)
    weights: weights array (n_components, n_samples)
    means = The means of each mixture component (n_components)
    variances: The variances of each mixture component (n_components)
    n_components: number of clusters
    pi: mixture weight components (n_components)

    Returns:
    means: The updated means of each mixture component (n_components)
    variances: The updated variances of each mixture component (n_components)
    """
    r = []
    for j in range(n_components):
        # first eqn in the article [r_ic = (pi1*pdf1) / (pi1*pdf1 + pi2*pdf2...)], the probability that each
        # data point belongs to each component
        r.append((weights[j] * pi[j]) / (np.sum([weights[i] * pi[i] for i in range(n_components)], axis=0)))
        # 5th eqn from article
        means[j] = np.sum(r[j] * X) / (np.sum(r[j]))
        # 6th eqn from article
        variances[j] = np.sum(r[j] * np.square(X - means[j])) / (np.sum(r[j]))
        # 4th eqn from article
        pi[j] = np.mean(r[j])
    rsum = np.sum(r, axis=0)

    return variances, means, pi


def plot_intermediate_steps(means, variances, density=False, save=False, file_name=None):
    plot_pdf(mu1, sigma1, alpha=0.0, linestyle='r--', label='Original Distibutions')
    plot_pdf(mu2, sigma2, alpha=0.0, linestyle='r--', label='Original Distibutions')
    plot_pdf(mu3, sigma3, alpha=0.0, linestyle='r--', label='Original Distibutions')

    color_gen = (x for x in ['green', 'blue', 'orange'])

    for mu, sigma in zip(means, variances):
        plot_pdf(mu, sigma, alpha=0.5, label='d', color=next(color_gen))
    if save or file_name is not None:
        step = file_name.split("_")[1]
        plt.title(f"step: {step}")
        plt.savefig(f"steps/{file_name}.png", bbox_inches='tight')
    plt.show()
def train_gmm(data, n_compenents=3, n_steps=50, plot_intermediate_steps_flag=True):
    """ Training step of the GMM model

    Parameters
    ----------
    data : array-like, shape (n_samples,)
        The data.
    n_components : int
        The number of clusters
    n_steps: int
        number of iterations to run
    """

    # intilize model parameters at the start
    means, variances, pi = random_init(n_compenents)

    for step in range(n_steps):
        # perform E step
        weights = step_expectation(data, n_compenents, means, variances)
        # perform M step
        variances, means, pi = step_maximization(X, weights, means, variances, n_compenents, pi)
        if plot_intermediate_steps_flag: plot_intermediate_steps(means, variances, )  # file_name=f'step_{step+1}')
    plot_intermediate_steps(means, variances, )

    return means, variances, pi



n_samples = 100
mu1, sigma1 = -5, 1.2
mu2, sigma2 = 5, 1.8
mu3, sigma3 = 0, 1.6

x1 = np.random.normal(loc = mu1, scale = np.sqrt(sigma1), size = n_samples)
x2 = np.random.normal(loc = mu2, scale = np.sqrt(sigma2), size = n_samples)
x3 = np.random.normal(loc = mu3, scale = np.sqrt(sigma3), size = n_samples)
X = np.concatenate((x1,x2,x3))

# plot the made up data for review
plot_pdf(mu1,sigma1,label=r"$\mu={} \ ; \ \sigma={}$".format(mu1,sigma1), color='green')
plot_pdf(mu2,sigma2,label=r"$\mu={} \ ; \ \sigma={}$".format(mu2,sigma2), color='blue')
plot_pdf(mu3,sigma3,label=r"$\mu={} \ ; \ \sigma={}$".format(mu3,sigma3), color='red')
plt.legend()
plt.show()

means, variances, pi = train_gmm(X,n_steps=300, plot_intermediate_steps_flag=False)

zzz=1