import matplotlib.pyplot as plt
import numpy as np

# without loss of generality, let's assume any list with five 0s and five 1s as given in the problem
# given_data = [0, 1, 0, 0, 1, 1, 1, 0, 1, 0]

# uniform prior distribution 
def uniform_dist():
    uniform_prior_dist = []
    for t in np.arange(0.4, 0.61, 0.01):
        # likelihoods = list(map(lambda x: (t ** x) * ((1 - t) ** (1 - x)), given_data))
        # likelihood = np.prod(likelihoods)
        likelihood = (t ** 5) * ((1 - t) ** 5)
        prior = 1
        posterior = likelihood * prior * 2772 # normalize
        uniform_prior_dist.append(posterior)
        #print "unif " + "theta: " + str(t) + " posterior: " + str(posterior)
 
    plt.plot(np.arange(0.4, 0.61, 0.01), uniform_prior_dist, 'ro')
    plt.savefig('plot_uniform.png')
    plt.close()

# beta prior distribution
def beta_dist():
    beta_prior_dist = []
    for t in np.arange(0.4, 0.61, 0.01):
        # likelihoods = list(map(lambda x: (t ** x) * ((1 - t) ** (1 - x)), given_data))
        # likelihood = np.prod(likelihoods)
        likelihood = (t ** 5) * ((1 - t) ** 5)
        prior = (t ** 2) * ((1 - t) ** 1)
        posterior = likelihood * prior * 24024 # normalize
        beta_prior_dist.append(posterior)
        #print "beta " + "theta: " + str(t) + " posterior: " + str(posterior)
    plt.plot(np.arange(0.4, 0.61, 0.01), beta_prior_dist, 'bs')
    plt.savefig('plot_beta.png')
    plt.close()

if __name__ == "__main__":
    uniform_dist()
    beta_dist()
