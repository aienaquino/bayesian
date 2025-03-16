# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 19:04:31 2025

@author: Aiensexy
"""
import numpy as np
import matplotlib.pyplot as plt
 
# Generate some synthetic data
np.random.seed(50)
true_mu =10
true_sigma = 6
data = np.random.normal(true_mu, true_sigma, size=300)
 
# Define the prior hyperparameters
prior_mu_mean = 0
prior_mu_precision = 1  # Variance = 1 / precision
prior_sigma_alpha = 2
prior_sigma_beta = 2  # Beta = alpha / beta
 
# Update the prior hyperparameters with the data
posterior_mu_precision = prior_mu_precision + len(data) / true_sigma**2
posterior_mu_mean = (prior_mu_precision * prior_mu_mean + np.sum(data)) / posterior_mu_precision
 
posterior_sigma_alpha = prior_sigma_alpha + len(data) / 2
posterior_sigma_beta = prior_sigma_beta + np.sum((data - np.mean(data))**2) / 2
 
# Calculate the posterior parameters
posterior_mu = np.random.normal(posterior_mu_mean, 1 / np.sqrt(posterior_mu_precision), size=20000)
posterior_sigma = np.random.gamma(posterior_sigma_alpha, 1 / posterior_sigma_beta, size=20000)
 
# Plot the posterior distributions
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(posterior_mu, bins=40, density=True, color='magenta', edgecolor='blue')
plt.title('Posterior distribution of $\mu$')
plt.xlabel('$\mu$')
plt.ylabel('Density')
 
plt.subplot(1, 2, 2)
plt.hist(posterior_sigma, bins=40, density=True, color='yellow', edgecolor='blue')
plt.title('Posterior distribution of $\sigma$')
plt.xlabel('$\sigma$')
plt.ylabel('Density')
 
plt.tight_layout()
plt.show()
 
# Calculate summary statistics
mean_mu = np.mean(posterior_mu)
std_mu = np.std(posterior_mu)
print("Mean of mu:", mean_mu)
print("Standard deviation of mu:", std_mu)
 
mean_sigma = np.mean(posterior_sigma)
std_sigma = np.std(posterior_sigma)
print("Mean of sigma:", mean_sigma)
print("Standard deviation of sigma:", std_sigma)

