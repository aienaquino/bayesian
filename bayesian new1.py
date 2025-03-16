# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 19:23:59 2025

@author: Aiensexy
"""
import numpy as np
import matplotlib.pyplot as plt
 
# Observed data
num_visitors = 1500  # Total number of visitors at the bank
num_transaction = 100  # Number of transaction(desired type of transaction)
 
# Prior hyperparameters for the Beta distribution
prior_alpha = 1  # Shape parameter
prior_beta = 1   # Shape parameter
 
# Update the prior with the observed data to get the posterior parameters
posterior_alpha = prior_alpha + num_transaction
posterior_beta = prior_beta + (num_visitors - num_transaction)
 
# Generate samples from the posterior Beta distribution
posterior_samples = np.random.beta(posterior_alpha, posterior_beta, size=10000)
 
# Plot the posterior distribution
plt.figure(figsize=(12, 6))
plt.hist(posterior_samples, bins=30, density=True, color='yellow', edgecolor='black', alpha=0.7)
plt.title('Posterior Distribution of Conversion Rate')
plt.xlabel('Conversion Rate')
plt.ylabel('Density')
plt.xlim(0, 0.1)  # Limiting x-axis to focus on conversion rates close to zero
plt.show()
 
# Calculate summary statistics
mean_conversion_rate = posterior_alpha / (posterior_alpha + posterior_beta)
mode_conversion_rate = (posterior_alpha - 1) / (posterior_alpha + posterior_beta - 2)  # Mode of the Beta distribution
 
print("Mean conversion rate:", mean_conversion_rate)
print("Mode conversion rate:", mode_conversion_rate)
