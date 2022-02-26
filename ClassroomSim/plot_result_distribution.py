from genericpath import exists
import ClassSimPackage
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from helpers import sample_trunc_normal
from statistics import median


# Goal: use argparse/yaml to read input parameters, then sample from prior, generate risk distribution, plot histogram
# the same as in https://colab.research.google.com/drive/1whAh1bvhvaMi42XsapHwNxgNMo2VqFmF#scrollTo=e3roVfd-YryT 
# make sure to allow varying the fraction masked

mask_mean, mask_sd = 0.145, 0.0536
mask_fraction = 0.3
prevalence = 0.005
p_vax = 0.95
weights_VE_transmission = [0.2, 0.2, 0.2, 0.2, 0.2]
weights_VE_susceptible = [0.2, 0.2, 0.2, 0.2, 0.2]

# TODO: figure out how to pass in param configs using a yaml file

def plot_outcome_distribution(
    prevalence = prevalence, 
    distancing = 1, 
    p_vax = p_vax,
    weights_VE_transmission = weights_VE_transmission,
    weights_VE_susceptible = weights_VE_susceptible,
    num_samples = 100000):
    
    sample_weights = np.kron(weights_VE_susceptible, weights_VE_transmission)

    result = []

    # load the means of the simulated number of secondary infections at the specified distancing and p_vax
    # for *varying* values of VE_transmission and VE_susceptible
    sec_infs_diff_VE = np.load('/home/yz685/ClassroomSimPackage/Outcomes_vary_params/' + str(distancing) + \
        '_ft_distancing/' + str(p_vax) + '_p_vax.npy')[:,3]
    
    for _ in range(num_samples):

        i = np.random.choice(25, 1, p = sample_weights)[0]

        masking_eff = sample_trunc_normal(mask_mean, mask_sd)

        sampled_sec_infs = sec_infs_diff_VE[i] * (mask_fraction * masking_eff + (1-mask_fraction)) * prevalence / 50

        # TODO: scale up according to class hours in a semester

        result.append(sampled_sec_infs)
    
    #return result
    plot_results_and_compute_quantiles(result, distancing, p_vax, num_samples)

def plot_results_and_compute_quantiles(result, distancing, p_vax, num_samples):
    quantiles = []
    for q in [0.05, 0.5, 0.95]:
        print(str(q)+'-quantile of per-person infection risk: ', np.quantile(result, q))
        quantiles.append(np.quantile(result, q))

    plt.figure(figsize = (7,5))
    plt.hist(result, bins=100, density = False)
    plt.title('Distribution of infection risk per person')
    plt.xlabel('Risk of infection per person')
    plt.xlim([0, quantiles[-1]*1.5])
    plt.ylabel('Percentage of simulated outcomes')
    plt.gca().set_xticklabels(['{:.4f}%'.format(p*100) for p in plt.gca().get_xticks()]) 
    plt.gca().set_yticklabels(['{:.2f}%'.format(p*100/num_samples) for p in plt.gca().get_yticks()]) 


    plt.axvline(quantiles[0], color='g', linestyle='dashed', linewidth=1, \
                label='5% quantile {:.4f}% '.format(quantiles[0]*100))
    plt.axvline(quantiles[1], color='r', linestyle='dashed', linewidth=1, \
                label='median {:.4f}% '.format(quantiles[1]*100))
    plt.axvline(quantiles[2], color='purple', linestyle='dashed', linewidth=1, \
                label='95% quantile {:.4f}% '.format(quantiles[2]*100))

    plt.legend()
    plt.savefig('/home/yz685/ClassroomSimPackage/Results/'+ str(distancing) + \
        '_ft_distancing_' + str(p_vax) + '_p_vax.pdf')
        
if __name__ == '__main__':
    plot_outcome_distribution()