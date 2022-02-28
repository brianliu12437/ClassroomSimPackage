from genericpath import exists
import ClassSimPackage
import pickle
import os, sys
sys.path.append(os.path.abspath(os.path.join('..')))
import numpy as np
import matplotlib.pyplot as plt
from helpers import sample_trunc_normal
import yaml

# Goal: use argparse/yaml to read input parameters, then sample from prior, generate risk distribution, plot histogram
# the same as in https://colab.research.google.com/drive/1whAh1bvhvaMi42XsapHwNxgNMo2VqFmF#scrollTo=e3roVfd-YryT 
# make sure to allow varying the fraction masked

# TODO: figure out how to pass in param configs using a yaml file
# TODO: add another transmissibility_scaling param to account for Delta --> Omicron

def plot_outcome_distribution(params, num_samples = 100000):

    prevalence = params['prevalence']
    distancing = params['distancing']
    p_vax = params['p_vax']
    class_type = params['class_type']

    sample_weights = np.kron(params['weights_VE_transmission'], params['weights_VE_susceptible'])

    result = []

    # load the means of the simulated number of secondary infections at the specified distancing and p_vax
    # for *varying* values of VE_transmission and VE_susceptible
    
    if not params['aerosol_only']:
        sec_infs_diff_VE = np.load('Outcomes_vary_params/' + str(distancing) + \
            '_ft_distancing/' + class_type + '/' + str(p_vax) + '_p_vax.npy')[:,3]
    else:
        sec_infs_diff_VE = np.load('Outcomes_vary_params/' + str(distancing) + \
            '_ft_distancing/' + class_type + '/' + str(p_vax) + '_p_vax.npy')[:,5]
    
    print('start sampling from prior')

    for _ in range(num_samples):

        i = np.random.choice(len(sample_weights), 1, p = sample_weights)[0]

        masking_eff = sample_trunc_normal(params['mask_eff_mean'], params['mask_eff_sd'])
        Omicron_mult = sample_trunc_normal(params['Omicron_mult_mean'], params['Omicron_mult_sd'])

        sampled_sec_infs = sec_infs_diff_VE[i] * (params['fraction_masked'] * masking_eff + (1-params['fraction_masked'])) * prevalence / 50

        sampled_sec_infs = sampled_sec_infs * Omicron_mult * params['class_hours_per_semester']
        if 'student_faculty_population_mult' in params:
            sampled_sec_infs = sampled_sec_infs * params['student_faculty_population_mult']


        result.append(sampled_sec_infs)
    
    print('finished sampling from prior, start plotting')
    
    #return result
    plot_results_and_compute_quantiles(result, distancing, p_vax, prevalence, class_type, num_samples, params['aerosol_only'])

def plot_results_and_compute_quantiles(result, distancing, p_vax, prevalence, class_type, num_samples, aerosol_only):
    quantiles = []
    for q in [0.05, 0.5, 0.95]:
        print(str(q)+'-quantile of per-person infection risk: ', np.quantile(result, q))
        quantiles.append(np.quantile(result, q))

    plt.figure(figsize = (7,5))
    plt.hist(result, bins=100, density = False)

    subject = 'instructor' if aerosol_only else 'student'
    title = 'Distribution of infection risk per {} \n prevalence {:.2f}%, {:.2f}% vaccinated, {} ft distancing, {} class'.format(subject, prevalence*100, p_vax*100, distancing, class_type)

    plt.title(title)
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
    plt.savefig('/home/yz685/ClassroomSimPackage/Results/'+ str(prevalence) + '_prevalence_' + str(distancing) + \
        '_ft_distancing_' + class_type + '_' + str(p_vax) + '_vaccinated_' + subject + '.pdf')
        
if __name__ == '__main__':

    params = yaml.load(open(sys.argv[1]), Loader = yaml.FullLoader)
    print(params)

    plot_outcome_distribution(params, num_samples = 10000)