from genericpath import exists
import ClassSimPackage
import pickle
import os, sys
sys.path.append(os.path.abspath(os.path.join('..')))
import numpy as np
import matplotlib.pyplot as plt
from helpers import sample_trunc_normal, sample_trunc_lognormal
import yaml
import time


def plot_outcome_distribution(params: dict, num_samples: int = 10000):
    """
    Args:
        params
        num_samples
    Returns:
        makes and saves a histogram plot for the simulation results
    """

    # prevalence = params['prevalence'] # TODO: should we be sampling prevalence?
    distancing = params['distancing']
    p_vax = params['p_vax']
    class_type = params['class_type']

    sample_weights = np.kron(
        params['weights_VE_susceptible'],
        params['weights_VE_transmission'] 
    )

    result = []

    # load the means of the simulated number of secondary infections at the 
    # specified distancing and p_vax for *varying* values of VE_transmission 
    # and VE_susceptible


    all_results = np.load('Outcomes_vary_params/' + str(distancing) + \
            '_ft_distancing/' + class_type + '/' + str(p_vax)+'_p_vax.npy')
    print('all_results', all_results)


    if not params['aerosol_only']:
        sec_infs_diff_VE = np.load('Outcomes_vary_params/' + str(distancing) + \
            '_ft_distancing/' + class_type + '/' + str(p_vax)+'_p_vax.npy')[:,3]
    else:
        sec_infs_diff_VE = np.load('Outcomes_vary_params/' + str(distancing) + \
            '_ft_distancing/' + class_type + '/' + str(p_vax)+'_p_vax.npy')[:,5]
        
    print('start sampling from prior')

    start_time = time.time()

    for iter in range(num_samples):

        if iter % 1000 == 0:
            current_time = time.time()
            print(f'iteration {iter}')
            print(f'{current_time - start_time} seconds since last checkpoint')
            start_time = current_time

        i = np.random.choice(len(sample_weights), 1, p = sample_weights)[0]

        # get the corresponding sampled VE_susceptible
        VE_susceptible_param_idx = i // len(params['weights_VE_transmission'])
        # VE_sus = params['weights_VE_susceptible'][VE_susceptible_param_idx]
        VE_sus = params['VE_susceptible_vals'][VE_susceptible_param_idx]
        # print('VE_sus', VE_sus)
        # print('sampled index i', i)
        # print('logged VE sus', all_results[i][0])

        masking_eff = sample_trunc_normal(
            params['mask_eff_mean'], 
            params['mask_eff_sd']
        )

        student_infections = sample_trunc_lognormal(
            mean = params['student_inf_mean'],
            sd = params['student_inf_sd'],
            threshold = params['student_inf_so_far']
        )
        prevalence = student_infections/(params['total_student_population']*28)

        sampled_sec_infs = sec_infs_diff_VE[i] * (
            params['fraction_masked'] * masking_eff + \
                (1-params['fraction_masked'])
        ) * prevalence 

        sampled_sec_infs = sampled_sec_infs * params['class_hours_per_semester'] 

        if params["Omicron"]:
            Omicron_mult = sample_trunc_normal(
                params['Omicron_mult_mean'], 
                params['Omicron_mult_sd']
            )
            sampled_sec_infs = sampled_sec_infs * Omicron_mult

        if params['aerosol_only']:
            sampled_sec_infs = sampled_sec_infs * \
                params['total_student_population'] * \
                params['instructor_teaching_frac'] / \
                params['instructor_population'] * VE_sus

        result.append(sampled_sec_infs)
    
    print('finished sampling from prior, start plotting')
    
    subject = params['instructor_type'] if 'instructor_type' in params \
        else 'student'

    #return result
    plot_results_and_compute_quantiles(
        result = result, 
        distancing = distancing, 
        p_vax = p_vax,  
        class_type = class_type, 
        num_samples = num_samples, 
        aerosol_only = params['aerosol_only'],
        subject = subject
    )

def plot_results_and_compute_quantiles(
    result: list, 
    distancing: int, 
    p_vax: float, 
    class_type: str, 
    num_samples: int, 
    aerosol_only: bool,
    subject: str
):
    quantiles = []
    for q in [0.05, 0.5, 0.95]:
        print(
            str(q)+'-quantile of per-person infection risk: ', 
            np.quantile(result, q)
        )
        quantiles.append(np.quantile(result, q))

    plt.figure(figsize = (7,5))
    plt.hist(result, bins=100, density = False)

    title = 'Distribution of infection risk per {} \n 1 ACH, {:.2f}% vaccinated, {} ft distancing, {} class'.format(subject, p_vax*100, distancing, class_type)

    plt.title(title)
    plt.xlabel('Risk of infection per person')
    plt.xlim([0, quantiles[-1]*1.5])
    plt.ylabel('Percentage of simulated outcomes')
    plt.gca().set_xticklabels(
        ['{:.2f}%'.format(p*100) for p in plt.gca().get_xticks()]
    ) 
    plt.gca().set_yticklabels(
        ['{:.2f}%'.format(p*100/num_samples) for p in plt.gca().get_yticks()]
    ) 

    plt.axvline(quantiles[0], color='g', linestyle='dashed', linewidth=2, 
                label='5% quantile {:.4f}% '.format(quantiles[0]*100))
    plt.axvline(quantiles[1], color='r', linestyle='dashed', linewidth=2, 
                label='median {:.4f}% '.format(quantiles[1]*100))
    plt.axvline(quantiles[2], color='purple', linestyle='dashed', linewidth=2, 
                label='95% quantile {:.4f}% '.format(quantiles[2]*100))

    plt.legend()
    # plt.savefig('/home/yz685/ClassroomSimPackage/Results/'+ str(prevalence) + 
    #      '_prevalence_' + str(distancing) + '_ft_distancing_' + class_type +  
    #      '_' + str(p_vax) + '_vaccinated_' + subject + '.pdf')
    plt.savefig('/home/yz685/ClassroomSimPackage/Results/'+ str(distancing) + 
        '_ft_distancing_' + class_type + '_' + str(p_vax) + '_vaccinated_' + 
        subject + '.pdf')
        
if __name__ == '__main__':

    params = yaml.load(open(sys.argv[1]), Loader = yaml.FullLoader)
    print(params)

    plot_outcome_distribution(params, num_samples = 100000)