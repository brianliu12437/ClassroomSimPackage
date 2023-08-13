"""
This script computes the risk of infection for a single classroom, varying the
distancing level, class type, vaccination rate and the VE parameters. 
For each parameter setting, it estimates the per-class risk of infection 
and saves the results in a 2D array, with each row of the form
[VE_susceptible, VE_transmission, p_vax,
mean_num_sec_infs, sd_num_sec_infs,
mean_num_sec_infs_aerosol_only, sd_num_sec_infs_aerosol_only].
"""

import simulate_one_classroom
import pickle
import os
import numpy as np
import random

N = 50
angle = 15
time = 1
# class_types = ['breathing', 'speaking', 'singing']
class_types = ['breathing']
ntrials = 500
air_exchanges_per_hour = 1
seating_function = simulate_one_classroom.generate_clumpy_plan
aerosol_params = {
    'inhale_air_rate': 6.8,
    'dose_response_constant': 1440,
    # "nominal" corresponding to viral load of 10^8 copies / mL
    'nominal_breathe_virus_emitted_hourly': 3300, 
    'nominal_talk_virus_emitted_hourly' : 27300,
    'nominal_sing_virus_emitted_hourly' : 330000,
    'nominal_heavy_breathe_virus_emitted_hourly': 3300*15,
    # distribution of viral load over orders of magnitude from 10^5 to 10^11
    'viral_load_distribution': [0.12, 0.22, 0.3, 0.23, 0.103, 0.0236, 0.0034]    
}

# distancing_vals = [1, 3, 6] 
distancing_vals = [1]
# p_vax_vals = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
p_vax_vals = [0.9]
VE_susceptible_vals = [0.4, 0.42, 0.66, 0.76, 0.79, 0.88]
VE_transmission_vals = [0, 0.5, 0.71]

def load_rooms(distancing):
    if distancing == 1:
        with open("../Data/small_room_info.pickle" , 'rb') as handle:
            layout = pickle.load(handle)
            pixels_per_foot = layout['pixels_per_foot']
            room_vol = layout['volume']
            room = layout['plan']

    elif distancing == 3:
        with open("../Data/medium_room_info.pickle" , 'rb') as handle:
            layout = pickle.load(handle)
            pixels_per_foot = layout['pixels_per_foot']
            room_vol = layout['volume']
            room = layout['plan']

    else: 
        with open("../Data/big_room_info.pickle" , 'rb') as handle:
            layout = pickle.load(handle)
            pixels_per_foot = layout['pixels_per_foot']
            room_vol = layout['volume']
            room = layout['plan']
    
    return room, room_vol, pixels_per_foot


if __name__ == '__main__':

    np.random.seed(0)
    random.seed(0)

    for distancing in distancing_vals:
        
        room, room_vol, pixels_per_foot = load_rooms(distancing)

        for class_type in class_types:

            results_dir = 'Outcomes_vary_params/' + str(distancing) + '_ft_distancing/' + class_type + '/'
            os.makedirs(results_dir, exist_ok = True)
        
            for p_vax in p_vax_vals:
                # store VE x VE results in a table for each (distancing, p_vax)
                print('computing {} ft distancing, {} fraction vaccination'.format(distancing, p_vax))
                results_all_VEs = []
                for VE_susceptible in VE_susceptible_vals:
                    for VE_transmission in VE_transmission_vals:
                        VE_params = {'VE_susceptible': VE_susceptible, 'VE_transmission': VE_transmission}
                        result = simulate_one_classroom.simulate_classroom(N,p_vax,room,seating_function,time,angle,class_type,
                                    room_vol,pixels_per_foot,air_exchanges_per_hour,
                                    VE_params, aerosol_params, ntrials)
                        results_all_VEs.append([VE_susceptible, VE_transmission] + result)
                
                # np.save(results_dir + str(p_vax) + '_p_vax_random_seating.npy', results_all_VEs)
                np.save(results_dir + str(p_vax) + '_p_vax.npy', results_all_VEs)
                print('saved results for {} ft distancing, {} class, {} fraction vaccination'.format(distancing, class_type, p_vax))