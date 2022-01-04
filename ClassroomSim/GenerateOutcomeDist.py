from genericpath import exists
import ClassSimPackage
import pickle
import os
import numpy as np

N = 50
angle = 15
time = 1
class_type = 'breathing'
ntrials = 100
air_exchanges_per_hour = 1
seating_function = ClassSimPackage.generate_clumpy_plan
aerosol_params = {
    'inhale_air_rate': 6.8,
    'dose_response_constant': 1440,
    'nominal_breathe_virus_emitted_hourly': 3300,# corresponding to 10^8 copies / mL
    'nominal_talk_virus_emitted_hourly' : 27300,
    'nominal_sing_virus_emitted_hourly' : 330000,
    'nominal_heavy_breathe_virus_emitted_hourly': 3300*15,
    'viral_load_distribution': [0.12, 0.22, 0.3, 0.23, 0.103, 0.0236, 0.0034] # over orders of magnitude from 10^5 to 10^11
}

distancing_vals = [1, 3, 6]
p_vax_vals = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
VE_susceptible_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
VE_transmission_vals = [0.1, 0.3, 0.5, 0.7, 0.9]

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

    for distancing in distancing_vals:
        results_dir = 'Outcomes_vary_params/' + str(distancing) + '_ft_distancing/'
        os.makedirs(results_dir, exists_ok = True)
        
        room, room_vol, pixels_per_foot = load_rooms(distancing)
        
        for p_vax in p_vax_vals:
            # store VE x VE results in a table for each (distancing, p_vax)
            results_all_VEs = []
            for VE_susceptible in VE_susceptible_vals:
                for VE_transmission in VE_transmission_vals:
                    VE_params = {'VE_susceptible': VE_susceptible, 'VE_transmission': VE_transmission}
                    result = ClassSimPackage.simulate_classroom(N,p,room,seating_function,time,angle,class_type,
                                room_vol,pixels_per_foot,air_exchanges_per_hour,
                                VE_params, aerosol_params, ntrials)
                    results_all_VEs.append([VE_susceptible, VE_transmission] + result)
            
            np.save(results_dir + str(p_vax) + '_p_vax.npy', results_all_VEs)