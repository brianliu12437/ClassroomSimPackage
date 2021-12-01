
import numpy as np
import pandas as pd
import itertools
import random
import math
import warnings

#helper functions
def Diff(li1, li2):
    """
    Difference of sets
    """
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif


def flip(p):
    """
    Biased coin flipper
    """
    return 1 if random.random() < p else 0

# plan generation functions
def generate_clumpy_plan(N,p,room, clump_size = 3):
    """ Generates a seating plan where the unvaccinated students sit together
    in clumps. Represents the worst case scenario.
    """
    Nvax = np.random.binomial(N,p)
    Nunvax = N-Nvax
    room = room.drop('UnvaxSpot',axis = 1).reset_index()
    grid = room.copy()
    grid['seating'] = 'E'

    if Nunvax > 0:
        clump_size = min(clump_size,Nunvax)
        num_clumps = math.floor(Nunvax/clump_size)
        remainder = Nunvax - num_clumps*clump_size
        clump = 0

        while clump < (num_clumps):
            ind1 = np.random.choice(room['index'].values,replace = False)
            grid['seating'].loc[grid['index'] == ind1] = 'U'
            room = room.drop(ind1, axis = 0)
            x_temp = grid['x'].loc[grid['index'] == ind1].values[0]
            y_temp = grid['y'].loc[grid['index'] == ind1].values[0]
            temp = room.copy()
            temp['dist_infected'] = ((temp['x'] -x_temp) ** 2 + (temp['y'] - y_temp) ** 2) ** 0.5
            temp = temp.sort_values('dist_infected', ascending = True).head(clump_size-1)
            grid['seating'].loc[grid['index'].isin(temp['index'].values)] = 'U'
            room = room.drop(temp['index'].values, axis = 0)
            clump = clump + 1

        remainder_ind = np.random.choice(grid['index'].loc[grid['seating'] == 'E'],remainder,replace = False)
        grid['seating'].loc[grid['index'].isin(remainder_ind)] = 'U'

    vax_ind = np.random.choice(grid['index'].loc[grid['seating'] == 'E'],Nvax,replace = False)
    grid['seating'].loc[grid['index'].isin(vax_ind)] = 'V'

    return grid

def generate_random_plan(N,p,room):
    """ Generates a seating plan where students sit randomly in the room
    """
    Nvax = round(N*p)
    Nunvax = N-Nvax
    room = room.drop('UnvaxSpot',axis = 1)
    grid = room.copy()
    grid = grid.reset_index()
    temp = list(np.append(np.append(np.repeat('V',Nvax),\
            np.repeat('U',Nunvax)),np.repeat('E',len(grid)-Nvax-Nunvax)))
    random.shuffle(temp)
    grid['seating'] = temp
    return grid


    #simulator functions

def distance_to_prob(d,susceptible_status,source_status,
                    time,pixels_per_foot,class_type,class_risk_params):

    d = d/pixels_per_foot*0.3048
    susceptible_const = 1
    source_const = 1

    if susceptible_status == 'V':
        susceptible_const = 1 - class_risk_params['vax_effectiveness']
    if source_status == 'V':
        source_const = (1 - class_risk_params['vax_effectiveness_transmission'])

    if class_type == 'speaking':
        source_const = source_const*6
    elif class_type == 'singing':
        source_const = source_const*48
    elif class_type == 'heavy_breathing':
        source_const = source_const*15

    prob = 2.4*susceptible_const  *(1-np.exp(-1 * source_const *\
                     0.0135 *time * (-0.1819*np.log(d)+0.43276)/d))

    prob = float(prob)
    return max(prob,0)


def generate_aerosol_risk(room_vol, vax_infected, unvax_infected, time,
                    class_type,class_risk_params,aerosol_params):

    if class_type == 'breathing':
        v = aerosol_params['nominal_breathe_virus_emitted_hourly']
    elif class_type == 'speaking':
        v = aerosol_params['nominal_talk_virus_emitted_hourly']
    elif class_type == 'singing':
        v = aerosol_params['nominal_sing_virus_emitted_hourly']
    elif class_type == 'heavy_breathing':
        v = aerosol_params['nominal_heavy_breathe_virus_emitted_hourly']
    elif class_type == 'no_aerosol':
        return 0

    hourly_virus_array = np.array([v/1000, v/100, v/10, v, v*10, v*100, v*1000])
    dose_array = hourly_virus_array * aerosol_params['inhale_air_rate'] / room_vol

    if vax_infected == 1:
        dose_array = dose_array * (1 - class_risk_params['vax_effectiveness_transmission'])

    effective_dose_array = dose_array / aerosol_params['dose_response_constant']
    unvax_susceptible_risk_array = 1 - np.exp(-effective_dose_array)
    unvax_susceptible_risk = np.dot(unvax_susceptible_risk_array, np.array(aerosol_params['viral_load_distribution']))
    unvax_susceptible_risk_over_time = 2.4*unvax_susceptible_risk * time

    return unvax_susceptible_risk_over_time




def simulate_single_trial(room,vax_infected, unvax_infected,
                                    time,angle,class_type,room_vol,N,
                                    pixels_per_foot, air_exchanges_per_hour,
                                    class_risk_params,aerosol_params):

    room = room.reset_index()
    vax_infect_id = random.sample(list(room[room['seating'] == 'V']['index'].values), vax_infected)
    unvax_infect_id = random.sample(list(room[room['seating'] == 'U']['index'].values), unvax_infected)
    infected = room[room['index'].isin(np.append(vax_infect_id,unvax_infect_id))]
    uninfected = room[ (~room['index'].isin(np.append(vax_infect_id,unvax_infect_id))) & \
                      (room['seating'] != 'E')  ]

    if vax_infected == 1:
        source_status = 'V'
    elif unvax_infected == 1:
        source_status = 'U'

    infected_x = infected['x']
    infected_y = infected['y']

    unvax_aerosol_risk = (1/air_exchanges_per_hour)*\
        generate_aerosol_risk(room_vol, vax_infected, unvax_infected, time,\
                            class_type,class_risk_params,aerosol_params)

    p_infections = []
    for i,row in uninfected.iterrows():
        x = row['x']
        y = row['y']
        v_infect = [x-infected_x,y - infected_y]
        v_vert = [0,10000]
        dot = v_infect[1]*v_vert[1]
        mag1 = 10000
        mag2 = np.sqrt((x-infected_x)**2+(y - infected_y)**2)
        theta =  math.acos(dot/(mag1*mag2))*180/math.pi
        susceptible_status = row['seating']
        susceptible_aerosol_risk = unvax_aerosol_risk

        if susceptible_status == 'V':
            susceptible_aerosol_risk = susceptible_aerosol_risk * (1-class_risk_params['vax_effectiveness'])

        if theta < 90 + angle:
            dist = np.sqrt((infected_x-x)**2+(infected_y-y)**2)
            p = distance_to_prob(dist,susceptible_status,source_status,\
                                time,pixels_per_foot,class_type,class_risk_params)
            p = max(p,susceptible_aerosol_risk)
            p_infections.append(p)
        else:
            p_infections.append(susceptible_aerosol_risk)

    return np.sum(p_infections),unvax_aerosol_risk


def simulate_classroom(N,p,room,seating_function,time,angle,class_type ,
                            room_vol,pixels_per_foot,air_exchanges_per_hour,
                            class_risk_params ,aerosol_params, ntrials):

    warnings.filterwarnings('ignore')
    trial = 0
    results = []
    aerosol_results = []
    while trial < ntrials:
        grid = seating_function(N,p,room)
        p_generate = (1-class_risk_params['vax_effectiveness'])*p/(1-class_risk_params['vax_effectiveness']*p)
        ind = flip(p_generate)

        if sum(grid['seating'] == 'V') == 0:
            ind = 0
        elif sum(grid['seating'] == 'U') == 0:
            ind = 1

        infect,aerosol = simulate_single_trial(grid,ind, 1-ind,
                                                time,angle,class_type,room_vol,N,
                                                pixels_per_foot, air_exchanges_per_hour,
                                                class_risk_params,aerosol_params)
        results.append(infect)
        aerosol_results.append(aerosol)
        trial = trial + 1

    return [p,np.mean(results),np.std(results), np.mean(aerosol_results), np.std(aerosol_results)]
