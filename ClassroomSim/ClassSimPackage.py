
import numpy as np
import pandas as pd
import itertools
import random
import math
import warnings
from helpers import *

# Generate seating plans
def generate_clumpy_plan(N, p_vax, room, clump_size = 3):
    """ 
    Generates a seating plan where the unvaccinated students sit together
    in clumps. Represents the worst case scenario.
    INPUTS:
    N: the number of students in the classroom
    p_vax: fraction of individuals in the room vaccinated
    room: type of room
    clump_size: the size of the "clump" of unvaccinated individuals that tend to sit together
    """
    Nvax = np.random.binomial(N, p_vax)
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

def generate_random_plan(N, p_vax, room):
    """ 
    Generates a seating plan where students sit randomly in the room
    """
    Nvax = round(N*p_vax)
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

def droplet_risk_by_distance(d, susceptible_status, source_status,
                            time, pixels_per_foot, class_type, VE_params):

    """
    Computes the risk that a susceptible person is infected through droplet transmission
    by a source case seated a certain distance away from him/her, depending on the vaccination status of both
    individuals, exposure time, VE parameters, and class type.
    (The risk of droplet transmission decreases with distance.)
    INPUTS:
    d: distance between source and susceptible, measured in pixels.
    susceptible status: 'V' = vaccinated, 'U' = unvaccinated
    source_status: 'V' = vaccinated, 'U' = unvaccinated
    time: length of exposure, measured in hours
    pixels_per_foot: a parameter for translating distance in feet to the number of pixels in the room layout
    class_type: one of ['breathing', 'speaking', 'singing', 'heavy_breathing']
    VE_params: dictionary for vaccine effectiveness parameters. 
               'VE_susceptible' = reduction in infection risk for a vaccinated susceptible person
               'VE_transmission' = reduction in the viral load emitted by a vaccinated infectious person
    """

    d = d/pixels_per_foot*0.3048
    susceptible_const = 1
    source_const = 1

    if susceptible_status == 'V':
        # probabiltiy of infection of a susceptible individual is scaled by susceptible_const if he/she is vaccinated
        susceptible_const = 1 - VE_params['VE_susceptible']
    if source_status == 'V':
        # viral load of the source is reduced by source_const if he/she is vaccinated
        source_const = (1 - VE_params['VE_transmission'])

    if class_type == 'speaking':
        source_const = source_const*6
    elif class_type == 'singing':
        source_const = source_const*48
    elif class_type == 'heavy_breathing':
        source_const = source_const*15

    droplet_transmission_prob = 2.4 * susceptible_const *(1-np.exp(-1 * source_const *\
                     0.0135 * time * (-0.1819*np.log(d)+0.43276)/d))

    droplet_transmission_prob = float(droplet_transmission_prob)
    return max(droplet_transmission_prob, 0)


def aerosol_risk(room_vol, source_is_vax, time,
                    class_type, VE_params, aerosol_params):
    """
    Computes the risk due to aerosol transmission for a susceptible person. 
    The risk of aerosol transmission is assumed to be uniform over all distances.
    INPUTS:
    room_vol: volume of the room, measured in cubic meters
    source_is_vax: 1 if the source is vaccinated, 0 otherwise
    time: exposure time, measured in hours
    class_type: one of ['breathing', 'speaking', 'singing', 'heavy_breathing', 'no_aerosol']. 
                The 'no_aerosol' scenario applies to outdoor settings, where aerosols are quickly diluted by airflow.
    VE_params: dictionary of vaccine effectiveness parameters. 
               'VE_susceptible' = reduction in infection risk for a vaccinated susceptible person
               'VE_transmission' = reduction in the viral load emitted by a vaccinated infectious person
    aerosol_params: dictionary of parameters for the aerosol transmission model
    """

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

    if source_is_vax == 1:
        dose_array = dose_array * (1 - VE_params['VE_transmission'])

    effective_dose_array = dose_array / aerosol_params['dose_response_constant']
    unvax_susceptible_risk_array = 1 - np.exp(-effective_dose_array)
    unvax_susceptible_risk = np.dot(unvax_susceptible_risk_array, np.array(aerosol_params['viral_load_distribution']))
    unvax_susceptible_risk_over_time = 2.4*unvax_susceptible_risk * time # scale by 2.4 for Delta

    return unvax_susceptible_risk_over_time



def simulate_single_trial(grid,source_is_vax, 
                            time,angle,class_type,room_vol, N,
                            pixels_per_foot, air_exchanges_per_hour,
                            VE_params,aerosol_params):
    """
    Computes two quantities:
    (1) the expected number of secondary infections given one source case in a room
    (2) the risk of aerosol transmission alone (this can be used to calculate the risk of a socially distant instructor)
    from running one trial of simulation.
    INPUTS:
    grid: a room plan with seating information (for each person, pixel of the seating location and vaccination status)
    source_is_vax: 1 if source is vaccinated, 0 otherwise
    time: length of exposure, measured in hours
    angle: defines the "cone of exposure" in the droplet transmission model. 
            Susceptibles outside of this cone are assumed to be not at risk.
    class_type: one of ['breathing', 'speaking', 'singing', 'heavy_breathing', 'no_aerosol']. 
    room_vol: volume of the room, measured in cubic meters
    N: number of individuals in the room
    pixels_per_foot: a parameter for translating distance in feet to the number of pixels in the room layout
    air_exchanges_per_hour: rate of ventilation that reduces the risk of aerosol transmission. 
            We assume that 1 air exchange per hour reduces the risk by half, 2 reduces by third, etc.
            This parameter does not affect the droplet transmission risk.
    VE_params: dictionary of vaccine effectiveness parameters. 
               'VE_susceptible' = reduction in infection risk for a vaccinated susceptible person
               'VE_transmission' = reduction in the viral load emitted by a vaccinated infectious person
    aerosol_params: dictionary of parameters for the aerosol transmission model    
    """

    grid = grid.reset_index()
    vax_source_id = random.sample(list(grid[grid['seating'] == 'V']['index'].values), source_is_vax)
    unvax_source_id = random.sample(list(grid[grid['seating'] == 'U']['index'].values), 1-source_is_vax)
    infected = grid[grid['index'].isin(np.append(vax_source_id, unvax_source_id))]
    uninfected = grid[ (~grid['index'].isin(np.append(vax_source_id, unvax_source_id))) & \
                      (grid['seating'] != 'E')  ]

    if source_is_vax == 1:
        source_status = 'V'
    else:
        source_status = 'U'

    infected_x = infected['x']
    infected_y = infected['y']

    # 1 air exchange per hour --> aerosols reduced by half;
    # 2 air exchanges per hour --> aerosols reduced by a third, etc.
    unvax_aerosol_risk = 1/(air_exchanges_per_hour+1)*\
        aerosol_risk(room_vol, source_is_vax, time,\
                            class_type, VE_params, aerosol_params)

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
            susceptible_aerosol_risk = susceptible_aerosol_risk * (1-VE_params['VE_susceptible'])

        # if the susceptible is seated within the cone of exposure, he/she is subject to
        # the risk of droplet transmission
        if theta < 90 + angle:
            dist = np.sqrt((infected_x-x)**2+(infected_y-y)**2)
            p = droplet_risk_by_distance(dist,susceptible_status,source_status,\
                                time,pixels_per_foot,class_type,VE_params)
            p = max(p,susceptible_aerosol_risk)
            p_infections.append(p)
        # if the susceptible is outside the cone, he/she is subject to aerosol risk only
        else:
            p_infections.append(susceptible_aerosol_risk)

    return np.sum(p_infections), unvax_aerosol_risk


def simulate_classroom(N, p_vax, room, seating_function, time, angle, class_type,
                            room_vol, pixels_per_foot, air_exchanges_per_hour,
                            VE_params, aerosol_params, ntrials):
    """
    Simulate ntrials number of trials. Returns a 5-dimensional array with
    (1) p_vax: probability that a person is vaccinated
    (2) mean and SD of the number of secondary infections generated by one infectious person
    (2) mean and SD of the risk of aerosol transmission for an unvaccinated person 
    INPUTS:
    seating_function: generate_random_plan() or generate_clumpy_plan(). Allocates vaccinated
                and unvaccinated individuals to eligible seats in a room.
    ntrials: the number of trials to run the simulation for.
    """

    warnings.filterwarnings('ignore')
    trial = 0
    results = []
    aerosol_results = []
    while trial < ntrials:
        grid = seating_function(N, p_vax, room)
        # probability that an infected person is vaccinated
        p_inf_is_vax = (1-VE_params['VE_susceptible'])*p_vax/(1-VE_params['VE_susceptible']*p_vax)
        inf_is_vax = flip(p_inf_is_vax)

        if sum(grid['seating'] == 'V') == 0:
            inf_is_vax = 0
        elif sum(grid['seating'] == 'U') == 0:
            inf_is_vax= 1

        infect, aerosol = simulate_single_trial(grid, inf_is_vax,
                                                time, angle, class_type, room_vol, N,
                                                pixels_per_foot, air_exchanges_per_hour,
                                                VE_params, aerosol_params)
        results.append(infect)
        aerosol_results.append(aerosol)
        trial = trial + 1

    return [p_vax, np.mean(results), np.std(results), np.mean(aerosol_results), np.std(aerosol_results)]


# TODO: run multiple parameter values (rather than point estimates)