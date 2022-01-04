import argparse
import sys

import numpy as np
import pandas as pd
import pickle

import ClassSimPackage

#Parse in command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--distancing', dest='distancing', choices = ['1','3','6'], type = str,
            help="Social distancing configuration in feet: [1,3,6]")

parser.add_argument('--masking', dest='masking', choices = [True,False], type = bool, help= \
            "Masks required: [True, False]")

parser.add_argument('--behavior', dest='behavior', choices = ['breathing','speaking' \
            ,'singing','heavy_breathing','no_aerosol'], type = str,
            help= "Classroom behavior: \
            ['breathing','speaking','singing','heavy_breathing','no_aerosol']")

parser.add_argument('--frac_vax', dest='proportion_class_vaccinated',type = float,
            help= "Proportion of the class vaccinated")

parser.add_argument('--VE_susceptible', dest='vaccine_efficacy',type = float, nargs='+', action='append',
            help= "Vaccine efficacy values for a susceptible person (default: discrete uniform dist over supplied values)")

parser.add_argument('--VE_transmission', dest='vaccine_efficacy_transmission', type = float, nargs='+', action='append',
            help= "Vaccine efficacy values against transmission (default: discrete uniform dist over supplied values)")

parser.add_argument('--seating_pattern', dest='seating', choices = ['random','clumpy'], type = str,
            help= "Classroom seating options: [True, False]")

parser.add_argument('--hours_in_class', dest='time',type = float,
            help= "Time (in hours) spent in classroom")

parser.add_argument('--ACH', dest='air_exchanges_per_hour',type = float,
            help= "Air exchanges per hour in the room")

parser.add_argument('--ntrials', dest = 'ntrials', type = int,
            help = 'Number of repetitions to run the simulator')

parser.add_argument('--prevalence', dest = 'prevalence', type = float, 
            help = 'Prevalence among attendees')

parser.add_argument('--vax_rate', dest = 'vax_rate', type = float,
            help = 'Fraction of attendees vaccinated')

parser.add_argument('--nsamples', dest = 'nsamples', type = int,
            help = 'Number of samples for generating the outcome distribution')


args = parser.parse_args()

#load in room types
if args.distancing == '1':
    with open("../Data/small_room_info.pickle" , 'rb') as handle:
        layout = pickle.load(handle)
        pixels_per_foot = layout['pixels_per_foot']
        room_vol = layout['volume']
        room = layout['plan']

elif args.distancing == '3':
    with open("../Data/medium_room_info.pickle" , 'rb') as handle:
        layout = pickle.load(handle)
        pixels_per_foot = layout['pixels_per_foot']
        room_vol = layout['volume']
        room = layout['plan']

else: # 6 ft distancing
    with open("../Data/big_room_info.pickle" , 'rb') as handle:
        layout = pickle.load(handle)
        pixels_per_foot = layout['pixels_per_foot']
        room_vol = layout['volume']
        room = layout['plan']

#load seating function
if args.seating == 'random':
    seating_function = ClassSimPackage.generate_random_plan
else:
    seating_function = ClassSimPackage.generate_clumpy_plan

#Start Sim
N = 50
angle = 15
p = args.proportion_class_vaccinated
time = args.time
class_type = args.behavior
ntrials = args.ntrials
air_exchanges_per_hour = args.air_exchanges_per_hour

# TODO: change this for every VE-VE combo
# take cartesian product between the two arguments
VE_params = {
            'VE_transmission':args.vaccine_efficacy_transmission,
            'VE_susceptible': args.vaccine_efficacy
            }

aerosol_params = {
    'inhale_air_rate': 6.8,
    'dose_response_constant': 1440,
    'nominal_breathe_virus_emitted_hourly': 3300,# corresponding to 10^8 copies / mL
    'nominal_talk_virus_emitted_hourly' : 27300,
    'nominal_sing_virus_emitted_hourly' : 330000,
    'nominal_heavy_breathe_virus_emitted_hourly': 3300*15,
    'viral_load_distribution': [0.12, 0.22, 0.3, 0.23, 0.103, 0.0236, 0.0034] # over orders of magnitude from 10^5 to 10^11
}

# TODO: add default distribution for masking effectiveness (I imagine this won't change much, so we don't need argparse for it)

mask_eff_mean, mask_eff_sd = 0.145, 0.0536

test = ClassSimPackage.simulate_classroom(N,p,room,seating_function,time,angle,class_type,
                            room_vol,pixels_per_foot,air_exchanges_per_hour,
                            VE_params, aerosol_params, ntrials)
print(test)


# TODO: add function for sampling from results with varying VE-VE combo and masking effectiveness distribution
# TODO: function for plotting the histogram and printing quantiles
