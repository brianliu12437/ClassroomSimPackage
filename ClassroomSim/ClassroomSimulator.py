import argparse
import sys

import numpy as np
import pandas as pd
import pickle

from ClassSimPackage import simulator as sim

#Parse in command line arguments
parser = argparse.ArgumentParser()

parser.add_argument(dest='distancing', choices = ['1','3','6'], type = str,
            help="Social distancing configuration in feet: [1,3,6]")

parser.add_argument(dest='masking', choices = [True,False], type = bool, help= \
            "Masks required: [True, False]")

parser.add_argument(dest='behavior', choices = ['breathing','speaking' \
            ,'singing','heavy_breathing'], type = str,
            help= "Classroom behavior: ['breathing','speaking','singing']")

parser.add_argument(dest='proportion_class_vaccinated',type = float,
            help= "Proportion of the class vaccinated")

parser.add_argument(dest='vaccine_efficacy',type = float,
            help= "Vaccine efficacy")

parser.add_argument(dest='vaccine_efficacy_transmission',type = float,
            help= "Vaccine efficacy against transmission")

parser.add_argument(dest = 'ntrials', type = int,
            help = 'Number of repitions to run the simulator')

parser.add_argument(dest='seating', choices = ['random','clumpy'], type = str,
            help= "Classroom seating options: [True, False]")

args = parser.parse_args()

#load in room types
if args.distancing == '1':
    with open("../Data/small_room_info.pickle" , 'rb') as handle:
        layout = pickle.load(handle)
        pixels_per_foot = layout['pixels_per_foot']
        volume = layout['volume']
        plan = layout['plan']

elif args.distancing == '3':
    with open("../Data/medium_room_info.pickle" , 'rb') as handle:
        layout = pickle.load(handle)
        pixels_per_foot = layout['pixels_per_foot']
        volume = layout['volume']
        plan = layout['plan']

else:
    with open("../Data/big_room_info.pickle" , 'rb') as handle:
        layout = pickle.load(handle)
        pixels_per_foot = layout['pixels_per_foot']
        volume = layout['volume']
        plan = layout['plan']

#load in aerosol risks:
if args.behavior == 'breathing':
    nominal_virus_emitted_hourly = 3300

elif args.behavior == 'speaking':
    nominal_virus_emitted_hourly = 27300

elif args.behavior == 'singing':
    nominal_virus_emitted_hourly = 330000

else:
    nominal_virus_emitted_hourly = 3300*15
