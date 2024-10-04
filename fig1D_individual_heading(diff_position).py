#!/usr/bin/env python

#### 7/5 NOTE: Need to add in code to get rid of wrapping issue Otherwise code seems to work pretty well

#### NOTE: Need to double check if this calculates the mean rotated heading properly


import pandas as pd
import csv
import matplotlib.pyplot as plt
from dateutil import parser
import numpy as np
import scipy as sp
import os
import glob
from circstats import difference, wrapdiff
import matplotlib.ticker as ticker

# def full_arctan(x,y):
#     return np.arctan(y/x)+np.pi if x<0 else np.arctan(y/x)
def full_arctan(x,y):                                           #makes all the angles between -180 to 180
    angle = np.arctan(y/x)+np.pi if x<0 else np.arctan(y/x)
    return angle if angle <= np.pi else angle -2*np.pi 

def angle_avg(data):
    return full_arctan(np.cos(data*np.pi/180).sum(),np.sin(data*np.pi/180).sum())*180/np.pi


def sun_angle(data):
    if data == 19:
        return 135
    elif data == 57:
        return 45
    elif data == 93:
        return -45
    elif data == 129:
        return -135
    else:
        return 0
    
plt.rcParams["figure.figsize"] =[9.00, 3.00]
plt.rcParams["figure.autolayout"] = True

path = r'C:\Users\angel\OneDrive - email.ucr.edu\Desktop\Giraldo Lab\Data\Long Flight\20min_4suns_HCS' #Change this to where data is
experiment_names = glob.glob(os.path.join(path, "*.csv"))
experiment_names.sort(reverse=False)


for experiment_name in experiment_names:
    df = pd.read_csv(experiment_name)
    experiment_data = df.values
    df['Image Time'] = df['Image Time'].apply(parser.parse)
    sun_change_indexes = [0]+[i for i in range(1,len(experiment_data)) if experiment_data[i,3]!=experiment_data[i-1,3]]+[len(experiment_data)] 
    #print('sun_change_indexes:', sun_change_indexes)
    #sun_change_indexes show dark period(no sun stimulus)
    sun_changes = sun_change_indexes[:-1]
    sun_periods = [df[sun_changes[i-1]:sun_changes[i]] for i in range(1,len(sun_changes))]+[df[sun_changes[-1]:-1]]

    #trial_period = 1200
    #time_delay = 0
    trial_period = 900
    sun_position = []
    heading_angles=[] ###Change range according to the number of suns
    rotated_heading_angles = []
    ##This is calculated per fly do not need 

    for i in range(len(sun_periods)):
        if i!=0:
            sun_period = sun_periods[i]
            delayed_sun_period = sun_period.loc[[(frame_time - sun_period['Image Time'].iloc[0]).seconds<=trial_period for frame_time in sun_period['Image Time']]]
            print('delayed sun period')
            print(delayed_sun_period)
            headings= np.array(delayed_sun_period['Heading Angle'])

            sun_position.append(sun_angle(delayed_sun_period['Sun Position'].iloc[0]))

            heading_angles.append(angle_avg(delayed_sun_period['Heading Angle'])) #### 1. When plotting raw data without sun correction
            
    sun_position = np.array(sun_position)
    heading_angles= np.array(heading_angles)   


    rotated_heading_angles.append(difference(heading_angles, sun_position, deg=True)) ### 2. When plotting raw data with sun correction

    print('sun_positions', sun_position)
    print('heading_angles',heading_angles)
    print('rotated_heading_angles:', rotated_heading_angles)

    ### Plot raw data
    sun_change_ticks = sun_change_indexes[:-1] if sun_change_indexes[-1] >= len(df) else sun_change_indexes
    sun_change_times = [df['Frame'].iloc[i] for i in sun_change_ticks]
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Angle', fontsize=18)
    plt.ylim(180,-180)
    fig = plt.figure(figsize=(9,3))
    fig.set_facecolor('w')
    ax = fig.add_subplot(1,1,1)
    for axis in ['left','bottom']:
            ax.spines[axis].set_linewidth(3.0)
    ax.xaxis.set_tick_params(width=3.0, length=6.0, direction = 'in')
    ax.yaxis.set_tick_params(width=3.0, length=6.0, direction = 'in') 

    ax.set_yticks((-180, 0, 180))
    ax.set_xticks(sun_change_ticks)
    #ax.set_xticks((0, 750, 7500, 15000, 22500, 30000, 37500))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_bounds(-180,180)
    ax.spines['bottom'].set_bounds(0,sun_change_indexes[-1] )
    ax.set_yticklabels(('-180', '0', '180'), color='k', fontsize=18)
    #ax.set_xticklabels(('0','750', '7500', '15000', '22500', '30000', '37500'), color='k', fontsize=18)
    ax.set_xticklabels(sun_change_times, rotation='horizontal', fontsize =18)

    for i in range(len(sun_change_indexes)-2):                          #Plots red arrow to indicate sun position change
        plt.arrow(sun_change_indexes[i+1], 240, 0, -20, color='r', width=100, head_width=600, head_length=20)


    heading_angles_array = df['Heading Angle'].values
    print('heading_angles_array')
    print(heading_angles_array)
    sun_position_array = df['Sun Position'].values
    #Plots all the heading during flight but as the difference between raw heading angle and sun position (Absolute Heading) 
    # Sun is positioned at 0 
    # rotated_angles = difference(heading_angles_array, sun_position_array, deg = True)
    # plt.plot(df['Frame'], rotated_angles,color = 'k', linewidth=2, label='First Flight') #for plotting rotated_heading
    # #plt.axhline(angle_avg(rotated_heading_angles[0]), color='blue') 

    # for i in range(len(sun_change_indexes)-2):
    #     constant_sun_range = (sun_change_indexes[i+1], sun_change_indexes[i+2])
    #     rotated_angles_with_sun= rotated_angles[sun_change_indexes[i+1]: sun_change_indexes[i+2]]
    #     avg_rotated_angle_with_sun = angle_avg(rotated_angles_with_sun)
    #     plt.plot(constant_sun_range,  [avg_rotated_angle_with_sun for _ in constant_sun_range], color ='blue', linewidth = 3)
    
    # total_sun_range= (sun_change_indexes[1], sun_change_indexes[-1])
    # plt.plot(total_sun_range, [[0]for _ in total_sun_range], color='r', linestyle ='--', linewidth = 3)


#Plot raw heading with actual sun position change
    for i in range(len(sun_change_indexes)-1):  
        if i !=0:
            constant_sun_range = range(sun_change_indexes[i],sun_change_indexes[i+1])
            print(constant_sun_range)
            headings_per_sun = heading_angles_array[sun_change_indexes[i]: sun_change_indexes[i+1]]
            print('headings per sun:', headings_per_sun)
            avg_heading_per_sun = angle_avg(headings_per_sun)
            print('average heading per period:', avg_heading_per_sun)
            plt.plot(constant_sun_range, [sun_angle(experiment_data[sun_change_indexes[i],3]) for _ in constant_sun_range],color ='red',  linestyle ='--',linewidth = 3)
            plt.plot(constant_sun_range, [avg_heading_per_sun for _ in constant_sun_range], color='blue', linewidth =3 ) #to show mean heading angle per sun period

            
    plt.plot(df['Frame'], heading_angles_array,color = 'k', linewidth=2, label='Raw heading')
    #Automatically saves figure in dedicated directory as fly number
    csv_filename = os.path.basename(experiment_name)
    fly_number = csv_filename.split('_')[0]
    plt.title(fly_number)
    output_directory = r'C:\Users\angel\OneDrive - email.ucr.edu\Desktop\Giraldo Lab\Data\Long Flight\20min_4suns_HCS\20min_4suns_raw_data'#Change this to fwhere you want to save images
    output_filename = f"{fly_number}.png"
    output_path = os.path.join(output_directory, output_filename)

    plt.savefig(output_path, transparent=True, dpi=600)

plt.show()

