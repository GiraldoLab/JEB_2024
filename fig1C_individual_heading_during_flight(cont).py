#!/usr/bin/env python

#Plots each fly's heading during continuous flight 

import pandas as pd
import csv
import matplotlib.pyplot as plt
from dateutil import parser
import numpy as np
import glob
import os
import matplotlib.ticker as ticker


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
    elif data == 97:
        return -55
    else:
        return 0
    

# frame_rate = 25 #frames per second

# def convert_frame_to_time(frame):
#     return frame/ frame_rate
plt.rcParams["figure.figsize"] =[9.0, 3.0]
plt.rcParams["figure.autolayout"] = True



path = r'C:\Users\angel\OneDrive - email.ucr.edu\Desktop\Giraldo Lab\Data\Long Flight\30min_control_HCS'
experiment_names = glob.glob(os.path.join(path, "*.csv"))
experiment_names.sort(reverse=False)

for experiment_name in experiment_names:
    csv_filename = os.path.basename(experiment_name)
    print(csv_filename)
    df = pd.read_csv(csv_filename)
    experiment_data = df.values
    df['Image Time'] = df['Image Time'].apply(parser.parse)
    #df['Frame Time'] = df['Frame'].apply(convert_frame_to_time)
    sun_change_indexes = [0]+[i for i in range(1,len(experiment_data)) if experiment_data[i,3]!=experiment_data[i-1,3]]+[len(experiment_data)] 
    print('sun_change_indexes:', sun_change_indexes) 
    print(range(len(sun_change_indexes)))
    #sun_change_indexes show dark period(no sun stimulus)
    sun_changes = sun_change_indexes[:-1]
    sun_periods = [df[sun_changes[i-1]:sun_changes[i]] for i in range(1,len(sun_changes))]+[df[sun_changes[-1]:-1]]
    #print('Sun Period:', sun_periods)


    time_delay = 0
    trial_period = 900
    heading_angles=[[],[]]

    for i in range(len(sun_periods)):
        if i!=0:
            sun_period = sun_periods[i]
            delayed_sun_period = sun_period.loc[[(frame_time - sun_period['Image Time'].iloc[0]).seconds<= trial_period for frame_time in sun_period['Image Time']]]
            heading_angles[i//2].append(angle_avg(delayed_sun_period['Heading Angle']))



    #plt.title("Fly Raw Heading Data", fontsize=24)
    frame_rate = 25.0 #frames per second

    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Angle', fontsize=18)
    plt.ylim(180,-180)
    fig = plt.figure(figsize=(9,3))
    fig.set_facecolor('w')
    ax = fig.add_subplot(1,1,1)
    for axis in ['left','bottom']:
            ax.spines[axis].set_linewidth(3.0)
    ax.xaxis.set_tick_params(width=3, length=6.0, direction = 'in')
    ax.yaxis.set_tick_params(width=3, length=6.0, direction = 'in') 
    #ax.yaxis.set_major_formatter(ticker.FuncFormatter(convert_frame_to_time))
    ax.set_yticks((-180, 0, 180))
    ax.set_xticks([i/frame_rate for i in (0, 750, 7500, 15000, 22500, 30000, 37500, 46000)])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_bounds(-180,180)
    ax.spines['bottom'].set_bounds(0, sun_change_indexes[-1]/frame_rate)  # convert frame number to seconds for set_bounds
    ax.set_yticklabels(('-180', '0', '180'), color='k', fontsize=18)
    ax.set_xticklabels([str(int(i/frame_rate)) for i in [0, 750, 7500, 15000, 22500, 30000, 37500, 46000]], color='k', fontsize=18)

    # Plot an arrow where sun changes
for i in range(len(sun_change_indexes)-2):
    plt.arrow(sun_change_indexes[i+1]/frame_rate, 240, 0, -20, color='r', width=100, head_width=600, head_length=20)  # convert frame number to seconds for arrow positions
    plt.plot(df['Frame']/frame_rate, df['Heading Angle'],color = 'k', linewidth=2, label='First Flight')
    for i in range(len(sun_change_indexes)-1):  
        if i!=0:
            constant_sun_range = range(sun_change_indexes[i],sun_change_indexes[i+1])
            #print('constant_sun_range:', [sun_angle(experiment_data[sun_change_indexes[i],3]) for _ in constant_sun_range])
            first_sun = range(sun_change_indexes[1],sun_change_indexes[2])
        
            fly_number = csv_filename.split('_')[0]
    
            plt.plot(constant_sun_range,[sun_angle(experiment_data[sun_change_indexes[i],3]) for _ in constant_sun_range],color ='red',  linestyle ='--',linewidth = 3)
            plt.plot(first_sun,[heading_angles[0][0] for _ in first_sun] , color='blue', linewidth =3)

    plt.title(fly_number)
    output_directory = r'C:\Users\angel\OneDrive - email.ucr.edu\Desktop\Giraldo Lab\Data\Long Flight\30min_control_HCS\30_20min Control Raw Data'
    output_filename = f"{fly_number}.png"
    output_path = os.path.join(output_directory, output_filename)

    plt.savefig(output_path, transparent=True, dpi=600)

plt.show()

