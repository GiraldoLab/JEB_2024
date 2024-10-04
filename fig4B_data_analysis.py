#!/usr/bin/env python
# Data analysis code for fig4B - plots the heading difference between the first sun period headings and the third sun period headings as a polar plot when the sun moves 15 degrees every 5 minutes
# Plots heading difference for the rotated headings and the real world headings

from turtle import delay
import matplotlib.pyplot as plt
import csv
import pandas as pd
from dateutil import parser
import numpy as np
import scipy as sp
import os
import glob
from circstats import difference, wrapdiff

#Calculates mean angle(degrees)
def full_arctan(x,y):                                           #makes all the angles between -180 to 180
    angle = np.arctan(y/x)+np.pi if x<0 else np.arctan(y/x)
    return angle if angle <= np.pi else angle -2*np.pi 
def angle_avg(data):
    return full_arctan(np.cos(data*np.pi/180).sum(),np.sin(data*np.pi/180).sum())*180/np.pi

#Calculates mean angle(radians)
def circmean(alpha,axis=None):  ##when angles are in radians
    mean_angle = np.arctan2(np.mean(np.sin(alpha),axis),np.mean(np.cos(alpha),axis))
    return mean_angle

#Calculates angle variance(radians)
def circvar(alpha,axis=None):
    if np.ma.isMaskedArray(alpha) and alpha.mask.shape!=():
        N = np.sum(~alpha.mask,axis)
    else:
        if axis is None:
            N = alpha.size
        else:
            N = alpha.shape[axis]
    R = np.sqrt(np.sum(np.sin(alpha),axis)**2 + np.sum(np.cos(alpha),axis)**2)/N
    V = 1-R
    return V

#Calculates angle variance(degrees)
def angle_var(data):
    return 1-np.sqrt(np.sin(data*np.pi/180).sum()**2 + np.cos(data*np.pi/180).sum()**2)/len(data)

def angle_std(variance):
    return np.sqrt(-2*np.log(1-variance))

def angle_strength(angle_std):
    return 1/(angle_std)

def yamartino_angle_std(variance):
    e = np.sqrt(1-(1-variance)**2)
    return np.arcsin(e)*(1+(2/np.sqrt(3)-1)*e**3)

def circdiff(alpha, beta):
    D = np.arctan2(np.sin(alpha*np.pi/180-beta*np.pi/180),np.cos(alpha*np.pi/180-beta*np.pi/180))
    return D

def led_to_angle():
    # Define the LED positions and corresponding angles
    led_positions = [19, 57, 93, 129]
    led_angles = [135, 45, -45, -135]

    # Initialize an empty array for the angles
    angles = np.zeros(142)  # LEDs #0-141 are visible

    # Compute the angles for each LED
    for i in range(4):
        start_led = led_positions[i]
        end_led = led_positions[(i + 1) % 4]
        
        if end_led < start_led:  # handle wrap-around
            end_led += 142
            
        start_angle = led_angles[i]
        end_angle = led_angles[(i + 1) % 4]
        
        num_leds = end_led - start_led
        led_angles_in_segment = np.linspace(start_angle, end_angle, num_leds, endpoint=False)
        
        for j, led in enumerate(range(start_led, end_led)):
            angles[led % 142] = led_angles_in_segment[j]

    return angles

# Function to get the angle for a given LED
def get_led_angle(led_number):
    led_to_angle_map = led_to_angle()
    return led_to_angle_map[led_number]

path = '' #path for fig4B data

#Sorts all the csv files in order in 'path' and sorts it to A trials and B trials
experiment_names = glob.glob(os.path.join(path, "*.csv"))
experiment_names.sort(reverse=False)

first_heading_angles=[]
first_heading_vars=[]
first_sun_positions =[]

second_heading_angles=[]
second_heading_vars=[]
second_sun_positions =[]

third_heading_angles=[]
third_heading_vars=[]
third_sun_positions =[]

experiments =[]
output= {}
time_period = 300 

for experiment_name in experiment_names:
    experiments.append(pd.read_csv(experiment_name))

for experiment in experiments:
    output[str(experiment_name)] = {}
    experiment['Image Time'] = experiment['Image Time'].apply(parser.parse)
    experiment_data = experiment.values
    sun_change_indexes = [0]+[i for i in range(1,len(experiment_data)) if experiment_data[i,3]!=experiment_data[i-1,3]]
    sun_periods = [experiment[sun_change_indexes[i-1]:sun_change_indexes[i]] for i in range(1,len(sun_change_indexes))]+[experiment[sun_change_indexes[-1]:-1]] 

    first_sun_period = sun_periods[1].loc[[(frame_time - sun_periods[1]['Image Time'].iloc[0]).seconds<=time_period for frame_time in sun_periods[1]['Image Time']]]
    first_sun_position = get_led_angle(first_sun_period['Sun Position'].iloc[0]) 
    first_avg_angle = angle_avg(first_sun_period['Heading Angle'])
    first_var_angle = angle_var(first_sun_period['Heading Angle'])
    
    first_heading_angles.append(first_avg_angle)            
    first_heading_vars.append(first_var_angle)
    first_sun_positions.append(first_sun_position)


    second_sun_period = sun_periods[2].loc[[(frame_time - sun_periods[2]['Image Time'].iloc[0]).seconds<=time_period for frame_time in sun_periods[2]['Image Time']]]
    second_sun_position = get_led_angle(second_sun_period['Sun Position'].iloc[0]) 
    second_avg_angle = angle_avg(second_sun_period['Heading Angle'])
    second_var_angle = angle_var(second_sun_period['Heading Angle'])
    
    second_heading_angles.append(second_avg_angle)            
    second_heading_vars.append(second_var_angle)
    second_sun_positions.append(second_sun_position)


    third_sun_period = sun_periods[3].loc[[(frame_time - sun_periods[3]['Image Time'].iloc[0]).seconds<=time_period for frame_time in sun_periods[3]['Image Time']]]
    third_sun_position = get_led_angle(third_sun_period['Sun Position'].iloc[0]) 
    third_avg_angle = angle_avg(third_sun_period['Heading Angle'])
    third_var_angle = angle_var(third_sun_period['Heading Angle'])

    third_heading_angles.append(third_avg_angle)            
    third_heading_vars.append(third_var_angle)
    third_sun_positions.append(third_sun_position)


        
first_heading_angles = np.array(first_heading_angles)   
first_heading_vars= np.array(first_heading_vars)
first_sun_positions = np.array(first_sun_positions)

second_heading_angles = np.array(second_heading_angles)   
second_heading_vars= np.array(second_heading_vars)
second_sun_positions = np.array(second_sun_positions)

third_heading_angles= np.array(third_heading_angles)   
third_heading_vars = np.array(third_heading_vars)
third_sun_positions = np.array(third_sun_positions)


##Apply cutt off filter of vecstrength<0.2 ###

FirstSunAngleVarsCutOff = np.array([])
SecondSunAngleVarsCutOff = np.array([])
ThirdSunAngleVarsCutOff = np.array([])


FirstSunheadingsCutOff = np.array([])
SecondSunheadingsCutOff = np.array([])
ThirdSunheadingsCutOff = np.array([])


FirstSunPositionsCutOff = np.array([])
SecondSunPositionsCutOff = np.array([])
ThirdSunPositionsCutOff = np.array([])


for x in range (first_heading_vars.size):
    if first_heading_vars[x] > 0.8 or second_heading_vars[x] > 0.8 or third_heading_vars[x]>0.8:                                     
        print(x)
        pass
    else:
        FirstSunAngleVarsCutOff = np.append(FirstSunAngleVarsCutOff, first_heading_vars[x])
        FirstSunheadingsCutOff = np.append(FirstSunheadingsCutOff, first_heading_angles[x])
        FirstSunPositionsCutOff = np.append(FirstSunPositionsCutOff, first_sun_positions[x])

        SecondSunAngleVarsCutOff = np.append(SecondSunAngleVarsCutOff, second_heading_vars[x])
        SecondSunheadingsCutOff = np.append(SecondSunheadingsCutOff, second_heading_angles[x])
        SecondSunPositionsCutOff = np.append(SecondSunPositionsCutOff, second_sun_positions[x])

        ThirdSunAngleVarsCutOff = np.append(ThirdSunAngleVarsCutOff, third_heading_vars[x])
        ThirdSunheadingsCutOff = np.append(ThirdSunheadingsCutOff, third_heading_angles[x])
        ThirdSunPositionsCutOff = np.append(ThirdSunPositionsCutOff, third_sun_positions[x])


        
FirstSunAngleVarsCutOff = FirstSunAngleVarsCutOff[1:]
FirstSunheadingsCutOff = FirstSunheadingsCutOff[1:]
FirstSunPositionsCutOff = FirstSunPositionsCutOff[1:]

SecondSunAngleVarsCutOff = SecondSunAngleVarsCutOff[1:]
SecondSunheadingsCutOff = SecondSunheadingsCutOff[1:]
SecondSunPositionsCutOff = SecondSunPositionsCutOff[1:]

ThirdSunAngleVarsCutOff = ThirdSunAngleVarsCutOff[1:]
ThirdSunheadingsCutOff = ThirdSunheadingsCutOff[1:]
ThirdSunPositionsCutOff = ThirdSunPositionsCutOff[1:]

FirstSunVecStrengthCutOff = 1-FirstSunAngleVarsCutOff 
SecondSunVecStrengthCutOff = 1-SecondSunAngleVarsCutOff
ThirdSunVecStrengthCutOff = 1-ThirdSunAngleVarsCutOff


first_sun_rotated_heading_angles=difference(FirstSunheadingsCutOff, FirstSunPositionsCutOff, deg=True)
second_sun_rotated_heading_angles=difference(SecondSunheadingsCutOff, SecondSunPositionsCutOff, deg=True)
third_sun_rotated_heading_angles=difference(ThirdSunheadingsCutOff, ThirdSunPositionsCutOff, deg=True)


heading_differenceA = difference(first_sun_rotated_heading_angles, second_sun_rotated_heading_angles, deg=True)
heading_differenceB = difference(second_sun_rotated_heading_angles, third_sun_rotated_heading_angles, deg=True)
heading_differenceC = difference(first_sun_rotated_heading_angles, third_sun_rotated_heading_angles, deg=True)

real_world_30deg_heading_diff = difference(FirstSunheadingsCutOff, ThirdSunheadingsCutOff, deg= True)
real_world_15deg_heading_diff1= difference(FirstSunheadingsCutOff, SecondSunheadingsCutOff, deg =True)
real_world_15deg_heading_diff2 = difference(SecondSunheadingsCutOff, ThirdSunheadingsCutOff, deg= True)


#% Plots Sea Urchin and runs Rayleigh Test
from circstats import confmean
from astropy.stats import rayleightest
from circstats import confmean
from collections import Counter

def sea_urchin_with_stacked_symbols(circmeans, vecstrengths, bin_num, hist_start,hist_spacing, figname):
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111, projection = 'polar')
    ax.plot(circmeans, vecstrengths, color = 'black', linewidth = 2, zorder=20) 
    circmeans=np.array(circmeans)
    pop_mean = (circmean(circmeans[0,:]))
    pop_var = (circvar(circmeans[0,:])) #added 5/9/24
    conf_in_95 = confmean(circmeans[0,:])
    CI_arc_r = np.ones(10000)
    print ('pop_mean = ', pop_mean)
    print('pop_var', pop_var)
    print ( 'conf_in_95 = ', conf_in_95)
    ax.plot((0, pop_mean), (0, 1), linewidth=2, color='r', zorder=22) # plot the circular mean
    CI_arc = np.linspace(pop_mean-conf_in_95, pop_mean+conf_in_95, num=10000) % (2*np.pi)

    bins = np.linspace(0, 2*np.pi, bin_num+1, endpoint = True)
    digitized = np.digitize(circmeans[0,:], bins)
    z = Counter(digitized)
    ax.grid(False)
    circle = plt.Circle((0.0, 0.0), 1., transform=ax.transData._b, edgecolor=([0.9, 0.9, 0.9]), facecolor= ([0.9, 0.9, 0.9]), zorder=10)
    ax.plot(CI_arc, CI_arc_r, color= 'r', linewidth=2, zorder=50)
    ax.scatter(0, 0, s=75, color='r', marker= '+', linewidth = 2, zorder=25)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_zero_location ("N")
    ax.set_theta_direction(-1)
    ax.add_artist(circle)
    ax.spines['polar'].set_visible(False) 
    for bin_index, angle in enumerate(bins):
        count = z[bin_index]
        bin_spacing = 2*np.pi/bin_num
        bin_center = angle - (bin_spacing/2)  
        if count >0:
            hist_r_pos= np.linspace(hist_start, hist_start+(hist_spacing*(count-1)), count, endpoint = True)
            ax.scatter([bin_center]*count, np.linspace(hist_start, hist_start+(hist_spacing*(count-1)), count, endpoint = True),s=95, marker = '.', color = 'black', zorder=20)
            ax.set_theta_offset(np.pi/2)
    ax.axis('off') #
    ax.grid(False)
    ax.set_rmax(1.5)
    plt.tight_layout()
    fig.savefig(figname, transparent=True, dpi=600)

def PlotSeaUrchin(headings, vec_strengths, figname):
   
    #convert angles and angles_to_plot to radians and positive
    allFliesMeanAnglesCutOffRad= np.deg2rad(headings)
    a= np.array([np.mod(i+(2*np.pi), 2*np.pi) for i in allFliesMeanAnglesCutOffRad])
    r_A= vec_strengths
    AToPlot = np.concatenate((a[:, np.newaxis], np.zeros_like(a)[:, np.newaxis]), axis=1).T
    RAToPlot = np.concatenate((r_A[:, np.newaxis], np.zeros_like(r_A)[:, np.newaxis]), axis=1).T

    list_of_positive_angles = [(q +2*np.pi)%(2*np.pi) for q in allFliesMeanAnglesCutOffRad]
    list_of_positive_angles_ToPlot = [(q +2*np.pi)%(2*np.pi) for q in AToPlot]
    
    # Run Rayleigh test
    rayleigh_result = rayleightest(allFliesMeanAnglesCutOffRad) 
    print('Rayleigh Test Results:', rayleigh_result)
        
    # plot
    sea_urchin_with_stacked_symbols(circmeans= list_of_positive_angles_ToPlot, vecstrengths= RAToPlot, bin_num=90, hist_start=1.1, hist_spacing=0.09, figname= figname)


PlotSeaUrchin(heading_differenceC, np.ones_like(heading_differenceC), 'Rotated heading 30deg heading change')
PlotSeaUrchin(real_world_30deg_heading_diff, np.ones_like(real_world_30deg_heading_diff), 'Real World 30deg heading change')

plt.show()
