#!/usr/bin/env python
# Data analysis code for fig4C - plots the heading difference for each 30deg sun position change as a polar plot when the sun moves 5 degrees every 100s
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
from scipy import interpolate
from scipy.interpolate import make_interp_spline


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


def sun_movement_direction(first_led_position, second_led_position): ##Edited 7/28
    diff = second_led_position - first_led_position
    return 'counterclockwise' if diff > 0 else 'clockwise'



path = '' #Path to fig4C data

#Sorts all the csv files in order in 'path' and sorts it to A trials and B trials
experiment_names = glob.glob(os.path.join(path, "*.csv"))
experiment_names.sort(reverse=False)

heading_angles_list=[] 
heading_vars_list=[]
sun_positions =[]
trial_period = 100

first_100s_heading_angles_list =[]
first_100s_heading_vars_list =[]
first_100s_sun_positions =[]

first_heading_angles=[]
first_heading_vars=[]
first_sun_positions =[]

mid_100s_heading_angles_list =[]
mid_100s_heading_vars_list =[]
mid_100s_sun_positions =[]

mid_heading_angles=[]
mid_heading_vars=[]
mid_sun_positions =[]

last_100s_heading_angles_list =[]
last_100s_heading_vars_list =[]
last_100s_sun_positions =[]

last_heading_angles=[]
last_heading_vars=[]
last_sun_positions =[]

experiments =[]
output= {}
#time_delay = 0  #seconds, for last 3100ss set to 120

for experiment_name in experiment_names:
    experiments.append(pd.read_csv(experiment_name))

for experiment in experiments:
    output[str(experiment_name)] = {}
    experiment['Image Time'] = experiment['Image Time'].apply(parser.parse)
    experiment_data = experiment.values
    sun_change_indexes = [0]+[i for i in range(1,len(experiment_data)) if experiment_data[i,3]!=experiment_data[i-1,3]]

    sun_periods = [experiment[sun_change_indexes[i-1]:sun_change_indexes[i]] for i in range(1,len(sun_change_indexes))]+[experiment[sun_change_indexes[-1]:-1]]

    #First Last 100s heading
    first_100s_sun_period = sun_periods[1].loc[[(frame_time - sun_periods[1]['Image Time'].iloc[0]).seconds<=trial_period for frame_time in sun_periods[1]['Image Time']]]
    first_sun_position = get_led_angle(first_100s_sun_period['Sun Position'].iloc[0])
    first_100s_sun_positions.append(first_sun_position)
    first_avg_angle = angle_avg(first_100s_sun_period['Heading Angle'])
    first_100s_heading_angles_list.append(first_avg_angle)
    first_var_angle = angle_var(first_100s_sun_period['Heading Angle'])
    first_100s_heading_vars_list.append(first_var_angle)

    mid_100s_sun_period = sun_periods[6].loc[[(frame_time - sun_periods[6]['Image Time'].iloc[0]).seconds<=trial_period for frame_time in sun_periods[6]['Image Time']]]
    mid_sun_position = get_led_angle(mid_100s_sun_period['Sun Position'].iloc[0])
    mid_100s_sun_positions.append(mid_sun_position)
    mid_avg_angle = angle_avg(mid_100s_sun_period['Heading Angle'])
    mid_100s_heading_angles_list.append(mid_avg_angle)
    mid_var_angle = angle_var(mid_100s_sun_period['Heading Angle'])
    mid_100s_heading_vars_list.append(first_var_angle)

    last_100s_sun_period = sun_periods[12].loc[[(frame_time - sun_periods[12]['Image Time'].iloc[0]).seconds<=trial_period for frame_time in sun_periods[12]['Image Time']]]
    last_sun_position = get_led_angle(last_100s_sun_period['Sun Position'].iloc[0])
    last_100s_sun_positions.append(last_sun_position)
    last_avg_angle = angle_avg(last_100s_sun_period['Heading Angle'])
    last_100s_heading_angles_list.append(last_avg_angle)
    last_var_angle = angle_var(last_100s_sun_period['Heading Angle'])
    last_100s_heading_vars_list.append(first_var_angle)

first_heading_angles= np.array(first_100s_heading_angles_list)
first_heading_vars = np.array(first_100s_heading_vars_list)
first_sun_positions = np.array(first_100s_sun_positions)
first_heading_vecstrength = 1 - first_heading_vars

mid_heading_angles= np.array(mid_100s_heading_angles_list)
mid_heading_vars = np.array(mid_100s_heading_vars_list)
mid_sun_positions = np.array(mid_100s_sun_positions)
mid_heading_vecstrength = 1 - mid_heading_vars

last_heading_angles= np.array(last_100s_heading_angles_list)
last_heading_vars = np.array(last_100s_heading_vars_list)
last_sun_positions = np.array(last_100s_sun_positions)
last_heading_vecstrength = 1 - last_heading_vars

##Apply cutt off filter of vecstrength<0.2 ###
FirstAngleVarsCutOff = np.array([])
MidAngleVarsCutOff = np.array([])
LastAngleVarsCutOff = np.array([])

FirstheadingsCutOff = np.array([])
MidheadingsCutOff = np.array([])
LastheadingsCutOff = np.array([])

FirstSunPositionsCutOff = np.array([])
MidSunPositionsCutOff = np.array([])
LastSunPositionsCutOff = np.array([])


for x in range (first_heading_vars.size):
    if first_heading_vars[x] > 0.8 or mid_heading_vars[x] > 0.8 or last_heading_vars[x]>0.8:                                     
        print(x)
        pass
    else:
        FirstAngleVarsCutOff = np.append(FirstAngleVarsCutOff, first_heading_vars[x])
        FirstheadingsCutOff = np.append(FirstheadingsCutOff, first_heading_angles[x])
        FirstSunPositionsCutOff = np.append(FirstSunPositionsCutOff, first_sun_positions[x])

        MidAngleVarsCutOff = np.append(MidAngleVarsCutOff, mid_heading_vars[x])
        MidheadingsCutOff = np.append(MidheadingsCutOff, mid_heading_angles[x])
        MidSunPositionsCutOff = np.append(MidSunPositionsCutOff, mid_sun_positions[x])

        LastAngleVarsCutOff = np.append(LastAngleVarsCutOff, last_heading_vars[x])
        LastheadingsCutOff= np.append(LastheadingsCutOff, last_heading_angles[x])
        LastSunPositionsCutOff = np.append(LastSunPositionsCutOff, last_sun_positions[x])

        
FirstAngleVarsCutOff = FirstAngleVarsCutOff[1:]
FirstheadingsCutOff= FirstheadingsCutOff[1:]
FirstSunPositionsCutOff = FirstSunPositionsCutOff[1:]

MidAngleVarsCutOff = MidAngleVarsCutOff[1:]
MidheadingsCutOff = MidheadingsCutOff[1:]
MidSunPositionsCutOff = MidSunPositionsCutOff[1:]

LastAngleVarsCutOff = LastAngleVarsCutOff[1:]
LastheadingsCutOff = LastheadingsCutOff[1:]
LastSunPositionsCutOff = LastSunPositionsCutOff[1:]

FirstVecStrengthCutOff = 1-FirstAngleVarsCutOff
MidVecStrengthCutOff = 1-MidAngleVarsCutOff
LastVecStrengthCutOff = 1-LastAngleVarsCutOff

first_sun_rotated_headings = difference(FirstheadingsCutOff, FirstSunPositionsCutOff, deg = True)
mid_sun_rotated_headings = difference(MidheadingsCutOff, MidSunPositionsCutOff, deg = True)
last_sun_rotated_headings = difference(LastheadingsCutOff, LastSunPositionsCutOff, deg = True)

heading_differenceA = difference(first_sun_rotated_headings, last_sun_rotated_headings, deg = True)
heading_differenceB = difference(first_sun_rotated_headings, mid_sun_rotated_headings, deg = True)
heading_differenceC = difference(mid_sun_rotated_headings, last_sun_rotated_headings, deg= True)

real_world_heading_differenceA = difference(FirstheadingsCutOff, LastheadingsCutOff, deg= True)
real_world_heading_differenceB = difference(FirstheadingsCutOff, MidheadingsCutOff, deg= True)
real_world_heading_differenceC = difference(MidheadingsCutOff, LastheadingsCutOff, deg= True)


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
    ax.plot(CI_arc, CI_arc_r, color= 'r', linewidth=2, zorder=50) #plot the umbrella
    ax.scatter(0, 0, s=75, color='r', marker= '+', linewidth = 2, zorder=25)

    ax.set_theta_offset(np.pi/2)
    ax.set_theta_zero_location ("N")
    ax.set_theta_direction(-1)
    ax.add_artist(circle)
    ax.spines['polar'].set_visible(True) # can make True for labels
    for bin_index, angle in enumerate(bins):

        count = z[bin_index]
        bin_spacing = 2*np.pi/bin_num
        bin_center = angle - (bin_spacing/2)  
        if count >0:
            hist_r_pos= np.linspace(hist_start, hist_start+(hist_spacing*(count-1)), count, endpoint = True)

            ax.scatter([bin_center]*count, np.linspace(hist_start, hist_start+(hist_spacing*(count-1)), count, endpoint = True),s=95, marker = '.', color = 'black', zorder=20)
            ax.set_theta_offset(np.pi/2)
    ax.axis('off') 
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


PlotSeaUrchin(heading_differenceA, np.ones_like(heading_differenceA), '0-60deg Sun Heading Difference Polar plot')
PlotSeaUrchin(real_world_heading_differenceA, np.ones_like(real_world_heading_differenceA),'0-60 Real World Heading Difference Polar Plot' )
PlotSeaUrchin(heading_differenceB, np.ones_like(heading_differenceB), '0-30deg Sun Heading Difference Polar plot')
PlotSeaUrchin(real_world_heading_differenceB, np.ones_like(real_world_heading_differenceB),'0-30 Real World Heading Difference Polar Plot' )
PlotSeaUrchin(heading_differenceC, np.ones_like(heading_differenceC), '30-60deg Sun Heading Difference Polar plot')
PlotSeaUrchin(real_world_heading_differenceC, np.ones_like(real_world_heading_differenceC),'30-60 Real World Heading Difference Polar Plot' )

plt.show()
