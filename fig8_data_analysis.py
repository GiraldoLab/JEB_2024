#!/usr/bin/env python
# Plots polarplots, linear plots of each sun period headings for flight trials with a dual green-UV spectral cue for fig8
# Plots the heading difference as a polar plot and running resampling analysis(bootstrap analysis) for differences between Sun1 and Sun2

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
def circmean(alpha, axis =None):   ###This is when averaging angles in radians
    mean_angle = np.arctan2(np.mean(np.sin(alpha), axis),np.mean(np.cos(alpha), axis))
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

def sun_angle(data):
    if data == 19:
        return 135
    elif data == 57:
        return 45
    elif data == 93:
        return -45
    elif data == 129:
        return -135
    elif data == 152:
        return 135
    elif data == 168:
        return 45
    elif data == 183:
        return -45
    elif data == 198:
        return -135
    else:
        return 0

def anti_sun(angle):
    angle_map = {45: -135, -45: 135, 135: -45, -135: 45}
    return angle_map.get(angle, "Invalid input")

def sun_movement_direction(first_positions, second_positions): 
    diff = difference(np.array(first_positions), np.array(second_positions), deg =True)
    return 'counterclockwise' if diff > 0 else 'clockwise'


path = '' #Path to fig8 data

#Sorts all the csv files in order in 'path' and sorts it to A trials and B trials
experiment_names = glob.glob(os.path.join(path, "*.csv"))
experiment_names.sort(reverse=False)

first_heading_angles=[]
first_heading_vars=[]
first_grn_sun_positions =[]
first_uv_sun_positions =[]

second_heading_angles=[]
second_heading_vars=[]
second_grn_sun_positions =[]
second_uv_sun_positions =[]


experiments =[]
output= {}
time_period1 = 300


for experiment_name in experiment_names:
    experiments.append(pd.read_csv(experiment_name))

for experiment in experiments:
    output[str(experiment_name)] = {}
    experiment['Image Time'] = experiment['Image Time'].apply(parser.parse)
    experiment_data = experiment.values
    sun_change_indexes = [0]+[i for i in range(1,len(experiment_data)) if experiment_data[i,4]!=experiment_data[i-1,4]] #Changed to detect sun position changes in 'Sun position2' instead of 'Sun Position1'
    sun_periods = [experiment[sun_change_indexes[i-1]:sun_change_indexes[i]] for i in range(1,len(sun_change_indexes))]+[experiment[sun_change_indexes[-1]:-1]]

    first_sun_period = sun_periods[1].loc[[(frame_time - sun_periods[1]['Image Time'].iloc[0]).seconds<=time_period1 for frame_time in sun_periods[1]['Image Time']]]
    first_grn_sun_position = sun_angle(first_sun_period['Sun Position1'].iloc[0]) #to show where sun stimulus was
    first_uv_sun_position = sun_angle(first_sun_period['Sun Position2'].iloc[0]) 
    first_avg_angle = angle_avg(first_sun_period['Heading Angle'])
    first_var_angle = angle_var(first_sun_period['Heading Angle'])
    
    first_heading_angles.append(first_avg_angle)            
    first_heading_vars.append(first_var_angle)
    first_grn_sun_positions.append(first_grn_sun_position)
    first_uv_sun_positions.append(first_uv_sun_position)

    second_sun_period = sun_periods[2].loc[[(frame_time - sun_periods[2]['Image Time'].iloc[0]).seconds<=time_period1 for frame_time in sun_periods[2]['Image Time']]]
    second_grn_sun_position = sun_angle(second_sun_period['Sun Position1'].iloc[0])
    second_uv_sun_position = sun_angle(second_sun_period['Sun Position2'].iloc[0])
    second_avg_angle = angle_avg(second_sun_period['Heading Angle'])
    second_var_angle = angle_var(second_sun_period['Heading Angle'])
    
    second_heading_angles.append(second_avg_angle)            
    second_heading_vars.append(second_var_angle)
    second_grn_sun_positions.append(second_grn_sun_position)
    second_uv_sun_positions.append(second_uv_sun_position)

        
first_heading_angles = np.array(first_heading_angles)   
first_heading_vars= np.array(first_heading_vars)
first_grn_sun_positions = np.array(first_grn_sun_positions)
first_uv_sun_positions = np.array(first_uv_sun_positions)

second_heading_angles = np.array(second_heading_angles)   
second_heading_vars= np.array(second_heading_vars)
second_grn_sun_positions = np.array(second_grn_sun_positions)
second_uv_sun_positions = np.array(second_uv_sun_positions)


##Apply cut off filter of vecstrength<0.2 ###
FirstSunFliesAngleVarsCutOff = np.zeros (1)
FirstSunFliesMeanAnglesCutOff = np.zeros (1)
FirstGrnSunFliesCutOff = np.zeros (1)
FirstUVSunFliesCutOff = np.zeros (1)

SecondSunFliesAngleVarsCutOff = np.zeros (1)
SecondSunFliesMeanAnglesCutOff = np.zeros (1)
SecondGrnSunFliesCutOff = np.zeros(1)
SecondUVSunFliesCutOff = np.zeros(1)


for x in range (first_heading_vars.size):
    if first_heading_vars[x] > 0.8 or second_heading_vars[x] > 0.8 :                                      
        print(x)
        pass
     
    else:

        FirstSunFliesAngleVarsCutOff = np.append(FirstSunFliesAngleVarsCutOff, first_heading_vars[x])
        FirstSunFliesMeanAnglesCutOff= np.append(FirstSunFliesMeanAnglesCutOff, first_heading_angles[x])
        FirstGrnSunFliesCutOff  = np.append(FirstGrnSunFliesCutOff, first_grn_sun_positions[x])
        FirstUVSunFliesCutOff  = np.append(FirstUVSunFliesCutOff, first_uv_sun_positions[x])


        SecondSunFliesAngleVarsCutOff= np.append(SecondSunFliesAngleVarsCutOff, second_heading_vars[x])
        SecondSunFliesMeanAnglesCutOff = np.append(SecondSunFliesMeanAnglesCutOff , second_heading_angles[x])
        SecondGrnSunFliesCutOff = np.append(SecondGrnSunFliesCutOff, second_grn_sun_positions[x])
        SecondUVSunFliesCutOff = np.append(SecondUVSunFliesCutOff, second_uv_sun_positions[x])


        
FirstSunFliesAngleVarsCutOff = FirstSunFliesAngleVarsCutOff[1:]
FirstSunFliesMeanAnglesCutOff =FirstSunFliesMeanAnglesCutOff[1:]
FirstGrnSunFliesCutOff = FirstGrnSunFliesCutOff[1:]
FirstUVSunFliesCutOff =FirstUVSunFliesCutOff[1:]

SecondSunFliesAngleVarsCutOff = SecondSunFliesAngleVarsCutOff[1:]
SecondSunFliesMeanAnglesCutOff = SecondSunFliesMeanAnglesCutOff[1:]
SecondGrnSunFliesCutOff = SecondGrnSunFliesCutOff[1:]
SecondUVSunFliesCutOff = SecondUVSunFliesCutOff[1:]



FirstSunFliesVecStrengthCutOff = 1- FirstSunFliesAngleVarsCutOff
SecondSunFliesVecStrengthCutOff = 1- SecondSunFliesAngleVarsCutOff


rotated_heading_anglesA=difference(FirstSunFliesMeanAnglesCutOff, FirstGrnSunFliesCutOff, deg=True)
uv_rotated_heading_anglesA = difference(FirstSunFliesMeanAnglesCutOff, FirstUVSunFliesCutOff, deg= True)
rotated_heading_anglesB = difference(SecondSunFliesMeanAnglesCutOff, FirstGrnSunFliesCutOff, deg= True)
heading_difference = difference(rotated_heading_anglesA, rotated_heading_anglesB,deg=True )



def LinearPlot(list_x, list_y, list_x_error, list_y_error, x_axis = 'list_x', y_axis ='list_y', title =None):
    fig = plt.figure(figsize=(5,10))
    fig.set_facecolor('w')
    ax = fig.add_subplot(1,1,1)
    for axis in ['left','bottom']:
                ax.spines[axis].set_linewidth(3.0)
                ax.spines['left'].set_position(('outward', 20))
                ax.spines['bottom'].set_position(('outward', 0))
    ax.xaxis.set_tick_params(width=3.0, length=8.0, direction = 'in')
    ax.yaxis.set_tick_params(width=3.0, length=8.0, direction = 'in') 
    ax.scatter(list_x, list_y, s=35, color='k', zorder=10)
    ax.scatter(list_x, list_y +360.0, s=35, color='k', zorder=10)
    ax.errorbar(x=list_x, y=list_y, xerr=list_x_error*36, yerr= list_y_error*36, fmt='none', linewidth =3, ecolor=[.6,.6,.6], capsize = 0, zorder=5)
    ax.errorbar(x=list_x, y=list_y +360.0, xerr=list_x_error*36 , yerr= list_y_error*36, fmt='none', linewidth =3, ecolor=[.6,.6,.6], capsize = 0, zorder=5)
    ax.plot([-180,180], [-180,180], color='k', zorder=1, linewidth=3, linestyle='solid')
    ax.plot([-180,180], [180,540], color='k', zorder=1, linewidth=3, linestyle='solid')

    ax.set_title('Continuous Flight Trials', fontsize =18)
    ax.set_xticks((-180, 0, 180)) #for larger fig
    ax.set_yticks((-180, 0, 180,  360,  540)) # for large fig

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.spines['left'].set_bounds(-180,540)

    ax.spines['bottom'].set_bounds(-180,180)

    ax.set_yticklabels(('-180', '0', '180',  '360',  '540'), color='k', fontsize=12)
    ax.set_xticklabels(('-180', '0', '180'), color='k', fontsize=12)
    ax.set_ylabel(y_axis, fontsize=10)
    ax.set_xlabel(x_axis, fontsize=10)

    ax.axis('equal')
    if title is not None:  
        ax.set_title(title)
    fig.savefig(title, transparent=True, dpi=600)

LinearPlot(rotated_heading_anglesA, rotated_heading_anglesB, FirstSunFliesAngleVarsCutOff,SecondSunFliesAngleVarsCutOff,x_axis ='  First sun headings', y_axis = 'Second sun headings', title = 'heading change linearplot')


##############################
def BootstrapAnalysis(list_A, list_B, title=None, NUM_RESAMPLES=10000):
    observedDiffs = circdiff(list_B, list_A)
    observedDiffMean = np.mean(np.abs(observedDiffs))

    resampledDiffMeans = np.zeros(NUM_RESAMPLES, dtype='float')
    for resampleInd in range(NUM_RESAMPLES):
        resampledB = np.random.permutation(list_B)
        resampledDiffs = circdiff(resampledB, list_A)
        resampledDiffMean = np.mean(np.abs(resampledDiffs))
        resampledDiffMeans[resampleInd] = resampledDiffMean

    pval = np.sum(resampledDiffMeans <= observedDiffMean)/float(NUM_RESAMPLES)
    pval = np.around(pval, decimals=3)

    observed_diff=observedDiffMean*180./np.pi
    observed_diff=np.around(observed_diff, decimals=3)

    bootstrap_mean=np.mean(resampledDiffMeans)*180./np.pi
    bootstrap_mean=np.around(bootstrap_mean, decimals=3)

    print ('pval= ', pval)
    print ('observed diff= ', observed_diff)
    print ('bootstrap mean= ', bootstrap_mean)
    fig = plt.figure(figsize=(4,2.25))
    fig.set_facecolor('w')
    ax = fig.add_subplot(1, 1, 1)
    for axis in ['left','bottom']:
                ax.spines[axis].set_linewidth(3.0)
    ax.xaxis.set_tick_params(width=3.0, length = 6.0, direction = 'in')
    ax.yaxis.set_tick_params(width=3.0, length = 6.0, direction = 'in')
    ax.hist(resampledDiffMeans*180./np.pi, bins=15, histtype='stepfilled', color = [0.7, 0.7, 0.7])
    ax.axvline(observedDiffMean*180./np.pi, ymin=0.078, ymax=0.88, color='r', linewidth=2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.set_xlabel('Mean angle difference',  fontsize=14)
    ax.set_ylabel('Counts', fontsize=14)
    ax.set_xlim((-15, 200))
    ax.set_ylim((-200, 2500))
    ax.spines['bottom'].set_bounds(0, 180)
    ax.spines['left'].set_bounds(0, 2500)
    ax.set_xticks((0, 90, 180))
    ax.set_yticks((0, 2500))
    ax.set_yticklabels((0, 2500), fontsize=14)
    ax.set_xticklabels((0, 90, 180), fontsize=14)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if title is not None:  
        ax.set_title(title)
    fig.savefig(title, transparent=True, dpi=600)
    
BootstrapAnalysis(rotated_heading_anglesA, rotated_heading_anglesB, 'Bootstrap results with labels')





#% Plots Sea Urchin and runs Rayleigh Test
from circstats import confmean
from astropy.stats import rayleightest
from circstats import confmean
from collections import Counter

def sea_urchin_with_stacked_symbols(circmeans, vecstrengths, bin_num, hist_start,hist_spacing, figname):
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111, projection = 'polar')
    ax.plot(circmeans, vecstrengths, color = 'black', linewidth = 2, zorder=20) #urchin
    circmeans=np.array(circmeans)
    pop_mean = (circmean(circmeans[0,:]))
    pop_var = (circvar(circmeans[0,:])) #added 5/9/24
    conf_in_95 = confmean(circmeans[0,:])
    CI_arc_r = np.ones(10000)
    print ('pop_mean = ', pop_mean)
    print ( 'conf_in_95 = ', conf_in_95)
    print('pop_var', pop_var)
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
    ax.spines['polar'].set_visible(False) 
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


PlotSeaUrchin(rotated_heading_anglesA,FirstSunFliesVecStrengthCutOff, 'first suns polar plot')
PlotSeaUrchin(rotated_heading_anglesB, SecondSunFliesVecStrengthCutOff, 'second suns polar plot')
PlotSeaUrchin(heading_difference, np.ones_like(heading_difference), 'heading difference polar plot')

plt.show() 