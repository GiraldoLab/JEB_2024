#!/usr/bin/env python

# Plots every 5minute average heading differences in a scatterplot (fig1C)
# Plots the heading difference of first and last 5minutes in a polar plot (fig1C)


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
 

# Calculates mean heading angle of fly(degrees)
def full_arctan(x,y):                                      #makes all the angles between -180 to 180
    angle = np.arctan(y/x)+np.pi if x<0 else np.arctan(y/x)
    return angle if angle <= np.pi else angle -2*np.pi  
def angle_avg(data):
    return full_arctan(np.cos(data*np.pi/180).sum(),np.sin(data*np.pi/180).sum())*180/np.pi  
# Calculates mean heading angle of fly(radians)
def circmean(alpha,axis=None):  
    mean_angle = np.arctan2(np.mean(np.sin(alpha),axis),np.mean(np.cos(alpha),axis))
    return mean_angle

# Calculates variance(radians)
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

#Calculates variance(degrees)
def angle_var(data):                                
    return 1-np.sqrt(np.sin(data*np.pi/180).sum()**2 + np.cos(data*np.pi/180).sum()**2)/len(data)

def angle_std(variance):
    return np.sqrt(-2*np.log(1-variance))
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
    elif data == 97:
        return -55
    else:
        return 0



path = '' #Path to fig1C data  
experiment_names = glob.glob(os.path.join(path, "*.csv"))
experiment_names.sort(reverse=False)
heading_angles=[[],[],[],[]]
heading_vars=[[],[],[],[]]
sun_positions =[[],[],[],[]]

experiments =[]
output= {}
time_period = 300

trial_periods1 = list(range(0, 1200, 300))
trial_periods2= list(range(300, 1500, 300))

for experiment_name in experiment_names:
    experiments.append(pd.read_csv(experiment_name))

for experiment in experiments:
    output[str(experiment_name)] = {}
    experiment['Image Time'] = experiment['Image Time'].apply(parser.parse)
    experiment_data = experiment.values
    sun_change_indexes = [0]+[i for i in range(1,len(experiment_data)) if experiment_data[i,3]!=experiment_data[i-1,3]]
    sun_periods = [experiment[sun_change_indexes[i-1]:sun_change_indexes[i]] for i in range(1,len(sun_change_indexes))]+[experiment[sun_change_indexes[-1]:-1]]
 
    for i in range(len(sun_periods)):
        if not (i%2):
            continue
        sun_period = sun_periods[i]
 
        for i in range(4):
           
            restricted_sun_period = sun_period.loc[[(frame_time - sun_period['Image Time'].iloc[0]).seconds<=trial_periods2[i] and (frame_time - sun_period['Image Time'].iloc[0]).seconds>=trial_periods1[i] for frame_time in sun_period['Image Time']]]

            sun_position = sun_angle(restricted_sun_period['Sun Position'].iloc[0]) #to show where sun stimulus was 
            avg_angle = angle_avg(restricted_sun_period['Heading Angle'])
            var_angle = angle_var(restricted_sun_period['Heading Angle'])

            heading_angles[i%4].append(avg_angle)
            heading_vars[i%4].append(var_angle)
            sun_positions[i%4].append(sun_position)

heading_angles = np.array(heading_angles)
sun_positions = np.array(sun_positions)
heading_vars = np.array(heading_vars)


first_mean_headings = difference(np.array(heading_angles[0]), np.array(sun_positions[0]), deg= True)
second_mean_headings = difference(np.array(heading_angles[1]), np.array(sun_positions[1]), deg= True)
third_mean_headings = difference(np.array(heading_angles[2]), np.array(sun_positions[2]), deg= True)
fourth_mean_headings = difference(np.array(heading_angles[3]), np.array(sun_positions[3]), deg= True)

first_heading_vars = np.array(heading_vars[0])
second_heading_vars = np.array(heading_vars[1])
third_heading_vars = np.array(heading_vars[2])
fourth_heading_vars = np.array(heading_vars[3])

first_sun_positions = np.array(sun_positions[0])
second_sun_positions = np.array(sun_positions[1])
third_sun_positions = np.array(sun_positions[2])
fourth_sun_positions = np.array(sun_positions[3])


### Apply cutt off filter of vecstrength<0.2 ###

FirstSunVarsCutOff = np.zeros (1)
SecondSunVarsCutOff = np.zeros (1)
ThirdSunVarsCutOff = np.zeros (1)
FourthSunVarsCutOff = np.zeros (1)

FirstMeanAnglesCutOff = np.zeros (1)
SecondMeanAnglesCutOff = np.zeros (1)
ThirdMeanAnglesCutOff = np.zeros (1)
FourthMeanAnglesCutOff = np.zeros (1)

FirstSunPositionsCutOff = np.zeros (1)
SecondSunPositionsCutOff = np.zeros (1)
ThirdSunPositionsCutOff = np.zeros (1)
FourthSunPositionsCutOff = np.zeros (1)

for x in range (first_heading_vars.size):
    if first_heading_vars[x] > 0.8 or second_heading_vars[x] >0.8 or third_heading_vars[x] >0.8 or fourth_heading_vars[x] >0.8 :                                      
        print(x)
        pass
    else:
        FirstSunVarsCutOff = np.append(FirstSunVarsCutOff, first_heading_vars[x])
        FirstMeanAnglesCutOff= np.append(FirstMeanAnglesCutOff, first_mean_headings[x])
        FirstSunPositionsCutOff = np.append(FirstSunPositionsCutOff, first_sun_positions[x])

        SecondSunVarsCutOff = np.append(SecondSunVarsCutOff, second_heading_vars[x])
        SecondMeanAnglesCutOff= np.append(SecondMeanAnglesCutOff, second_mean_headings[x])
        SecondSunPositionsCutOff = np.append(SecondSunPositionsCutOff, second_sun_positions[x])

        ThirdSunVarsCutOff = np.append(ThirdSunVarsCutOff, third_heading_vars[x])
        ThirdMeanAnglesCutOff = np.append(ThirdMeanAnglesCutOff, third_mean_headings[x])
        ThirdSunPositionsCutOff = np.append(ThirdSunPositionsCutOff, third_sun_positions[x])

        FourthSunVarsCutOff = np.append(FourthSunVarsCutOff, fourth_heading_vars[x])
        FourthMeanAnglesCutOff = np.append(FourthMeanAnglesCutOff, fourth_mean_headings[x])
        FourthSunPositionsCutOff = np.append(FourthSunPositionsCutOff, fourth_sun_positions[x])



FirstSunVarsCutOff = FirstSunVarsCutOff [1:]
print('Average vector strength first 5min:', 1 - sum(FirstSunVarsCutOff )/len(FirstSunVarsCutOff ))
FirstMeanAnglesCutOff = FirstMeanAnglesCutOff[1:] 
FirstSunPositionsCutOff = FirstSunPositionsCutOff[1:] 

SecondSunVarsCutOff = SecondSunVarsCutOff [1:]
print('Average vector strength second 5min:', 1 - sum(SecondSunVarsCutOff )/len(SecondSunVarsCutOff ))
SecondMeanAnglesCutOff = SecondMeanAnglesCutOff[1:] 
SecondSunPositionsCutOff = SecondSunPositionsCutOff[1:] 

ThirdSunVarsCutOff = ThirdSunVarsCutOff [1:]
print('Average vector strength third 5min:', 1 - sum(ThirdSunVarsCutOff )/len(ThirdSunVarsCutOff ))
ThirdMeanAnglesCutOff = ThirdMeanAnglesCutOff[1:] 
ThirdSunPositionsCutOff = ThirdSunPositionsCutOff[1:] 

FourthSunVarsCutOff = FourthSunVarsCutOff [1:]
print('Average vector strength fourth 5min:', 1 - sum(FourthSunVarsCutOff )/len(FourthSunVarsCutOff ))
FourthMeanAnglesCutOff = FourthMeanAnglesCutOff[1:] 
FourthSunPositionsCutOff = FourthSunPositionsCutOff[1:] 



first_sun_rotated_heading_angles = difference(FirstMeanAnglesCutOff, FirstSunPositionsCutOff, deg= True)
second_sun_rotated_heading_angles = difference(SecondMeanAnglesCutOff, SecondSunPositionsCutOff, deg= True)
third_sun_rotated_heading_angles = difference(ThirdMeanAnglesCutOff, ThirdSunPositionsCutOff, deg= True)
fourth_sun_rotated_heading_angles = difference(FourthMeanAnglesCutOff, FourthSunPositionsCutOff, deg= True)


difference_one = difference(first_sun_rotated_heading_angles, second_sun_rotated_heading_angles, deg = True)
difference_two = difference(second_sun_rotated_heading_angles, third_sun_rotated_heading_angles, deg = True)
difference_three =difference(third_sun_rotated_heading_angles, fourth_sun_rotated_heading_angles, deg = True)
first_last_diff = difference(first_sun_rotated_heading_angles, fourth_sun_rotated_heading_angles, deg = True)



all_suns_heading_differences = []
all_suns_heading_differences.append(difference_one)
all_suns_heading_differences.append(difference_two)
all_suns_heading_differences.append(difference_three)
all_suns_heading_diffs_per_fly = np.transpose(all_suns_heading_differences)

heading_change =[]

heading_change.append(angle_avg(difference_one))
heading_change.append(angle_avg(difference_two))
heading_change.append(angle_avg(difference_three))

heading_change=np.array(heading_change)

#Plot the scatter plot for each 5 minute heading differece for each fly
fig = plt.figure(figsize=(10,5))
fig.set_facecolor('w')
ax = fig.add_subplot(1,1,1)
for axis in ['left','bottom']:
    ax.spines[axis].set_linewidth(3)
ax.xaxis.set_tick_params(width=3, length=6.0, direction = 'in')
ax.yaxis.set_tick_params(width=3, length=6.0, direction = 'in') 
for i in range(len(difference_one)):  
    ax.scatter(list(range(3)),all_suns_heading_diffs_per_fly[i], color ='silver', linewidth = 2)
    ax.plot(list(range(3)),all_suns_heading_diffs_per_fly[i], color ='silver', linewidth = 2)

ax.scatter(list(range(3)), heading_change, label=heading_change,color = 'k', linewidth= 3)
ax.plot(list(range(3)), heading_change, marker='o',color = 'k', linewidth =3)
ax.set_xticks((0,1,2))
ax.set_yticks(( -180, 0,   180))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_bounds(-180,180)
ax.spines['bottom'].set_bounds(0,2)
ax.set_yticklabels(( '-180', '0',  '180'), color='k', fontsize=10)
ax.set_xticklabels(('$T_{10-5}$','$T_{15-10}$', '$T_{20-15}$'), color='k', fontsize=10)
ax.set_ylabel(' Mean Heading Differences', fontsize=12)
ax.set_xlabel('Time Differences', fontsize=12)
fig.savefig('20min Control 5min flight period heading analysis not absolute diffs.png', transparent=True, dpi=600)
#fig.tight_layout()
plt.show()



#% Plots polar plot and runs Rayleigh Test
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
    pop_var = (circvar(circmeans[0,:])) 
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
   
    #Convert angles and angles_to_plot to radians and positive
    allFliesMeanAnglesCutOffRad= np.deg2rad(headings)
    a= np.array([np.mod(i+(2*np.pi), 2*np.pi) for i in allFliesMeanAnglesCutOffRad])
    r_A= vec_strengths
    AToPlot = np.concatenate((a[:, np.newaxis], np.zeros_like(a)[:, np.newaxis]), axis=1).T
    RAToPlot = np.concatenate((r_A[:, np.newaxis], np.zeros_like(r_A)[:, np.newaxis]), axis=1).T

    list_of_positive_angles = [(q +2*np.pi)%(2*np.pi) for q in allFliesMeanAnglesCutOffRad]
    list_of_positive_angles_ToPlot = [(q +2*np.pi)%(2*np.pi) for q in AToPlot]
    
    # Run Rayleigh test
    rayleigh_result = rayleightest(allFliesMeanAnglesCutOffRad) 
    print('Rayleigh Test Results: ',rayleigh_result)
        
    # plot
    sea_urchin_with_stacked_symbols(circmeans= list_of_positive_angles_ToPlot, vecstrengths= RAToPlot, bin_num=90, hist_start=1.1, hist_spacing=0.09, figname= figname)


PlotSeaUrchin(first_last_diff, np.ones_like(first_last_diff), 'First Last 5min Heading Difference')

# Show the plot
plt.show()

