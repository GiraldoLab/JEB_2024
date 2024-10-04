#Plots the average 20 minute heading for each individual fly in a polar plot 

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
 

# Calculates mean heading angle of fly (degrees)
def full_arctan(x,y):
    angle = np.arctan(y/x)+np.pi if x<0 else np.arctan(y/x)
    return angle if angle <= np.pi else angle -2*np.pi  

def angle_avg(data):
    return full_arctan(np.cos(data*np.pi/180).sum(),np.sin(data*np.pi/180).sum())*180/np.pi  
# Calculates mean heading angle of fly (radians)
def circmean(alpha,axis=None):  
    mean_angle = np.arctan2(np.mean(np.sin(alpha),axis),np.mean(np.cos(alpha),axis))
    return mean_angle

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

#Calculates variance 
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


 
#Path to fig1B csv files
path = ''
experiment_names = glob.glob(os.path.join(path, "*.csv"))
experiment_names.sort(reverse=False)
heading_angles=[]
heading_vars=[]
sun_positions =[]
experiments =[]
output= {}
time_period = 1200


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
        
        full_sun_period = sun_period.loc[[(frame_time - sun_period['Image Time'].iloc[0]).seconds<=time_period for frame_time in sun_period['Image Time']]]


        sun_position = sun_angle(full_sun_period['Sun Position'].iloc[0]) #to show where sun stimulus was 
        avg_angle = angle_avg(full_sun_period['Heading Angle'])
        var_angle = angle_var(full_sun_period['Heading Angle'])
        heading_angles.append(avg_angle)   
        heading_vars.append(var_angle)
        sun_positions.append(sun_position)

heading_angles = np.array(heading_angles)   
heading_vars= np.array(heading_vars)
sun_positions = np.array(sun_positions)


##Apply cutoff filter to exclude flies with a vector strength smaller than 0.2 ###
AngleVarsCutOff = np.zeros (1)
MeanAnglesCutOff = np.zeros (1)
SunCutOff = np.zeros (1)

for x in range (heading_vars.size):
    if  heading_vars[x] >0.8 :                                  
        print(x)
        pass
    else:
        AngleVarsCutOff = np.append(AngleVarsCutOff, heading_vars[x])
        MeanAnglesCutOff = np.append(MeanAnglesCutOff, heading_angles[x])
        SunCutOff = np.append(SunCutOff, sun_positions[x])

AngleVarsCutOff = AngleVarsCutOff[1:]
print('average vector strength first 5min:', 1 - sum(AngleVarsCutOff)/len(AngleVarsCutOff))
MeanAnglesCutOff = MeanAnglesCutOff[1:] 
SunCutOff = SunCutOff[1:] 

VecStrengthCutOff = 1 - AngleVarsCutOff

rotated_heading_angles = difference(MeanAnglesCutOff, SunCutOff , deg= True)




#% Plots Sea Urchin and runs Rayleigh Test
from circstats import confmean
from astropy.stats import rayleightest
from circstats import confmean
from collections import Counter

def sea_urchin_with_stacked_symbols(circmeans, vecstrengths, bin_num, hist_start,hist_spacing, figname):
    #fig = plt.figure(figsize = (10,10))
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111, projection = 'polar')
    #ax.scatter(circmeans, vecstrengths, color = 'black', zorder=20) #for scatter plot
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
    #CI_arc = np.linspace((pop_mean-conf_in_95), (pop_mean + conf_in_95), num=10000)
    CI_arc = np.linspace(pop_mean-conf_in_95, pop_mean+conf_in_95, num=10000) % (2*np.pi)

    bins = np.linspace(0, 2*np.pi, bin_num+1, endpoint = True)
    digitized = np.digitize(circmeans[0,:], bins)
    z = Counter(digitized)
    ax.grid(False)
    circle = plt.Circle((0.0, 0.0), 1., transform=ax.transData._b, edgecolor=([0.9, 0.9, 0.9]), facecolor= ([0.9, 0.9, 0.9]), zorder=10)
    ax.plot(CI_arc, CI_arc_r, color= 'r', linewidth=2, zorder=50) #plot the umbrella
    ax.scatter(0, 0, s=75, color='r', marker= '+', linewidth = 2, zorder=25)
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_zero_location ("N")
    ax.set_theta_direction(-1)
    ax.add_artist(circle)
    ax.spines['polar'].set_visible(False) 
    for bin_index, angle in enumerate(bins):
        #print angle, z[bin_index]
        count = z[bin_index]
        bin_spacing = 2*np.pi/bin_num
        bin_center = angle - (bin_spacing/2)  
        if count >0:
            hist_r_pos= np.linspace(hist_start, hist_start+(hist_spacing*(count-1)), count, endpoint = True)
            #print ' hist_r_pos ', hist_r_pos
            #print 'diff', np.diff(hist_r_pos)
            ax.scatter([bin_center]*count, np.linspace(hist_start, hist_start+(hist_spacing*(count-1)), count, endpoint = True),s=95, marker = '.', color = 'black', zorder=20)
            ax.set_theta_offset(np.pi/2)
    ax.axis('off') #original 'off'
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
    print('Rayleigh Test Results',rayleigh_result)
        
    # plot
    sea_urchin_with_stacked_symbols(circmeans= list_of_positive_angles_ToPlot, vecstrengths= RAToPlot, bin_num=90, hist_start=1.1, hist_spacing=0.09, figname= figname)

PlotSeaUrchin(rotated_heading_angles, VecStrengthCutOff, '20min Headings Polar plot')

plt.show()