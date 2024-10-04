#!/usr/bin/env python
# Plots polarplots, linear plots of each sun period headings for flight trials where the sun moved 15degrees per hour with a dual green-UV spectral cue(Each fly is tested 4 times, 1hr, 2hrs, and 6hr after the initial flight trial)
# Plots the heading difference as a polar plot and running resampling analysis(bootstrap analysis) for differences between 1hr, 2hr, 6hr trials

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
def full_arctan(x,y):
    angle = np.arctan(y/x)+np.pi if x<0 else np.arctan(y/x)
    return angle if angle <= np.pi else angle -2*np.pi  
def angle_avg(data):
    return full_arctan(np.cos(data*np.pi/180).sum(),np.sin(data*np.pi/180).sum())*180/np.pi  

# Calculates mean heading angle of fly(radians)
def circmean(alpha, axis =None):   
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


def sun_to_angle(led_number):
    angle_map = {57:45, 198: -135, 51:60, 195:-120, 45:75, 193:-105, 21:135, 183:-45, 39:90, 190:-90, 15:150, 180:-30, 93: -45, 152: 135, 99:-60, 155:120, 105:-75, 157:105, 129: -135, 167: 45}
    return angle_map.get(led_number, "Invalid input" )

allFliesMeanAnglesListA = list()
allFliesRotatedMeanAnglesListA = list()
allFliesAngleVarsListA = list()
allFliesGrnSunListA = list()
allFliesUVSunListA = list()

allFliesMeanAnglesListB = list()
allFliesRotatedMeanAnglesListB = list()
allFliesAngleVarsListB = list()
allFliesGrnSunListB = list()
allFliesUVSunListB = list()


allFliesMeanAnglesListC = list()
allFliesRotatedMeanAnglesListC = list()
allFliesAngleVarsListC = list()
allFliesGrnSunListC = list()
allFliesUVSunListC = list()

allFliesMeanAnglesListD = list()
allFliesRotatedMeanAnglesListD = list()
allFliesAngleVarsListD = list()
allFliesGrnSunListD = list()
allFliesUVSunListD = list()



path = '' #Path to fig9
#Sorts all the csv files in order in 'path' and sorts it to A trials and B trials
experiment_names = glob.glob(os.path.join(path, "*.csv"))
experiment_names.sort(reverse=False)
#Sorts all fly A trials in numerical order - file names needs to start from fly01,fly02,.. when naming file
flight_0hr = [f for f in experiment_names if f[-5:]=='A.csv']
flight_0hr.sort(reverse=False)
flight_1hr = [f for f in experiment_names if f[-5:]=='B.csv']
flight_1hr.sort(reverse=False)
flight_2hr= [f for f in experiment_names if f[-5:]=='C.csv']
flight_2hr.sort(reverse=False)
flight_6hr = [f for f in experiment_names if f[-5:]=='D.csv']
flight_6hr.sort(reverse=False)

experiments =[]
output= {}
time_delay = 0

for experiment_name in experiment_names:
    experiment = pd.read_csv(experiment_name)
    output[str(experiment_name)] = {}
    experiment['Image Time'] = experiment['Image Time'].apply(parser.parse)
    experiment_data = experiment.values
    sun_change_indexes = [0]+[i for i in range(1,len(experiment_data)) if experiment_data[i,3]!=experiment_data[i-1,3]]
    sun_periods = [experiment[sun_change_indexes[-1]:-1]]
    for sun_period in sun_periods:
        delayed_sun_period = sun_period.loc[[(frame_time - sun_period['Image Time'].iloc[0]).seconds>time_delay for frame_time in sun_period['Image Time']]]
        grn_sun_position = sun_to_angle(delayed_sun_period['Sun Position1'].iloc[0]) #to show where sun stimulus was during the experiment
        uv_sun_position = sun_to_angle(delayed_sun_period['Sun Position2'].iloc[0])
        heading_angle = angle_avg(delayed_sun_period['Heading Angle']) #original: pos_angles(delayed_sun_period['Heading Angle'])
        heading_angle_var = angle_var(delayed_sun_period['Heading Angle'])#original: pos_angles(delayed_sun_period['Heading Angle'])
        heading_angle_std = angle_std(heading_angle_var)
        heading_angle_yamartino_std = yamartino_angle_std(heading_angle_var)

        if experiment_name in flight_0hr:     
            allFliesMeanAnglesListA.append(heading_angle)
            allFliesAngleVarsListA.append(heading_angle_var)
            allFliesGrnSunListA.append(grn_sun_position)
            allFliesUVSunListA.append(uv_sun_position)


        elif experiment_name in flight_1hr:
            allFliesMeanAnglesListB.append(heading_angle)
            allFliesAngleVarsListB.append(heading_angle_var)
            allFliesGrnSunListB.append(grn_sun_position)
            allFliesUVSunListB.append(uv_sun_position)

        elif experiment_name in flight_2hr:
            allFliesMeanAnglesListC.append(heading_angle)
            allFliesAngleVarsListC.append(heading_angle_var)
            allFliesGrnSunListC.append(grn_sun_position)
            allFliesUVSunListC.append(uv_sun_position)

        elif experiment_name in flight_6hr:
            allFliesMeanAnglesListD.append(heading_angle)
            allFliesAngleVarsListD.append(heading_angle_var)
            allFliesGrnSunListD.append(grn_sun_position)
            allFliesUVSunListD.append(uv_sun_position)

allFliesMeanAnglesA = np.array(allFliesMeanAnglesListA)
allFliesAngleVarsA = np.array(allFliesAngleVarsListA)         
allFliesGrnSunA = np.array(allFliesGrnSunListA)
allFliesUVSunA = np.array(allFliesUVSunListA)

allFliesMeanAnglesB = np.array(allFliesMeanAnglesListB)
allFliesAngleVarsB = np.array(allFliesAngleVarsListB)
allFliesGrnSunB = np.array(allFliesGrnSunListB)
allFliesUVSunB = np.array(allFliesUVSunListB)


allFliesMeanAnglesC = np.array(allFliesMeanAnglesListC)
allFliesAngleVarsC = np.array(allFliesAngleVarsListC)         
allFliesGrnSunC = np.array(allFliesGrnSunListC)
allFliesUVSunC = np.array(allFliesUVSunListC)

allFliesMeanAnglesD = np.array(allFliesMeanAnglesListD)
allFliesAngleVarsD = np.array(allFliesAngleVarsListD)
allFliesGrnSunD = np.array(allFliesGrnSunListD)
allFliesUVSunD = np.array(allFliesUVSunListD)

## Apply cut off for vector strength <0.2 ##
allFliesAngleVarsACutOff = np.zeros (1)
allFliesAngleVarsBCutOff = np.zeros (1)
allFliesAngleVarsCCutOff = np.zeros (1)
allFliesAngleVarsDCutOff = np.zeros (1)

allFliesMeanAnglesACutOff = np.zeros (1)
allFliesMeanAnglesBCutOff = np.zeros (1)
allFliesMeanAnglesCCutOff = np.zeros (1)
allFliesMeanAnglesDCutOff = np.zeros (1)

allFliesGrnSunACutOff = np.zeros (1)
allFliesGrnSunBCutOff = np.zeros (1)
allFliesGrnSunCCutOff = np.zeros (1)
allFliesGrnSunDCutOff = np.zeros (1)  

allFliesUVSunACutOff = np.zeros (1)
allFliesUVSunBCutOff = np.zeros (1)
allFliesUVSunCCutOff = np.zeros (1)
allFliesUVSunDCutOff = np.zeros (1)

for x in range (allFliesAngleVarsA.size):
    if allFliesAngleVarsA[x] > 0.8 or allFliesAngleVarsB[x] > 0.8 or allFliesAngleVarsC[x] > 0.8 or allFliesAngleVarsD[x] > 0.8:                                      
        print(x)
        pass
    else:
        allFliesAngleVarsACutOff = np.append(allFliesAngleVarsACutOff, allFliesAngleVarsA[x])
        allFliesMeanAnglesACutOff = np.append(allFliesMeanAnglesACutOff, allFliesMeanAnglesA[x])
        allFliesGrnSunACutOff = np.append(allFliesGrnSunACutOff, allFliesGrnSunA[x])
        allFliesUVSunACutOff = np.append(allFliesUVSunACutOff, allFliesUVSunA[x])

        allFliesAngleVarsBCutOff = np.append(allFliesAngleVarsBCutOff, allFliesAngleVarsB[x])
        allFliesMeanAnglesBCutOff = np.append(allFliesMeanAnglesBCutOff, allFliesMeanAnglesB[x])
        allFliesGrnSunBCutOff = np.append(allFliesGrnSunBCutOff, allFliesGrnSunB[x])
        allFliesUVSunBCutOff = np.append(allFliesUVSunBCutOff, allFliesUVSunB[x])

        allFliesAngleVarsCCutOff = np.append(allFliesAngleVarsCCutOff, allFliesAngleVarsC[x])
        allFliesMeanAnglesCCutOff = np.append(allFliesMeanAnglesCCutOff, allFliesMeanAnglesC[x])
        allFliesGrnSunCCutOff = np.append(allFliesGrnSunCCutOff, allFliesGrnSunC[x])
        allFliesUVSunCCutOff = np.append(allFliesUVSunCCutOff, allFliesUVSunC[x])

        allFliesAngleVarsDCutOff = np.append(allFliesAngleVarsDCutOff, allFliesAngleVarsD[x])
        allFliesMeanAnglesDCutOff = np.append(allFliesMeanAnglesDCutOff, allFliesMeanAnglesD[x])
        allFliesGrnSunDCutOff = np.append(allFliesGrnSunDCutOff, allFliesGrnSunD[x])
        allFliesUVSunDCutOff = np.append(allFliesUVSunDCutOff, allFliesUVSunD[x])


allFliesAngleVarsACutOff = allFliesAngleVarsACutOff[1:]
allFliesMeanAnglesACutOff = allFliesMeanAnglesACutOff[1:]
allFliesGrnSunACutOff = allFliesGrnSunACutOff[1:]
allFliesUVSunACutOff = allFliesUVSunACutOff[1:]

allFliesAngleVarsBCutOff = allFliesAngleVarsBCutOff[1:]
allFliesMeanAnglesBCutOff = allFliesMeanAnglesBCutOff[1:]
allFliesGrnSunBCutOff = allFliesGrnSunBCutOff[1:]
allFliesUVSunBCutOff = allFliesUVSunBCutOff[1:]

allFliesAngleVarsCCutOff = allFliesAngleVarsCCutOff[1:]
allFliesMeanAnglesCCutOff = allFliesMeanAnglesCCutOff[1:]
allFliesGrnSunCCutOff = allFliesGrnSunCCutOff[1:]
allFliesUVSunCCutOff = allFliesUVSunCCutOff[1:]

allFliesAngleVarsDCutOff = allFliesAngleVarsDCutOff[1:]
allFliesMeanAnglesDCutOff = allFliesMeanAnglesDCutOff[1:]
allFliesGrnSunDCutOff = allFliesGrnSunDCutOff[1:]
allFliesUVSunDCutOff = allFliesUVSunDCutOff[1:]

print('0hrAtrialvecstrengthsavg:',1-sum(allFliesAngleVarsACutOff)/len(allFliesAngleVarsACutOff))
print('1hrtrialvecstrengthsavg:',1-sum(allFliesAngleVarsBCutOff)/len(allFliesAngleVarsBCutOff))
print('2hrtrialvecstrengthsavg:',1-sum(allFliesAngleVarsCCutOff)/len(allFliesAngleVarsCCutOff))
print('6hrtrialvecstrengthsavg:',1-sum(allFliesAngleVarsDCutOff)/len(allFliesAngleVarsDCutOff))


allFliesVecStrengthCutOffA = 1-allFliesAngleVarsACutOff
allFliesVecStrengthCutOffB = 1-allFliesAngleVarsBCutOff
allFliesVecStrengthCutOffC = 1-allFliesAngleVarsCCutOff
allFliesVecStrengthCutOffD = 1-allFliesAngleVarsDCutOff

rotated_heading_anglesA = difference(allFliesMeanAnglesACutOff, allFliesGrnSunACutOff, deg= True)
rotated_heading_anglesB = difference(allFliesMeanAnglesBCutOff, allFliesGrnSunBCutOff, deg= True)
rotated_heading_anglesC = difference(allFliesMeanAnglesCCutOff, allFliesGrnSunCCutOff, deg= True)
rotated_heading_anglesD = difference(allFliesMeanAnglesDCutOff, allFliesGrnSunDCutOff, deg= True)

heading_difference1hr= difference(rotated_heading_anglesA ,rotated_heading_anglesB, deg=True)
heading_difference2hr = difference(rotated_heading_anglesA, rotated_heading_anglesC, deg=True)
heading_difference6hr= difference(rotated_heading_anglesA ,rotated_heading_anglesD, deg=True)


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
    if title is not None:  # Add title if specified
        ax.set_title(title)
    #fig.tight_layout()
    fig.savefig(title, transparent=True, dpi=600)

LinearPlot(rotated_heading_anglesA, rotated_heading_anglesB, allFliesAngleVarsACutOff,allFliesAngleVarsACutOff,x_axis =' 0hr Sun Headings', y_axis = '1hr Sun Headings', title = '0hr_1hr Sun Headings')
LinearPlot(rotated_heading_anglesA, rotated_heading_anglesC, allFliesAngleVarsACutOff, allFliesAngleVarsCCutOff, x_axis =' 0hr Sun Headings',y_axis = '2hr Sun Headings', title = '0hr_2hr Sun Headings')
LinearPlot(rotated_heading_anglesA, rotated_heading_anglesD, allFliesAngleVarsACutOff,allFliesAngleVarsDCutOff,x_axis =' 0hr Sun Headings', y_axis = '6hr Sun Headings', title = '0hr_6hr Sun Headings')

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
    
# Call the function with your data
BootstrapAnalysis(rotated_heading_anglesA, rotated_heading_anglesB, '0hr_1hr Sun Bootstrap Results')
BootstrapAnalysis(rotated_heading_anglesA, rotated_heading_anglesC, '0hr_2hr Sun Bootstrap Results')
BootstrapAnalysis(rotated_heading_anglesA, rotated_heading_anglesD, '0hr_6hr Sun Bootstrap Results')



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
PlotSeaUrchin(rotated_heading_anglesA, allFliesVecStrengthCutOffA, '0hr Flight Polar plot')
PlotSeaUrchin(rotated_heading_anglesB, allFliesVecStrengthCutOffB, '1hr Flight Polar plot')
PlotSeaUrchin(rotated_heading_anglesC, allFliesVecStrengthCutOffC, '2hr Flight Polar plot')
PlotSeaUrchin(rotated_heading_anglesD, allFliesVecStrengthCutOffD, '6hr Flight Polar plot')
PlotSeaUrchin(heading_difference1hr, np.ones_like(heading_difference1hr), '0-1hr Heading Difference Polarplot' )
PlotSeaUrchin(heading_difference2hr, np.ones_like(heading_difference2hr), '0-2hr Heading Difference Polarplot')
PlotSeaUrchin(heading_difference6hr, np.ones_like(heading_difference6hr), '0hr-6hr Heading Difference Polarplot')



#### For Fig10C clockwise(CW) and counterclockwise(CCW) analysis ####
sun_diff = difference(allFliesGrnSunACutOff, allFliesGrnSunBCutOff, deg= True)
CW_heading_difference1=[] #for 0-1hr clockwise heading difference
CW_heading_difference2=[] #for 0-2hr clockwise heading difference
CW_heading_difference3 =[] #for 0-6hr clockwise heading difference

CCW_heading_difference1 =[] #for 0-1hr counterclockwise heading difference
CCW_heading_difference2 =[] #for 0-2hr counterclockwise heading difference
CCW_heading_difference3 =[] #for 0-6hr counterclockwise heading difference

for i in range(len(sun_diff)):

    if sun_diff[i]<0:
        CW_heading_difference1.append(heading_difference1hr[i])
        CW_heading_difference2.append(heading_difference2hr[i])
        CW_heading_difference3.append(heading_difference6hr[i])
    else:
        CCW_heading_difference1.append(heading_difference1hr[i])
        CCW_heading_difference2.append(heading_difference2hr[i])
        CCW_heading_difference3.append(heading_difference6hr[i])

#Plot polar plots
PlotSeaUrchin(CW_heading_difference1, np.ones_like(CW_heading_difference1), '0-1hr CW heading difference')
PlotSeaUrchin(CW_heading_difference2, np.ones_like(CW_heading_difference2), '0-2hr CW heading difference')
PlotSeaUrchin(CW_heading_difference3, np.ones_like(CW_heading_difference3), '0-6hr CW heading difference')

PlotSeaUrchin(CCW_heading_difference1, np.ones_like(CCW_heading_difference1), '0-1hr CCW heading difference')
PlotSeaUrchin(CCW_heading_difference2, np.ones_like(CCW_heading_difference2), '0-2hr CCW heading difference')
PlotSeaUrchin(CCW_heading_difference3, np.ones_like(CCW_heading_difference3), '0-6hr CCW heading difference')

plt.show() 