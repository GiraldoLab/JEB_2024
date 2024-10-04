# -*- coding: utf-8 -*-
"""
Created on Mon May  8 15:00:11 2017

@author: Ysabel
"""

'''
how to use:
1. run data_format_modify_histographs_cutoff for A
2. in ipython, type: A = allFliesVectorStrength
3. run data_format_modify_histographs_cutoff for B
4. in ipython, type: B = allFliesVectorStrength
5. run this file
'''
#Plots histogram of vector strengths shown in figure 6.
import numpy as np
import os
import matplotlib.pyplot as plt
plt.ion()
import warnings
import numpy.ma as ma
import datetime
import scipy as sp
import scipy.stats as stats
from matplotlib import gridspec
#import fly_plot_basics as fpb

def cumprobdist(data,ax=None,xmax=None,color ='k', plotArgs={}):
    if ax is None:
        ax = plt.axes()
    if xmax is None:
        xmax = np.max(data)
    elif xmax < np.max(data):
        warnings.warn('value of xmax lower than maximum of data')
        xmax = np.max(data)
    num_points = float(len(data))
    X = np.concatenate(([0.0],data,data,[xmax]))
    X.sort()
    #X = X[-1::-1]
    Y = np.concatenate(([0.0], np.arange(num_points), np.arange(num_points)+1, [num_points]))/num_points
    Y.sort()
    line = ax.plot(X,Y,color = color, **plotArgs)
    return line[0]

#From Fig6 cue conflict experiments, vector strengths of first suns, second sun green only, second sun uv only
#G=UV
first_suns = [0.94970302, 0.61128665, 0.72071588, 0.97074753, 0.59187742, 0.87995426, 0.97132691, 0.94961526, 0.5332515,  0.57125375, 0.95667494, 0.94788164, 0.90360289, 0.47015852, 0.71193528, 0.81797661, 0.90444374, 0.97779341, 0.83067945, 0.28463917, 0.72641423, 0.39305815, 0.8205908,  0.75477116, 0.60114169, 0.35825306, 0.85572113, 0.98153894, 0.69621656, 0.78617966, 0.89847789, 0.90882273, 0.94586629, 0.30265851, 0.93060242, 0.95408884, 0.79115937, 0.39207136, 0.93971864, 0.70840495]
#G
second_grn =[0.31309539, 0.88129192, 0.32662878, 0.45490661, 0.86083202, 0.84606402, 0.72128704, 0.65555328, 0.80477123, 0.37361364, 0.69255188, 0.84812189, 0.84723347, 0.24685772, 0.47695965, 0.72332616, 0.32579034, 0.92774928, 0.9085588,  0.64973342, 0.62159018]
#UV
second_uv = [0.98489282, 0.93929178, 0.91042521, 0.97994977, 0.95906716, 0.73351319, 0.9780449,  0.71779938, 0.63821322, 0.98973983, 0.99104029, 0.47551438, 0.52276562, 0.47273732, 0.96657928, 0.98621099, 0.69708203, 0.44950069, 0.99621637]

#From fig1 data
#Fig1.C G1
same_grn1 =[0.30876681, 0.69405866, 0.37880435, 0.97281503, 0.73111796, 0.90833377, 0.92475695, 0.88718937, 0.79688413, 0.78108206, 0.44119246, 0.26156735, 0.31734997, 0.42883358, 0.69011238, 0.96485702, 0.69011139, 0.72075318, 0.90561446, 0.95999395, 0.36453221, 0.51417206, 0.86621161, 0.5668335, 0.90257985, 0.53660541, 0.87666858, 0.97940186, 0.80481848, 0.94677363, 0.71133108, 0.95634977, 0.94815232, 0.93031083, 0.94292636, 0.98958336, 0.76106515, 0.85642392, 0.77243855]
#Fig1.C G2
same_grn2 =[0.61122112, 0.93276165, 0.84911064, 0.97982604, 0.32570881, 0.93418373, 0.95175869, 0.94904928, 0.6407256,  0.92116442, 0.97080601, 0.9680278, 0.97481226, 0.72656106, 0.85254854, 0.98452721, 0.78966856, 0.81760781, 0.81818731, 0.98266565, 0.40595388, 0.79886626, 0.63998284, 0.35995227, 0.91283895, 0.93827009, 0.98057595, 0.99136902, 0.97772805, 0.97260796, 0.97478281, 0.97182111, 0.97979182, 0.97864705, 0.96984786, 0.9964902, 0.99495061, 0.97709066, 0.98557274]


fig = plt.figure(figsize=(4,4))
fig.set_facecolor('w')
ax = fig.add_subplot(1, 1, 1)
for axis in ['left','bottom']:
            ax.spines[axis].set_linewidth(1.5)
ax.xaxis.set_tick_params(width=1.5, direction = 'in')
ax.yaxis.set_tick_params(width=1.5, direction = 'in')



cumprobdist(first_suns, ax=ax, xmax=1., color='k')
cumprobdist(second_grn, ax=ax, xmax=1. , color ='limegreen' )
cumprobdist(second_uv, ax=ax, xmax=1. , color ='mediumorchid')
cumprobdist(same_grn1, ax=ax, xmax =1., color ='deepskyblue')
cumprobdist(same_grn2, ax=ax, xmax=1., color ='dodgerblue')

ax.set_ylim((-.1, 1.1))
ax.set_xlim((-.1, 1.1))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_bounds(0.,1.)
ax.spines['left'].set_bounds(0.,1.)
ax.set_yticks((0, 0.5, 1))
ax.set_xticks((0, 0.5, 1))
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
ax.set_xlabel('Vector Strength')
ax.set_ylabel('Fraction of flies')
plt.show()
fig.savefig('cue_conflict_and_same_grn_suns.png', transparent=True, dpi =600)
#%% Plot each individual vector strength

