# import required packages
import os 
import csv
from sympy import symbols, init_printing
import numpy as np
import sympy
import itertools
import scipy
import datetime
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from pygom import DeterministicOde, Transition, SimulateOde, TransitionType, SquareLoss
from scipy.optimize import minimize

import pickle as pk
import jsonpickle as jpk

from cycler import cycler
import pwlf

import umap
import umap.plot

from sklearn.decomposition import PCA
from sklearn import cluster
from sklearn.cluster import AgglomerativeClustering
import hdbscan

import warnings
import math

# for FPCA:
#
import skfda
from skfda.datasets import fetch_growth
from skfda.exploratory.visualization import plot_fpca_perturbation_graphs
from skfda.preprocessing.dim_reduction.projection import FPCA
from skfda.representation.basis import BSpline, Fourier, Monomial

#############################################################################
# # DATA
# # from looking at dat data matrix definitions, need following country indexed dicts:
# #
# # longshort_c
# # longshort_cases_c
# # testing
# # first_thresh
# # longshort_cases_adj_c1

## maybe rationalize:
## deaths_raw
## cases_raw
## cases_adj_lin2020
## cases_adj_pwlfit
## cases_adj_nonlin   ( = old longshort_cases_adj_c)
## cases_adj_nonlinr

# two key parameters : mindeaths between 100 and 200 and mindays 150 to 160 
print('Getting data:')
from data import *

print('Constructing common synchronized deaths, case and testing data...');

# database == 'OWID' # OWID database: to use JHU database comment this line and use line below
database='JHU'     # JHU database: to use OWID database comment this line and use line above
report_correct = True     # whether to use reporting spike corrected data for clustering
daysync = 23       # needs to be same as value in data.py

"""
# We don't need most of this any more 
print('database',database, 'correct active',report_correct)

# for OWID database
# covid_owid_ts= {'confirmed':confirmed_owid,'deaths':deaths_owid,'recovered':recovered_owid, 'tests': tests_owid , 'stringency': stringency_owid,
#                 'population':population_owid,'population_density':population_density_owid,'gdp_per_capita':gdp_per_capita_owid}

total_deaths_x = get_data_owid_key('total_deaths',daysync)
new_deaths_spm_x = get_data_owid_key('new_deaths_smoothed_per_million',daysync)
total_cases_x = get_data_owid_key('total_cases',daysync)
total_cases_ppm_x = get_data_owid_key('total_cases_per_million',daysync)
new_cases_spm_x = get_data_owid_key('new_cases_smoothed_per_million',daysync)
testing_x = get_data_owid_key('new_tests_smoothed_per_thousand',daysync)

total_deaths_owid = {cc:total_deaths_x[cc] for cc in total_deaths_x if cc != 'dates' and cc != 'World'}
new_deaths_spm = {cc:new_deaths_spm_x[cc] for cc in new_deaths_spm_x if cc != 'dates' and cc != 'World'}
total_cases = {cc:total_cases_x[cc] for cc in total_cases_x if cc != 'dates' and cc != 'World'}
total_cases_ppm = {cc:total_cases_ppm_x[cc] for cc in total_cases_ppm_x if cc != 'dates' and cc != 'World'}
new_cases_spm = {cc:new_cases_spm_x[cc] for cc in new_cases_spm_x if cc != 'dates' and cc != 'World'}
testing = {cc:testing_x[cc] for cc in testing_x if cc != 'dates' and cc != 'World'}

print('done.')
"""

if database == 'OWID':
    if report_correct:
        total_deaths = total_deaths_cs_owid
        new_deaths_spm = new_deaths_c_spm_owid
        new_cases_spm = new_cases_c_spm_owid
    else:
        total_deaths = total_deaths_s_owid
        new_deaths_spm = new_deaths_spm_owid
        new_cases_spm = new_cases_spm_owid
elif database == 'JHU':
    if report_correct:
        total_deaths = total_deaths_cs_jhu     
        new_deaths_spm = new_deaths_c_spm_jhu
        new_cases_spm = new_cases_c_spm_jhu
    else:
        total_deaths = total_deaths_s_jhu
        new_deaths_spm = new_deaths_spm_jhu
        new_cases_spm = new_cases_spm_jhu


big = {cc:new_deaths_spm[cc] for cc in bcountries}
big_cases = {cc:new_cases_spm[cc] for cc in bcountries}
print('debug len(total_deaths)',len(total_deaths))
print('debug len(big)',len(big))


# badspikes = ['Peru','Bolivia','Chile','China','Equador','Kyrgystan']   # eliminate Peru and a few other countries because of bad spikes.

# synchronization method : by threshold on total deaths
print('synchronizing and trimming time series to common length...')
short_deaths = {}
short_cases = {}
short_testing = {}
short_reg_testing = {}
first_thresh = {}
thresh = 10   # better to use day when #total_deaths (ie cumulative) absolute first reaches 10 or perhaps 30 absolute as sync point & keep entire rest of trace
for cc in bcountries:
    tdates = len(total_deaths['Germany'])  # changed to a particular common country to get database indept (formerly using 'dates' entry in total_Deaths_x)
    for i in range(tdates):
        if total_deaths[cc][i] >= thresh:
            short_deaths[cc] = [big[cc][j] for j in range(i,len(big[cc]))]
            short_cases[cc] = [big_cases[cc][j] for j in range(i,len(big_cases[cc]))]
            short_testing[cc] = [testing[cc][j] for j in range(i,len(testing[cc]))]
            short_reg_testing[cc] = [reg_testing[cc][j] for j in range(i,len(reg_testing[cc]))]
            first_thresh.update({cc:i})
            break;
short_deaths_est =  min([len(short_deaths[x]) for x in short_deaths])
short_deaths_c = {cc:short_deaths[cc][:short_deaths_est] for cc in short_deaths} # this crops all country time series to the shortest one, currently not used
short_cases_est =  min([len(short_cases[x]) for x in short_cases])
short_cases_c = {cc:short_cases[cc][:short_cases_est] for cc in short_cases}
short_testing_est = min([len(short_testing[x]) for x in short_testing])
short_reg_testing_est = min([len(short_reg_testing[x]) for x in short_reg_testing])
short_testing_c = {cc:short_testing[cc][:short_testing_est] for cc in short_testing}
short_reg_testing_c = {cc:short_reg_testing[cc][:short_reg_testing_est] for cc in short_reg_testing} 

# choose subset of time series data that must have at least len mindays=160
mindays = 150 # changed from 160 to include more countries on Sep 24
longshort = {cc:short_deaths[cc] for cc in short_deaths if (len(short_deaths[cc])>=mindays)};
longshortest =  min([len(longshort[x]) for x in longshort])
longshort_c = {cc:longshort[cc][:longshortest] for cc in longshort}
lcountries = [cc for cc in longshort_c]

# scaled_cases = {cc:new_cases_spm[cc]/max(new_cases_spm[cc]) for cc in lcountries} # note that lcountries determined by death data

# select only traces with minimum length of mindays
longshort_cases = {cc:short_cases[cc] for cc in short_cases if (len(short_cases[cc])>=mindays)};
longshort_cases_est =  min([len(longshort_cases[x]) for x in longshort_cases])
clusdata_len = longshort_cases_est
longshort_cases_c = {cc:longshort_cases[cc][:longshort_cases_est] for cc in longshort_cases}
lccountries = longshort_cases.keys()


longshort_testing_c = {cc:short_testing[cc][:clusdata_len] for cc in short_testing}
longshort_reg_testing_c = {cc:short_reg_testing[cc][:clusdata_len] for cc in short_reg_testing}
big_testing_c = longshort_reg_testing_c


deaths_raw = longshort_c
cases_raw = longshort_cases_c
             
dat = np.array([longshort_cases_c[cc] for cc in longshort_cases_c])
testingtmp = np.linspace(0.1,1.0,len(dat[0]))
dat = [dd/testingtmp for dd in dat]
cases_adj_lin2020 = {lcountries[i]:dat[i] for i in range(len(dat))}

# cases w/ piecewise linear fit testing rampup
# testing = np.linspace(0.1,1.0,len(longshort_cases_c['Germany']))

reg_testing_lc = {cc:np.array(longshort_reg_testing_c[cc]) for cc in lcountries}
dat = np.array([longshort_cases_c[cc]/reg_testing_lc[cc] for cc in lcountries])
# dat = np.array([longshort_cases_c[cc]/testing_lc[cc][first_thresh[cc]:first_thresh[cc]+len(longshort_cases_c[cc])] for cc in lcountries])
cases_adj_pwlfit = {lcountries[i]:dat[i] for i in range(len(dat))}


print('making cases with nonlinear testing adjustment...')
cases_adj_nonlin = make_cases_adj_nonlin(longshort_testing_c,longshort_cases_c,K=2)
cases_adj_nonlinr = make_cases_adj_nonlin(longshort_reg_testing_c,longshort_cases_c,K=2)              
print('done.')
print('to change the nonlinear correction function, call make_cases_adj_nonlin(K), K=2 by default')

clusdata_all = {}
clusdata_all['deaths'] = deaths_raw
clusdata_all['cases'] = cases_raw
clusdata_all['cases_lin2020'] = cases_adj_lin2020
clusdata_all['cases_pwlfit'] = cases_adj_pwlfit
clusdata_all['cases_nonlin'] = cases_adj_nonlin
clusdata_all['cases_nonlinr'] = cases_adj_nonlinr


def plot_adj(country, data, adj = None, testing=None,  ndays=250, axis = None):
    ndays = 250
    if testing:
        # Ntests = regtests(testing,country)  # this does not work here, since data is already synchronized, use regularized data in testing 
        Ntests = testing[country]
    if axis is None:   
        fig, ax1 = plt.subplots(figsize=(12,8))
    else:
        ax1 = axis
    ax1.plot(data[country][:ndays]) 
    if adj is not None:  # already adjusted
        ax1.plot(adj[country][:ndays])
    ax1.set_title(country)
    ax1.set_ylabel('Cases/million')
    ax1.set_xlabel('day')
    if testing:
        ax2 = ax1.twinx()
        ax2.plot(Ntests[:ndays],color='red',alpha=0.4)
        ax2.set_ylabel('Testing/1000')


    if axis is None:
        plt.show()

def plot_all(countries,dat,adj=None,testing=None,ndays=250):
    max_cols=6
    max_rows=int(len(countries)/max_cols) + 1
    fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(24,4*max_rows))

    for idx, country  in enumerate(countries):
        row = idx // max_cols
        col = idx % max_cols
        plot_adj(country,dat,adj,testing,ndays,axis=axes[row,col])
    for idx in range(len(countries),max_rows*max_cols):
        row = idx // max_cols
        col = idx % max_cols
        axes[row, col].axis("off")
    #plt.subplots_adjust(wspace=.05, hspace=.05)
    fig.tight_layout()
    #for ax in fig.get_axes():
    #    ax.label_outer()
    plt.show()

def plot_all2(countries,dat,adj=None,testing=None,ndays=250):
    max_cols=6
    max_rows=int(len(countries)/max_cols) + 1
    fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(24,4*max_rows))

    for idx, country  in enumerate(countries):
        row = idx // max_cols
        col = idx % max_cols
        axes[row,col].plot(dat[country])
        if adj is not None:
            ax = axes[row,col].twinx()
            ax.plot(adj[country])
    for idx in range(len(countries),max_rows*max_cols):
        row = idx // max_cols
        col = idx % max_cols
        axes[row, col].axis("off")
    #plt.subplots_adjust(wspace=.05, hspace=.05)
    fig.tight_layout()
    #for ax in fig.get_axes():
    #    ax.label_outer()
    plt.show()

"""
Compute correlations between clusterings, component by component.  Gather best correlation between each cluster and clusterings of all other 15.
"""

def corcl(a,b):
    if len(set(a)) > 0 or len(set(b)) > 0:
        return len(set(a).intersection(set(b)))/float(len(set(a).union(set(b))))
    else:
        return 1 
    
def match(a,x):
    rtn = [i for i in range(len(a)) if a[i] == x]
    return rtn
    
def mxcor(m,n,nclus=3):
    cx = []
    for k in range(nclus):
        m1 = match(m,k)
        m2 = match(n,k)
        cx.append(corcl(m1,m2))
    return max(cx)

# corclasses = np.zeros((len(classes),len(classes)))
# for i in range(len(classes)-1):
#     cc = classes[i]
#     for j in range(i+1,len(classes)):
#         ccc = classes[j]
#         cx = []
#         corclasses[i,j] = mxcor(cc,ccc)
#         corclasses[j,i] = corclasses[i,j]

# for i in range(len(classes)):
#     corclasses[i,i] = 1.0
    
    
print('----------------------------------------')
print('Finished loading Cluster module')
print('----------------------------------------')


###########################################################
# to get ModelFit class definition:
exec(open('ClusterFit.py','r').read())
###########################################################
