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

#############################################################################
## DATA
## from looking at dat data matrix definitions, need following country indexed dicts:
##
## longshort_c
## longshort_cases_c
## testing
## first_thresh
## longshort_cases_adj_c

## maybe rationalize:
## deaths_raw
## cases_raw
## cases_adj_lin2020
## cases_adj_pwlfit
## cases_adj_nonlin   ( = old longshort_cases_adj_c)

print('Getting data:')
from data import *

total_deaths_x = get_data_owid_key('total_deaths')
total_deaths = {cc:total_deaths_x[cc] for cc in total_deaths_x if cc != 'dates' and cc != 'World'}
td_mx = [max(total_deaths[cc]) for cc in total_deaths]
countries = [cc for cc in total_deaths if max(total_deaths[cc])>=200]

new_deaths_spm = get_data_owid_key('new_deaths_smoothed_per_million')

# get rid of countries with trivially small new_deaths_spm:
# eliminate Peru because of bad spikes.
badspikes = ['Peru','Bolivia','Chile','China','Equador','Kyrgystan']                                             
mid = {cc:new_deaths_spm[cc] for cc in countries if max(new_deaths_spm[cc])>0.5 and max(new_deaths_spm[cc])<= 1} # note that several important countries between 0.5 and 1 per million
#print(len(mid))
#print(mid.keys())
big = {cc:new_deaths_spm[cc] for cc in countries if max(new_deaths_spm[cc])>0.5 and cc not in badspikes} # To Do:  replace badspike data with JHU.
#print(len(big))
#print(big.keys())
bcountries = big.keys()

scaled = {cc:new_deaths_spm[cc]/max(new_deaths_spm[cc]) for cc in countries}

# first synchronization method : replaced below by threshold on total deaths
short = {}
thresh = 0.01
for cc in big:
    for i in range(len(big[cc])):
        if big[cc][i] > thresh:
            short[cc] = [big[cc][j] for j in range(i,len(big[cc]))]
            break;
shortest = min([len(short[x]) for x in short])

short_t = {}
thresh = 0.01 # threshold of 0.01 per million ie 1/10^8 will not work well for medium sized countries
thresh = 10   # better to use day when #total_deaths (ie cumulative) absolute first reaches 10 or perhaps 30 absolute as sync point & keep entire rest of trace
for cc in big:
    tdates = len(total_deaths_x['dates'])
    for i in range(tdates):
        if total_deaths[cc][i] >= thresh:
            short_t[cc] = [big[cc][j] for j in range(i,len(big[cc]))]
            break;
short_test =  min([len(short_t[x]) for x in short_t])
short_c = {cc:short[cc][:shortest] for cc in short}

# must have at least len 160
longshort = {cc:short_t[cc] for cc in short_t if (len(short_t[cc])>=160)};
longshortest =  min([len(longshort[x]) for x in longshort])
longshort_c = {cc:longshort[cc][:longshortest] for cc in longshort}
lcountries = [cc for cc in longshort_c]

print('getting testing data...');
new_cases_spm = get_data_owid_key('new_cases_smoothed_per_million')
total_cases_x = get_data_owid_key('total_cases')
total_cases = {cc:total_cases_x[cc] for cc in total_cases_x if cc != 'dates' and cc != 'World'}
total_cases_ppm = get_data_owid_key('total_cases_per_million')
total_cases_ppm = {cc:total_cases_ppm[cc] for cc in total_cases_ppm if cc != 'dates' and cc != 'World'}
testing_x = get_data_owid_key('new_tests_smoothed_per_thousand')
testing = {cc:testing_x[cc] for cc in testing_x if cc != 'dates' and cc != 'World'}
print('done.')

# big_testing calculated from testing below : using piecewise linear approximation
# note first_thresh defined below needed to use testing in connection with synced data such as big

scaled_cases = {cc:new_cases_spm[cc]/max(new_cases_spm[cc]) for cc in lcountries}
big_cases = {cc:new_cases_spm[cc] for cc in bcountries} #  bcountries filters  max(new_deaths_spm[cc])>=0.5 and cc not in badspikes}

short_cases = {}
thresh = 0.01
for cc in big:
    for i in range(len(big[cc])):
        if big_cases[cc][i] > thresh:
            short_cases[cc] = [big_cases[cc][j] for j in range(i,len(big_cases[cc]))]
            break;
shortest = min([len(short_cases[x]) for x in short_cases])

short_cases_t = {}
thresh = 0.01 # threshold of 0.01 per million ie 1/10^8 will not work well for small countries
thresh = 10 # I think maybe better to use day when #total_deaths (ie cumulative) absolute first reaches 10 or perhaps 30 absolute as sync point & keep entire rest of trace
first_thresh = {}
for cc in big:
    tdates = len(total_deaths_x['dates'])
    for i in range(tdates):
        if total_deaths[cc][i] >= thresh:
            short_cases_t[cc] = [big_cases[cc][j] for j in range(i,len(big_cases[cc]))]
            first_thresh.update({cc:i})
            break;
short_cases_est =  min([len(short_cases_t[x]) for x in short_cases_t])
longshort_cases = {cc:short_cases_t[cc] for cc in short_cases_t if (len(short_cases_t[cc])>=160)};
longshort_cases_est =  min([len(longshort_cases[x]) for x in longshort_cases])
clusdata_len = longshort_cases_est
short_cases_c = {cc:short_cases[cc][:short_cases_est] for cc in short_cases}
longshort_cases_c = {cc:longshort_cases[cc][:longshort_cases_est] for cc in longshort_cases}
testing_c = {cc:testing[cc][:clusdata_len] for cc in testing}

print('doing piecwise linear fits...');
warnings.simplefilter('ignore')
big_testing={}
for i,cc in enumerate(big_cases):
    # testing_cap = np.array([max(t,0.1) for t in testing[cc]])
    testing_cap = testing[cc][50:]
    xxi = range(len(testing_cap))
    xHat=np.linspace(min(xxi), max(xxi), num=len(testing_cap))
    yyf = [Float(y) for y in testing_cap]
    if i<1000:
        my_pwlf = pwlf.PiecewiseLinFit(xxi, yyf)
        res = my_pwlf.fit(2,[0.],[0.1]) # force fit to go through point (0,0.1)
        # breaks = my_pwlf.fit(2,[0.],[0.1])
        slopes = my_pwlf.calc_slopes()
        pred = my_pwlf.predict(xHat)
        yHat = np.concatenate((np.array([0.1]*50),pred))
        yHat = np.array([max(t,0.1) for t in yHat])
        for i,y in enumerate(yHat):
            if i>0 and y<yHat[i-1]:
                yHat[i]=yHat[i-1]
        big_testing.update({cc:yHat.copy()})
    
big_testing_c = {cc:big_testing[cc][:clusdata_len] for cc in big_testing}
             
print('done.')

deaths_raw = longshort_c
cases_raw = longshort_cases_c
             

dat = np.array([longshort_cases_c[cc] for cc in longshort_cases_c])
testingtmp = np.linspace(0.1,1.0,len(dat[0]))
dat = [dd/testingtmp for dd in dat]
cases_adj_lin2020 = {lcountries[i]:dat[i] for i in range(len(dat))}


# cases w/ piecewise linear fit testing rampup
# testing = np.linspace(0.1,1.0,len(longshort_cases_c['Germany']))


testing_lc = {cc:np.array(big_testing[cc]) for cc in lcountries}
dat = np.array([longshort_cases_c[cc]/testing_lc[cc][first_thresh[cc]:first_thresh[cc]+len(longshort_cases_c[cc])]
                for cc in lcountries])
cases_adj_pwlfit = {lcountries[i]:dat[i] for i in range(len(dat))}


def regtests(testing,country,trampday1=50):
    Ntests = [tt for tt in testing[country]]
    tests = 0
    for i,tt in enumerate(testing[country]):
        if tt:
            break        
    tday1 = i
    if tday1 > trampday1:
        line = np.linspace(0.01,max(0.01,tt),i+1-trampday1)
    else:
        line = [tt]
    Ntests = [line[i-trampday1] if (i<tday1 and i>=trampday1) else tt for i,tt in enumerate(testing[country])]
    return Ntests


# for nonlinear testing adjustment:
def CaCo (Co, Nt, K=2):  # cases_actual / cases_observed given Nt=testing
    K1 = 25*(K-1)/(5.0-K)
    K2 = K1/5
    if Co > 0:
        rt = 1000*Nt/Co
        return (K1+rt)/(K2+rt)
    else:
        return 1

# # cases w/ nonlinear testing rampup
cases_adj_nonlin = {}


def make_cases_adj_nonlin(K=2):
    global cases_adj_nonlin
    cases_adj_nonlin={}
    testing_0p1 = {cc: [0.1 if math.isnan(t) else t for t in testing[cc]] for cc in big_cases}
    testing_0p1_c = {cc:testing_0p1[cc][-clusdata_len:] for cc in testing_0p1}
    cases_adj_nonlin = {cc:[CaCo(longshort_cases_c[cc][i],regtests(testing_0p1_c,cc)[i],2)*longshort_cases_c[cc][i] for i in range(len(longshort_cases_c[cc]))] for cc in longshort_cases_c}
    try:
        clusdata_all['cases_nonlin'] = {cc:cases_adj_nonlin[cc] for cc in cases_adj_nonlin}
    except:
        pass


print('making cases with nonlinear testing adjustment...')
make_cases_adj_nonlin()            
print('done.')
print('to change the nonlinear correction function, call make_cases_adj_nonlin(K), K=2 by default')

clusdata_all = {}
clusdata_all['deaths'] = deaths_raw
clusdata_all['cases'] = cases_raw
clusdata_all['cases_lin2020'] = cases_adj_lin2020
clusdata_all['cases_pwlfit'] = cases_adj_pwlfit
clusdata_all['cases_nonlin'] = cases_adj_nonlin


def plot_adj(country, data, adj = None, testing=None,  ndays=250, axis = None):
    ndays = 250
    if testing:
        Ntests = regtests(testing,country)
    if axis is None:   
        fig, ax1 = plt.subplots(figsize=(12,8))
    else:
        ax1 = axis
    ax1.plot(data[country][:ndays]) 
    if adj:  # already adjusted
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



#######################################################################
## ClusterFit class



#######################################################################
## ClusterFit class

class ClusterFit:
    """
    container class for fitting PCA, clustering
    """

    def __init__(self,
                 data,           # could be deaths/cases, raw/adjusted
                 Npca = 10,
                 outfile = ''):
        self.Npca = Npca
        self.data = data
        self.outfile = outfile
        self.dat = np.array([data[cc] for cc in data])
        # normalize the data
        for i in range(len(self.dat)):
            mx = max(self.dat[i])
            self.dat[i] = [dd/mx for dd in self.dat[i]]
        self.pca = PCA(Npca)
        self.pca.fit(self.dat)
        print('explained_variance_ratio:')
        print('explained_variance_ratio_' in dir(self.pca))
        print([x for x in dir(self.pca) if '__' not in x])
        #print(self.pca.explained_variance_ratio_)
        print('singular values:')
        #print(self.pca.singular_values_)

        self.fitted = self.pca.fit_transform(self.dat)
        self.smoothed = self.pca.inverse_transform(self.fitted)

    def plot_2components(self):
        plt.scatter(self.fitted[:,0],fitted[:,1]);

    def plot_all(self):
        max_cols=6
        max_rows=int(len(self.dat)/max_cols) + 1
        fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(20,3.5*max_rows))
        countries = [cc for cc in self.data]
        for idx, countrycode  in enumerate(countries):
            row = idx // max_cols
            col = idx % max_cols
            #axes[row, col].axis("off")
            axes[row, col].plot(self.dat[idx])
            axes[row, col].plot(self.smoothed[idx])
            axes[row, col].set_title(countrycode)
        for idx in range(len(lcountries),max_rows*max_cols):
            row = idx // max_cols
            col = idx % max_cols
            axes[row, col].axis("off")
        #plt.subplots_adjust(wspace=.05, hspace=.05)
        if self.outfile != '':
            plt.savefig(self.outfile)
        plt.show()

    def umap_cluster(self,random_state=0):
        self.um_fit = umap.UMAP(random_state=random_state,n_neighbors=5).fit(self.fitted)
        self.um_dat = [self.um_fit.embedding_[:,i] for i in range(2)]
        tdat = np.transpose(self.um_dat)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=4)
        self.labels = clusterer.fit_predict(tdat)
        print('hdbscan found',len(set(self.labels)),'clusters.')
        
        
    
    def plot_umap(self):
        plt.scatter(self.um_dat[0],self.um_dat[1],c=self.labels)
        
    def plot_pcas(self):
        max_cols = 5
        max_rows = self.Npca // max_cols
        if self.Npca%max_cols>0:
            max_rows = max_rows+1
        fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(20,max_rows*3.5))
        for i in range(10):
            foo = np.zeros(10)
            foo[i] = 1
            mypca = self.pca.inverse_transform(foo)
            row = i // max_cols
            col = i % max_cols
            #axes[row, col].axis("off")
            axes[row, col].plot(mypca)


