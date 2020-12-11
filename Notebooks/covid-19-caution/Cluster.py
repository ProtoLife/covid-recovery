# import required packages
import os 
import csv

import numpy as np
import warnings
import math

import matplotlib.dates as mdates
from matplotlib import pyplot as plt
import datetime
import itertools

import scipy
from scipy.optimize import minimize
from sympy import symbols, init_printing
import sympy
from pygom import DeterministicOde, Transition, SimulateOde, TransitionType, SquareLoss
from scipy.signal import find_peaks as find_peaks

from time import time
import pickle as pk
import jsonpickle as jpk

from cycler import cycler
import pwlf                 # piecewise linear fit 

# for clustering and PCA
#
import umap 
import umap.plot

from sklearn.decomposition import PCA
from sklearn import cluster
from sklearn.cluster import AgglomerativeClustering
import hdbscan

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

class BaseData:
    def __init__(self,base_data='data_all_base'):
        if base_data:                           # read in base data from file
            start=time()
            print('reading in data from',base_data,'...')
            try:
                with open('./pks/'+base_data+'.pk','rb') as fp:
                    foo = pk.load(fp)
                print('elapsed: ',time()-start)

                # make each element of the dictionary a variable named with key:
                for x in foo:
                    stmp = "self."+x+"= foo['"+x+"']"
                    exec(stmp)
                self.data_loaded = True
                self.base_data = base_data
            except:
                print('Error: database pk file not found','./pks/'+base_data+'.pk')
                return                
        else:
            print('Error: base_data not defined')
            return

class ClusterData:

    def regtests(self,testing,country,trampday1=50):
        """ regularize testing data by ramping up linearly from common trampday1 
            to value on first reported testing capacity
        """
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
    def CaCo (self, Co, Nt, K=2):  # cases_actual / cases_observed given Nt=testing
        K1 = 25*(K-1)/(5.0-K)
        K2 = K1/5
        if Co > 0:
            rt = 1000*Nt/Co
            return (K1+rt)/(K2+rt)
        else:
            return 1

    def make_cases_adj_nonlin(self,testing,cases,K=2):
        cases_adj_nonlin={}
        testing_0p1_c = testing_0p1 = {cc: [0.1 if math.isnan(t) else t for t in testing[cc]] for cc in testing if cc != 'dates'}
        cases_adj_nonlin = {cc:np.array([self.CaCo(cases[cc][i],self.regtests(testing_0p1_c,cc)[i],2)*cases[cc][i] for i in range(len(cases[cc]))]) for cc in cases if cc != 'dates'}
        return cases_adj_nonlin

    def __init__(self,based,clusdtype='std',cluster_data=False,report_correct=True,database='JHU',daysync=23,thresh=10,
                 mindays=150, mindeaths=200,mindeathspm=0.1,syncat='first major peak',K=2):

        self.clusdtype=clusdtype
        # self.base_data=bd
        self.cluster_data=cluster_data
        self.report_correct=report_correct
        self.database=database
        self.daysync=daysync
        self.thresh=thresh
        self.mindays=mindays
        self.mindeaths=mindeaths
        self.mindeathspm=mindeathspm
        self.syncat=syncat
        self.K=K


        if cluster_data:  # read in cluster data from file
            start=time()
            print('reading in data from','./pks/data_cluster_'+self.clusdtype+'.pk','...')
            with open('./pks/data_cluster_'+self.clusdtype+'.pk','rb') as fp:
                foo = pk.load(fp)
            print('elapsed: ',time()-start)

            # make each element of the dictionary a variable named with key:
            for x in foo:
                stmp = "self."+x+"= foo['"+x+"']"
                exec(stmp)
            self.cluster_data_loaded = True
        else:        # use class parameter cluster_data=None to regenerate cluster data from base data
            print('Constructing common synchronized deaths, case and testing data...');
            print('database',self.database,'report_correct',self.report_correct)
            print('mindeaths',self.mindeaths,'mindeathspm',self.mindeathspm)
            print('database',self.database,'report correction',self.report_correct)
            print('daysync',self.daysync,'thresh for deaths',self.thresh,'mindays',self.mindays)

            """
            Basic data series:
            total_deaths
            new_deaths_spm
            new_cases_spm

            where spm = 'smoothed per million'

            The code below chooses between
            -- OWID or JHU
            -- outlier corrected (report_correct==True) or not

            """

            if self.database == 'OWID':
                if report_correct:
                    self.total_deaths = based.total_deaths_cs_owid
                    self.new_deaths_spm = based.new_deaths_c_spm_owid
                    self.new_cases_spm = based.new_cases_c_spm_owid
                else:
                    self.total_deaths = based.total_deaths_s_owid
                    self.new_deaths_spm = based.new_deaths_spm_owid
                    self.new_cases_spm = based.new_cases_spm_owid
            elif self.database == 'JHU':
                if report_correct:
                    self.total_deaths = based.total_deaths_cs_jhu     
                    self.new_deaths_spm = based.new_deaths_c_spm_jhu
                    self.new_cases_spm = based.new_cases_c_spm_jhu
                else:
                    self.total_deaths = based.total_deaths_s_jhu
                    self.new_deaths_spm = based.new_deaths_spm_jhu
                    self.new_cases_spm = based.new_cases_spm_jhu

            """
            Clustering data:
            -- align initial boundary for death threshold
            -- filter for at least 150 days
            basic data to start with are big and big_cases, with bcountries, set in data.py, filtering common_countries (common to owid & jhu)
            """

            # mindeaths = 100
            # mindeathspm = 0.5
            self.countries_common = based.countries_common
            self.bcountries_1 = [cc for cc in self.countries_common if (max(based.total_deaths_cs_jhu[cc])>=self.mindeaths and max(based.total_deaths_cs_owid[cc])>=self.mindeaths)]
            self.bcountries = [cc for cc in self.bcountries_1 if (max(based.new_deaths_c_spm_jhu[cc])>=self.mindeathspm and max(based.new_deaths_c_spm_owid[cc])>=self.mindeathspm)]
            print('No of big common countries is',len(self.bcountries))
            print('---------------------------------')
            # from data.py:
            # bcountries_1 = [cc for cc in countries_common if (max(total_deaths_cs_jhu[cc])>=mindeaths and max(total_deaths_cs_owid[cc])>=mindeaths)]
            # bcountries = [cc for cc in bcountries_1 if (max(new_deaths_c_spm_jhu[cc])>=mindeathspm and max(new_deaths_c_spm_owid[cc])>=mindeathspm)]

            self.big = {cc:self.new_deaths_spm[cc] for cc in self.bcountries}
            self.big_cases = {cc:self.new_cases_spm[cc] for cc in self.bcountries}
            print('number of countries in total_deaths)',len(self.total_deaths))
            print('number of countries in big',len(self.big))


            # badspikes = ['Peru','Bolivia','Chile','China','Equador','Kyrgystan']   # eliminate Peru and a few other countries because of bad spikes.

            # synchronization method : by threshold on total deaths
            print('synchronizing and trimming time series to common length...')

            self.first_peak = {}
            self.first_thresh = {}
            self.tdates = len(self.total_deaths['Germany'])  # changed to a particular common country to get database indept (formerly using 'dates' entry in total_Deaths_x)
            self.daily_deaths = np.zeros(self.tdates,np.float)
            if self.syncat == 'first major peak':
                minfirstpeak = self.tdates  
                for cc in self.bcountries:
                    self.daily_deaths[0] = self.total_deaths[cc][0]
                    for i in range(1,self.tdates):
                        self.daily_deaths[i] = self.total_deaths[cc][i]-self.total_deaths[cc][i-1]
                    dmax = np.max(self.daily_deaths)
                    peaks = find_peaks(self.daily_deaths,distance=10,height=0.05*dmax)[0] # peaks only recorded if 5% or more of max 
                    self.first_peak.update({cc:self.tdates}) # no peaks value, set to beyond end of array, needs to be picked up on use
                    if len(peaks) >= 1:
                        self.first_peak.update({cc:peaks[0]})
                    if self.first_peak[cc] < minfirstpeak:
                        minfirstpeak = self.first_peak[cc]
                self.minfirstpeak = minfirstpeak
                print('minfirstpeak',minfirstpeak,'max possible length',self.tdates-minfirstpeak)
            else:
                minfirstthresh = self.tdates  
                for cc in self.bcountries:
                    for i in range(self.tdates):
                        if self.total_deaths[cc][i] >= self.thresh:
                            self.first_thresh.update({cc:i})
                            if self.first_thresh[cc] < minfirstthresh:
                                minfirstthresh = self.first_thresh[cc]
                            break;
                self.minfirstthresh = minfirstthresh
                print('minfirstthresh',minfirstthresh,'max possible length',self.tdates-minfirstthresh)
            self.short_deaths = {}
            self.short_cases = {}
            self.short_testing = {}
            self.short_reg_testing = {}

            for cc in self.bcountries:
                if self.syncat == 'first major peak':
                    i = self.first_peak[cc]-self.minfirstpeak
                else:
                    i = self.first_thresh[cc]
                self.short_deaths[cc] = [self.big[cc][j] for j in range(i,len(self.big[cc]))]
                self.short_cases[cc] = [self.big_cases[cc][j] for j in range(i,len(self.big_cases[cc]))]
                self.short_testing[cc] = [based.testing[cc][j] for j in range(i,len(based.testing[cc]))]
                self.short_reg_testing[cc] = [based.reg_testing[cc][j] for j in range(i,len(based.reg_testing[cc]))]

            self.short_deaths_est =  min([len(self.short_deaths[x]) for x in self.short_deaths])
            self.short_deaths_c = {cc:self.short_deaths[cc][:self.short_deaths_est] for cc in self.short_deaths} # this crops all country time series to the shortest one, currently not used
            self.short_cases_est =  min([len(self.short_cases[x]) for x in self.short_cases])
            self.short_cases_c = {cc:self.short_cases[cc][:self.short_cases_est] for cc in self.short_cases}
            self.short_testing_est = min([len(self.short_testing[x]) for x in self.short_testing])
            self.short_reg_testing_est = min([len(self.short_reg_testing[x]) for x in self.short_reg_testing])
            self.short_testing_c = {cc:self.short_testing[cc][:self.short_testing_est] for cc in self.short_testing}
            self.short_reg_testing_c = {cc:self.short_reg_testing[cc][:self.short_reg_testing_est] for cc in self.short_reg_testing} 

            # choose subset of time series data that must have at least len mindays=150
            # mindays = 150 # changed from 160 to include more countries on Sep 24
            self.longshort = {cc:self.short_deaths[cc] for cc in self.short_deaths if (len(self.short_deaths[cc])>=self.mindays)};
            self.longshortest =  min([len(self.longshort[x]) for x in self.longshort])
            self.longshort_c = {cc:self.longshort[cc][:self.longshortest] for cc in self.longshort}
            self.lcountries = [cc for cc in self.longshort_c]

            # scaled_cases = {cc:new_cases_spm[cc]/max(new_cases_spm[cc]) for cc in lcountries} # note that lcountries determined by death data

            # select only traces with minimum length of mindays
            self.longshort_cases = {cc:self.short_cases[cc] for cc in self.short_cases if (len(self.short_cases[cc])>=self.mindays)};
            self.longshort_cases_est =  min([len(self.longshort_cases[x]) for x in self.longshort_cases])
            self.clusdata_len = self.longshort_cases_est
            self.longshort_cases_c = {cc:self.longshort_cases[cc][:self.longshort_cases_est] for cc in self.longshort_cases}
            self.lccountries = list(self.longshort_cases.keys())


            self.longshort_testing_c = {cc:self.short_testing[cc][:self.clusdata_len] for cc in self.short_testing}
            self.longshort_reg_testing_c = {cc:self.short_reg_testing[cc][:self.clusdata_len] for cc in self.short_reg_testing}
            self.big_testing_c = self.longshort_reg_testing_c


            self.deaths_raw = self.longshort_c
            self.cases_raw = self.longshort_cases_c
                         
            dat = np.array([self.longshort_cases_c[cc] for cc in self.longshort_cases_c])
            testingtmp = np.linspace(0.1,1.0,len(dat[0]))
            dat = [dd/testingtmp for dd in dat]
            self.cases_adj_lin2020 = {self.lcountries[i]:dat[i] for i in range(len(dat))}

            self.reg_testing_lc = {cc:np.array(self.longshort_reg_testing_c[cc]) for cc in self.lcountries}
            dat = np.array([self.longshort_cases_c[cc]/self.reg_testing_lc[cc] for cc in self.lcountries])
            # dat = np.array([longshort_cases_c[cc]/testing_lc[cc][first_thresh[cc]:first_thresh[cc]+len(longshort_cases_c[cc])] for cc in lcountries])
            self.cases_adj_pwlfit = {self.lcountries[i]:dat[i] for i in range(len(dat))}


            print('making cases with nonlinear testing adjustment...')
            # note that we could instead construct these from cases_adj_nonlin(r)_(jhu,owid) by making daily and then synchronizing as above
            # this would be faster but more complicated 
            self.cases_adj_nonlin = self.make_cases_adj_nonlin(self.longshort_testing_c,self.longshort_cases_c,self.K)
            self.cases_adj_nonlinr = self.make_cases_adj_nonlin(self.longshort_reg_testing_c,self.longshort_cases_c,self.K)              
            print('done.')

            self.clusdata_all = {}
            self.clusdata_all['deaths'] = self.deaths_raw
            self.clusdata_all['cases'] = self.cases_raw
            self.clusdata_all['cases_lin2020'] = self.cases_adj_lin2020
            self.clusdata_all['cases_pwlfit'] = self.cases_adj_pwlfit
            self.clusdata_all['cases_nonlin'] = self.cases_adj_nonlin
            self.clusdata_all['cases_nonlinr'] = self.cases_adj_nonlinr
            self.datasets = [c for c in self.clusdata_all]
            self.cluster_data_loaded = True
        print('----------------------------------------')
        print('Finished loading Cluster module')
        print('----------------------------------------')

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
    
def get_cluster_data(clusdtype):
    #ClData=ClusterData(bd,clusdtype='JRP1',cluster_data=True)  # ideally this should be made to work
    print('reading in data from','./pks/data_cluster_'+clusdtype+'.pk','...')
    with open('./pks/data_cluster_'+clusdtype+'.pk','rb') as fp:
        foo = pk.load(fp)
    return foo['ClData']


###########################################################
# will want to move this !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# to get ModelFit class definition:
# exec(open('ClusterFit.py','r').read())
###########################################################

    
