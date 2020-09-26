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

print('Getting deaths, case and testing data...');

total_deaths_x = get_data_owid_key('total_deaths')
total_deaths = {cc:total_deaths_x[cc] for cc in total_deaths_x if cc != 'dates' and cc != 'World'}
new_deaths_spm_x = get_data_owid_key('new_deaths_smoothed_per_million')
new_deaths_spm = {cc:new_deaths_spm_x[cc] for cc in new_deaths_spm_x if cc != 'dates' and cc != 'World'}

total_cases_x = get_data_owid_key('total_cases')
total_cases = {cc:total_cases_x[cc] for cc in total_cases_x if cc != 'dates' and cc != 'World'}
total_cases_ppm_x = get_data_owid_key('total_cases_per_million')
total_cases_ppm = {cc:total_cases_ppm_x[cc] for cc in total_cases_ppm_x if cc != 'dates' and cc != 'World'}
new_cases_spm_x = get_data_owid_key('new_cases_smoothed_per_million')
new_cases_spm = {cc:new_cases_spm_x[cc] for cc in new_cases_spm_x if cc != 'dates' and cc != 'World'}

testing_x = get_data_owid_key('new_tests_smoothed_per_thousand')
testing = {cc:testing_x[cc] for cc in testing_x if cc != 'dates' and cc != 'World'}

print('done.')

td_mx = [max(total_deaths[cc]) for cc in total_deaths]
mindeaths = 100 
countries = [cc for cc in total_deaths if max(total_deaths[cc])>=mindeaths]

# get rid of countries with trivially small new_deaths_spm:
# eliminate Peru and a few other countries because of bad spikes.
badspikes = ['Peru','Bolivia','Chile','China','Equador','Kyrgystan']                                             
mid = {cc:new_deaths_spm[cc] for cc in countries if max(new_deaths_spm[cc])>0.5 and max(new_deaths_spm[cc])<= 1} # note that several important countries between 0.5 and 1 per million
#print(len(mid))
#print(mid.keys())
big = {cc:new_deaths_spm[cc] for cc in countries if max(new_deaths_spm[cc])>0.5 and cc not in badspikes} # To Do:  replace badspike data with JHU.
bcountries = big.keys()
big_cases = {cc:new_cases_spm[cc] for cc in bcountries} #  bcountries filters  max(new_deaths_spm[cc])>=0.5 and cc not in badspikes}
#print(len(big))
#print(big.keys())

scaled = {cc:new_deaths_spm[cc]/max(new_deaths_spm[cc]) for cc in countries}

# reg_testing calculated from testing below : using piecewise linear approximation
# note first_thresh defined below needed to use testing in connection with synced data such as big
print('doing piecwise linear fits to testing data ...');
warnings.simplefilter('ignore')
reg_testing={}
for i,cc in enumerate(bcountries):
    # testing_cap = np.array([max(t,0.1) for t in testing[cc]])
    testing_cap = testing[cc][50:] # we assume international common starting day 50 of begin of preparation of testing (linear ramp to first recorded data) 
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
        reg_testing.update({cc:yHat.copy()})    
print('done.')

# synchronization method : by threshold on total deaths
short_deaths = {}
short_cases = {}
short_testing = {}
short_reg_testing = {}
first_thresh = {}
thresh = 10   # better to use day when #total_deaths (ie cumulative) absolute first reaches 10 or perhaps 30 absolute as sync point & keep entire rest of trace
for cc in bcountries:
    tdates = len(total_deaths_x['dates'])
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

def make_cases_adj_nonlin(testing,K=2):
    global cases_adj_nonlin
    cases_adj_nonlin={}
    testing_0p1_c = testing_0p1 = {cc: [0.1 if math.isnan(t) else t for t in testing[cc]] for cc in testing}
    # testing_0p1_c = {cc:testing_0p1[cc][-clusdata_len:] for cc in testing_0p1}
    cases_adj_nonlin = {cc:[CaCo(longshort_cases_c[cc][i],regtests(testing_0p1_c,cc)[i],2)*longshort_cases_c[cc][i] for i in range(len(longshort_cases_c[cc]))] for cc in longshort_cases_c}
    try:
        clusdata_all['cases_nonlin'] = {cc:cases_adj_nonlin[cc] for cc in cases_adj_nonlin}
    except:
        pass
    return cases_adj_nonlin


print('making cases with nonlinear testing adjustment...')
cases_adj_nonlin = make_cases_adj_nonlin(longshort_testing_c)
cases_adj_nonlinr = make_cases_adj_nonlin(longshort_reg_testing_c)              
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


#######################################################################
# # ClusterFit class

class ClusterFit:
    """
    container class for fitting PCA, clustering
    """
    def __init__(self,
                 data,           # could be deaths/cases, raw/adjusted
                 Npca = 10,
                 fft = None,    # optionally True to do PCA on Fourier transformed data
                 outfile = ''):
        self.Npca = Npca
        self.data = data
        self.outfile = outfile
        self.dat = np.array([data[cc] for cc in data])

        self.pca = PCA(Npca)
        if fft == 'fft' or fft == 'powfft':
            self.fftdat = np.fft.rfft(self.dat) # last axis by default
            self.nfft = len(self.fftdat[0])
            if fft == 'powfft':
                self.fftpow = np.square(np.abs(self.fftdat))
                for i in range(len(self.fftpow)): # normalize data ignoring DC component
                    mx = max(self.fftpow[i])
                    self.fftpow[i] = [dd/mx for dd in self.fftpow[i]]
                self.lfftpow = np.log(self.fftpow)
                # self.pca.fit(self.fftpow)
                self.fitted = self.pca.fit_transform(self.lfftpow)
                self.smoothed = self.pca.inverse_transform(self.fitted)
                self.fft = 'powfft'
            else: # 'fft'
                # consider scaling data from all countries to same max freq amplitude per country of fft 
                self.rfft =  np.concatenate((np.real(self.fftdat),np.imag(self.fftdat)),axis = 1) # concatenate along 2nd axis
                # self.pca.fit(self.rfft)
                maxvals = np.zeros(len(self.dat))
                dcvals = np.zeros(len(self.dat))
                for i in range(len(self.rfft)): # normalize data ignoring DC component, scaling data from all countries to same max freq amplitude per country
                    dcvals[i] = self.rfft[i,0] # ignore DC component
                    self.rfft[i,0] = 0.
                    mx = maxvals[i] = max(self.rfft[i])
                    # mx = maxvals[i] = 1.0
                    self.rfft[i] = [dd/mx for dd in self.rfft[i]]
                self.fitted = self.pca.fit_transform(self.rfft)
                self.rsmoothed = self.pca.inverse_transform(self.fitted)
                self.fftsmoothed = np.transpose(np.array([self.rsmoothed[:,k] + self.rsmoothed[:,self.nfft+k]*1j for k in range(self.nfft)], dtype=np.cdouble))
                for i in range(len(data)):
                    self.fftsmoothed[i,:] =  self.fftsmoothed[i,:]*maxvals[i]
                self.fftsmoothed[:,0] = dcvals
                self.smoothed = np.fft.irfft(self.fftsmoothed,len(self.dat[0]))
                self.fft = 'fft'
        else:
            for i in range(len(self.dat)):   # normalize data
                mx = max(self.dat[i])
                self.dat[i] = [dd/mx for dd in self.dat[i]]
            # self.pca.fit(self.dat)
            self.fitted = self.pca.fit_transform(self.dat)
            self.smoothed = self.pca.inverse_transform(self.fitted)
            self.nfft = 0
            self.fft = None

        #print('explained_variance_ratio:')
        #print('explained_variance_ratio_' in dir(self.pca))
        #print([x for x in dir(self.pca) if '__' not in x])
        #print(self.pca.explained_variance_ratio_)
        #print('singular values:')
        #print(self.pca.singular_values_)

    def plot_2components(self):
        plt.scatter(self.fitted[:,0],fitted[:,1]);

    def cluster_plot_all(self):
        max_cols=6
        max_rows=int(len(self.dat)/max_cols) + 1
        fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(20,3.5*max_rows))
        if self.fft == 'powfft' or self.fft == 'fft':
            axes2 = np.array([[ax.twinx() for ax in axesrow] for axesrow in axes])         
        countries = [cc for cc in self.data]

        if len(self.clus_labels) == len(countries): 
            print('sorting countries according to cluster labels') 
            self.clus_argsort = np.lexsort((countries,self.clus_labels))
            scountries = [countries[self.clus_argsort[i]] for i in range(len(countries))]
        else:
            scountries = countries

        for id, countrycode  in enumerate(countries):
            row = id // max_cols
            col = id % max_cols
            if len(self.clus_labels) == len(countries):
                idx = self.clus_argsort[id]
            else:
                idx = id
            axes[row, col].plot(self.dat[idx])
            if self.fft == 'powfft':
                axes2[row, col].plot(self.smoothed[idx],color='red')
                # axes2[row, col].set_yscale('log') # not required, data is already logarithmic
            elif self.fft == 'fit':
                axes2[row, col].plot(self.smoothed[idx],color='orange')
            else:
                axes[row, col].plot(self.smoothed[idx])
            axes[row, col].set_title(countries[idx])
        for idx in range(len(countries),max_rows*max_cols):
            row = idx // max_cols
            col = idx % max_cols
            axes[row, col].axis("off")
            if self.fft == 'powfft':
                axes2[row, col].axis("off")
        #plt.subplots_adjust(wspace=.05, hspace=.05)
        if self.outfile != '':
            plt.savefig(self.outfile)
        plt.show()

    def hdbscan(self,min_size=4):
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size)
        tdat = self.dat
        self.clus_labels = self.clusterer.fit_predict(tdat)
        validity = hdbscan.validity.validity_index(tdat, self.clus_labels)
        print('cluster validity index =',validity)
        print('cluster validity of each cluster:')
        validity = hdbscan.validity.validity_index(tdat, self.clus_labels,per_cluster_scores=True)
        for i,v in enumerate(validity):
            print('cluster',self.clus_labels[i],'validity =',validity[i])
            

    def plot_fpca(self):
        dat_disc = skfda.representation.grid.FDataGrid(dat,list(range(len(dat[0]))))
        fpca_disc = FPCA(n_components=10)
        fpca_disc.fit(dat_disc)
        fpca_disc.components_.plot()        

    def hdbscan_fpca(self,min_size=4,min_samples=3,n_components=5,diag=True):
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size,min_samples=min_samples)
        dat_disc = skfda.representation.grid.FDataGrid(dat,list(range(len(dat[0]))))
        fpca_disc = FPCA(n_components=n_components)
        fpca_disc.fit(dat_disc)
        self.fpca_transform = fpca_disc.transform(dat_disc)
        self.clus_labels = self.clusterer.fit_predict(self.fpca_transform)
        if diag:
            try:
                validity = hdbscan.validity.validity_index(self.fpca_transform, self.clus_labels)
                labels = self.clus_labels
                print('hdbscan_min_clus=',min_size,':  ',n_components ,'FPCAcomponents:  ',
                      len(set([x for x in labels if x>-1])),'clusters;  ',
                      sum([1 for x in labels if x>-1]),'clustered;  ',sum([1 for x in labels if x==-1]),'unclustered; ','validity =',np.round(validity,3))
            except:
                validity=None
                labels = self.clus_labels
                print('hdbscan_min_clus=',min_size,':  ',n_components ,'FPCAcomponents:  ',
                  len(set([x for x in labels if x>-1])),'clusters;  ',
                  sum([1 for x in labels if x>-1]),'clustered;  ',sum([1 for x in labels if x==-1]),'unclustered; ','validity =',validity)        

    def hdbscan_pca(self,min_size=4):
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size)
        tdat = self.fitted
        print('shape of cluster data = ',tdat.shape)
        self.clus_labels = self.clusterer.fit_predict(tdat)
        validity = hdbscan.validity.validity_index(tdat, self.clus_labels)
        print('cluster validity index =',validity)
        print('cluster validity of each cluster:')
        validity = hdbscan.validity.validity_index(tdat, self.clus_labels,per_cluster_scores=True)
        for i,v in enumerate(validity):
            print('cluster',self.clus_labels[i],'validity =',validity[i])

    def umap(self,random_state=0,n_neighbors=10):
        self.um_fit = umap.UMAP(random_state=random_state,n_neighbors=n_neighbors).fit(self.fitted)
        self.um_dat = [self.um_fit.embedding_[:,i] for i in range(2)]

    def umap_cluster(self,random_state=0,min_size=4,diag=True,n_neighbors=10):
        self.um_fit = umap.UMAP(random_state=random_state,n_neighbors=n_neighbors).fit(self.fitted)
        self.um_dat = [self.um_fit.embedding_[:,i] for i in range(2)]
        tdat = np.transpose(self.um_dat)

        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size)
        self.clus_labels = self.clusterer.fit_predict(tdat)
        self.clus_probs = self.clusterer.probabilities_
        if diag:
            print('hdbscan found',len(set(self.clus_labels)),'clusters.')
        
    def umap_best_cluster(self,Nclus=3,Ntries=50,minsize=4,ranstate=0,n_neighbors=10):
        clusall = []
        clus = {}
        clus['probs'] = []
        clus['idx'] = []
        for i in range(ranstate,ranstate+Ntries):
            self.umap_cluster(random_state=i,min_size=minsize,diag=False,n_neighbors=n_neighbors)
            if len(set(self.clus_labels)) == Nclus:
                clus['probs'].append(np.mean(self.clus_probs))
                clus['idx'].append(i)
        print('found',len(clus['probs']),'clusterings with size',Nclus,'clusters')
        if len(clus['probs'])>1:
            idx = np.argsort(clus['probs'])[-1:][0]
        elif len(clus['probs']) == 1:
            idx = 0
        else:
            print("Failed to find a cluster with",Nclus,"components")
            return
        self.umap_cluster(random_state=clus['idx'][idx],min_size=minsize,diag=False,n_neighbors=n_neighbors)

    
    def plot_umap(self):
        labs = [x for x in self.clus_labels]
        for i in range(len(labs)):
            if labs[i]<0:
                labs[i] = None
        plt.scatter(self.um_dat[0],self.um_dat[1],c=labs)
        xx = [self.um_dat[0][i] for i in range(len(labs)) if labs[i]==None]
        yy = [self.um_dat[0][i] for i in range(len(labs)) if labs[i]==None]
        #print(xx)
        #print(yy)
        plt.scatter(xx,yy,color='red')   
        
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
            if self.fft == 'fft':
                fftmypca = np.array([mypca[k] + mypca[self.nfft+k]*1j for k in range(self.nfft)], dtype=np.cdouble) 
                mypca = np.fft.irfft(fftmypca)
            row = i // max_cols
            col = i % max_cols
            #axes[row, col].axis("off")
            axes[row, col].plot(mypca)
           
