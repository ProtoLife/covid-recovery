# 
# 1. 1.9.2020 Managed to convert ODE models for economic extension to transition model ready for stochastic simulation, using separate birth death list
#             See section on SC2UIR model. Not done for other two economic extensions yet
# 2. 1.9.2020 Implemented stochastic simulation (Tau-leap method) using PyGom inbuilt capabilities: for SCIR simulation only so far
#             Neeed to use integer N>>1, not 1.0, for stochastic simulation. Calculates in a few minutes for N=10000, rescaled ICUfrac to 0.02 (x10). N=100000 didn't finish in 10m.

# # Model Definitions



# import required packages
import os 
import csv
from sympy import symbols, init_printing
import numpy as np
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sb
from matplotlib import pyplot as plt
import sympy
import itertools
import scipy
import datetime
import matplotlib.dates as mdates
from pygom import DeterministicOde, Transition, SimulateOde, TransitionType, SquareLoss
from scipy.optimize import minimize

import pickle as pk
import jsonpickle as jpk

from cycler import cycler
import datetime
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import pwlf
import sys
import copy
#from IPython.core.display import display, HTML
#display(HTML("<style>.container { width:100% !important; }</style>"))

import pprint
ppr = pprint.PrettyPrinter()

print('loading data.py...')
from data import *
print('done with data.py.')

savefigs = False # whether to save specific figures for paper to .../figures directory

def Float(x):
    try:
        rtn = float(x)
    except:
        rtn = float('NaN')
    return rtn

class ModelFit:
    """ We collect all information related to a fit between a pygom model and a set of data in this class
        It has access to the model structure and defines all required parameters and details of fit """

    def dumpparams(self,run_id=''): # Have to add self since this will become a method
        """stores params in a file './params/Model_Name.pk
        This stuff needs modules os, sys, pickle as pk.'"""
        mname = self.modelname
        country = self.dbparams['country']
        rname = self.run_id
        dirnm = os.getcwd()

        if run_id != '':            # if run_id, turn it into self.run_id and use it for output filename
            if run_id != rname:
                print("warning: changing run_id from ",rname,'to',run_id)
                self.run_id = run_id
        else:
            run_id = self.run_id # should always be something from __init__
        pfile = dirnm+'/params/'+run_id+'.pk'

        try:
            all_params = {'params':self.params, 
                          'sbparams':self.sbparams,
                          'fbparams':self.fbparams,
                          'cbparams':self.cbparams,
                          'dbparams':self.dbparams,
                          'initial_values':self.initial_values 
                          }
            with open(pfile,'wb') as fp:
                pk.dump(all_params,fp)
            print('dumped params to',pfile)
        except:
            print('problem dumping params to ',pfile)
    def loadparams(self,run_id=''): 
        """loads params from same file.  returns None if any problem finding the file.
        This stuff needs modules os, sys, pickle as pk."""
        if run_id == '':
            run_id = self.run_id
        else:
            print("warning: changing run_id from ",self.run_id,'to',run_id)
            self.run_id = run_id
            
        dirnm = os.getcwd()
        pfile = dirnm+'/params/'+run_id+'.pk'
        try:
            with open(pfile,'rb') as fp:
                all_params = pk.load(fp)
                print('loaded params from ',pfile,':')
        except:
            print("no file available with this run_id",pfile)
            return None

        print('-------  params from file:')
        ppr.pprint(all_params)
        # check to see that
        for pp in ['params','sbparams','fbparams','cbparams','dbparams']:
            try:
                ppp = eval('self.'+pp) # fail first time when ModelFit doesn't have params.
                selfkk = [kk for kk in ppp]
                newkk = [k for k in all_params[pp]]
                if newkk != selfkk:
                    print("params don't match when loading the params from ",pfile)
                    print('old keys:',selfkk)
                    print('new keys:',newkk)
                    return None
            except:
                pass            # ok to fail 1st time
        try:
            self.params = all_params['params']
            self.model.parameters = self.params
            self.sbparams = all_params['sbparams']
            self.fbparams = all_params['fbparams']
            self.cbparams = all_params['cbparams']
            self.dbparams = all_params['dbparams']
            self.initial_values = all_params['initial_values'] # will get copied properly?
        except:
            print('problem loading the params from ',pfile)
            return None
        return True



    def difference(self,datain):
        dataout = np.zeros(np.shape(datain))
        for i in range(1,len(datain)):
            dataout[i,...] = datain[i,...]-datain[i-1,...]
        return dataout
        
    def rolling_average(self,datain,period):
        (tmax,n) = np.shape(datain)
        dataout = np.zeros((tmax,n),dtype=float)
        moving_av = np.zeros(n,dtype=float)
        for k in range(len(datain)):
            if k-period >= 0:
                moving_av[:] = moving_av[:] - datain[k-7,...]
            moving_av[:] = moving_av[:] + datain[k,...]
            dataout[k] = moving_av/min(float(period),float(k+1))
        return dataout

    def plotdata(self,dtypes=['confirmed','deaths']):
        if type(dtypes)==str:
            dtypes = [dtypes]
        xx = np.array(range(len(self.tdata)-1))
        print(len(xx))
        print([(x,len(self.data[x])) for x in dtypes])

        for dt in dtypes:
            try:
                yy = self.data[dt]
            except:
                print("data type '"+dt+"' not found.")
            try:
                plt.plot(xx,yy)
            except:
                print("couldn't plot xx,yy",xx,yy)
        plt.show()

    def solveplot(self, species=['confirmed'],summing='daily',averaging='weekly',mag = {'deaths':10},axes=None,
                   scale='linear',plottitle= '',label='',newplot = True, gbrcolors=False, figsize = None):
        """
        solve ODEs and plot for fitmodel indicated
        
        species : alternatives 'all', 'EI', 'confirmed', 'deaths', ...
        tmax : max time for simulation
        summing: type of summing smoothing options : 'daily', ...
        averaging : None, 'daily', 'weekly'
        fitdata : data to fit
        axes : previous axes to plot on [None]
        scale : alternative 'linear' or 'log'
        plottitle : title for plot
        label : label for curve when called as part of multicurve plot
        newplot : whether to open new plot True/False
        gbrcolors : color types to use
        figsize : size of fig in inches (binary tuple)
        """
       
        # tmax = self.tsim[-1]
        # tvec=np.arange(0,tmax,1)

        if not isinstance(species,list):
            lspecies = [species]
        else:
            lspecies = species

        dspecies = [dt if dt != 'caution_fraction' else 'stringency' for dt in lspecies]
        mags = [mag[dt] if dt in mag.keys() else 1 for dt in dspecies]

        tvec = self.tsim
        tvec1 = tvec[1:]
        if not self.data is {}:
            fitdata = np.transpose(np.array([self.data[dt] for dt in dspecies]))
        else:
            fitdata = None
        if not fitdata is None:
            tmaxf = len(fitdata)
            if fitdata.ndim != 2:
                print("error in number of dimensions of array")
            else:
                print("fit data ",np.shape(fitdata))
            tvecf=np.arange(0,tmaxf,1)
            tvecf1 = tvecf[1:]
        
        if newplot:
            axes = None
            if (figsize == None):
                figsize=(8,6)
            plt.figure(figsize=figsize)
            # fig, axeslist = plt.subplots(1, nmodels, figsize=(nmodels*8,6))
               
        smodel = self.modelname
        model = self.model

        soln = scipy.integrate.odeint(model.ode, model.initial_values[0], tvec[1::])
        #Plot
        # ax = axeslist[nm]
        if axes == None: 
            ax = axes = plt.subplot(1,1,1)
        else:
            ax = axes
        if scale == 'log': #Plot on log scale
            ax.semilogy()
            ax.set_ylim([0.00000001,1.0])
            
        if summing == 'daily':
            ssoln = self.difference(soln)
            if not fitdata is None:
                sfit = self.difference(fitdata)
        else:
            ssoln = soln
            if not fitdata is None:
                sfit = fitdata
                
        if averaging == 'weekly':
            srsoln = self.rolling_average(ssoln,7)
            if not fitdata is None:
                srfit = self.rolling_average(sfit,7)
        else:
            srsoln = ssoln
            if not fitdata is None:
                srfit = sfit
                    
        for ns,species in enumerate(lspecies):
            if species == 'confirmed':
                suma = np.sum(srsoln[:,model.confirmed],axis=1)*mags[ns]
                if not fitdata is None:
                    ax.plot(tvec1,suma,label=label,color='green')
                    fita = srfit[1::,ns]*mags[ns]/self.fbparams['FracConfirmedDet']/self.population # confirmed cases data, corrected by FracConfirmedDet
                    ax.plot(tvecf1,fita,'o',label=label,color='green')
                else:
                    ax.plot(tvec1,suma,label=label)
            if species == 'recovered':
                suma = np.sum(srsoln[:,model.recovered],axis=1)*mags[ns]  
                if not fitdata is None:
                    ax.plot(tvec1,suma,label=label,color='blue')
                    fita = srfit[1::,ns]*mags[ns]/self.fbparams['FracRecoveredDet']/self.population # recovered cases data, corrected by FracRecoveredDet
                    ax.plot(tvecf1,fita,'o',label=label,color='blue')
                else:
                    ax.plot(tvec1,suma,label=label)
            elif species == 'deaths':
                suma = np.sum(srsoln[:,model.deaths],axis=1)*mags[ns]
                if not fitdata is None:
                    ax.plot(tvec1,suma,label=label,color='red')
                    fita = srfit[1::,ns]*mags[ns]/self.fbparams['FracDeathsDet']/self.population # deaths cases data, corrected by FracDeathsDet
                    ax.plot(tvecf1,fita,'o',label=label,color='red')
                else:
                    ax.plot(tvec1,suma,label=label)
            elif species == 'EI':
                ax.plot(tvec1,soln[:,model.ei],label=label)
                # ax.plot(tvec1,soln[:,model.ei],label="%s" % count)
                if 'I3' in model.modelname: 
                    plt.legend(("E","I1","I2","I3"))
                elif 'E' in model.modelname: 
                    plt.legend(("E","I"))
                else:
                    plt.legend(("I"))
            elif species == 'caution_fraction':
                #print('model name',model.modelname)
                susc = soln[:,model.S_c]
                suma = np.sum(soln[:,model.all_susceptibles],axis=1)
                old_settings = np.seterr(divide='ignore') #
                suma = np.divide(susc,suma)
                np.seterr(**old_settings)  # reset to default
                if not fitdata is None:
                    ax.plot(tvec1,suma,label=label,color='green')
                    fita = srfit[1::,ns]*mags[ns] # caution fraction from data (stringency) with correciton to unit scale via mags
                    ax.plot(tvecf1,fita,'o',label=label,color='green')
                else:
                    ax.plot(tvec1,suma,label=label)               
            elif species == 'all':
                ax.plot(tvec1,soln,label=label)
                if 'I3' in model.modelname:
                    if 'C3'in model.modelname:
                        pspecies=("S","E","I1","I2","I3","R","D","Ic","Sc","Ec")
                    elif 'C' in model.modelname:
                        pspecies=("S","E","I1","I2","I3","R","D","Sc")
                    else:
                        pspecies=("S","E","I1","I2","I3","R","D")
                elif 'E' in model.modelname:
                    if 'C3'in model.modelname:
                        pspecies=("S","E","I","R","D","Ic","Sc","Ec")
                    else:
                        pspecies=("S","E","I","R","D","Sc")                
                else:
                    if 'C2'in model.modelname:
                        pspecies=("S","I","R","D","Ic","Sc")
                    else:
                        pspecies=("S","I","R","D","Sc")
                plt.legend(pspecies)
                
        plt.xlabel("Time (days)")
        plt.ylabel("Fraction of population")
        plt.title(model.modelname +' '+plottitle)
        self.soln = soln
        self.dumpparams()       # dump every plot;  could be changed by sliders
        return

    def prparams(self):
        print('params:')
        ppr.pprint(self.params)
        print('sbparams:')
        ppr.pprint(self.sbparams)
        print('pfbarams:')
        ppr.pprint(self.fbparams)
        print('cbparams:')
        ppr.pprint(self.cbparams)
        print('dbparams:')
        ppr.pprint(self.dbparams)

    def __init__(self,modelname,model=None,country='Germany',run_id='',datatypes='all',data_src='owid',startdate=None,stopdate=None,simdays=None):
        global make_model,covid_ts,covid_owid_ts
        dirnm = os.getcwd()
        if run_id == '':                       # construct default run_id from mname and country
            if country != '':
                stmp = modelname+'_'+country
            else:
                stmp = modelname
            pfile = dirnm+'/params/'+stmp+'.pk'
            run_id = stmp
        self.run_id = run_id

        ######################################
        # set up model
        self.modelname = modelname
        if model:
            self.model = model
            if self.model.modelname != modelname:
                print("warning:  changing model from",modelname,'to',self.model.modelname)
        else:
            #model_d = make_model(modelname)                # I still prefer this I think, but 
            model_d = copy.deepcopy(fullmodels[modelname])  # should avoid modifying fullmodels at all from fits, otherwise never clear what parameters are
            self.model = model_d['model']
            if not self.loadparams(run_id):
                print('using default set of parameters for model type',modelname)
        self.params   = model_d['params']
        self.cbparams = model_d['cbparams']
        self.sbparams = model_d['sbparams']
        self.fbparams = model_d['fbparams']
        self.dbparams = model_d['dbparams']
        self.initial_values = model_d['initial_values']

        # set up data and times for simulation
        if data_src == 'jhu':
            ts = covid_ts
        elif data_src == 'owid':
            ts = covid_owid_ts
        else:
            print('data_src',data_src,'not yet hooked up: OWID data used instead')
            ts = covid_owid_ts
        self.country = country
        self.population = population_owid[country][0]

        fmt_jhu = '%m/%d/%y'
        dates_t = [datetime.datetime.strptime(dd,fmt_jhu) for dd in ts['confirmed']['dates'] ] # ts dates stored in string format of jhu fmt_jhu = '%m/%d/%y'
        firstdate_t =  dates_t[0]
        lastdate_t =  dates_t[-1]
        if startdate:
            startdate_t = datetime.datetime.strptime(startdate,fmt_jhu)
        else:
            startdate_t = firstdate_t
        if stopdate:
            stopdate_t = datetime.datetime.strptime(stopdate,fmt_jhu)
            print('stopdate',stopdate) 
        else:
            stopdate_t = lastdate_t
        if (startdate_t - firstdate_t).days < 0:
            print('start date out of data range, setting to data first date',ts['confirmed']['dates'][0])
            startdate_t = firstdate_t
            daystart = 0
        else:
            daystart = (startdate_t- firstdate_t).days
        if (stopdate_t - startdate_t).days > (lastdate_t - startdate_t).days:
            print('stop date out of data range, setting to data last date',ts['confirmed']['dates'][-1])
            stopdate_t = lastdate_t
        datadays = (stopdate_t-startdate_t).days + 1            
        if simdays: # simdays allowed greater than datadays to enable predictions
            if simdays < datadays:
                stopdate_t = startdate_t + datetime.timedelta(days=simdays-1)  # if simulation for shorter time than data, restrict data to this
                datadays = (stopdate_t-startdate_t).days + 1    
        else:
            simdays = datadays
        self.dates = [date.strftime(fmt_jhu) for date in dates_t if date>=startdate_t and date <= lastdate_t]
        self.tsim = np.linspace(0, simdays -1, simdays)
        self.tdata = np.linspace(0, datadays -1, datadays)

        if datatypes == 'all' or not datatypes:
            if data_src == 'owid':
                datatypes = ['confirmed','deaths','tests', 'stringency']
            else:
                datatypes = ['confirmed','deaths','recovered']
        self.data = {}
        for dt in datatypes:
            self.data.update({dt:ts[dt][country][daystart:datadays]}) 

        self.startdate = startdate_t.strftime(fmt_jhu)
        self.stopdate = stopdate_t.strftime(fmt_jhu)

def make_model(mod_name):
    """ make models of types ['SIR','SCIR','SC2IR','SEIR','SCEIR','SC3EIR','SEI3R','SCEI3R','SC3EI3R','SC2UIR','SC3UEIR','SC3UEI3R']"""
    rtn = {}
    I_0 =  0.00003

    if mod_name == 'SIR':
        state = ['S', 'I', 'R', 'D']
        param_list = ['beta', 'gamma','mu','N']
        transition = [
            Transition(origin='S', destination='I', equation='beta*I*S',
                       transition_type=TransitionType.T),
            Transition(origin='I', destination='R', equation='gamma*I',
                       transition_type=TransitionType.T),
            Transition(origin='I', destination='D', equation='mu*I',
                       transition_type=TransitionType.T)    
        ]

        model = DeterministicOde(state, param_list, transition=transition)
        model.modelname='SIR'
        model.ei=1
        model.confirmed=slice(1,4)  # cases 1-3 i.e. I, R and D
        model.recovered=slice(2,3)
        model.deaths=slice(3,4)
        model.I_1 = 1
        x0 = [1.0-I_0, I_0, 0.0, 0.0]
        model.initial_values = (x0, 0) # 0 for t[0]

        rtn['state'] = state
        rtn['param_list'] = param_list
        rtn['model'] = model
        return rtn

    if mod_name == 'SCIR':
        state = ['S', 'I', 'R', 'D', 'S_c']
        param_list = ['beta', 'gamma', 'mu', 'c_0', 'c_1', 'c_2', 'N']

        transition = [
            Transition(origin='S', destination='I', equation='beta*I*S',
                       transition_type=TransitionType.T),
            Transition(origin='S', destination='S_c', equation='c_2*I*S',
                       transition_type=TransitionType.T),
            Transition(origin='S_c', destination='S', equation='c_1*S_c',
                       transition_type=TransitionType.T),
            Transition(origin='S_c', destination='I', equation='c_0*beta*I*S_c',
                       transition_type=TransitionType.T),
            Transition(origin='I', destination='R', equation='gamma*I',
                       transition_type=TransitionType.T),
            Transition(origin='I', destination='D', equation='mu*I',
                       transition_type=TransitionType.T)    
            ]

        model = DeterministicOde(state, param_list, transition=transition)
        global SCIR_modelS
        SCIR_modelS = SimulateOde(state, param_list , transition=transition)
        model.modelname='SCIR'
        model.ei=1
        model.confirmed=slice(1,4)  # cases 1-3 i.e. I, R and D
        model.recovered=slice(2,3)
        model.deaths=slice(3,4)
        model.all_susceptibles=[0,4]
        model.S_c=4
        model.I_1 = 1
        x0_SCIR = [1.0-I_0, I_0, 0.0, 0.0, 0.0]
        model.initial_values = (x0_SCIR, 0)

        rtn['state'] = state
        rtn['param_list'] = param_list
        rtn['model'] = model
        return rtn

    if mod_name == 'SC2IR':
        state = ['S', 'I', 'R', 'D', 'I_c', 'S_c']
        param_list = ['beta', 'gamma', 'mu', 'c_0', 'c_1', 'c_2', 'N']

        transition = [
            Transition(origin='S', destination='I', equation='beta*(I+c_0*I_c)*S',
                       transition_type=TransitionType.T),
            Transition(origin='S', destination='S_c', equation='c_2*(I+I_c)*S',
                       transition_type=TransitionType.T),
            Transition(origin='S_c', destination='S', equation='c_1*S_c',
                       transition_type=TransitionType.T),
            Transition(origin='S_c', destination='I_c', equation='c_0*beta*(I+c_0*I_c)*S_c',
                       transition_type=TransitionType.T),
            Transition(origin='I', destination='R', equation='gamma*I',
                       transition_type=TransitionType.T),
            Transition(origin='I', destination='D', equation='mu*I',
                       transition_type=TransitionType.T),
            Transition(origin='I', destination='I_c', equation='c_2*(I+I_c)*I',
                       transition_type=TransitionType.T),
            Transition(origin='I_c', destination='R', equation='gamma*I_c',
                       transition_type=TransitionType.T),
            Transition(origin='I_c', destination='I', equation='c_1*I_c',
                       transition_type=TransitionType.T),
            Transition(origin='I_c', destination='D', equation='mu*I_c',
                       transition_type=TransitionType.T)  #, 
            ]

        model = DeterministicOde(state, param_list, transition=transition)
        model.modelname='SC2IR'

        model.ei=1
        model.confirmed=slice(1,5)  # cases 1-3 i.e. I, R and D
        model.recovered=slice(2,3)
        model.deaths=slice(3,4)
        model.all_susceptibles=[0,5]
        model.S_c=5
        model.I_1 = 1
        x0_SC2IR = [1.0-I_0, I_0, 0.0, 0.0, 0.0, 0.0]
        model.initial_values = (x0_SC2IR, 0)

        rtn['state'] = state
        rtn['param_list'] = param_list
        rtn['model'] = model
        return rtn
    
    if mod_name == 'SEIR':
        state = ['S', 'E', 'I', 'R', 'D']
        param_list = ['beta', 'alpha', 'gamma', 'mu', 'N']

        transition = [
            Transition(origin='S', destination='E', equation='beta*I*S',
                       transition_type=TransitionType.T),
            Transition(origin='E', destination='I', equation='alpha*E',
                       transition_type=TransitionType.T),
            Transition(origin='I', destination='R', equation='gamma*I',
                       transition_type=TransitionType.T),
            Transition(origin='I', destination='D', equation='mu*I',
                       transition_type=TransitionType.T)    
            ]

        model = DeterministicOde(state, param_list, transition=transition)
        model.modelname='SEIR'
        model.ei=slice(1,3) # cases 1,2 i.e. E and I
        model.confirmed=slice(2,5)  # cases 2-4 i.e. I, R and D, not E
        model.recovered=slice(3,4)
        model.deaths=slice(4,5)
        model.I_1 = 2
        x0_SEIR = [1.0-I_0, 0.0, I_0, 0.0, 0.0]
        model.initial_values = (x0_SEIR, 0)

        rtn['state'] = state
        rtn['param_list'] = param_list
        rtn['model'] = model
        return rtn

    if mod_name == 'SCEIR':
        state = ['S', 'E', 'I', 'R', 'D', 'S_c']
        param_list = ['beta', 'alpha', 'gamma', 'mu', 'c_0', 'c_1', 'c_2', 'N']

        transition = [
            Transition(origin='S', destination='E', equation='beta*I*S',
                       transition_type=TransitionType.T),
            Transition(origin='S', destination='S_c', equation='c_2*I*S',
                       transition_type=TransitionType.T),
            Transition(origin='S_c', destination='S', equation='c_1*S_c',
                       transition_type=TransitionType.T),
            Transition(origin='S_c', destination='E', equation='c_0*beta*I*S_c',
                       transition_type=TransitionType.T),
            Transition(origin='E', destination='I', equation='alpha*E',
                       transition_type=TransitionType.T),
            Transition(origin='I', destination='R', equation='gamma*I',
                       transition_type=TransitionType.T),
            Transition(origin='I', destination='D', equation='mu*I',
                       transition_type=TransitionType.T)    
            ]

        model = DeterministicOde(state, param_list, transition=transition)
        model.modelname='SCEIR'
        model.ei=slice(1,3) # cases 1,2 i.e. E,I
        model.confirmed=slice(2,5)  # cases 2-4 i.e. I, R and D, not E
        model.recovered=slice(3,4)
        model.deaths=slice(4,5)
        model.all_susceptibles=[0,5]
        model.S_c=5
        model.I_1 = 2
        x0_SCEIR = [1.0-I_0, 0.0, I_0, 0.0, 0.0, 0.0]
        model.initial_values = (x0_SCEIR, 0)
        rtn['state'] = state
        rtn['param_list'] = param_list
        rtn['model'] = model
        return rtn

    if mod_name == 'SC3EIR':
        state = ['S', 'E', 'I', 'R', 'D', 'I_c', 'S_c', 'E_c']
        param_list = ['beta', 'alpha', 'gamma', 'mu', 'c_0', 'c_1', 'c_2', 'N']

        transition = [
            Transition(origin='S', destination='E', equation='beta*(I+c_0*I_c)*S',
                       transition_type=TransitionType.T),
            Transition(origin='S', destination='S_c', equation='c_2*(I+I_c)*S',
                       transition_type=TransitionType.T),
            Transition(origin='S_c', destination='S', equation='c_1*S_c',
                       transition_type=TransitionType.T),
            Transition(origin='S_c', destination='E_c', equation='c_0*beta*(I+c_0*I_c)*S_c',
                       transition_type=TransitionType.T),
            Transition(origin='E', destination='I', equation='alpha*E',
                       transition_type=TransitionType.T),
            Transition(origin='E', destination='E_c', equation='c_2*(I+I_c)*E',
                       transition_type=TransitionType.T),
            Transition(origin='E_c', destination='I_c', equation='alpha*E_c',
                       transition_type=TransitionType.T),
            Transition(origin='E_c', destination='E', equation='c_1*E_c',
                       transition_type=TransitionType.T),
            Transition(origin='I', destination='R', equation='gamma*I',
                       transition_type=TransitionType.T),
            Transition(origin='I', destination='I_c', equation='c_2*(I+I_c)*I',
                       transition_type=TransitionType.T),
            Transition(origin='I', destination='D', equation='mu*I',
                       transition_type=TransitionType.T),
            Transition(origin='I_c', destination='R', equation='gamma*I_c',
                       transition_type=TransitionType.T),
            Transition(origin='I_c', destination='I', equation='c_1*I_c',
                       transition_type=TransitionType.T),
            Transition(origin='I_c', destination='D', equation='mu*I_c',
                       transition_type=TransitionType.T)
            ]

        model = DeterministicOde(state, param_list, transition=transition)
        model.modelname='SC3EIR'
        model.ei=slice(1,3) # cases 1,2 i.e. E,I  # note E_c and I_c not included
        model.confirmed=slice(2,6)  # cases 2-5 i.e. I, R, D, and I_c, not E, E_c
        model.recovered=slice(3,4)
        model.deaths=slice(4,5)
        model.all_susceptibles=[0,6]
        model.S_c=6
        model.I_1 = 2
        x0_SC3EIR = [1.0-I_0, 0.0, I_0, 0.0, 0.0, 0.0, 0.0, 0.0]
        model.initial_values = (x0_SC3EIR, 0)

        rtn['state'] = state
        rtn['param_list'] = param_list
        rtn['model'] = model
        return rtn

    if mod_name == 'SEI3R':
        state = ['S', 'E', 'I_1', 'I_2','I_3','R','D']
        param_list = ['beta_1', 'beta_2','beta_3','alpha', 'gamma_1', 'gamma_2', 'gamma_3',
                      'p_1','p_2','mu','N']

        transition = [
            Transition(origin='S', destination='E', equation='(beta_1*I_1+beta_2*I_2+beta_3*I_3)*S',
                       transition_type=TransitionType.T),
            Transition(origin='E', destination='I_1', equation='alpha*E',
                       transition_type=TransitionType.T),
            Transition(origin='I_1', destination='R', equation='gamma_1*I_1',
                       transition_type=TransitionType.T),
            Transition(origin='I_2', destination='R', equation='gamma_2*I_2',
                       transition_type=TransitionType.T),
            Transition(origin='I_3', destination='R', equation='gamma_3*I_3',
                       transition_type=TransitionType.T),
            Transition(origin='I_1', destination='I_2', equation='p_1*I_1',
                       transition_type=TransitionType.T),
            Transition(origin='I_2', destination='I_3', equation='p_2*I_2',
                       transition_type=TransitionType.T),
            Transition(origin='I_3', destination='D', equation='mu*I_3',
                       transition_type=TransitionType.T)    
            ]


        model = DeterministicOde(state, param_list, transition=transition)
        model.modelname='SEI3R'
        model.ei=slice(1,5)
        model.confirmed=slice(2,7)  # cases 2-6 i.e. I1, I2, I3, R and D
        model.recovered=slice(5,6)
        model.deaths=slice(6,7)
        model.I_1 = 2
        x0_SEI3R = [1.0-I_0, 0.0, I_0, 0.0, 0.0, 0.0, 0.0]
        model.initial_values = (x0_SEI3R, 0)

        rtn['state'] = state
        rtn['param_list'] = param_list
        rtn['model'] = model
        return rtn

    if mod_name == 'SCEI3R':
        state = ['S', 'E', 'I_1', 'I_2','I_3','R','D','S_c']
        param_list = ['beta_1', 'beta_2','beta_3','alpha', 'gamma_1', 'gamma_2', 'gamma_3',
                      'p_1','p_2','mu','c_0','c_1','c_2','N']

        transition = [
            Transition(origin='S', destination='E', equation='(beta_1*I_1+beta_2*I_2+beta_3*I_3)*S',
                       transition_type=TransitionType.T),
            Transition(origin='S', destination='S_c', equation='c_2*I_3*S',
                       transition_type=TransitionType.T),
            Transition(origin='S_c', destination='S', equation='c_1*S_c',
                       transition_type=TransitionType.T),
            Transition(origin='S_c', destination='E', equation='c_0*(beta_1*I_1+beta_2*I_2+beta_3*I_3)*S_c',
                       transition_type=TransitionType.T),
            Transition(origin='E', destination='I_1', equation='alpha*E',
                       transition_type=TransitionType.T),
            Transition(origin='I_1', destination='R', equation='gamma_1*I_1',
                       transition_type=TransitionType.T),
            Transition(origin='I_2', destination='R', equation='gamma_2*I_2',
                       transition_type=TransitionType.T),
            Transition(origin='I_3', destination='R', equation='gamma_3*I_3',
                       transition_type=TransitionType.T),
            Transition(origin='I_1', destination='I_2', equation='p_1*I_1',
                       transition_type=TransitionType.T),
            Transition(origin='I_2', destination='I_3', equation='p_2*I_2',
                       transition_type=TransitionType.T),
            Transition(origin='I_3', destination='D', equation='mu*I_3',
                       transition_type=TransitionType.T)    
            ]


        model = DeterministicOde(state, param_list, transition=transition)
        model.modelname='SCEI3R'
        model.ei=slice(1,5)
        model.confirmed=slice(2,7)  # cases 2-6 i.e. I1, I2, I3, R and D
        model.recovered=slice(5,6)
        model.deaths=slice(6,7)
        model.all_susceptibles=[0,7]
        model.S_c=7
        model.I_1 = 2
        x0_SCEI3R = [1.0-I_0, 0.0, I_0, 0.0, 0.0, 0.0, 0.0, 0.0]
        model.initial_values = (x0_SCEI3R, 0)

        rtn['state'] = state
        rtn['param_list'] = param_list
        rtn['model'] = model
        return rtn

    if mod_name == 'SC3EI3R':
        state = ['S', 'E', 'I_1', 'I_2','I_3', 'R', 'D', 'I_c', 'S_c', 'E_c']
        param_list = ['beta_1', 'beta_2','beta_3','alpha', 'gamma_1', 'gamma_2', 'gamma_3',
                      'p_1','p_2','mu','c_0','c_1','c_2','N']

        transition = [
            Transition(origin='S', destination='E', equation='(beta_1*I_1+beta_2*I_2+beta_3*I_3+c_0*beta_1*I_c)*S',
                       transition_type=TransitionType.T),
            Transition(origin='S_c', destination='E_c', equation='c_0*(beta_1*I_1+beta_2*I_2+beta_3*I_3+c_0*beta_1*I_c)*S_c',
                       transition_type=TransitionType.T),
            Transition(origin='S', destination='S_c', equation='c_2*I_3*S',
                       transition_type=TransitionType.T),
            Transition(origin='S_c', destination='S', equation='c_1*S_c',
                       transition_type=TransitionType.T),
            Transition(origin='E', destination='I_1', equation='alpha*E',
                       transition_type=TransitionType.T),
            Transition(origin='E', destination='E_c', equation='c_2*I_3*E',
                       transition_type=TransitionType.T),
            Transition(origin='E_c', destination='I_c', equation='alpha*E_c',
                       transition_type=TransitionType.T),
            Transition(origin='E_c', destination='E', equation='c_1*E_c',
                       transition_type=TransitionType.T),
            Transition(origin='I_1', destination='R', equation='gamma_1*I_1',
                       transition_type=TransitionType.T),
            Transition(origin='I_1', destination='I_c', equation='c_2*I_3*I_1',  # error corrected I_1, mistakenly was I_c 
                       transition_type=TransitionType.T),    
            Transition(origin='I_c', destination='R', equation='gamma_1*I_c',
                       transition_type=TransitionType.T),
            Transition(origin='I_c', destination='I_1', equation='c_1*I_c',
                       transition_type=TransitionType.T),    
            Transition(origin='I_2', destination='R', equation='gamma_2*I_2',
                       transition_type=TransitionType.T),
            Transition(origin='I_3', destination='R', equation='gamma_3*I_3',
                       transition_type=TransitionType.T),
            Transition(origin='I_1', destination='I_2', equation='p_1*I_1',
                       transition_type=TransitionType.T),
            Transition(origin='I_c', destination='I_2', equation='p_1*I_c',
                       transition_type=TransitionType.T),
            Transition(origin='I_2', destination='I_3', equation='p_2*I_2',
                       transition_type=TransitionType.T),
            Transition(origin='I_3', destination='D', equation='mu*I_3',
                       transition_type=TransitionType.T)
            ]


        model = DeterministicOde(state, param_list, transition=transition)
        model.modelname='SC3EI3R'
        model.ei=slice(1,5) # 1,2,3,4 i.e. E,I_1,I_2,I_3 – not E_c and I_c 
        model.confirmed=slice(2,8)  # cases 2-7 i.e. I1, I2, I3, R, D and I_c
        model.recovered=slice(5,6)
        model.deaths=slice(6,7)
        model.all_susceptibles=[0,8]
        model.S_c=8
        model.I_1 = 2
        x0_SC3EI3R = [1.0-I_0, 0.0, I_0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        model.initial_values = (x0_SC3EI3R, 0)

        rtn['state'] = state
        rtn['param_list'] = param_list
        rtn['model'] = model
        return rtn

    if mod_name == 'SC2UIR':
        state = ['S', 'I', 'R', 'D', 'I_c', 'S_c', 'S_u', 'W']
        param_list = ['beta', 'gamma', 'mu', 'c_0', 'c_1', 'c_2', 'k_u', 'k_1', 'k_w','kappa', 'N']

        transition = [
            Transition(origin='S', equation='-beta*(I+c_0*I_c)*S+c_1*S_c-c_2*(I+I_c)*S-k_u*(1-W)*S+k_1*S_u'),
            Transition(origin='S_c', equation='-c_0*beta*(I+c_0*I_c)*S_c-c_1*S_c+c_2*(I+I_c)*S-k_u*(1-W)*S_c'),
            Transition(origin='S_u', equation='-beta*(I+c_0*I_c)*S_u+k_u*(1-W)*(S+S_c)-k_1*S_u'),
            Transition(origin='I', equation='beta*(I+c_0*I_c)*S-gamma*I-mu*I+c_1*I_c-c_2*(I+I_c)*I'),
            Transition(origin='I_c', equation='c_0*beta*(I+c_0*I_c)*S_c-gamma*I_c-mu*I_c-c_1*I_c+c_2*(I+I_c)*I'),
            Transition(origin='R', equation='gamma*(I+I_c)'),
            Transition(origin='D', equation='mu*(I+I_c)'),
            Transition(origin='W', equation='k_w*W*(1-kappa*S_c-W)')
            ]

        model = DeterministicOde(state, param_list, ode=transition)
        model.modelname='SC2UIR'
        model.ei=1                  # case 1 i.e. I  # note I_c not included
        model.confirmed=slice(1,5)  # cases 1-4 i.e. I, R, D, and I_c
        model.recovered=slice(2,3)
        model.deaths=slice(3,4)
        model.all_susceptibles=[0,5,6]
        model.S_c=5
        model.I_1 = 1
        x0_SC2UIR = [1.0-I_0, I_0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        model.initial_values = (x0_SC2UIR, 0)

        rtn['state'] = state
        rtn['param_list'] = param_list
        rtn['model'] = model
        return rtn

    if mod_name == 'SC2UIR':
        state = ['S', 'I', 'R', 'D', 'I_c', 'S_c', 'S_u', 'W']
        param_list = ['beta', 'gamma', 'mu', 'c_0', 'c_1', 'c_2', 'k_u', 'k_1', 'k_w','kappa', 'N']

        transition = [
            Transition(origin='S', destination='I', equation='beta*(I+c_0*I_c)*S', transition_type=TransitionType.T),
            Transition(origin='S', destination='S_c', equation='c_2*(I+I_c)*S', transition_type=TransitionType.T),
            Transition(origin='S', destination='S_u', equation='k_u*(1-W)*S', transition_type=TransitionType.T),
            Transition(origin='S_c', destination='S', equation='c_1*S_c', transition_type=TransitionType.T),
            Transition(origin='S_c', destination='I_c', equation='c_0*beta*(I+c_0*I_c)*S_c', transition_type=TransitionType.T),
            Transition(origin='S_c', destination='S_u', equation='k_u*(1-W)*S_c', transition_type=TransitionType.T),
            Transition(origin='S_u', destination='S', equation='k_1*S_u', transition_type=TransitionType.T),   
            Transition(origin='S_u', destination='I', equation='beta*(I+c_0*I_c)*S_u', transition_type=TransitionType.T),    
            Transition(origin='I', destination='I_c', equation='c_2*(I+I_c)*I', transition_type=TransitionType.T),    
            Transition(origin='I', destination='R', equation='gamma*I', transition_type=TransitionType.T), 
            Transition(origin='I', destination='D', equation='mu*I', transition_type=TransitionType.T), 
            Transition(origin='I_c', destination='I', equation='c_1*I_c', transition_type=TransitionType.T),
            Transition(origin='I_c', destination='R', equation='gamma*I_c', transition_type=TransitionType.T), 
            Transition(origin='I_c', destination='D', equation='mu*I_c', transition_type=TransitionType.T),
            Transition(origin='W', destination='D', equation='0*W', transition_type=TransitionType.T)
            ]
        bdlist =     [Transition(origin='W',equation='k_w*W*(1-kappa*S_c-W)', transition_type=TransitionType.B)
            ]
        model = DeterministicOde(state, param_list, transition=transition)
        model.birth_death_list = bdlist
        model.modelname='SC2UIR'
        model.ei=1                  # case 1 i.e. I  # note I_c not included
        model.confirmed=slice(1,5)  # cases 1-4 i.e. I, R, D, and I_c
        model.recovered=slice(2,3)
        model.deaths=slice(3,4)
        model.all_susceptibles=[0,5,6]
        model.S_c=5
        model.I_1 = 1
        x0_SC3UEIR = [1.0-I_0, 0.0, I_0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        model.initial_values = (x0_SC3UEIR, 0)

        rtn['state'] = state
        rtn['param_list'] = param_list
        rtn['model'] = model
        return rtn

    if mod_name == 'SC3UEIR':
        state = ['S', 'E', 'I', 'R', 'D', 'I_c', 'S_c', 'E_c', 'S_u', 'W']
        param_list = ['beta', 'alpha', 'gamma', 'mu', 'c_0', 'c_1', 'c_2', 'k_u', 'k_1', 'k_w','kappa', 'N']

        transition = [
            Transition(origin='S', equation='-beta*(I+c_0*I_c)*S+c_1*S_c-c_2*(I+I_c)*S-k_u*(1-W)*S+k_1*S_u'),
            Transition(origin='S_c', equation='-c_0*beta*(I+c_0*I_c)*S_c-c_1*S_c+c_2*(I+I_c)*S-k_u*(1-W)*S_c'),
            Transition(origin='S_u', equation='-beta*(I+c_0*I_c)*S_u+k_u*(1-W)*(S+S_c)-k_1*S_u'),
            Transition(origin='E', equation='beta*(I+c_0*I_c)*(S+S_u)-alpha*E+c_1*E_c-c_2*(I+I_c)*E'),
            Transition(origin='E_c', equation='c_0*beta*(I+c_0*I_c)*S_c-alpha*E_c-c_1*E_c+c_2*(I+I_c)*E'),
            Transition(origin='I', equation='alpha*E-gamma*I-mu*I+c_1*I_c-c_2*(I+I_c)*I'),
            Transition(origin='I_c', equation='alpha*E_c-gamma*I_c-mu*I_c-c_1*I_c+c_2*(I+I_c)*I'),
            Transition(origin='R', equation='gamma*(I+I_c)'),
            Transition(origin='D', equation='mu*(I+I_c)'),
            Transition(origin='W', equation='k_w*W*(1-kappa*S_c-W)')
            ]

        model = DeterministicOde(state, param_list, ode=transition)
        model.modelname='SC3UEIR'
        model.ei=slice(1,3) # cases 1,2 i.e. E,I  # note E_c and I_c not included
        model.confirmed=slice(2,6)  # cases 2-5 i.e. I, R, D, and I_c, not E, E_c
        model.recovered=slice(3,4)
        model.deaths=slice(4,5)
        model.all_susceptibles=[0,6,8]
        model.S_c=6
        model.I_1 = 2
        x0_SC3UEIR = [1.0-I_0, 0.0, I_0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        model.initial_values = (x0_SC3UEIR, 0)

        rtn['state'] = state
        rtn['param_list'] = param_list
        rtn['model'] = model
        return rtn

    if mod_name == 'SC3UEI3R':
        state = ['S', 'E', 'I_1', 'I_2', 'I_3', 'R', 'D',  'I_c', 'S_c', 'E_c', 'S_u', 'W'] # order important to allow correct plot groupings
        param_list = ['beta_1', 'beta_2', 'beta_3', 'p_1', 'p_2', 'alpha', 
                      'gamma_1', 'gamma_2', 'gamma_3','mu', 'c_0', 'c_1', 'c_2', 'k_u', 'k_1', 'k_w', 'kappa', 'N'] # order also important

        transition = [
            Transition(origin='S', equation='-(beta_1*(I_1+c_0*I_c)+beta_2*I_2+beta_3*I_3)*S+c_1*S_c-c_2*(I_3)*S-k_u*(1-W)*S+k_1*S_u'),
            Transition(origin='S_c', equation='-c_0*(beta_1*(I_1+c_0*I_c)+beta_2*I_2+beta_3*I_3)*S_c-c_1*S_c+c_2*(I_3)*S-k_u*(1-W)*S_c'),
            Transition(origin='S_u', equation='-(beta_1*(I_1+c_0*I_c)+beta_2*I_2+beta_3*I_3)*S_u+k_u*(1-W)*(S+S_c)-k_1*S_u'),
            Transition(origin='W', equation='k_w*W*(1-kappa*S_c-W)'),
            Transition(origin='E', equation='beta_1*(I_1+c_0*I_c)*(S+S_u)-alpha*E-c_2*(I_3)*E+c_1*E_c'),
            Transition(origin='E_c', equation='c_0*beta_1*(I_1+c_0*I_c)*S_c-alpha*E_c+c_2*(I_3)*E-c_1*E_c'),
            Transition(origin='I_1', equation='alpha*E-gamma_1*I_1-p_1*I_1-c_2*(I_3)*I_1+c_1*I_c'),
            Transition(origin='I_c', equation='alpha*E_c-gamma_1*I_c-p_1*I_c+c_2*(I_3)*I_1-c_1*I_c'), # changed to I_c, prints better
            Transition(origin='I_2', equation='p_1*(I_1+I_c)-gamma_2*I_2-p_2*I_2'),
            Transition(origin='I_3', equation='p_2*I_2-gamma_3*I_3-mu*I_3'),     # error corrected, this is equation for I_3 not I_2
            Transition(origin='R', equation='gamma_1*(I_1+I_c)+gamma_2*I_2+gamma_3*I_3'),
            Transition(origin='D', equation='mu*I_3')
            ]

        model = DeterministicOde(state, param_list, ode=transition)
        model.modelname='SC3UEI3R'  # following needs to be adjusted for new models, NB add new species at end to preserve slice subsets
        model.ei=slice(1,5)         # 1,2,3,4 i.e. E,I_1,I_2,I_3 – not E_c and I_c 
        model.confirmed=slice(2,8)  # cases 2-7 i.e. I1, I2, I3, R, D and I_c
        model.recovered=slice(5,6)  # case 5 R
        model.deaths=slice(6,7)     # case 6 D
        model.all_susceptibles=[0,8,10]
        model.S_c=8
        model.I_1 = 2
        x0_SC3UEI3R = [1.0-I_0, 0.0, I_0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        model.initial_values = (x0_SC3UEI3R, 0)

        rtn['state'] = state
        rtn['param_list'] = param_list
        rtn['model'] = model
        return rtn

def param_copy(model):
    params = model.parameters
    newparams = {}
    pkeys1 = list(model.params.keys())
    pkeys2 = list(model.parameters.keys())
    for i in range(len(pkeys1)):
        newparams[pkeys1[i]] = params[pkeys2[i]]
    print(newparams)
    model.parameters=newparams
    
def param_modify(model,param,value):
    params = model.parameters
    newparams = {}
    pkeys1 = list(model.params.keys())
    pkeys2 = list(model.parameters.keys())
    for i in range(len(pkeys1)):
        newparams[pkeys1[i]] = params[pkeys2[i]]
    newparams[param]=value
    print(newparams)
    model.parameters=newparams
    
# param_modify(SCIR_model,'beta',0.721)  # requires .params to be set (see below)

def vector2params_old(b,a,g,p,u,c,k,N,modelname):
    """this earlier version of arameter translation routine is kept here for reference
    allows the construction of model specific parameters for different models from a single set
    based on SEI3R model with vector b,g,p as well as vector caution c and economics k
    later modified for better correspondence between SEIR and SEI3R and derivates """
    if 'I3' in modelname:  # models with hospitalization
        params = {
            'beta_1' : b[1],
            'beta_2' : b[2],
            'beta_3' : b[3],
            'alpha' : a,
            'gamma_1': g[1],
            'gamma_2': g[2],
            'gamma_3': g[3],
            'p_1'    : p[1],
            'p_2'    : p[2],
            'mu'    : u}
    elif 'E' in modelname:
        params = {
            'beta' : b[1],  # see above for explanations
            'alpha' : a, 
            'gamma': g[1]+g[2]*(p[1]/(g[2]+p[2]))+g[3]*(p[1]/(g[2]+p[2]))*(p[2]/(g[3]+u)),
            'mu'    : u*(p[1]/(g[2]+p[2])*(p[2]/(g[3]+u)))}    
    else:
        params = {
            'beta' : b[1],  # see above for explanations
            'gamma': g[1]+g[2]*(p[1]/(g[2]+p[2]))+g[3]*(p[1]/(g[2]+p[2]))*(p[2]/(g[3]+u)),
            'mu'    : u*(p[1]/(g[2]+p[2])*(p[2]/(g[3]+u)))}
            
    if 'C' in modelname: # models with caution  
        params['c_0'] = c[0]
        params['c_1'] = c[1]
        if 'I3' in modelname: # models with hospitalization
            params['c_2'] = c[2]
        else:
            params['c_2'] = c[2]*FracCritical
        
    if 'U' in modelname: # models with economic correction to caution  
        params['k_u'] = k[0]
        params['k_1'] = k[1]
        params['k_w'] = k[2]
        params['kappa'] = k[3]
        
    params['N'] = N
    return params

def vector2params(b,a,g,p,u,c,k,N,FracCritical,modelname):
    """allows the construction of model specific parameters for different models from a single set
    based on SEI3R model with vector b,g,p as well as vector caution c and economics k"""
    if 'I3' in modelname:  # models with hospitalization
        params = {
            'beta_1' : b[1],
            'beta_2' : b[2],
            'beta_3' : b[3],
            'alpha' : a,
            'gamma_1': g[1],
            'gamma_2': g[2],
            'gamma_3': g[3],
            'p_1'    : p[1],
            'p_2'    : p[2],
            'mu'    : u}
    elif 'E' in modelname:
        irat = 1 + p[1]/(g[2]+p[2]) + p[2]/(g[3]+u)
        #irat = 1
        params = {
            'beta' : b[1],  # see above for explanations
            'alpha' : a, 
            'gamma': (g[1]+g[2]*(p[1]/(g[2]+p[2]))+g[3]*(p[1]/(g[2]+p[2]))*(p[2]/(g[3]+u)))/irat,
            'mu'    : u*(p[1]/(g[2]+p[2])*(p[2]/(g[3]+u))/irat)}
    else:
        irat = 1 + p[1]/(g[2]+p[2]) + p[2]/(g[3]+u)
        #irat = 1
        params = {
            #'beta' : np.sqrt(b[1]*a),  # see above for explanations
            'beta' : b[1],  # see above for explanations
            'gamma': (g[1]+g[2]*(p[1]/(g[2]+p[2]))+g[3]*(p[1]/(g[2]+p[2]))*(p[2]/(g[3]+u)))/irat,
            'mu'    : u*(p[1]/(g[2]+p[2])*(p[2]/(g[3]+u))/irat)}
            
    if 'C' in modelname: # models with caution  
        params['c_0'] = c[0]
        params['c_1'] = c[1]
        if 'I3' in modelname: # models with hospitalization
            params['c_2'] = c[2]
        else:
            params['c_2'] = c[2]*FracCritical
        
    if 'U' in modelname: # models with economic correction to caution  
        params['k_u'] = k[0]
        params['k_1'] = k[1]
        params['k_w'] = k[2]
        params['kappa'] = k[3]
        
    params['N'] = N
    return params

def params2vector(params,modelname='SC3UEI3R'):  # requires I3 in modelname
    b = [None,None,None,None]
    g = [None,None,None,None]
    p = [None,None,None]
    c = [None,None,None]
    k = [None,None,None,None]
    b[0]=0.0
    b[1]=params['beta_1']
    b[2]=params['beta_2']
    b[3]=params['beta_3']
    g[0]=0.0
    g[1]=params['gamma_1']
    g[2]=params['gamma_2']
    g[3]=params['gamma_3']
    p[0]=0.0
    p[1]=params['p_1']
    p[2]=params['p_2']
    a=params['alpha']
    u=params['mu']
    N=params['N']
    if 'C' in modelname: # models with caution 
        c[0]=params['c_1']
        c[1]=params['c_2']
        c[2]=params['c_3']
    if 'U' in modelname: # models with economic correction to caution  
        k[0] = params['k_u']
        k[1] = params['k_1']
        k[2] = params['k_w']
        k[3] = params['kappa']
    return (b,a,g,p,u,c,k,N)

def base2vectors(sbparams,cbparams,fbparams):
    """ converts dictionary of bae parameters to vector of parameters and then to pygom simulation parameters"""
    Exposure =sbparams['Exposure']
    IncubPeriod = sbparams['IncubPeriod']
    DurMildInf = sbparams['DurMildInf']
    FracMild = sbparams['FracMild']
    FracSevere = sbparams['FracSevere']
    FracCritical = sbparams['FracCritical']
    CFR = sbparams['CFR']
    TimeICUDeath = sbparams['TimeICUDeath']
    DurHosp = sbparams['DurHosp']
    ICUFrac = sbparams['ICUFrac']
    I0 = sbparams['I0']

    CautionFactor = cbparams['CautionFactor']
    CautionRetention = cbparams['CautionRetention']
    CautionICUFrac = cbparams['CautionICUFrac']
    EconomicRetention =  cbparams['EconomicRetention']
    EconomicCostOfCaution = cbparams['EconomicCostOfCaution']
    
    FracConfirmedDet = fbparams['FracConfirmedDet']
    FracRecoveredDet = fbparams['FracRecoveredDet']
    FracDeathsDet = fbparams['FracDeathsDet']
    
    N=1
    b=np.zeros(4)     # beta
    g=np.zeros(4)     # gamma
    p=np.zeros(3)     # progression
    c=np.zeros(3)     # caution
    k=np.zeros(4)     # economic caution

    a=1/IncubPeriod                       # transition rate from exposed to infected
    b=Exposure*np.array([0,1,0,0])/N      # hospitalized cases don't transmit
    u=(1/TimeICUDeath)*(CFR/FracCritical) # death rate from ICU
    g[3]=(1/TimeICUDeath)-u               # recovery rate

    p[2]=(1/DurHosp)*(FracCritical/(FracCritical+FracSevere))
    g[2]=(1/DurHosp)-p[2]

    g[1]=(1/DurMildInf)*FracMild
    p[1]=(1/DurMildInf)-g[1]

    c[0]=CautionFactor
    c[1]=1/CautionRetention
    c[2]=1/(N*ICUFrac*CautionICUFrac)     # this is the rate coefficient giving 1/day at I3 = denominator

    k[0]=1/EconomicRetention              # assumes default rate is same as 1
    k[1]=1/EconomicRetention              # this is always correct
    k[2]=1/EconomicRetention              # assumes default rate is same as 1
    k[3]=EconomicCostOfCaution
    
    return(b,a,g,p,u,c,k,N,FracCritical,I0)

def base2params(sbparams,cbparams,fbparams,smodel):
    b,a,g,p,u,c,k,N,FracCritical,I0 = base2vectors(sbparams,cbparams,fbparams)
    return(vector2params(b,a,g,p,u,c,k,N,FracCritical,smodel))

def vectors2base(b,a,g,p,u,c,k,N,I0,ICUFrac):
    """ converts vector of parameters back to dictionaries of base parameters
        assumes only one parameter for bvector in the form b*[0,1,0,0]"""
    Exposure = b[1] # assuming b has structure b[1]*[0,1,0,0]
    IncubPeriod = a

    FracMild = g[1]/(g[1]+p[1])
    FracSevere = (p[1]/(g[1]+p[1]))*(g[2]/(g[2]+p[2]))
    # FracCritical = (g[1]/(g[1]+p[1]))*(p[2]/(g[2]+p[2]))
    FracCritical = 1 - FracMild -FracSevere         # not independent
    CFR = (u/(g[3]+u)*(p[2]/(g[2]+p[2]))*(p[1]/(g[1]+p[1]))  
    IncubPeriod =1/(g[1]+p[1])
    DurHosp = 1/(g[2]+p[2])
    TimeICUDeath = 1/(g(3)+u)

    CautionFactor = c[0]
    CautionRetention = 1/c[1]
    CautionICUFrac = 1/(N*c[2]*ICUFrac)
    
    EconomicStriction =  1/k[0]
    EconomicRetention =  1/k[1]
    EconomyRelaxation = 1/k[2]
    EconomicCostOfCaution =  k[3]
    
    sbparams = {'Exposure':Exposure,'IncubPeriod':IncubPeriod,'DurMildInf':DurMildInf,
                'FracMild':FracMild,'FracSevere':FracSevere,'FracCritical':FracCritical,
                'CFR':CFR,'TimeICUDeath':TimeICUDeath,'DurHosp':DurHosp,'ICUFrac':ICUFrac,'I0':I0}
    cbparams = {'CautionFactor':CautionFactor,'CautionRetention':CautionRetention,'CautionICUFrac':CautionICUFrac,
                'EconomicRetention':EconomicRetention,'EconomicCostOfCaution':EconomicCostOfCaution}
  
    
    return(sbparams,cbparams)

def base2ICs(I0,N,smodel,model):
    (x0old,t0) = model.initial_values
    nstates = len(x0old)
    x0 = [0.]*nstates
    x0[0] = N*(1-I0)
    if model.I_1 < nstates:
        x0[model.I_1] = N*I0

    else:
        print('error, initial infectives location out of bounds',model.I_1,'not <',nstates)
    return (x0,t0)

def default_params(sbparams=None,cbparams=None,fbparams=None,dbparams=None):
    """ supply default parameters for those not already defined as arguments 
        does not check if non None arguments are correctly defined """
    if not sbparams:      # standard params
        Exposure=0.25     # Rate coefficient for exposure per individual in contact per day
        IncubPeriod=5     #Incubation period, days 
        DurMildInf=10     #Duration of mild infections, days
        FracMild=0.8      #Fraction of infections that are mild
        FracSevere=0.15   #Fraction of infections that are severe
        FracCritical=0.05 #Fraction of infections that are critical
        CFR=0.02          #Case fatality rate (fraction of infections resulting in death)
        TimeICUDeath=7    #Time from ICU admission to death, days
        DurHosp=11        #Duration of hospitalization, days
        ICUFrac= 0.001    # Fraction of ICUs relative to population size N
        I0 = 0.00003      # Fraction of population initially infected
    if not sbparams:      # standard params set 2 from Germany fit
        Exposure=0.4     # Rate coefficient for exposure per individual in contact per day
        IncubPeriod=5     #Incubation period, days 
        DurMildInf=10     #Duration of mild infections, days
        FracMild=0.7      #Fraction of infections that are mild
        FracSevere=0.20   #Fraction of infections that are severe
        FracCritical=0.1  #Fraction of infections that are critical
        CFR=0.05          #Case fatality rate (fraction of infections resulting in death)
        TimeICUDeath=5    #Time from ICU admission to death, days
        DurHosp=4         #Duration of hospitalization, days
        ICUFrac= 0.001    # Fraction of ICUs relative to population size N
        I0 = 0.0000003    # Fraction of population initially infected

        sbparams = {'Exposure':Exposure,'IncubPeriod':IncubPeriod,'DurMildInf':DurMildInf,
                   'FracMild':FracMild,'FracSevere':FracSevere,'FracCritical':FracCritical,
                   'CFR':CFR,'TimeICUDeath':TimeICUDeath,'DurHosp':DurHosp,'ICUFrac':ICUFrac,'I0':I0}
    if not cbparams:          # Model extension by John McCaskill to include caution 
        CautionFactor= 0.3    # Fractional reduction of exposure rate for cautioned individuals
        CautionRetention= 14. # Duration of cautionary state of susceptibles (4 weeks)
        CautionICUFrac= 0.25  # Fraction of ICUs occupied leading to 90% of susceptibles in caution 
        EconomicRetention = CautionRetention # Duration of economic dominant state of susceptibles (here same as caution, typically longer)
        EconomicCostOfCaution = 0.5 # Cost to economy of individual exercising caution
    if not cbparams:          # Model extension by John McCaskill to include caution  # set 2 
        CautionFactor= 0.1    # Fractional reduction of exposure rate for cautioned individuals
        CautionRetention= 1/0.015 # Duration of cautionary state of susceptibles (4 weeks)
        CautionICUFrac= 0.1   # Fraction of ICUs occupied leading to 90% of susceptibles in caution 
        EconomicRetention = CautionRetention # Duration of economic dominant state of susceptibles (here same as caution, typically longer)
        EconomicCostOfCaution = 0.5 # Cost to economy of individual exercising caution

    cbparams = {'CautionFactor':CautionFactor,'CautionRetention':CautionRetention,'CautionICUFrac':CautionICUFrac,
                'EconomicRetention':EconomicRetention,'EconomicCostOfCaution':EconomicCostOfCaution}

    if not fbparams:          # Model fitting extension to allow for incomplete detection
        FracConfirmedDet=1.0  # Fraction of recovered individuals measured : plots made with this parameter
        FracRecoveredDet=FracConfirmedDet # Fraction of recovered individuals measured
        FracDeathsDet=1.0
    if not fbparams:          # Model fitting extension to allow for incomplete detection
        FracConfirmedDet=1.0 # Fraction of recovered individuals measured : plots made with this parameter
        FracRecoveredDet=FracConfirmedDet # Fraction of recovered individuals measured
        FracDeathsDet=1.0

        fbparams = {'FracConfirmedDet':FracConfirmedDet,'FracRecoveredDet':FracRecoveredDet,'FracDeathsDet':FracDeathsDet}

    if not dbparams:     # extra data-related params for defining a run, including possible fitting with sliders:
        dbparams = {'country':'Germany','data_src':'owid'}

    return [sbparams,cbparams,fbparams,dbparams]

# Set up multimodel consistent sets of parameters, based on standard set defined by Dr. Alison Hill for SEI3RD 
def parametrize_model(smodel,sbparams=None,cbparams=None,fbparams=None,dbparams=None):
    [sbparams,cbparams,fbparams,dbparams] = default_params(sbparams,cbparams,fbparams,dbparams)
    dbparams['run_name'] = smodel # default value when no country yet
    b,a,g,p,u,c,k,N,FracCritical,I0 = base2vectors(sbparams,cbparams,fbparams)
    fullmodel = make_model(smodel)
    model = fullmodel['model']
    params_in=vector2params(b,a,g,p,u,c,k,N,FracCritical,smodel)
    model.initial_values = base2ICs(I0,N,smodel,model)
    model.parameters = params_in # sets symbolic name parameters
    fullmodel['params'] = params_in    # sets string params
    fullmodel['sbparams'] = sbparams
    fullmodel['cbparams'] = cbparams
    fullmodel['fbparams'] = fbparams
    fullmodel['dbparams'] = dbparams
    fullmodel['initial_values'] = model.initial_values
    return fullmodel


smodels = ['SIR','SCIR','SC2IR','SEIR','SCEIR','SC3EIR','SEI3R','SCEI3R','SC3EI3R','SC2UIR','SC3UEIR','SC3UEI3R'] # full set
# smodels = ['SEI3R','SC3EI3R','SC3UEI3R'] # short list for debugging
 
# Initialize all models

cmodels = {}
fullmodels = {}
print('making the models...')
for smodel in smodels:
    fullmodel = parametrize_model(smodel)
    fullmodels[smodel] = fullmodel
    # take fullmodel['model'] so that modelnm is same model as before
    # for backward compatibility
    cmodels[smodel] = fullmodel['model']
    modelnm = smodel+'_model'
    exec(modelnm+" = fullmodel['model']")
    print(smodel)
            
            
    
    # fullmodels[smodel] = make_model(smodel)
    # cmodels[smodel] = fullmodels[smodel]['model']
    # params_in=vector2params(b,a,g,p,u,c,k,N,FracCritical,smodel)
    # cmodels[smodel].initial_values = base2ICs(I0,N,smodel,cmodels)
    # fullmodels[smodel]['model'].parameters = params_in # sets symbolic name parameters
    # fullmodels[smodel]['model'].params = params_in    # sets string params
    # cmodels[smodel].parameters = params_in
    # cmodels[smodel].sbparams = sbparams
    # cmodels[smodel].cbparams = cbparams
    # cmodels[smodel].fbparams = fbparams
    # dbparams['run_name'] = smodel # default value when no country yet
    # cmodels[smodel].dbparams = dbparams
    # modelnm = smodel+'_model'
    # exec(modelnm+" = cmodels[smodel]")
    # print(smodel)

print('done with the models.')
    
