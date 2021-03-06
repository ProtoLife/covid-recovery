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

# print('loading data.py...')
# from data import *
# print('done with data.py.')
print('Assuming data already loaded, so no "from data import *" required.')

savefigs = False # whether to save specific figures for paper to .../figures directory

def Float(x):
    try:
        rtn = float(x)
    except:
        rtn = float('NaN')
    return rtn

###########################################################
# to get ModelFit class definition:
exec(open('ModelFit.py','r').read())
###########################################################

C_2s = 1000.   # scaling factor for c_2, to allow fit parameter c_2 to be of commensurate magnitude to other parameters

def make_model(mod_name):
    """ make models of types ['SIR','SCIR','SC2IR','SEIR','SCEIR','SC3EIR','SEI3R','SCEI3R','SC3EI3R','SC2UIR','SC3UEIR','SC3UEI3R']"""
    global C_2s           # scaling factor for c_2
    rtn = {}
    I_0 =  0.00003
    c_2s = '%f*' % C_2s   # string equation substitute scaled constant

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
            Transition(origin='S', destination='S_c', equation=c_2s+'c_2*I*S',
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
            Transition(origin='S', destination='S_c', equation=c_2s+'c_2*(I+I_c)*S',
                       transition_type=TransitionType.T),
            Transition(origin='S_c', destination='S', equation='c_1*S_c',
                       transition_type=TransitionType.T),
            Transition(origin='S_c', destination='I_c', equation='c_0*beta*(I+c_0*I_c)*S_c',
                       transition_type=TransitionType.T),
            Transition(origin='I', destination='R', equation='gamma*I',
                       transition_type=TransitionType.T),
            Transition(origin='I', destination='D', equation='mu*I',
                       transition_type=TransitionType.T),
            Transition(origin='I', destination='I_c', equation=c_2s+'c_2*(I+I_c)*I',
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
            Transition(origin='S', destination='S_c', equation=c_2s+'c_2*I*S',
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
            Transition(origin='S', destination='S_c', equation=c_2s+'c_2*(I+I_c)*S',
                       transition_type=TransitionType.T),
            Transition(origin='S_c', destination='S', equation='c_1*S_c',
                       transition_type=TransitionType.T),
            Transition(origin='S_c', destination='E_c', equation='c_0*beta*(I+c_0*I_c)*S_c',
                       transition_type=TransitionType.T),
            Transition(origin='E', destination='I', equation='alpha*E',
                       transition_type=TransitionType.T),
            Transition(origin='E', destination='E_c', equation=c_2s+'c_2*(I+I_c)*E',
                       transition_type=TransitionType.T),
            Transition(origin='E_c', destination='I_c', equation='alpha*E_c',
                       transition_type=TransitionType.T),
            Transition(origin='E_c', destination='E', equation='c_1*E_c',
                       transition_type=TransitionType.T),
            Transition(origin='I', destination='R', equation='gamma*I',
                       transition_type=TransitionType.T),
            Transition(origin='I', destination='I_c', equation=c_2s+'c_2*(I+I_c)*I',
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
            Transition(origin='S', destination='S_c', equation=c_2s+'c_2*I_3*S',
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
            Transition(origin='S', destination='S_c', equation=c_2s+'c_2*I_3*S',
                       transition_type=TransitionType.T),
            Transition(origin='S_c', destination='S', equation='c_1*S_c',
                       transition_type=TransitionType.T),
            Transition(origin='E', destination='I_1', equation='alpha*E',
                       transition_type=TransitionType.T),
            Transition(origin='E', destination='E_c', equation=c_2s+'c_2*I_3*E',
                       transition_type=TransitionType.T),
            Transition(origin='E_c', destination='I_c', equation='alpha*E_c',
                       transition_type=TransitionType.T),
            Transition(origin='E_c', destination='E', equation='c_1*E_c',
                       transition_type=TransitionType.T),
            Transition(origin='I_1', destination='R', equation='gamma_1*I_1',
                       transition_type=TransitionType.T),
            Transition(origin='I_1', destination='I_c', equation=c_2s+'c_2*I_3*I_1',  # error corrected I_1, mistakenly was I_c 
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
            Transition(origin='S', equation='-beta*(I+c_0*I_c)*S+c_1*S_c-%f*c_2*(I+I_c)*S-k_u*(1-W)*S+k_1*S_u' % C_2s),
            Transition(origin='S_c', equation='-c_0*beta*(I+c_0*I_c)*S_c-c_1*S_c+%f*c_2*(I+I_c)*S-k_u*(1-W)*S_c' % C_2s),
            Transition(origin='S_u', equation='-beta*(I+c_0*I_c)*S_u+k_u*(1-W)*(S+S_c)-k_1*S_u'),
            Transition(origin='I', equation='beta*(I+c_0*I_c)*S-gamma*I-mu*I+c_1*I_c-%f*c_2*(I+I_c)*I' % C_2s),
            Transition(origin='I_c', equation='c_0*beta*(I+c_0*I_c)*S_c-gamma*I_c-mu*I_c-c_1*I_c+%f*c_2*(I+I_c)*I' % C_2s),
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
            Transition(origin='S', destination='S_c', equation=c_2s+'c_2*(I+I_c)*S', transition_type=TransitionType.T),
            Transition(origin='S', destination='S_u', equation='k_u*(1-W)*S', transition_type=TransitionType.T),
            Transition(origin='S_c', destination='S', equation='c_1*S_c', transition_type=TransitionType.T),
            Transition(origin='S_c', destination='I_c', equation='c_0*beta*(I+c_0*I_c)*S_c', transition_type=TransitionType.T),
            Transition(origin='S_c', destination='S_u', equation='k_u*(1-W)*S_c', transition_type=TransitionType.T),
            Transition(origin='S_u', destination='S', equation='k_1*S_u', transition_type=TransitionType.T),   
            Transition(origin='S_u', destination='I', equation='beta*(I+c_0*I_c)*S_u', transition_type=TransitionType.T),    
            Transition(origin='I', destination='I_c', equation=c_2s+'c_2*(I+I_c)*I', transition_type=TransitionType.T),    
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
            Transition(origin='S', equation='-beta*(I+c_0*I_c)*S+c_1*S_c-%f*c_2*(I+I_c)*S-k_u*(1-W)*S+k_1*S_u' % C_2s),
            Transition(origin='S_c', equation='-c_0*beta*(I+c_0*I_c)*S_c-c_1*S_c+%f*c_2*(I+I_c)*S-k_u*(1-W)*S_c' % C_2s),
            Transition(origin='S_u', equation='-beta*(I+c_0*I_c)*S_u+k_u*(1-W)*(S+S_c)-k_1*S_u'),
            Transition(origin='E', equation='beta*(I+c_0*I_c)*(S+S_u)-alpha*E+c_1*E_c-%f*c_2*(I+I_c)*E' % C_2s),
            Transition(origin='E_c', equation='c_0*beta*(I+c_0*I_c)*S_c-alpha*E_c-c_1*E_c+%f*c_2*(I+I_c)*E' % C_2s),
            Transition(origin='I', equation='alpha*E-gamma*I-mu*I+c_1*I_c-%f*c_2*(I+I_c)*I' % C_2s),
            Transition(origin='I_c', equation='alpha*E_c-gamma*I_c-mu*I_c-c_1*I_c+%f*c_2*(I+I_c)*I' % C_2s),
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
            Transition(origin='S', equation='-(beta_1*(I_1+c_0*I_c)+beta_2*I_2+beta_3*I_3)*S+c_1*S_c-%f*c_2*(I_3)*S-k_u*(1-W)*S+k_1*S_u' % C_2s),
            Transition(origin='S_c', equation='-c_0*(beta_1*(I_1+c_0*I_c)+beta_2*I_2+beta_3*I_3)*S_c-c_1*S_c+%f*c_2*(I_3)*S-k_u*(1-W)*S_c' % C_2s),
            Transition(origin='S_u', equation='-(beta_1*(I_1+c_0*I_c)+beta_2*I_2+beta_3*I_3)*S_u+k_u*(1-W)*(S+S_c)-k_1*S_u'),
            Transition(origin='W', equation='k_w*W*(1-kappa*S_c-W)'),
            Transition(origin='E', equation='beta_1*(I_1+c_0*I_c)*(S+S_u)-alpha*E-%f*c_2*(I_3)*E+c_1*E_c' % C_2s),
            Transition(origin='E_c', equation='c_0*beta_1*(I_1+c_0*I_c)*S_c-alpha*E_c+%f*c_2*(I_3)*E-c_1*E_c' % C_2s),
            Transition(origin='I_1', equation='alpha*E-gamma_1*I_1-p_1*I_1-%f*c_2*(I_3)*I_1+c_1*I_c' % C_2s),
            Transition(origin='I_c', equation='alpha*E_c-gamma_1*I_c-p_1*I_c+%f*c_2*(I_3)*I_1-c_1*I_c' % C_2s), # changed to I_c, prints better
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
    print('make-model:  ERROR:  could not make model',mod_name);
    return None

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

def vector2params(b,a,g,p,u,c,k,N,modelname):
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
            # params['c_2'] = c[2]*FracCritical  # this can be calculated explicitly in next line
            params['c_2'] = c[2]*(p[1]/(g[1]+p[1]))*(p[2]/(g[2]+p[2]))

    if '_A' in modelname:
        if 'I3' in modelname: # models with hospitalization
            params['c_3'] = c[3]
        else:
            params['c_3'] = c[3]*(p[1]/(g[1]+p[1]))*(p[2]/(g[2]+p[2]))      
        
    if 'U' in modelname: # models with economic correction to caution  
        params['k_u'] = k[0]
        params['k_1'] = k[1]
        params['k_w'] = k[2]
        params['kappa'] = k[3]
        
    params['N'] = N
    return params

def params2vector(params,modelname='SC3UEI3R'):  # requires I3 in modelname
    if 'I3' not in modelname:
        print("Error in params2vector:  must have 'I3' in modelname.")
        return None
    b = [None,None,None,None]
    g = [None,None,None,None]
    p = [None,None,None]
    c = [None,None,None]
    k = [None,None,None,None]

    a=params['alpha']

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
    u=params['mu']       # equivalent to p[3]

    N=params['N']

    if 'C' in modelname: # models with caution 
        c[0]=params['c_0']
        c[1]=params['c_1']
        c[2]=params['c_2']
    if '_A' in modelname: # models with age structure
        c[3]=params['c_3']

    if 'U' in modelname: # models with economic correction to caution  
        k[0] = params['k_u']
        k[1] = params['k_1']
        k[2] = params['k_w']
        k[3] = params['kappa']
    return (b,a,g,p,u,c,k,N)

def base2vectors(sbparams,cbparams,fbparams):
    """ converts dictionary of base parameters to vector of parameters and then to pygom simulation parameters"""
    global C_2s # scaling factor for c_2 (1000) to allow c_2 parameter to be same order of magnitude as other parameters
    Exposure =sbparams['Exposure']
    IncubPeriod = sbparams['IncubPeriod']
    DurMildInf = sbparams['DurMildInf']
    FracMild = sbparams['FracMild']
    FracCritical = sbparams['FracCritical']
    FracSevere=1-FracMild-FracCritical
    CFR = sbparams['CFR']
    TimeICUDeath = sbparams['TimeICUDeath']
    DurHosp = sbparams['DurHosp']
    ICUFrac = sbparams['ICUFrac']
    I0 = 10**sbparams['logI_0']

    CautionFactor = cbparams['CautionFactor']
    CautionRetention = cbparams['CautionRetention']
    CautionExposure = cbparams['CautionExposure']
    CautionExposureYoung = cbparams['CautionExposureYoung'] # optional additional parameter, by default CautionExposure

    EconomicStriction =  cbparams['EconomicStriction']
    EconomicRetention =  cbparams['EconomicRetention']
    EconomyRelaxation =  cbparams['EconomyRelaxation']
    EconomicCostOfCaution = cbparams['EconomicCostOfCaution']
    
    FracConfirmedDet = fbparams['FracConfirmedDet']
    FracRecoveredDet = FracConfirmedDet
    FracDeathsDet = fbparams['FracDeathsDet']
    
    N=1
    b=np.zeros(4)     # beta
    g=np.zeros(4)     # gamma
    p=np.zeros(3)     # progression
    c=np.zeros(4)     # caution
    k=np.zeros(4)     # economic caution

    a=1/IncubPeriod                       # transition rate from exposed to infected
    b=Exposure*np.array([0,1,0,0])/N      # hospitalized cases don't transmit
    u=(1/TimeICUDeath)*(CFR/FracCritical) # death rate from ICU
    g[3]=(1/TimeICUDeath)-u               # recovery rate

    p[2]=(1/DurHosp)*(FracCritical/(1-FracMild))
    g[2]=(1/DurHosp)-p[2]

    g[1]=(1/DurMildInf)*FracMild
    p[1]=(1/DurMildInf)-g[1]

    c[0]=CautionFactor
    c[1]=1/CautionRetention
    c[2]=1/(N*(ICUFrac*C_2s)*CautionExposure)     # this is the rate coefficient giving 1/day at I3 = denominator
    c[3]=1/(N*(ICUFrac*C_2s)*CautionExposureYoung)

    k[0]=1/EconomicStriction              
    k[1]=1/EconomicRetention            
    k[2]=1/EconomyRelaxation   
    k[3]=EconomicCostOfCaution
    
    return(b,a,g,p,u,c,k,N,I0)

def base2params(sbparams,cbparams,fbparams,smodel):
    b,a,g,p,u,c,k,N,I0 = base2vectors(sbparams,cbparams,fbparams)
    return(vector2params(b,a,g,p,u,c,k,N,smodel))

def vectors2base(b,a,g,p,u,c,k,N,I0,ICUFrac):
    """ converts vector of parameters back to dictionaries of base parameters
        assumes only one parameter for bvector in the form b*[0,1,0,0]"""
    global C_2s
    Exposure          = b[1]*N # assuming b vector has structure b*[0,1,0,0]
    IncubPeriod       = 1.0/a

    FracMild          = g[1]/(g[1]+p[1])  
    FracCritical       = (g[1]/(g[1]+p[1]))*(p[2]/(g[2]+p[2]))
    #FracSevere        = (p[1]/(g[1]+p[1]))*(g[2]/(g[2]+p[2])) 
    FracSevere        = 1 - FracMild -FracCritical            
    CFR               = (u/(g[3]+u))*(p[2]/(g[2]+p[2]))*(p[1]/(g[1]+p[1]))  
    DurMildInf        = 1/(g[1]+p[1])
    DurHosp           = 1/(g[2]+p[2])
    TimeICUDeath      = 1/(g[3]+u)

    CautionFactor     = c[0]
    if c[1]:
        CautionRetention  = 1/c[1]
    else:
        CautionRetention  = None
    if c[2]:
        CautionExposure    = 1/(N*c[2]*(C_2s*ICUFrac))
    else:
        CautionExposure    = None
    if c[3]:
        CautionExposureYoung    = 1/(N*c[3]*(C_2s*ICUFrac))
    else:
        CautionExposureYoung    = CautionExposure
    
    if k[0]:
        EconomicStriction     =  1.0/k[0]
    else:
        EconomicStriction     =  None
    if k[1]:
        EconomicRetention     =  1.0/k[1]
    else:
        EconomicRetention     = None
    if k[2]:
        EconomyRelaxation     =  1.0/k[2]
    else:
        EconomyRelaxation     = None
    EconomicCostOfCaution =  k[3]

    sbparams = {'Exposure':Exposure,'IncubPeriod':IncubPeriod,'DurMildInf':DurMildInf,
                'FracMild':FracMild,'FracCritical':FracCritical,'CFR':CFR,
                'TimeICUDeath':TimeICUDeath,'DurHosp':DurHosp,'ICUFrac':ICUFrac,'logI_0':np.log10(I0)}
    cbparams = {'CautionFactor':CautionFactor,'CautionRetention':CautionRetention,
                'CautionExposure':CautionExposure,'CautionExposureYoung':CautionExposureYoung,            
                'EconomicStriction':EconomicStriction,'EconomicRetention':EconomicRetention,
                'EconomyRelaxation':EconomyRelaxation,'EconomicCostOfCaution':EconomicCostOfCaution}
   
    return (sbparams,cbparams)

def base2ICs(I0,N,smodel,model,age_structure=None):

    (x0old,t0) = model.initial_values
    nstates = len(x0old)
    x0 = [0.]*nstates
    if age_structure:
        first_infected_agegroup = int(age_structure//4)
    else: 
        first_infected_agegroup = 0 
    x0[first_infected_agegroup] = N*(1-I0)   # assumes susceptibles (first in state list)
    # x0[0] = N*(1-I0)
    if model.I_1 < nstates: # age structure dealt with in model specific model.I_1
        x0[model.I_1] = N*I0
    else:
        print('error, initial infectives location out of bounds',model.I_1,'not <',nstates)
    return (x0,t0)

def default_params(sbparams=None,cbparams=None,fbparams=None,dbparams=None):
    """ supply default parameters for those not already defined as arguments 
        does not check if non None arguments are correctly defined """

    if not sbparams:      # standard params set 2 from Germany fit
        Exposure=0.4     # Rate coefficient for exposure per individual in contact per day
        IncubPeriod=5     #Incubation period, days 
        DurMildInf=10     #Duration of mild infections, days
        FracMild=0.7      #Fraction of infections that are mild
        FracCritical=0.10   #Fraction of infections that are critical
        CFR=0.05          #Case fatality rate (fraction of infections resulting in death)
        TimeICUDeath=5    #Time from ICU admission to death, days
        DurHosp=4         #Duration of hospitalization, days
        ICUFrac= 0.001    # Fraction of ICUs relative to population size N
        logI_0 = np.log10(0.0000003)  # Fraction of population initially infected

        sbparams = {'Exposure':Exposure,'IncubPeriod':IncubPeriod,'DurMildInf':DurMildInf,
                   'FracMild':FracMild,'FracCritical':FracCritical,'CFR':CFR,
                   'TimeICUDeath':TimeICUDeath,'DurHosp':DurHosp,'ICUFrac':ICUFrac,'logI_0':logI_0}

    if not cbparams:          # Model extension by John McCaskill to include caution                     # set 2 based on Germany fit
        CautionFactor= 0.1    # Fractional reduction of exposure rate for cautioned individuals
        CautionRetention= 60. # Duration of cautionary state of susceptibles (8 weeks)
        CautionExposure= 0.1  # Rate of transition to caution per (individual per ICU) per day
        CautionExposureYoung= 0.1  # Rate of transition to caution per (individual per ICU) per day for young people
        EconomicStriction = 30.
        EconomicRetention = 60. # Duration of economic dominant state of susceptibles (here same as caution, typically longer)
        EconomyRelaxation = 60.
        EconomicCostOfCaution = 0.5 # Cost to economy of individual exercising caution

        cbparams = {'CautionFactor':CautionFactor,'CautionRetention':CautionRetention,
                    'CautionExposure':CautionExposure,'CautionExposureYoung':CautionExposureYoung,
                    'EconomicStriction':EconomicStriction,'EconomicRetention':EconomicRetention,
                    'EconomyRelaxation':EconomyRelaxation,'EconomicCostOfCaution':EconomicCostOfCaution}

    if not fbparams:          # Model fitting extension to allow for incomplete detection
        FracConfirmedDet=1.0 # Fraction of recovered individuals measured : plots made with this parameter
        FracRecoveredDet=FracConfirmedDet # Fraction of recovered individuals measured
        FracDeathsDet=1.0

        fbparams = {'FracConfirmedDet':FracConfirmedDet,'FracDeathsDet':FracDeathsDet}

    if not dbparams:     # extra data-related params for defining a run, including possible fitting with sliders:
        dbparams = {'country':'Germany','data_src':'owid'}

    return [sbparams,cbparams,fbparams,dbparams]

# Set up multimodel consistent sets of parameters, based on standard set defined by Dr. Alison Hill for SEI3RD 
def parametrize_model(smodel,sbparams=None,cbparams=None,fbparams=None,dbparams=None,age_structure=None):
    if sbparams == None or cbparams==None or fbparams==None or dbparams==None:
        [sbparams,cbparams,fbparams,dbparams] = default_params(sbparams,cbparams,fbparams,dbparams)
        dbparams['run_name'] = smodel # default value when no country yet
    b,a,g,p,u,c,k,N,I0 = base2vectors(sbparams,cbparams,fbparams)
    fullmodel = make_model(smodel)
    model = fullmodel['model']
    params_in=vector2params(b,a,g,p,u,c,k,N,smodel)
    if age_structure:
        model.initial_values = base2ICs(I0,N,smodel,model,age_structure=age_structure)
    else:
        model.initial_values = base2ICs(I0,N,smodel,model)
    # model.baseparams = list(sbparams)+list(cbparams)+list(fbparams)
    model.parameters = params_in # sets symbolic name parameters
    fullmodel['params'] = params_in    # sets string params
    fullmodel['sbparams'] = sbparams
    fullmodel['cbparams'] = cbparams
    fullmodel['fbparams'] = fbparams
    fullmodel['dbparams'] = dbparams
    fullmodel['initial_values'] = model.initial_values  # this line probably not required, since already initialized in make_model
    return fullmodel


# smodels = ['SIR','SCIR','SC2IR','SEIR','SCEIR','SC3EIR','SEI3R','SCEI3R','SC3EI3R','SC2UIR','SC3UEIR','SC3UEI3R'] # full set
# smodels = ['SEIR','SC3EIR','SC3UEIR','SEI3R','SC3EI3R','SC3UEI3R'] # partial set with comparison
smodels = ['SEI3R','SC3EI3R','SC3UEI3R'] # short list, others can be added if required from notebook
# samodels = ['SIR_A4','SC2IR_A4','SEI3R_A4','SC3EI3R_A4','SC3UEI3R_A4'] 
samodels = []

# Initialize all models
cmodels = {}
fullmodels = {}
print('making the models...')
for smodel in smodels+samodels:
    if '_A' in smodel:
        [smodel_root,age_str] = smodel.split("_A")
        try:
            age_structure = int(age_str)
        except:
            print("Error in parameterize_model, age suffix is not an integer.")
    else:
        smodel_root = smodel
        age_structure = None

    if smodel_root not in possmodels:
        print('root model name',smodel_root,'not yet supported')
    else: 
        fullmodel = parametrize_model(smodel_root,age_structure=age_structure)
        fullmodels[smodel] = fullmodel
        # take fullmodel['model'] so that modelnm is same model as before for backward compatibility
        cmodels[smodel] = fullmodel['model']
        modelnm = smodel+'_model'
        exec(modelnm+" = fullmodel['model']")
        print(smodel)          

print('done with the models.')
