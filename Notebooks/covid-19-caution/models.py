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
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import pwlf

from cycler import cycler
import datetime
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import pwlf
import sys
#from IPython.core.display import display, HTML
#display(HTML("<style>.container { width:100% !important; }</style>"))



savefigs = False # whether to save specific figures for paper to .../figures directory


def dumpparams(self): # Have to add self since this will become a method
    mname = self.modelname
    dirnm = os.getcwd()
    pfile = dirnm+'/params/'+mname+'.pk'
    try:
        params = self.params.copy()
        with open(pfile,'wb') as fp:
            pk.dump(params,fp,protocol=pk.HIGHEST_PROTOCOL)
        print('dumped params to',pfile)
    except:
        print('problem dumping params to ',pfile)


def loadparams(self): # Have to add self since this will become a method
    mname = self.modelname
    dirnm = os.getcwd()
    pfile = dirnm+'/params/'+mname+'.pk'
    try:
        with open(pfile,'rb') as fp:
            params = pk.load(fp)
            print('loaded params from ',pfile,':')
    except:
        print("problem loading",pfile)
        return None


    nms = [x.name for x in self.param_list]
    try:
        self.parameters = params.copy()
        self.params = params.copy()
    except:
        print('problem loading the params; none loaded')
        return None
    return True

OdeClass = DeterministicOde().__class__
setattr(OdeClass,'dumpparams', dumpparams)
setattr(OdeClass,'loadparams', loadparams)

def Float(x):
    try:
        rtn = float(x)
    except:
        rtn = float('NaN')
    return rtn


def  print_ode2(self):
        '''
        Prints the ode in symbolic form onto the screen/console in actual
        symbols rather than the word of the symbol.
        
        Based on the PyGOM built-in but adapted for Jupyter
        Corrected by John McCaskill to avoid subscript format error
        '''
        A = self.get_ode_eqn()
        B = sympy.zeros(A.rows,2)
        for i in range(A.shape[0]):
            B[i,0] = sympy.symbols('d' + '{' + str(self._stateList[i]) + '}'+ '/dt=')
            B[i,1] = A[i]

        return B


# Jupyter Specifics
from IPython.display import display, HTML
from ipywidgets.widgets import interact, interactive, IntSlider, FloatSlider, Layout, ToggleButton, ToggleButtons, fixed
display(HTML("<style>.container { width:100% !important; }</style>"))
style = {'description_width': '100px'}
slider_layout = Layout(width='99%')


# ## Caution Extensions to SIR Model

# ### SIR model

# #### Equations
# 
# \begin{equation}
# \begin{split}
# \dot{S} &= -\beta I S\\
# \dot{I} &= \beta I S - \gamma I - \mu I\\
# \dot{R} & = \gamma I \\
# \dot{D} & = \mu I
# \end{split}
# \end{equation}
# 
# 
# #### Variables
# * $S$: Susceptible individuals
# * $I$: Infected individuals 
# * $R$: individuals who have recovered from disease and are now immune
# * $D$: Dead individuals
# * $N=S+I+R+D$ Total population size (constant)
# 
# #### Parameters
# * $\beta$ rate at which infected individuals contact susceptibles and infect them
# * $\gamma$ rate at which infected individuals recover from disease and become immune
# * $\mu$ death rate for infected individuals

# #### Implementation
# Using PyGOM, we will set up my simple SCIR model ODE system
# PyGOM – A Python Package for Simplifying Modelling with Systems of Ordinary Differential Equations https://arxiv.org/pdf/1803.06934.pdf

# In[8]:


# set up the symbolic SIR model, actually SIRD including deaths
def make_model(mod_name):
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

# these earlier parameter translation routines are kept here for reference
# currently not used
# they allow the construction of model specific parameters for different models from a single set
def vector2params_old(b,a,g,p,u,c,k,N,modelname):
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

# vector2params allows the construction of model specific parameters for different models from a single set
def vector2params(b,a,g,p,u,c,k,N,modelname):
    global FracCritical
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

# Set up multimodel consistent sets of parameters
Exposure=0.25     # Rate coefficient for exposure per individual in contact per day
IncubPeriod=5     #Incubation period, days 
DurMildInf=10     #Duration of mild infections, days
FracMild=0.8      #Fraction of infections that are mild
FracSevere=0.15   #Fraction of infections that are severe
FracCritical=0.05 #Fraction of infections that are critical
CFR=0.02          #Case fatality rate (fraction of infections resulting in death)
TimeICUDeath=7    #Time from ICU admission to death, days
DurHosp=11        #Duration of hospitalization, days

# Model extension by John McCaskill to include caution 
CautionFactor= 0.3    # Fractional reduction of exposure rate for cautioned individuals
CautionRetention= 14. # Duration of cautionary state of susceptibles (4 weeks)
CautionICUFrac= 0.25  # Fraction of ICUs occupied leading to 90% of susceptibles in caution 
ICUFrac= 0.001        # Fraction of ICUs relative to population size N

EconomicCostOfCaution = 0.5 # Cost to economy of individual exercising caution

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

k[0]=c[1]
k[1]=c[1]
k[2]=c[1]
k[3]=EconomicCostOfCaution



    
smodels = ['SIR','SCIR','SC2IR','SEIR','SCEIR','SC3EIR','SEI3R','SCEI3R','SC3EI3R','SC2UIR','SC3UEIR','SC3UEI3R']

cmodels = {}
fullmodels = {}
for smodel in smodels:
    fullmodels[smodel] = make_model(smodel)
    cmodels[smodel] = fullmodels[smodel]['model']
    params_in=vector2params(b,a,g,p,u,c,k,N,smodel)
    fullmodels[smodel]['model'].parameters = params_in
    cmodels[smodel].parameters = params_in
    modelnm = smodel+'_model'
    exec(modelnm+" = cmodels[smodel]")
    
