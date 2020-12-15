import lmfit
import copy
import os
import io
# import functools  # partial can be used to provide extra parameters to callbacks, currently commented out
from time import time
from ipywidgets import widgets
from ipywidgets.widgets import interact, interactive, interactive_output, fixed, Widget             
from ipywidgets.widgets import interact, interactive, IntSlider, FloatSlider, Layout, ToggleButton, ToggleButtons, fixed, Widget
from ipywidgets.widgets import HBox, VBox, Label, Dropdown, Checkbox, Output
from IPython.display import display,clear_output

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# print('In ModeFit.py test print')

from model_fits_age import *

class ModelFit:
    """ We collect all information related to a fit between a pygom model and a set of data in this class
        It has access to the model structure and defines all required parameters and details of fit """
    def __init__(self,modelname='SEIR',basedata=None,model=None,country='',run_id='',datatypes='all',fit_targets=['confirmed','deaths'],
                 data_src='owid',startdate=None,stopdate=None,simdays=None,new=True,fit_method='leastsq',param_class='base',
                 agestructure=1):
        """
        if run_id is '', self.run_id takes a default value of default_run_id = modelname+'_'+country
        if run_id is not '', it is used as self.run_id, used in turn for param filename.
        except that if run_id starts with character '_', it is appended to the default run_id,
        i.e. if run_id[0]=='_': self.run_id = default_run_id+run_id 
        """
        # print('HERE in init of ModelFit')
        global make_model,possmodels,agemodels

        self.agestructure = agestructure
        if int(agestructure) > 1 and modelname in agemodels :   # modelname value from widget
            modelname_a = modelname+'_A'+str(agestructure)
        elif int(agestructure) > 1:  # age structure not yet implemented for this model type
            modelname_a = modelname
            agestructure=1
        else:
            modelname_a = modelname
        self.modelname = modelname_a

        self.data_src = data_src
        self.fit_method = fit_method
        self.new = new
        self.model = model
        self.param_class = param_class

        if basedata==None:
            print("Error:  must specify base data with arg basedata.")
        if self.data_src == 'jhu':
            self.data = basedata.covid_ts
        elif self.data_src == 'owid':
            self.data = basedata.covid_owid_ts
        else:
            print("Error:  data_src must be one of jhu or owid.")
        self.basedata = basedata

        self.startdate = startdate
        self.stopdate = stopdate
        self.simdays = simdays
        self.datatypes = datatypes
        self.fit_targets = fit_targets

        dirnm = os.getcwd()
        # construct default name for file / run_id
        if country != '':
            defnm = modelname+'_'+country
        else:
            defnm = modelname

        if run_id == '':                         # use default name
            self.run_id = defnm
        elif run_id[0]=='_':                     # use run_id as addon to default
            self.run_id = defnm+run_id
        else:
            self.run_id = run_id                 # use specified name
        #print('=============',self.run_id)
        pfile = dirnm+'/params/'+self.run_id+'.pk'


        ######################################
        # set up model
        self.setup_model(modelname)

        ################################################################
        # For scan, country='' and will be set up in scan loop
        if country != '':
            self.setup_data(country,data_src)

    def setup_model(self,modelname):
        self.modelname = modelname
        if self.model:
            if self.model.modelname != modelname:
                print("warning:  changing model from",modelname,'to',self.model.modelname)
                self.modelname = modelname
        else:
            if modelname not in fullmodels:
                if '_A' in modelname:
                    [modelname_root,age_str] = modelname.split("_A")
                    try:
                        age_structure = int(age_str)
                    except:
                        print("Error in setup_model, age suffix is not an integer.")
                        return
                else:
                    modelname_root = modelname
                    age_structure = None
                if modelname_root not in possmodels:
                    print('root model name',modelname_root,'not yet supported')
                    return
                else:
                    print('Adding model',modelname,'to stored models.')
                    fullmodel = parametrize_model(modelname_root,modelname,age_structure=age_structure)
                    fullmodels[modelname] = fullmodel

            model_d = copy.deepcopy(fullmodels[modelname])  # should avoid modifying fullmodels at all from fits, otherwise never clear what parameters are


            self.model = model_d['model']
            if self.new:
                    #print('using default set of parameters for model type',modelname)
                    self.odeparams   = model_d['params']
                    self.cbparams = model_d['cbparams']
                    self.sbparams = model_d['sbparams']
                    self.fbparams = model_d['fbparams']
                    self.dbparams = model_d['dbparams']
                    self.initial_values = model_d['initial_values']
                    self.age_structure = model_d['age_structure']
            else:
                if not self.loadparams(self.run_id):
                    #print('Problem loading paramfile for',run_id,'... using default set of parameters for model type',modelname)
                    self.odeparams   = model_d['params']
                    self.cbparams = model_d['cbparams']
                    self.sbparams = model_d['sbparams']
                    self.fbparams = model_d['fbparams']
                    self.dbparams = model_d['dbparams']
                    self.initial_values = model_d['initial_values']
                    self.age_structure = model_d['age_structure']
        # self.baseparams = list(self.sbparams)+list(self.cbparams)+list(self.fbparams) # caused ode/base switching problems
        self.baseparams = {**self.sbparams,**self.cbparams,**self.fbparams}  # now a merged dictionary

        if self.param_class == 'ode':
            self.params = copy.deepcopy(self.odeparams)
        elif self.param_class == 'base':
            self.params = copy.deepcopy(self.baseparams)
        else:
            print("Error:  bad param_class.")
            return None
        


    def setup_data(self,country,data_src):

        if data_src not in ['jhu','owid','cluster']:
            print('data_src',data_src,'not yet hooked up: use jhu, owid or cluster data instead')
            return None
        else:
            self.data_src = data_src
            self.dbparams['data_src']=data_src
            if self.data_src == 'jhu':
                self.data = self.basedata.covid_ts
            elif self.data_src == 'owid':
                self.data = self.basedata.covid_owid_ts
            else:
                print("Error:  data_src must be one of jhu or owid.")
        ts = self.data
        # NB: countries in ts are common countries, keyed using simple string names (common name for datasets) 
        self.country_str = country_str = country
        self.country = country
        self.dbparams['country']=country
        self.population = self.basedata.population_owid[self.country_str][-2] # -2 seems to get all countries population (no zeros)

        fmt_jhu = '%m/%d/%y'
        if self.data_src == 'owid' or self.data_src == 'jhu':
            dates_t = [datetime.datetime.strptime(dd,fmt_jhu) for dd in ts['deaths']['dates'] ] # ts dates stored in string format of jhu fmt_jhu = '%m/%d/%y'
            firstdate_t =  dates_t[0]
            lastdate_t =  dates_t[-1]
            if self.startdate:
                startdate_t = datetime.datetime.strptime(self.startdate,fmt_jhu)
            else:
                startdate_t = firstdate_t
            if self.stopdate:
                stopdate_t = datetime.datetime.strptime(self.stopdate,fmt_jhu)
                #print('stopdate',self.stopdate) 
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
            else:
                datadays = (lastdate_t-startdate_t).days + 1
            self.dates = [date.strftime(fmt_jhu) for date in dates_t if date>=startdate_t and date <= lastdate_t]
        elif self.data_src == 'cluster':
            datadays = len(ts['deaths'][country])
            if self.simdays: # simdays allowed greater than datadays to enable predictions
                if self.simdays < datadays:
                    datadays = self.simdays
            self.startdate = '02/01/20' # fake first date for cluster time series
            startdate_t = datetime.datetime.strptime(self.startdate,fmt_jhu)
            daystart = 0
            self.dates = [startdate_t + datetime.timedelta(days=x) for x in range(datadays)] # fake dates
            stopdate_t = self.dates[-1]
        else:
            print("Error:  can't deal with data_src", self.data_src)
            return None

        if self.simdays: # simdays allowed greater than datadays to enable predictions
            if self.simdays < datadays:
                stopdate_t = startdate_t + datetime.timedelta(days=self.simdays-1)  # if simulation for shorter time than data, restrict data to this
                datadays = (stopdate_t-startdate_t).days + 1    
                self.tsim = np.linspace(0, datadays -1, datadays)
            else:
                self.tsim = np.linspace(0, self.simdays -1, self.simdays)
        else:
            self.tsim = np.linspace(0, datadays -1, datadays)

        if self.datatypes == 'all':
            self.datatypes = [x for x in ts]

        self.tsdata = {}
        for dt in self.datatypes:
            if dt not in ts:
                print('datatype error:')
                print(dt,'not in ts for data_src',self.data_src)
                return None
            if ts[dt] is not None:
                try:
                    self.tsdata[dt] = ts[dt][country][daystart:datadays].copy()
                except Exception as e:
                        print('problem with',dt,'country',country)
                        print(e)
                    
            #self.data.update({dt:ts[dt][country][daystart:datadays]}) 

        self.startdate = startdate_t.strftime(fmt_jhu)
        self.stopdate = stopdate_t.strftime(fmt_jhu)

        for targ in self.fit_targets:
            if targ not in self.tsdata:
                print('Error: fit target',targ,'is not available in datatypes',self.datatypes)
                return None
        self.fit_data = 'default'

    def  print_ode(self):
        '''
        Prints the ode in symbolic form onto the screen/console in actual
        symbols rather than the word of the symbol.
        
        Based on the PyGOM built-in but adapted for Jupyter
        Corrected by John McCaskill to avoid subscript format error
        '''
        A = self.model.get_ode_eqn()
        B = sympy.zeros(A.rows,2)
        for i in range(A.shape[0]):
            B[i,0] = sympy.symbols('d' + '{' + str(self.model._stateList[i]) + '}'+ '/dt=')
            B[i,1] = A[i]
        return B

    def dumpparams(self,run_id=''): # Have to add self since this will become a method
        """stores params in a file './params/Model_Name.pk'
        This stuff needs modules os, sys, pickle as pk.
        If run_id is nonempty, it is used to construct the filename, and self.run_id is set to its value."""
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
        self.paramfile = pfile

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
            #print('dumped params to',pfile)
        except:
            print('problem dumping params to ',pfile)
    def loadparams(self,run_id=''): 
        """loads params from same file.  returns None if any problem finding the file.
        This stuff needs modules os, sys, pickle as pk.
        If run_id is nonempty, it is used to construct the filename, and self.run_id is set to its value."""
        if run_id == '':
            run_id = self.run_id
        elif self.run_id != run_id:
            print("warning: changing run_id from ",self.run_id,'to',run_id)
            self.run_id = run_id
            
        dirnm = os.getcwd()
        pfile = dirnm+'/params/'+run_id+'.pk'
        self.paramfile = pfile

        try:
            with open(pfile,'rb') as fp:
                all_params = pk.load(fp)
                print('loaded params from ',pfile,':')
        except:
            print("For this run_id, a fresh file: ",pfile)
            return None

        #print('-------  params from file:')
        #ppr.pprint(all_params)
        # check to see that all params being loaded match params of model, if not: fail.
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

    def checkparams(self,pinit):
        cnt = 0
        for pp in pinit:
            mn = pinit[pp][1]
            mx = pinit[pp][2]
            if pp in self.params:
                cur = self.params[pp]
            elif pp == 'logI_0':
                cur = self.sbparams[pp]
            else:
                print("Error: couldn't find param",pp)
            win = mx-mn
            if (cur-mn)/win < 0.05:
                cnt = cnt+1
                perc = np.round(100*(cur-mn)/win,2)
                print('Param',pp,'within',perc,'% of min.')
            if (mx-cur)/win < 0.05:
                cnt = cnt+1
                perc = np.round(100*(mx-cur)/win,2)
                print('Param',pp,'within',perc,'% of max.')
        if cnt==0:
            print("All params are away from boundaries.")
        # print("Finished checkparams.")

    def check_params(self):
        print('len(self.odeparams) = ',len(self.odeparams))
        print(self.odeparams)
        print('model.num_param = ',self.model.num_param)
        print(self.model.parameters)

    def set_param(self,param,value):
        if self.param_class == 'ode':
            self.set_ode_param(param,value)
        elif self.param_class == 'base':
            self.set_base_param(param,value)
        else:
            print('set_param Error: bad param_class =',self.param_class)

    def set_ode_param(self,param,value):
        # print('--------------------------- new set param call ----------------------------------')
        # print('In set_param with param',param,'with value',value,'self',self)
        # print('self.model.parameters',self.model.parameters)
        # print('---------------------------------------------------------------------------------')
        plist = [p.name for p in list(self.model.param_list)]
        if param not in plist and param != 'logI_0':
            print('Error:  param name',param,'is not a parameter for this',self.modelname,'model.')
        else:
            self.odeparams[param] = value
            if param != 'logI_0':
                tmp = {param:value}
                self.model.parameters = tmp # pygom magic sets the right parameter in the model.parameters dictionary.
            # self.model.parameters = self.odeparams[param]   # this has problem with initial condition parameter logI_0


    def set_base_param(self,param,value):
        """sets base parameter and converts to ode parameters for simulation
           note that this process is pretty inefficient, operating one by one on parameters
           initial condition logI_0 is not set by this routine
        """
        if param not in self.baseparams:
            print('Error:  param name',param,'is not a base parameter for this',self.modelname,'model.')
            eprint(self.baseparams)
        else:
            self.baseparams[param] = value
        if param in list(self.sbparams):
            self.sbparams[param] = value
        elif param in list(self.cbparams):
            self.cbparams[param] = value  
        elif param in list(self.fbparams):
            self.fbparams[param] = value 

        b,a,g,p,u,c,k,N,I0 = base2vectors(self.sbparams,self.cbparams,self.fbparams)
        params_in=vector2params(b,a,g,p,u,c,k,N,self.modelname)
        #print('in set_base_param',param,value,'------------------')
        #print('params_in',params_in)
        #print('params',self.params)
        for pm in self.odeparams:
            self.odeparams[pm] = params_in[pm] # NB: vector2params returns all params including for U model
        self.model.parameters = self.odeparams # pygom magic sets the right parameter in the model.parameters dictionary.

    def set_initial_values(self,ival,t0=None):
        # consistency check:
        if len(self.initial_values[0]) != len(self.model.initial_values[0]):
            print('warning: inconsistent initial values in model.')
        if len(ival) != len(self.model.initial_values[0]):
            print('error:  initial value must be of length', len(self.model.initial_values[0]))
        self.model.initial_values[0] = [x for x in ival]
        self.initial_values[0] = [x for x in ival]
        if t0 is not None:
            self.model.initial_values[1] = t0
            self.initial_values[1] = t0

    def set_I0(self,logI_0):
        self.sbparams['logI_0']=logI_0
        I0 = 10**logI_0
        self.model.initial_values[0][0] = 1.0 - I0
        self.model.initial_values[0][self.model.I_1] = I0    # use model specific position of initial infective compartment
        self.initial_values[0][0] = 1.0 - I0
        self.initial_values[0][self.model.I_1] = I0
        # print('Setting I0 value at index',self.model.I_1,'to',I0)

    def refresh_base(self):
        """
        converts sim params stored in self.params (obtained after fit) into  base params, 
        storing in self.sbparams and self.cbparams.
        Logic:  apply params2vector, followed by vectors2base, both found in model_fits_nodata.py
        """
        vec = params2vector(self,self.odeparams,self.modelname) # returns (b,a,g,p,u,c,k,N)
        # print('params',self.modelname,self.params)
        # print('vec',vec)
        I0 = np.power(10,self.sbparams['logI_0'])
        ICUFrac = self.sbparams['ICUFrac']
        sb,cb = vectors2base(*vec,I0,ICUFrac) # returns (sbparams, cbparams)
        if set([x for x in sb]) != set([x for x in self.sbparams]):
            print("Error:  sbparams mismatch in refresh_base().  Params unchanged.")
        else:
            self.sbparams = copy.deepcopy(sb)
        if set([x for x in cb]) != set([x for x in self.cbparams]):
            print("Error:  cbparams mismatch in refresh_base().  Params unchanged.")
        else:
            self.cbparams = copy.deepcopy(cb)
        self.baseparams = {**self.sbparams,**self.cbparams,**self.fbparams}

    def difference(self,datain,exceptions=[]):
        dataout = np.zeros(np.shape(datain))
        for i in range(1,len(datain)):
            dataout[i,...] = datain[i,...]-datain[i-1,...]
        for ns in exceptions:
            for i in range(0,len(datain)):
                dataout[i,ns] = datain[i,ns]
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
        xx = np.array(range(len(self.tsim)-1))
        print(len(xx))
        print([(x,len(self.tsdata[x])) for x in dtypes])

        for dt in dtypes:
            try:
                yy = self.tsdata[dt]
            except:
                print("data type '"+dt+"' not found.")
            try:
                plt.plot(xx,yy)
            except:
                print("couldn't plot xx,yy",xx,yy)
        plt.show()


    def transfer_fit_to_params_init(self,params_init_min_max):
        """ used to transfer current fit parameters as initial parameter values to an existing
            initialization structure params_init_min_max
            only those parameters in params_init_min_max will have initial values updated
        """
        plist = (self.odeparams,self.sbparams,self.cbparams,self.fbparams,self.dbparams)
        for ptype in plist:
            for p in ptype:
                if p in params_init_min_max:
                    pv = params_init_min_max[p]
                    if len(pv) == 4:
                        params_init_min_max[p] = (ptype[p],pv[1],pv[2],pv[3])
                    else:
                        params_init_min_max[p] = (ptype[p],pv[1],pv[2])
        return params_init_min_max

    def plot_age_groups(self,slice_,age_groups,srsoln,magsns,tvec1,ax,color,label):
        step = age_groups*slice_.step if slice_.step else age_groups
        suma_age_cum_last = 0
        for age in range(age_groups):
            suma_age = np.sum(srsoln[:,slice(slice_.start+age,slice_.stop,step)],axis=1)*magsns
            suma_age_cum = suma_age.copy() if age == 0 else suma_age_cum + suma_age
            ax.fill_between(tvec1,suma_age_cum_last,suma_age_cum,color=color,alpha=float(age)/age_groups);
            ax.plot(tvec1,suma_age_cum,label=label,color=color,alpha=0.2,linewidth=1);
            suma_age_cum_last = suma_age_cum
            # line, = ax.plot(...
            # line.set_dashes([2,2,2+age,2])

    def solveplot(self, species=['confirmed'],summing='daily',averaging='weekly',mag = {'deaths':10},axis=None,
                  scale='linear',plottitle= '',label='',newplot = True, gbrcolors=False, figsize = None,
                  outfile = None,datasets=['confirmed_corrected_smoothed'],age_groups=None,background='white'):
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

        if age_groups:
            print('age_groups',age_groups)

        smodel = self.modelname
        model = self.model

        if not isinstance(species,list):
            lspecies = [species]
            ldatasets = [datasets]
        else:
            lspecies = species
            ldatasets = datasets

        exceptions = []
        dexceptions = []
        for dt in ldatasets:
            if dt not in [x for x in self.tsdata]:
                print('Error:  ',dt,'not in data')

        dspecies = [dt if dt != 'caution_fraction' else 'stringency' for dt in lspecies]
        mags = [mag[dt] if dt in mag.keys() else 1 for dt in dspecies]
        if 'economy' in lspecies and 'U' in smodel:
            exceptions.append(model.W)
        if 'economy' in ldatasets and 'U' in smodel:
            dexceptions.append(ldatasets.index('economy'))
        tvec = self.tsim
        tvec1 = tvec[1:]
        if not self.tsdata is {}:
            fitdata = np.transpose(np.array([self.tsdata[dt] for dt in ldatasets]))
            fitsmoothed = False
            for dt in datasets:
                if 'smoothed' in dt:
                    fitsmoothed = True
        else:
            fitdata = None
        if not fitdata is None:
            tmaxf = len(fitdata)
            if fitdata.ndim != 2:
                print("error in number of dimensions of array")
            tvecf=np.arange(0,tmaxf,1)
            tvecf1 = tvecf[1:]
        
        if newplot:
            axis = None
            if (figsize == None):
                figsize=(8,6)
            plt.figure(figsize=figsize)
            # fig, axeslist = plt.subplots(1, nmodels, figsize=(nmodels*8,6))
            

        self.soln = scipy.integrate.odeint(model.ode, model.initial_values[0], tvec[1::])
        # print('debug, calling scipy integrate on self.model with IC', model.initial_values[0])
        #Plot
        # ax = axeslist[nm]
        if axis == None: 
            ax = axis = plt.subplot(1,1,1);
        else:
            ax = axis;
        ax.set_facecolor(background)
        ax1 = None

        if scale == 'log': #Plot on log scale
            ax.semilogy();
            ax.set_ylim([0.00000001,1.0]);
            
        if summing == 'daily':
            ssoln = self.difference(self.soln,exceptions=exceptions)
            if not fitdata is None:
                sfit = self.difference(fitdata,exceptions=dexceptions)
        else:
            ssoln = self.soln
            if not fitdata is None:
                sfit = fitdata
                
        if averaging == 'weekly':
            srsoln = self.rolling_average(ssoln,7)
            if not fitdata is None:
                if not fitsmoothed:               
                    srfit = self.rolling_average(sfit,7)
                else:
                    srfit = sfit
        else:
            srsoln = ssoln
            if not fitdata is None:
                srfit = sfit             
        for ns,species in enumerate(lspecies):
            if species == 'confirmed':
                suma = np.sum(srsoln[:,model.confirmed],axis=1)*mags[ns]
                if not fitdata is None:
                    fita = srfit[1::,ns]*mags[ns]/self.fbparams['FracConfirmedDet']/self.population # confirmed cases data, corrected by FracConfirmedDet
                    ax.plot(tvecf1,fita,'o',label=label,color='green');
                ax.plot(tvec1,suma,label=label,color='green');
                if age_groups:
                    print('age',age_groups,model.confirmed)
                    self.plot_age_groups(model.confirmed,age_groups,srsoln,mags[ns],tvec1,ax,'green',label);
            if species == 'recovered':
                suma = np.sum(srsoln[:,model.recovered],axis=1)*mags[ns]  
                if not fitdata is None:
                    fita = srfit[1::,ns]*mags[ns]/self.fbparams['FracRecoveredDet']/self.population # recovered cases data, corrected by FracRecoveredDet
                    ax.plot(tvecf1,fita,'o',label=label,color='blue');
                ax.plot(tvec1,suma,label=label,color='blue');
                if age_groups:
                    self.plot_age_groups(model.recovered,age_groups,srsoln,mags[ns],tvec1,ax,'blue',label);
            elif species == 'deaths':
                suma = np.sum(srsoln[:,model.deaths],axis=1)*mags[ns]
                if not fitdata is None:
                    fita = srfit[1::,ns]*mags[ns]/self.fbparams['FracDeathsDet']/self.population # deaths cases data, corrected by FracDeathsDet
                    ax.plot(tvecf1,fita,'o',label=label,color='red',alpha=0.2);
                ax.plot(tvec1,suma,label=label,color='darkred');
                if age_groups:
                    print('age',age_groups,model.deaths)
                    self.plot_age_groups(model.deaths,age_groups,srsoln,mags[ns],tvec1,ax,'darkred',label);
            elif species == 'EI':
                ax.plot(tvec1,self.soln[:,model.ei],label=label);
                if age_groups:
                    self.plot_age_groups(model.ei,age_groups,self.soln,mags[ns],tvec1,ax,'blue',label);
                # ax.plot(tvec1,self.soln[:,model.ei],label="%s" % count)
                if 'I3' in model.modelname: 
                    plt.legend(("E","I1","I2","I3"));
                elif 'E' in model.modelname: 
                    plt.legend(("E","I"));
                else:
                    plt.legend(("I"));
            elif species == 'caution_fraction':
                if 'C' in smodel:
                    #print('model name',model.modelname)
                    if age_groups and isinstance(model.all_susceptibles,list):
                        print('all susceptibles',model.all_susceptibles)
                        suma = np.zeros(len(tvec1),float)
                        for sl in model.all_susceptibles:
                            if isinstance(sl,slice):
                                suma[:] = suma[:] + np.sum(self.soln[:,sl],axis=1)
                            elif isinstance(sl,int):
                                suma[:] = suma[:] + self.soln[:,sl]
                    else:
                        suma = np.sum(self.soln[:,model.all_susceptibles],axis=1)
                    if age_groups:
                        susc = np.sum(self.soln[:,model.S_c],axis=1)
                    else:
                        susc = self.soln[:,model.S_c]
                    old_settings = np.seterr(divide='ignore') #
                    suma = np.divide(susc,suma)
                    np.seterr(**old_settings)  # reset to default
                    if len(lspecies) > 1:
                        ax1 = ax.twinx();
                    else:
                        ax1 = ax;
                    if not fitdata is None and ns<len(ldatasets):
                        fita = srfit[1::,ns]*mags[ns] # caution fraction from data (stringency) with correciton to unit scale via mags
                        ax1.plot(tvecf1,fita,'o',label=label,color='orange');
                    ax1.plot(tvec1,suma,label=label,color='orange');             
            elif species == 'all':
                ax.plot(tvec1,self.soln,label=label);
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
                if 'F' in model.modelname:
                    pspecies.append("Sf")
                plt.legend(pspecies);
            elif species == 'economy':
                if 'U' in smodel:
                    suma = srsoln[:,model.W]*mags[ns]
                    if 'caution_fraction' not in lspecies:
                        if len(lspecies) > 1:
                            ax1 = ax.twinx();
                        else:
                            ax1 = ax;
                    if not fitdata is None and ns<len(ldatasets):
                        fita = srfit[1::,ns]*mags[ns] # caution fraction from data (stringency) with correciton to unit scale via mags
                        ax1.plot(tvecf1,fita,'o',label=label,color='blue');  
                    ax1.plot(tvec1,suma,label=label,color='blue');                         
                
        plt.xlabel("Time (days)");
        plt.ylabel("Fraction of population");

        plt.title(model.modelname+' '+self.country+' '+plottitle);
        if outfile:
            plt.savefig(outfile,bbox_inches='tight');
        self.dumpparams();       # dump every plot;  could be changed by sliders
        plt.close()
        self.fig = ax.figure;


    def prparams(self,outfile = ''):
        """
        pretty print all params.
        If outfile is not '', params are printed to it, in the form of a dictionary that can be read back in.
        """
        if outfile != '':
            with open(outfile,'w') as out:
                pp = pprint.PrettyPrinter(stream=out)
                pp.pprint({'params':self.params,
                           'sbparams':self.sbparams,
                           'fbparams':self.fbparams,
                           'cbparams':self.cbparams,
                           'dbparams':self.dbparams,
                           'initial_values':self.initial_values})
        else:
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
            print('initial_values:')
            ppr.pprint(self.initial_values)


    def getparams(self):
        rtn = {}
        for pp in ['params','sbparams','fbparams','cbparams','dbparams']:
            ppp = eval('self.'+pp) # fail first time when ModelFit doesn't have params.
            rtn[pp] = ppp
        return rtn

    def get_fitdata(self,species=['deaths'],datasets=['deaths_corrected_smoothed']):
        # NB species is same as fit_targets and datasets the same as fit_data
        if not isinstance(species,list):    # this correction only reqd if get_fitdata or solve4fit called externally
            lspecies = [species]
            ldatasets =[datasets]
        else:
            lspecies = species
            ldatasets =datasets
            if not len(datasets)==len(lspecies):
                print('Error in input to get_fitdata: species and datasets parameters not same length')
        # 
        tvec = self.tsim
        tvec1 = tvec[1:]
        fitdata = {}
        if not self.tsdata is {}:
            for i,ls in enumerate(lspecies):
                ds = ldatasets[i]
                if ls == 'confirmed':     
                    datmp = self.tsdata[ds] # confirmed cases data, corrected by FracConfirmedDet
                    fitdata[ls] = [x/self.fbparams['FracConfirmedDet']/self.population for x in datmp]
                elif ls == 'deaths':
                    datmp = self.tsdata[ds] # deaths cases data, corrected by FracDeathsDet
                    fitdata[ls] = [x/self.fbparams['FracDeathsDet']/self.population for x in datmp]
                else:
                    fitdata[ls] = np.array(self.tsdata[ds])

        else:
            print('missing fit data')
            for ls in lspecies:
                fitdata[ls] = None
        return fitdata

    def solve4fit(self,species = ['deaths'],datasets=['deaths_corrected_smoothed']):
        fitdata = self.get_fitdata(species,datasets)
        lspecies = [x for x in fitdata]
        tmaxf = len(fitdata[lspecies[0]])            

        tvec = self.tsim
        tvecf=np.arange(0,tmaxf,1)
        tvecf1 = tvecf[1:]
        # print('In solve4fit debug, self', self,'self.model.parameters:',self.model.parameters)
        self.soln = scipy.integrate.odeint(self.model.ode, self.model.initial_values[0], tvec)
        rtn = {}
        slices = {}
        for ls in lspecies:
            if ls == 'deaths':
                slices['deaths'] = self.model.deaths
            if ls == 'confirmed':
                slices['confirmed'] = self.model.confirmed

        for ls in lspecies:
            if ls == 'deaths':
                weight = 30.    # slightly overweight deaths (expect 20x for Germany) compared with 'confirmed', to give more weight to deaths
            else:
                weight = 1.
            rtn[ls] = {}
            rtn[ls]['data'] = weight*np.array(fitdata[ls])
            rtn[ls]['soln'] = weight*np.sum(self.soln[:,slices[ls]],axis=1) #  sum over all species in 'confirmed' or only one species for 'deaths'
            rtn[ls]['resid'] = rtn[ls]['soln']-rtn[ls]['data']
        return rtn

    def solve4fitlog(self,species = ['deaths'],datasets=['deaths_corrected_smoothed']):
        """
        like solve4fit() but take log of data and soln before computing residual.
        """
        fitdata = self.get_fitdata(species,datasets)
        lspecies = [x for x in fitdata]
        tmaxf = len(fitdata[lspecies[0]])            

        tvec = self.tsim
        tvecf=np.arange(0,tmaxf,1)
        tvecf1 = tvecf[1:]
        self.soln = scipy.integrate.odeint(self.model.ode, self.model.initial_values[0], tvec)
        rtn = {}
        slices = {}
        self.logresid = {}
        for ls in lspecies:
            if ls == 'deaths':
                slices['deaths'] = self.model.deaths
            if ls == 'confirmed':
                slices['confirmed'] = self.model.confirmed

        for ls in lspecies:
            if ls == 'deaths':  # equivalent of weight change in solve4fit
                offset = 1.0/self.population
            else:
                offset = 10.0/self.population
            rtn[ls] = {}
            rtn[ls]['data'] = np.array(fitdata[ls])
            rtn[ls]['soln'] = np.sum(self.soln[:,slices[ls]],axis=1) #  sum over all species in 'confirmed'

            fdata = rtn[ls]['data']
            fdat = np.maximum(fdata,0)
            lfdat = np.log10(offset+fdat)

            sdata = rtn[ls]['soln']
            sdat = np.maximum(sdata,0)
            lsdat = np.log10(offset+sdat)
            rtn[ls]['resid'] = lsdat - lfdat
            # self.logresid = [sdat,lsdat,fdat,lfdat,lsdat-lfdat]
            self.logresid[ls] = (lsdat-lfdat).copy() # reduces amount of information stored for efficiency
        return rtn

    def fit(self,params_init_min_max,checkdict,fit_targets='default',fit_data='default',diag=True,report=True,conf_interval=False,fit_kws={}):
        """ fits parameters described in params_init_min_max, format 3 or 4-tuple (val,min,max,step)
            from class 'ode' or 'base', using method fit_method, and fit target quantitites fit_targets
            to data specified in fit_data, with option of diagnosis output diag
        """
        fit_method = self.fit_method
        # process input parameters ------------------------------------------------------------------------------------------
        # 1. param_class
        param_class = self.param_class
        print('fit: param_class = ',param_class)
        if param_class not in ['ode','base']:
            print('parameters must be either all in class ode or base currently, not',param_class) # logI_0 is in both classes
            return
        
        # 2. fit_targets    
        if fit_targets == 'default':
            fit_targets = self.fit_targets
        elif isinstance(fit_targets, str):
            fit_targets = [fit_targets]
        if len(set(fit_targets).difference(['confirmed','deaths'])) != 0:
            fit_targets = list(set(fit_targets).intersection(['confirmed','deaths']))
            if len(fit_targets) == 0:
                fit_targets = ['confirmed','deaths']
            print('can only fit deaths or confirmed for now, proceeding with',fit_targets)
        self.fit_targets = fit_targets
        
        # 3. fit_data
        if fit_data == 'default':
            self.fit_data = fit_data = [fit_target+'_corrected_smoothed' for fit_target in fit_targets]
        elif isinstance(fit_data, str):
            fit_data = [fit_data]
        if len(fit_data) == len(fit_targets):
            self.fit_data = fit_data
        else:
            eprint('fit_targets and fit_data must have same length',len(fit_targets),len(fit_data))
            eprint('proceeding with default')
            self.fit_data = fit_data = [fit_target+'_corrected_smoothed' for fit_target in fit_targets]
        
        # 4. params_init_min_max
        for pp in params_init_min_max:
            if pp is not 'logI_0': # add any other special ad hoc params here...
                if param_class == 'ode':
                    if pp not in list(self.model.param_list):
                        print(pp,':  bad param for',self.model.modelname,'model.')
                        return
                elif param_class == 'base':
                    if pp not in self.baseparams:
                        print(pp,':  bad base param for',self.model.modelname,'model.')
                        return
        for pp in params_init_min_max:
            if len(params_init_min_max[pp]) < 3: # length may be 3 or 4 (including set data for sliders)
                print('params_init_min_max has incorrect form.')
                print('should be dictionary with each entry as tuple (initial_value,min,max).')
                print('or dictionary with each entry as tuple (initial_value,min,max,step).')
                return

        # prepare parameters for lmfit ------------------------------------------------------------------------------------
        params_lmf = lmfit.Parameters()
        for pp in params_init_min_max:
            if not checkdict[pp+'_fix'].value:
                params_lmf.add(pp, params_init_min_max[pp][0],
                               min=params_init_min_max[pp][1],
                               max=params_init_min_max[pp][2])

        ## set initial params for fit
        for x in params_lmf:
            self.set_param(x, params_lmf[x].value)
        if 'logI_0' in params_lmf: # set other ad hoc params in both sets like this
                self.set_I0(params_lmf['logI_0'].value) 

        ## modify resid here for other optimizations -----------------------------------------------------------------------
        def resid(pars,*args):
            # print('------------------------------- new resid call ------------------------------------')
            if args:
                modelfit = args[0]
            else:
                print('Error in resid args, is not tuple containing modelfit instance',args)
                return
            parvals = pars.valuesdict()
            for x in parvals:
                if x in modelfit.params:
                    modelfit.set_param(x, parvals[x])
                    # print('in resid setting parameter',x,parvals[x])
                elif x != 'logI_0' and x in modelfit.baseparams:
                    modelfit.set_base_param(x, parvals[x])
                    # print('in resid setting base parameter',x,parvals[x])
            # print('')
            if 'logI_0' in params_lmf:
                modelfit.set_I0(parvals['logI_0'])    

            # maybe try log(1+xxx) by using solve4fitlog
            fittry = modelfit.solve4fitlog(modelfit.fit_targets,modelfit.fit_data) # use solve4fitlog to get residuals as log(soln)-log(data)
            #rmsres2 = np.sqrt(np.sum(np.square(resd)))
            #print('resid: ',rmsres2)
            fitresid = np.concatenate([fittry[fit_target]['resid'] for fit_target in modelfit.fit_targets])
            return fitresid
            # return [fittry[fit_target]['resid'] for fit_target in modelfit.fit_targets]

        def lsq(diffs):
            """
            scalar function to minimize in lmfit.minimize() specified by reduce_fcn parameter
            """
            return np.sqrt(np.sum(np.square(diffs))) # corrected to produce scalar as required

        ## do the fit -------------------------------------------------------------------------------------------------------
        try:
            if diag:
                start = time()
                self.residall = []
                self.paramall = []
                def per_iteration(pars, iteration, resd, *args, **kws):
                    rmsres2 = np.sqrt(np.sum(np.square(resd)))
                    self.residall.append(rmsres2)                    
                    self.paramall.append(pars.copy())
                fit_output = lmfit.minimize(resid, params_lmf, method=fit_method,args=(self,),iter_cb=per_iteration,**fit_kws)
                # fit_output = lmfit.minimize(resid, params_lmf, method=fit_method,args=(self,),iter_cb=per_iteration,reduce_fcn=lsq,**fit_kws)

                print('elapsed time = ',time()-start)
                self.checkparams(params_init_min_max)
                lmfit.report_fit(fit_output)
                if (fit_method == 'leastsq') and conf_interval:
                    print('calculating Confidence Intervals')
                    ci = lmfit.conf_interval(mini, fit_output)
                    print('Confidence Intervals')
                    lmfit.printfuncs.report_ci(ci)                    
            elif report:
                # fit_output = lmfit.minimize(resid, params_lmf, method=fit_method,args=(self,),**fit_kws)
                if (fit_method == 'leastsq') and conf_interval:
                    mini = lmfit.Minimizer(resid, params_lmf, fcn_args=(self,),**fit_kws)
                    fit_output = mini.minimize()
                    lmfit.report_fit(fit_output)
                    print('calculating Confidence Intervals')
                    ci = lmfit.conf_interval(mini, fit_output)
                    print('Confidence Intervals')
                    lmfit.printfuncs.report_ci(ci)
                else:
                    fit_output = lmfit.minimize(resid, params_lmf, method=fit_method,args=(self,),**fit_kws)
                    lmfit.report_fit(fit_output)
            else:
                fit_output = lmfit.minimize(resid, params_lmf, method=fit_method,args=(self,),**fit_kws)
        except Exception as e:
            print('Problem with fit...')
            print(e)

        ## set model params to fitted values, dump to file --------------------------------------------------------------------
        if 'fit_output' in locals():
            for x in fit_output.params:
                if x in self.params:
                    self.set_param(x, fit_output.params[x].value)
                elif 'logI_0' in fit_output.params:
                    self.set_I0(fit_output.params['logI_0'].value)
                elif 'I0' in fit_output.params:
                    logI_0 = np.log10(fit_output.params['I0'])
                    self.set_I0(logI_0)                    
            # refresh base prams from any changed sim params
            # note that user must be careful if optimizing on both base params and sim params.
            # refresh_base() will convert sim params to base, and use them for current values.
            self.refresh_base() 

            # bundle fit results
            self.fit_output = fit_output
            all_params = {'params':self.params, 
                          'sbparams':self.sbparams,
                          'fbparams':self.fbparams,
                          'cbparams':self.cbparams,
                          'dbparams':self.dbparams,
                          'initial_values':self.initial_values 
            }
            self.all_params = copy.deepcopy(all_params)

            ## dump new fitted values.
            self.dumpparams()
        else:
            print('Problem with fit, model params not changed')



    def slidefitplot(self,figsize = (15,15),**myparams):
        """
        perform plot of confirmed cases and deaths with current values of slider parameters
        stored in teh dictionary myparams
        note currently deaths are here magnified by x10
        """
        country = myparams['country']
        data_src = myparams['data_src']
        if self.country != country or self.data_src != data_src:
            self.country = country
            self.data_src = data_src
            self.setup_data(country,data_src)

        param_class = self.param_class
        for pm in myparams:
            if (pm is 'param_class') or (pm is 'figsize') or (pm is 'country') or (pm is 'data_src'):
                continue
            if pm is 'logI_0':
                self.set_I0(myparams[pm])
            else:
                if param_class == 'ode':
                    if pm not in self.params:
                        print('Error:  this',self.modelname,'does not have ode parameter',pm)
                        return
                    else:
                        self.set_param(pm,myparams[pm])
                elif param_class == 'base':
                    if pm not in list(self.sbparams) + list(self.cbparams) + list(self.fbparams):
                        print('Error:  this',self.modelname,'does not have base parameter',pm)
                        return
                    else:
                        self.set_base_param(pm,myparams[pm])
                # print('new parameters',self.model.parameters)
        self.solveplot(species=['deaths','confirmed','caution_fraction','economy'],mag = {'deaths':10},
                       datasets=['deaths_corrected_smoothed','confirmed_corrected_smoothed'],age_groups=self.age_structure,figsize = figsize)


class Scan(ModelFit):
    def __init__(self,*,countries,scanplot=True,params_init_min_max,**kwargs):
        super().__init__(**kwargs)
        cnt=0
        # max_rows = 2   # for short test...
        
        self.countries = countries
        self.scanplot = scanplot
        self.params_init_min_max = params_init_min_max
        self.scan_params = {}
        self.scan_fitdata = {}
        self.run_id = self.run_id+'_scan'
        #for idx, country  in enumerate(short_countries):

    def scan(self):
        start = time()
        cnt=0
        max_cols=8
        max_rows=int(len(self.countries)/max_cols) + 1
        if max_rows==1:
            max_rows = 2
        if self.scanplot:
            fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(24,2.6*max_rows))

        for idx, country  in enumerate(self.countries):
            print(idx,country)
            row = idx // max_cols
            col = idx % max_cols
            ###############################################
            ## do the fit
            try:
                super().setup_data(country,self.data_src)
                super().fit(self.params_init_min_max,fit_method='leastsq',diag=False,fit_targets=['deaths'],fit_data=['deaths_corrected_smoothed'],report=False)
                if self.scanplot:
                    super().solveplot(species=['deaths'],datasets=['deaths_corrected_smoothed'],axis=axes[row,col],newplot=False)  
                self.scan_params[country] = copy.deepcopy(self.all_params)
                self.scan_fitdata[country] = self.fit_output
            except Exception as e:
                print('Problem...')
                print(sys.exc_info()[0])
                print(e)
            if self.scanplot:
                axes[row,col].set_title(country)
            cnt = cnt+1
            #if cnt==15:   # for short test
            #    break
            ###############################################

        if self.scanplot:
            for idx in range(cnt,max_rows*max_cols):
                row = idx // max_cols
                col = idx % max_cols
                axes[row, col].axis("off")
            fig.tight_layout()
            plt.savefig('./pdfs/'+self.run_id+'.pdf')
            plt.show()
        self.dump()
        finish = time()
        print('Total elapsed time for ',len(self.countries),'countries:',finish-start)
        print('Time per country:',float(finish-start)/len(self.countries))

    def dump(self):
        filename = './pks/'+self.run_id+'.pk'
        scan_all = {}
        scan_all['params'] = self.scan_params
        scan_all['fitdata'] = self.scan_fitdata
        with open(filename,'bw') as fp:
            pk.dump(scan_all,fp)


class SliderFit(ModelFit):
    """
    derived class to add sliders.
    Usage mode:
    * adjust sliders
    * call fit() to fit, starting at slider values.
    """
    global sim_param_inits

    def __init__(self,*,params_init_min_max=None,basedata=None,model=None,datatypes='all',fit_targets=['confirmed','deaths'],
                 startdate=None,stopdate=None,simdays=None,new=True,fit_method='leastsq',
                 modelnames_widget=None,
                 modelage_widget=None,
                 countries_widget=None,
                 datasrcs_widget=None,
                 paramtypes_widget=None,
                 runid_widget=None,
                 modify_cur=False,**kwargs):

        if basedata is None:
            print("SliderFit Error: basedata cannot be None")
            return

        ###########################################
        ## set widget defaults:

        chosen_model = 'SC3FUEI3R'
        chosen_country = 'Australia'
        chosen_paramtype = 'base'
        chosen_age = 1
        chosen_data_src = 'jhu'
        chosen_run_id = 'test0'

        countries_common = basedata.countries_common
        paramtypes = ['base','ode']
        datasrcs = ['jhu','owid']
        agegroups = [1,4,8,16]
        
        if  modelnames_widget is None:
            modelnames_widget = Dropdown(options=possmodels,description='model',layout={'width': 'max-content'},value=chosen_model)
        else:
            self.modelnames_widget = modelnames_widget
        if  modelage_widget is None:
            modelage_widget = Dropdown(options=agegroups,description='age grps',layout={'width': 'max-content'},value=chosen_age)
        else:
            self.modelage_widget = modelage_widget
        if  countries_widget is None:
            countries_widget = Dropdown(options=countries_common,description='countries',layout={'width': 'max-content'},value=chosen_country)
        else:
            self.countries_widget = countries_widget
        if  datasrcs_widget is None:
            datasrcs_widget = RadioButtons(options=datasrcs,value='jhu',description='data src',disabled=False,layout={'width': 'max-content'}) 
        else:
            self.datasrcs_widget = datasrcs_widget
        if  paramtypes_widget is None:
            paramtypes_widget = Dropdown(options=paramtypes,description='param class',style={'description_width': 'initial'}, layout={'width': 'max-content'},value=chosen_paramtype)
        else:
            self.paramtypes_widget = paramtypes_widget
        if  runid_widget is None:
            runid_widget = Text(value='First up',placeholder='Enter run id',description='Run_id:',disabled=False)
        else:
            self.runid_widget = runid_widget

        ###########################################
        ## Call base class __init__

        # for reference:  __init__ call of base ModelFit class:
        # def __init__(self,modelname,basedata=None,model=None,country='',run_id='',datatypes='all',fit_targets=['confirmed','deaths'],
        #              data_src='owid',startdate=None,stopdate=None,simdays=None,new=True,fit_method='leastsq',param_class='base',
        #              countries_widget=fixed('United Kingdom'),datasrcs_widget=fixed('jhu')):
        super().__init__(basedata = basedata,
                         modelname = self.modelnames_widget.value,
                         country = self.countries_widget.value,
                         data_src = self.datasrcs_widget.value,
                         param_class = self.paramtypes_widget.value,
                         run_id = self.runid_widget.value,
                         **kwargs) # **kwargs passes all the rest                         
                         
                         
        cnt=0
        # max_rows = 2   # for short test...
        if params_init_min_max == None:
            # grab defaults
            if self.param_class == 'ode':
                self.params_init_min_max = sim_param_inits[self.modelname]
            elif self.param_class == 'base':
                self.params_init_min_max = default_base_params(modelname=self.modelname)
        else:
            self.params_init_min_max = params_init_min_max

        self.slidedict = {}     # will be set by allsliderparams()
        self.checkdict = {}

        #############################################
        ## set observes for the widgets
        self.modelnames_widget.observe(self.on_param_change,names='value')
        self.modelage_widget.observe(self.on_param_change,names='value')
        self.countries_widget.observe(self.on_param_change,names='value')
        self.datasrcs_widget.observe(self.on_param_change,names='value')
        self.paramtypes_widget.observe(self.on_param_change,names='value')
        self.runid_widget.observe(self.on_param_change,names='value')

#        if not modify_cur:
        self.makeslbox(modify_cur)

    def on_param_change(self,change):
        """
        modelnames_widget = Dropdown(options=possmodels,description='model',layout={'width': 'max-content'},value=chosen_model)
        modelage_widget = Dropdown(options=agegroups,description='age grps',layout={'width': 'max-content'},value=chosen_age)
        countries_widget = Dropdown(options=countries_common,description='countries',layout={'width': 'max-content'},value=chosen_country)
        paramtypes_widget = Dropdown(options=paramtypes,description='param class',style={'description_width': 'initial'}, layout={'width': 'max-content'},value=chosen_paramtype)
        runid_widget = Text(value='First up',placeholder='Enter run id',description='Run_id:',disabled=False)
        datasrcs_widget = RadioButtons(options=datasrcs,value='jhu',description='data src',disabled=False,layout={'width': 'max-content'}) 
        """
        global agemodels
        widg = change['owner']
        val = change['new']
        widg_desc =widg.description
        if self is None:
            print('Error, on_param_change called with self None')
            return
                                            # first set values to those in the structure
        modelname = self.modelname
        agestructure = self.agestructure
        country = self.country
        data_src = self.data_src
        param_class = self.param_class
        run_id = self.run_id
                                            # update the value that changed from the widget
        if widg_desc == 'model':
            modelname = val
        elif widg_desc == 'age grps':
            agestructure = val
        elif widg_desc == 'countries':
            country = val
        elif widg_desc == 'data src':
            data_src = val
        elif widg_desc == 'param class':
            param_class = val
        elif widg_desc == 'Run_id:':
            run_id = val
                                            # now construct full model name from base name and agestructure
        if int(agestructure) > 1 and modelname in agemodels :   # modelname value from widget
            modelname_a = modelname+'_A'+str(agestructure)
        elif int(agestructure) > 1:  # age structure not yet implemented for this model type
            modelname_a = modelname
            agestructure=1
            widg.value = agestructure   # correct value back to 1 (or None)
            # self.modelage_widget.value = agestructure # correct value back to 1 (or None)
        else:
            modelname_a = modelname

        bd = self.basedata
        if widg_desc in ['model','age grps','param class','Run_id:']:
            self.__init__(params_init_min_max=self.params_init_min_max,basedata=self.basedata,model=self.model,datatypes=self.datatypes,fit_targets=self.fit_targets,
                          startdate=self.startdate,stopdate=self.stopdate,simdays=self.simdays,new=self.new,fit_method=self.fit_method,
                          modelnames_widget=self.modelnames_widget,
                          modelage_widget=self.modelage_widget,
                          countries_widget=self.countries_widget,
                          datasrcs_widget=self.datasrcs_widget,
                          paramtypes_widget=self.paramtypes_widget,
                          runid_widget=self.runid_widget,
                          modify_cur=True  )
            self.transfer_cur_to_plot()
        elif widg_desc in ['countries','data src']:
            self.setup_data(country,data_src);
            self.transfer_cur_to_plot();

        #if not MyModel is None:
        #    print('displaying with MyModel',MyModel)
        #    display(MyModel.slbox)
        #    print('after display')

    def on_slider_param_change(self,change):
        pm = change['owner'].description
        # print('ospc:',pm)
        val = change['new']
        # print('ospc:',val)     
        self.set_param(pm,float(val))
        self.transfer_cur_to_plot();

    def transfer_cur_to_plot(self):
        if self.param_class == 'ode':
            pdic = self.odeparams
        elif self.param_class == 'base':
            pdic = self.baseparams
        x_dic = {}
        x_dic.update({'country':self.country})
        x_dic.update({'data_src':self.data_src})

        with self.slfitplot:
            self.slidefitplot(figsize=(6,6),**pdic,**x_dic);
#            self.solveplot(species=['deaths','confirmed','caution_fraction','economy'],mag = {'deaths':10},
#                           datasets=['deaths_corrected_smoothed','confirmed_corrected_smoothed'],age_groups=self.age_structure,figsize = (6,6))
            clear_output(wait=True)
            display(self.fig)
            #self.slidefitplot(figsize=(6,6),**pdic,**x_dic);

    def makeslbox(self,modify_cur):
        #################################
        ## set up widgets

        self.allsliderparams(modify_cur)  # sets self.slidedict = dictionary of sliders and self.checkdict of fixed items
        #print('sliderdict',self.slidedict.keys())
        self.slidedict.update({'param_class':fixed(self.param_class)})
        if not modify_cur:
            self.fit_button = widgets.Button(description="Fit from current params",layout=widgets.Layout(border='solid 1px'))
        fit_output_text = 'Fit output will be displayed here.'
        if modify_cur:
            self.fit_display_widget.value = fit_output_text
        else:
            self.fit_display_widget = widgets.Textarea(value=fit_output_text,disabled=False,
                                              layout = widgets.Layout(height='320px',width='520px'))
            self.fitbox = VBox([Label('Fit output data'),self.fit_display_widget])
            fittypes = ['leastsq','nelder','differential_evolution','slsqp','shgo','cobyla','lbfgsb','bfgs','basinhopping','dual_annealing']
            self.fittypes_widget = Dropdown(options=fittypes,description='fit meth',layout={'width': 'max-content'},value='leastsq')

            self.slfitplot=Output(layout=Layout(height='500px', width = '500px'))
        self.transfer_cur_to_plot();

        #slfitplot = interactive_o5688utput(self.slidefitplot,self.slidedict)   # disrupts slider value updates
        if modify_cur:
            self.sliderbox.close()
            self.sliders.close()
            self.checks.close()

        self.sliders=VBox([w1 for w1 in list(self.slidedict.values()) if isinstance(w1,Widget) and w1 != self.countries_widget and w1 != self.datasrcs_widget],
                     layout = widgets.Layout(height='400px',width='520px'))
        self.checks= VBox([w1 for w1 in list(self.checkdict.values()) if isinstance(w1,Widget)],
                     layout = widgets.Layout(height='400px',width='280px'))
        self.sliderbox = VBox([HBox([self.fit_button,self.fittypes_widget]),
                          HBox([VBox([Label('Adjustable params:'),self.sliders]),VBox([Label('Fixed/Adjustable params:'),self.checks])])])

        if modify_cur:
            self.slbox.close()
        self.slbox=HBox([self.slfitplot,self.sliderbox,self.fitbox])
        if modify_cur:
            display(self.slbox)

        # if not modify_cur:  display(self.slbox)
        #import functools
        #def on_button_clicked(b, rs_="some_default_string"):
        #    fun(rs_)
        #button.on_clicked(functools.partial(on_button_clicked, rs_="abcdefg"))

        ##############################################
        # activate click button
        def do_the_fit(b):
            try:
                old_stdout = sys.stdout
                sys.stdout = mystdout = io.StringIO()
                ## do the fit
                self.fit_display_widget.value = "Processing fit, please wait ..." #jsm
                #print("just before fit")
                self.fit()
                #print("just after fit")
                self.fit_display_widget.value = mystdout.getvalue()   #  fit_output_widget global.
            finally:
                sys.stdout = old_stdout

        self.fit_button.on_click(do_the_fit)

        ################################
        ## hook up fittypes_widget...
        def update_fittype(*args):
            self.fit_method = self.fittypes_widget.value
            do_the_fit(None)

        self.fittypes_widget.observe(update_fittype,'value')


    def allsliderparams(self,modify_cur):
        """
            construct dictionary of slider widgets corresponding to 
            input params_init_min_max is the dictionary of tuples for parameter optimization (3 or 4-tuples)
            pimm is short name for params_init_min_max
        """
        
        param_class = self.param_class
        pimm = self.params_init_min_max
        if pimm == {}:
            print('missing non empty dictionary params_init_min_max')
            return
        elif len(pimm[list(pimm.keys())[0]]) != 4:
            print('dictionary params_init_min_max must contain tuples with 4 entries (val,min,max,step)')
            return

        if modify_cur:
            slidedict = {}
            checkdict = {}
        else:
            slidedict = self.slidedict
            checkdict = self.checkdict            
        if slidedict == {}:
            slider_layout = Layout(width='400px', height='12px')
            check_layout = Layout(width='240px', height='12px')
            style = {'description_width': 'initial'}
            modelname=self.modelname
            # slidedict.update({'figsize':fixed((8,5))})

            if param_class == 'ode':
                slidedict.update({'param_class':fixed('ode')})
                for pm in pimm:
                    if ((not 'C_' in pm) or 'C' in modelname) and ((not 'k_' in pm) or 'U' in modelname) and ((not 'k_4' in pm) or '_A' in modelname) and ((not 'C_4' in pm) or 'F' in modelname):
                        slidedict.update({pm:FloatSlider(min=pimm[pm][1],max=pimm[pm][2],step=pimm[pm][3],value=pimm[pm][0],description=pm,
                                                        style=style,
                                                        layout=slider_layout,
                                                        continuous_update=False,readout_format='.3f')})
                        #slidedict[pm].observe(functools.partial(self.on_slider_param_change,pm),names='value') # this might have been an alternative
                        slidedict[pm].observe(self.on_slider_param_change,names='value')
                        checkdict.update({pm+'_fix':Checkbox(value=True,description=pm,disabled=False,layout=check_layout,style=style)})
            elif param_class == 'base':
                slidedict.update({'param_class':fixed('base')})
                for pm in pimm:
                    if ((not 'Caution' in pm) or 'C' in modelname) and ((not 'Econom' in pm) or 'U' in modelname) and ((not 'Young' in pm) or '_A' in modelname) and ((not 'Fatigue' in pm) or 'F' in modelname):
                        slidedict.update({pm:FloatSlider(min=pimm[pm][1],max=pimm[pm][2],step=pimm[pm][3],value=pimm[pm][0],description=pm,
                                                        style=style,
                                                        layout=slider_layout,
                                                        continuous_update=False,readout_format='.3f')})
                        slidedict[pm].observe(self.on_slider_param_change,names='value')
                        checkdict.update({pm+'_fix':Checkbox(value=True,description=pm,disabled=False,layout=check_layout,style=style)})
        else:
            modelname=self.modelname
            if param_class == 'ode':
                for pm in pimm:
                    if ((not 'C_' in pm) or 'C' in modelname) and ((not 'k_' in pm) or 'U' in modelname) and ((not 'k_4' in pm) or '_A' in modelname) and ((not 'C_4' in pm) or 'F' in modelname):
                        slidedict[pm].value=pimm[pm][0]
                        slidedict[pm].min=pimm[pm][1]
                        slidedict[pm].max=pimm[pm][2]
                        slidedict[pm].step=pimm[pm][3]  
            elif param_class == 'base':
                for pm in pimm:
                    if ((not 'Caution' in pm) or 'C' in modelname) and ((not 'Econom' in pm) or 'U' in modelname) and ((not 'Young' in pm) or '_A' in modelname) and ((not 'Fatigue' in pm) or 'F' in modelname):          
                        slidedict[pm].value=pimm[pm][0]
                        slidedict[pm].min=pimm[pm][1]
                        slidedict[pm].max=pimm[pm][2]
                        slidedict[pm].step=pimm[pm][3]  
        self.slidedict = slidedict
        self.checkdict = checkdict

    def transfer_cur_to_params_init(self):
        """ used to transfer current parameters as initial parameter values to an existing
            initialization structure params_init_min_max
            only those parameters in params_init_min_max will have initial values updated
        taken from ModelFit.transfer_fit_to_params_init()
        Only difference:  takes no arg, returns no value, acts on self.params_init_min_max.
        """
        plist = (self.odeparams,self.sbparams,self.cbparams,self.fbparams,self.dbparams)
        for ptype in plist:
            for p in ptype:
                curval = ptype[p]
                if p in self.params_init_min_max:
                    pv = self.params_init_min_max[p]
                    if len(pv) == 4:
                        self.params_init_min_max[p] = (curval,pv[1],pv[2],pv[3])
                    else:
                        self.params_init_min_max[p] = (curval,pv[1],pv[2])

    def transfer_cur_to_sliders(self):
        self.unobserve_sliders()
        plist = (self.odeparams,self.sbparams,self.cbparams,self.fbparams,self.dbparams)
        # eprint('in transfer_cur_to_sliders, sbparams',self.sbparams)
        for ptype in plist:
            for p in ptype.keys():
                if p in self.slidedict.keys():
                    # eprint('transferring ',p,'value was',self.slidedict[p].value,'value is',ptype[p])

                    self.slidedict[p].value = ptype[p]
        self.reobserve_sliders()


    def unobserve_sliders(self):
        plist = (self.odeparams,self.sbparams,self.cbparams,self.fbparams,self.dbparams)
        # eprint('in transfer_cur_to_sliders, sbparams',self.sbparams)
        for ptype in plist:
            for p in ptype.keys():
                if p in self.slidedict.keys():
                    # eprint('transferring ',p,'value was',self.slidedict[p].value,'value is',ptype[p])
                    self.slidedict[p].unobserve_all()

    def reobserve_sliders(self):
        plist = (self.odeparams,self.sbparams,self.cbparams,self.fbparams,self.dbparams)
        # eprint('in transfer_cur_to_sliders, sbparams',self.sbparams)
        for ptype in plist:
            for p in ptype.keys():
                if p in self.slidedict.keys():
                    # eprint('transferring ',p,'value was',self.slidedict[p].value,'value is',ptype[p])
                    self.slidedict[p].observe(self.on_slider_param_change,names='value') # restore observe for plot

    def fit(self,**kwargs):
        # print('entering fit')
        # self.checkparams
        self.transfer_cur_to_params_init()
        # eprint(self.params_init_min_max)
        super().fit(self.params_init_min_max,self.checkdict,**kwargs)
        #eprint('self',self,'base params',self.baseparams)
        #eprint('sbparams',self.sbparams)
        # print('params',self.params)
        # next line should be same as
        # self.params_init_min_max = self.transfer_fit_to_params_init(self.params_init_min_max)
        self.transfer_cur_to_params_init()
        self.transfer_cur_to_sliders() # does not redraw plot
        self.transfer_cur_to_plot();   # does redraw plot

        #eprint('after transfer')
        
