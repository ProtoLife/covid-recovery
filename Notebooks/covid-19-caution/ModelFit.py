import lmfit
from time import time

class ModelFit:
    """ We collect all information related to a fit between a pygom model and a set of data in this class
        It has access to the model structure and defines all required parameters and details of fit """

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

    def set_param(self,param,value):
        # print('--------------------------- new set param call ----------------------------------')
        # print('In set_param with param',param,'with value',value,'self',self)
        # print('self.model.parameters',self.model.parameters)
        # print('---------------------------------------------------------------------------------')
        plist = [p.name for p in list(self.model.param_list)]
        if param not in plist:
            print('Error:  param name',param,'is not a parameter for this',self.modelname,'model.')
        self.params[param] = value
        #tmp = {param:value}
        #self.model.parameters = tmp # pygom magic sets the right parameter in the model.parameters dictionary.
        self.model.parameters =self.params


    def set_base_param(self,param,value):
        """sets base parameter and converts to ode parameters for simulation
           note that this process is pretty inefficient, operating one by one on parameters
           initial condition logI_0 is not set by this routine
        """
        if param not in self.baseparams:
            print('Error:  param name',param,'is not a base parameter for this',self.modelname,'model.')
        if param in list(self.sbparams):
            self.sbparams[param] = value
        elif param in list(self.cbparams):
            self.cbparams[param] = value  
        elif param in list(self.fbparams):
            self.fbparams[param] = value 
        b,a,g,p,u,c,k,N,I0 = base2vectors(self.sbparams,self.cbparams,self.fbparams)
        params_in=vector2params(b,a,g,p,u,c,k,N,smodel)
        for pm in self.params:
            self.params[pm] = params_in[pm] # NB: vector2params returns all params including for U model
        self.model.parameters = self.params # pygom magic sets the right parameter in the model.parameters dictionary.

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

    def slidefitplot(self,param_class='ode',**myparams):
        """
        perform plot of confirmed cases and deaths with current values of slider parameters
        stored in teh dictionary myparams
        note currently deaths are here magnified by x10
        """
        for pm in myparams:
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
        self.solveplot(species=['deaths','confirmed'],mag = {'deaths':10},datasets=['deaths_corrected_smoothed','confirmed_corrected_smoothed'],figsize = (15,15))

    def allsliderparams(self,params_init_min_max={}):
        """
            construct dictionary of slider widgets corresponding to 
            input params_init_min_max is the dictionary of tuples for parameter optimization (3 or 4-tuples)
            pimm is short name for params_init_min_max
        """
        pimm = params_init_min_max
        if pimm == {}:
            print('missing non empty dictionary params_init_min_max')
            return
        elif len(pimm[list(pimm.keys())[0]]) != 4:
            print('dictionary params_init_min_max must contain tuples with 4 entries (val,min,max,step)')
            return
        slidedict = {}
        for pm in pimm:
            slidedict.update({pm:FloatSlider(min=pimm[pm][1],max=pimm[pm][2],step=pimm[pm][3],value=pimm[pm][0],description=pm,
                                style=style,layout=slider_layout,continuous_update=False,readout_format='.3f')})
        return slidedict 

    def transfer_fit_to_params_init(self,params_init_min_max):
        """ used to transfer current fit parameters as initial parameter values to an existing
            initialization structure params_init_min_max
            only those parameters in params_init_min_max will have initial values updated
        """
        plist = (self.params,self.sbparams,self.cbparams,self.fbparams,self.dbparams)
        for ptype in plist:
            for p in ptype:
                if p in params_init_min_max:
                    pv = params_init_min_max[p]
                    if len(pv) == 4:
                        params_init_min_max[p] = (ptype[p],pv[1],pv[2],pv[3])
                    else:
                        params_init_min_max[p] = (ptype[p],pv[1],pv[2])
        return params_init_min_max

    def solveplot(self, species=['confirmed'],summing='daily',averaging='weekly',mag = {'deaths':10},axis=None,
                   scale='linear',plottitle= '',label='',newplot = True, gbrcolors=False, figsize = None, outfile = None,datasets=['confirmed_corrected_smoothed']):
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
            ldatasets = [datasets]
        else:
            lspecies = species
            ldatasets = datasets

        dspecies = [dt if dt != 'caution_fraction' else 'stringency' for dt in lspecies]
        mags = [mag[dt] if dt in mag.keys() else 1 for dt in dspecies]

        tvec = self.tsim
        tvec1 = tvec[1:]
        if not self.data is {}:
            fitdata = np.transpose(np.array([self.data[dt] for dt in datasets]))
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
               
        smodel = self.modelname
        model = self.model

        self.soln = scipy.integrate.odeint(model.ode, model.initial_values[0], tvec[1::])
        # print('debug, calling scipy integrate on self.model with IC', model.initial_values[0])
        #Plot
        # ax = axeslist[nm]
        if axis == None: 
            ax = axis = plt.subplot(1,1,1)
        else:
            ax = axis
        if scale == 'log': #Plot on log scale
            ax.semilogy()
            ax.set_ylim([0.00000001,1.0])
            
        if summing == 'daily':
            ssoln = self.difference(self.soln)
            if not fitdata is None:
                sfit = self.difference(fitdata)
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
                    ax.plot(tvec1,suma,label=label,color='darkred')
                    fita = srfit[1::,ns]*mags[ns]/self.fbparams['FracDeathsDet']/self.population # deaths cases data, corrected by FracDeathsDet
                    ax.plot(tvecf1,fita,'o',label=label,color='red',alpha=0.2)
                else:
                    ax.plot(tvec1,suma,label=label)
            elif species == 'EI':
                ax.plot(tvec1,self.soln[:,model.ei],label=label)
                # ax.plot(tvec1,self.soln[:,model.ei],label="%s" % count)
                if 'I3' in model.modelname: 
                    plt.legend(("E","I1","I2","I3"))
                elif 'E' in model.modelname: 
                    plt.legend(("E","I"))
                else:
                    plt.legend(("I"))
            elif species == 'caution_fraction':
                #print('model name',model.modelname)
                susc = self.soln[:,model.S_c]
                suma = np.sum(self.soln[:,model.all_susceptibles],axis=1)
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
                ax.plot(tvec1,self.soln,label=label)
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
        if outfile:
            plt.savefig(outfile,bbox_inches='tight')
        self.dumpparams()       # dump every plot;  could be changed by sliders
        return

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
        if not self.data is {}:
            for i,ls in enumerate(lspecies):
                ds = ldatasets[i]
                if ls == 'confirmed':     
                    datmp = self.data[ds] # confirmed cases data, corrected by FracConfirmedDet
                    fitdata[ls] = [x/self.fbparams['FracConfirmedDet']/self.population for x in datmp]
                elif ls == 'deaths':
                    datmp = self.data[ds] # deaths cases data, corrected by FracDeathsDet
                    fitdata[ls] = [x/self.fbparams['FracDeathsDet']/self.population for x in datmp]
                else:
                    fitdata[ls] = np.array(self.data[ds])

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
        for ls in lspecies:
            if ls == 'deaths':
                slices['deaths'] = self.model.deaths
            if ls == 'confirmed':
                slices['confirmed'] = self.model.confirmed

        for ls in lspecies:
            rtn[ls] = {}
            rtn[ls]['data'] = np.array(fitdata[ls])
            rtn[ls]['soln'] = self.soln[:,slices[ls]][:,0]

            mn = min([x for x in fitdata[ls] if x>0])
            fdat = [x if x > 0 else mn for x in fitdata[ls]]
            lfdat = np.array([np.log(x) for x in fdat])

            sdata = rtn[ls]['soln']
            mn = min([x for x in sdata if x>0])
            sdat = [x if x > 0 else mn for x in sdata]
            lsdat = np.array([np.log(x) for x in sdat])
            rtn[ls]['resid'] = lsdat - lfdat
            self.logresid = [sdat,lsdat,fdat,lfdat,lsdat-lfdat]
        return rtn

    def fit(self,params_init_min_max,param_class='ode',fit_method='leastsq',fit_targets='default',fit_data='default',diag=True,report=True,conf_interval=False,fit_kws={}):
        """ fits parameters described in params_init_min_max, format 3 or 4-tuple (val,min,max,step)
            from class 'ode' or 'base', using method fit_method, and fit target quantitites fit_targets
            to data specified in fit_data, with option of diagnosis output diag
        """
        # process input parameters ------------------------------------------------------------------------------------------
        # 1. param_class
        if param_class not in ['ode','base']:
            print('parameters must be either all in class ode or base currently, not',param_class) # logI_0 is in both classes
            return
        
        # 2. fit_targets    
        if fit_targets == 'default':
            fit_targets = self.fit_targets
        elif isinstance(fit_targets, str):
            fit_targets = [fit_targets]
        if len(set(fit_targets).difference(['deaths','confirmed'])) != 0:
            fit_targets = list(set(fit_targets).intersection(['deaths','confirmed']))
            if len(fit_targets) == 0:
                fit_targets = ['deaths']
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
            print('fit_targets and fit_data must have same length',len(fit_targets),len(fit_data))
            print('proceeding with default')
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
            params_lmf.add(pp,params_init_min_max[pp][0],
                           min=params_init_min_max[pp][1],
                           max=params_init_min_max[pp][2])

        ## set initial params for fit
        for x in params_lmf:
            if x in self.params:
            # if x in list(self.model.param_list):
                self.set_param(x, params_lmf[x].value)
            elif x != 'logI_0' and x in self.baseparams:
                self.set_base_param(x, params_lmf[x].value)
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

            fittry = modelfit.solve4fit(modelfit.fit_targets,modelfit.fit_data) # use solve4fitlog to get residuals as log(soln)-log(data)
            #rmsres2 = np.sqrt(np.sum(np.square(resd)))
            #print('resid: ',rmsres2)
            fitresid = np.concatenate([fittry[fit_target]['resid'] for fit_target in modelfit.fit_targets])
            return fitresid
            # return [fittry[fit_target]['resid'] for fit_target in modelfit.fit_targets]

        def lsq(diffs):
            """
            scalar function to minimize in lmfit.minimize() specified by reduce_fcn parameter
            """
            return np.sqrt(np.square(diffs))

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
                outfit = lmfit.minimize(resid, params_lmf, method=fit_method,args=(self,),iter_cb=per_iteration,**fit_kws)
                # outfit = lmfit.minimize(resid, params_lmf, method=fit_method,args=(self,),iter_cb=per_iteration,reduce_fcn=lsq,**fit_kws)

                print('elapsed time = ',time()-start)
                lmfit.report_fit(outfit)
            elif report:
                # outfit = lmfit.minimize(resid, params_lmf, method=fit_method,args=(self,),**fit_kws)
                if (fit_method == 'leastsq') and conf_interval:
                    mini = lmfit.Minimizer(resid, params_lmf, fcn_args=(self,),**fit_kws)
                    outfit = mini.minimize()
                    lmfit.report_fit(outfit)
                    print('calculating Confidence Intervals')
                    ci = lmfit.conf_interval(mini, outfit)
                    print('Confidence Intervals')
                    lmfit.printfuncs.report_ci(ci)
                else:
                    outfit = lmfit.minimize(resid, params_lmf, method=fit_method,args=(self,),**fit_kws)
                    lmfit.report_fit(outfit)
            else:
                outfit = lmfit.minimize(resid, params_lmf, method=fit_method,args=(self,),**fit_kws)
        except Exception as e:
            print('Problem with fit...')
            print(e)

        ## set model params to fitted values, dump to file --------------------------------------------------------------------
        if 'outfit' in locals():
            for x in outfit.params:
                if x in self.params:
                    self.set_param(x, outfit.params[x].value)
                elif 'logI_0' in outfit.params:
                    self.set_I0(outfit.params['logI_0'].value)
                elif 'I0' in outfit.params:
                    logI_0 = np.log10(outfit.params['I0'])
                    self.set_I0(logI_0)                    
                    
            ## dump new fitted values.
            self.outfit = outfit
            self.dumpparams()
        else:
            print('Problem with fit, model params not changed')


    def __init__(self,modelname,model=None,country='Germany',run_id='',datatypes='all',data_src='owid',startdate=None,stopdate=None,simdays=None,new=False):
        """
        if run_id is '', self.run_id takes a default value of default_run_id = modelname+'_'+country
        if run_id is not '', it is used as self.run_id, used in turn for param filename.
        except that if run_id starts with character '_', it is appended to the default run_id,
        i.e. if run_id[0]=='_': self.run_id = default_run_id+run_id 
        """
        global make_model,covid_ts,covid_owid_ts
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
        self.modelname = modelname
        if model:
            self.model = model
            if self.model.modelname != modelname:
                print("warning:  changing model from",modelname,'to',self.model.modelname)
                self.modelname = modelname
        else:
            #model_d = make_model(modelname)                # I still prefer this I think, but 
            model_d = copy.deepcopy(fullmodels[modelname])  # should avoid modifying fullmodels at all from fits, otherwise never clear what parameters are
            self.model = model_d['model']
            if new:
                    #print('using default set of parameters for model type',modelname)
                    self.params   = model_d['params']
                    self.cbparams = model_d['cbparams']
                    self.sbparams = model_d['sbparams']
                    self.fbparams = model_d['fbparams']
                    self.dbparams = model_d['dbparams']
                    self.initial_values = model_d['initial_values']
            else:
                if not self.loadparams(self.run_id):
                    #print('Problem loading paramfile for',run_id,'... using default set of parameters for model type',modelname)
                    self.params   = model_d['params']
                    self.cbparams = model_d['cbparams']
                    self.sbparams = model_d['sbparams']
                    self.fbparams = model_d['fbparams']
                    self.dbparams = model_d['dbparams']
                    self.initial_values = model_d['initial_values']
                    
        self.baseparams = list(self.sbparams)+list(self.cbparams)+list(self.fbparams)
        # set up data and times for simulation
        if data_src == 'jhu':
            ts = covid_ts
        elif data_src == 'owid':
            ts = covid_owid_ts
        else:
            print('data_src',data_src,'not yet hooked up: OWID data used instead')
            ts = covid_owid_ts

        self.country_str = country_str = country
        if data_src == 'owid':
            self.country = country
        elif data_src == 'jhu':
            if country in ['Australia', 'China', 'Denmark', 'France', 'Netherlands', 'United Kingdom']:
                self.country = country = (self.country_str,'Total')
            else:
                self.country = country = (self.country_str,'')
        else:
            print('data_src not yet implemented, using default owid')
            data_src = 'owid'
            self.country = country

        self.population = population_owid[self.country_str][-2] # -2 seems to get all countries population (no zeros)

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
                datatypes = ['confirmed','deaths','tests', 'stringency','deaths_corrected_smoothed','confirmed_corrected_smoothed','new_deaths_corrected_smoothed','new_confirmed_corrected_smoothed']
            else:
                datatypes = ['confirmed','deaths','recovered','deaths_corrected_smoothed','confirmed_corrected_smoothed','recovered_corrected_smoothed','new_deaths_corrected_smoothed','new_confirmed_corrected_smoothed','new_recovered_corrected_smoothed']
        self.data = {}
        for dt in datatypes:
            self.data.update({dt:ts[dt][country][daystart:datadays]}) 

        self.startdate = startdate_t.strftime(fmt_jhu)
        self.stopdate = stopdate_t.strftime(fmt_jhu)

        self.fit_targets = ['deaths']
        self.fit_data = 'default'

