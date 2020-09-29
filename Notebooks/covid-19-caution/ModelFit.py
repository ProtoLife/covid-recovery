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
            print('dumped params to',pfile)
        except:
            print('problem dumping params to ',pfile)
    def loadparams(self,run_id=''): 
        """loads params from same file.  returns None if any problem finding the file.
        This stuff needs modules os, sys, pickle as pk.
        If run_id is nonempty, it is used to construct the filename, and self.run_id is set to its value."""
        if run_id == '':
            run_id = self.run_id
        else:
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
            print("no file available with this run_id: ",pfile)
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

    def get_fitdata(self,species=['deaths']):
        if not isinstance(species,list):
            lspecies = [species]
        else:
            lspecies = species
        # 
        tvec = self.tsim
        tvec1 = tvec[1:]
        fitdata = {}
        if not self.data is {}:
            for ls in lspecies:
                if ls == 'deaths':
                    datmp = self.data[ls] # confirmed cases data, corrected by FracConfirmedDet
                    fitdata[ls] = [x/self.fbparams['FracConfirmedDet']/self.population for x in datmp]
                else:
                    fitdata[ls] = np.array(self.data[ls])

        else:
            print('missing fit data')
            for ls in lspecies:
                fitdata[ls] = None
        return fitdata

    def solvefit(self,species = ['deaths']):
        fitdata = self.get_fitdata(species)
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
            rtn[ls]['resid'] = rtn[ls]['soln']-rtn[ls]['data']

        return rtn


    def solveplot(self, species=['confirmed'],summing='daily',averaging='weekly',mag = {'deaths':10},axes=None,
                   scale='linear',plottitle= '',label='',newplot = True, gbrcolors=False, figsize = None, outfile = None):
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

        self.soln = scipy.integrate.odeint(model.ode, model.initial_values[0], tvec[1::])
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
        print('=============',self.run_id)
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
                    print('using default set of parameters for model type',modelname)
                    self.params   = model_d['params']
                    self.cbparams = model_d['cbparams']
                    self.sbparams = model_d['sbparams']
                    self.fbparams = model_d['fbparams']
                    self.dbparams = model_d['dbparams']
                    self.initial_values = model_d['initial_values']
            else:
                if not self.loadparams(self.run_id):
                    print('Problem loading paramfile for',run_id,'... using default set of parameters for model type',modelname)
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

