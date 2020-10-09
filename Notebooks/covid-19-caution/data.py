import csv
import numpy as np
import datetime
import warnings
import math
import pwlf

# ----------------------------------------- functions for extracting and processing data ---------------------------------
covid_owid = []               # defined globally to allow access to raw data read in for owid
countries_jhu_str_total = []  # defined globally for convenience in coutnry conversions
owid_to_jhu_str_country = {}  # defined globally for convenience in coutnry conversions

def Float(x):
    try:
        rtn = float(x)
    except:
        rtn = float('NaN')
    return rtn

def get_data(jhu_file):
    dat = []
    with open(jhu_file, newline='') as csvfile:
        myreader = csv.reader(csvfile, delimiter=',')
        popdat = []
        i = 0
        for row in myreader:
            if i != 0:
                poplist = []
                j = 0
                for elt in row[:-1]: # delete last day from dated data to allow same length data as for owid 
                    if j >= 4:
                        poplist.append(int(elt))
                    elif j == 0:
                        poplist.append(elt)
                    elif j == 1:
                        poplist[0]=(elt,poplist[0]) # delete last day from dates to allow same length data as for owid 
                    j = j+1
                popdat.append(poplist)
            else:
                popdat.append(row[:-1]) 
            # print(popdat[i])
            i = i + 1;
    # dates
    popdat0=['dates']
    for elt in popdat[0][4:]:  
        popdat0.append(elt)
    popdat[0] = [pop for pop in popdat0]
    # print('popdat[0]',popdat[0])
    # totals over all countries
    totals = np.zeros(len(popdat[0])-1,dtype=int)  
    # print('debug length of totals is',len(totals))
    for row in popdat[1:]:
        totals = totals + np.array(row[1:]) 
    totals = list(np.asarray(totals))
    # print(totals)
    popkeyed = {poplist[0]: poplist[1:] for poplist in popdat} 
    popkeyed.update({'dates':popdat[0][1:]}) 
    popkeyed.update({('World',''):totals})
    # del popkeyed[('d','a')]
    # assemble totals for countries with multiple regions
    total = np.zeros(len(popkeyed['dates']),dtype=int)      
    poptotkeyed = {}
    for country,tseries in popkeyed.items():
        if country!='dates' and country[1] != '': # it seems that UK is single exception with both '' and non '' regions, UK total is then UK overseas
            countrytotal = (country[0],'Total')
            if countrytotal in poptotkeyed:
                # print(country,popkeyed[country],poptotkeyed[countrytotal])
                total = np.array(tseries)[:]+np.array(poptotkeyed[countrytotal])[:]
            else:
                total =  np.array(tseries)[:]                        
            poptotkeyed.update({countrytotal:list(total)})
    for countrytotal,tseries in poptotkeyed.items():
        total = np.array(tseries)
        popkeyed.update({countrytotal:list(total)})
    # print('First four dates:',popkeyed['dates'][0:4])
    return popkeyed

def jhu_to_owid_str_country_md(countries_owid): 
    jhu_to_owid_str_country = {}
    for cc in countries_owid:
        jhu_to_owid_str_country.update({cc:cc})
    jhu_to_owid_str_country.update({
        'Burma':'Myanmar',
        'Cabo Verde':'Cape Verde',
        'Congo (Brazzaville)':'Congo',
        'Congo (Kinshasa)':'Democratic Republic of Congo',
        'Czechia':'Czech Republic',
        'Diamond Princess':'Diamond Princess',
        'Eswatini':'Swaziland',
        'Holy See':'Vatican',
        'Korea, South':'South Korea',
        'MS Zaandam':'MS Zaandam',
        'North Macedonia':'Macedonia',
        'Taiwan*':'Taiwan',
        'Timor-Leste':'Timor',
        'US':'United States',
        'West Bank and Gaza':'Palestine',
        'dates':'dates'
    })
    return jhu_to_owid_str_country

def owid_to_jhu_str_country_md(countries_owid)
    owid_to_jhu_str_country = {}
    for cc in countries_owid:
        owid_to_jhu_str_country.update({cc:cc})
    owid_to_jhu_str_country.update({
        'Myanmar':'Burma',
        'Cape Verde':'Cabo Verde',
        'Congo':'Congo (Brazzaville)',
        'Democratic Republic of Congo':'Congo (Kinshasa)',
        'Czech Republic':'Czechia',
        'Diamond Princess':'Diamond Princess',
        'Swaziland':'Eswatini',
        'Vatican':'Holy See',
        'South Korea':'Korea, South',
        'MS Zaandam':'MS Zaandam',
        'Macedonia':'North Macedonia',
        'Taiwan':'Taiwan*',
        'Timor':'Timor-Leste',
        'United States':'US',
        'Palestine':'West Bank and Gaza',
        'dates':'dates'
    })
    return owid_to_jhu_str_country

def owid_to_jhu_country(cc):
    global countries_jhu_str_total
    global owid_to_jhu_str_country
    cc_j = owid_to_jhu_str_country[cc]
    if cc_j in countries_jhu_str_total:
        return (cc_j,'Total')
    else:
        return (cc_j,'')

def expand_data(covid_ts,database='jhu'):
    """ input time series dictionary : JHU or OWID
        expands data in the three direct cumulative raw data types 
        'deaths','confirmed','recovered'
        to both daily (new...) and smoothed daily (new_..._smoothed) types
        the former is a simple difference between successive days
        the latter is a seven day rolling average of the difference data
        in addition we create the reporting glitch corrected versions of the two new datasets above 
        i.e. new_..._corrected and new_..._corrected_smoothed
        then we produce also cumulative versions of the smoothed and corrected smoothed sets
        works for both JHU and OWID dictionaries
    """
    file =open('data_corrections_'+database+'.csv',"w+")
    file.write("dtype,country,day,mean,correction,sigma\n")

    new_covid_ts = covid_ts.copy()
    if database == 'jhu':
        basetypes = ['deaths','confirmed','recovered']
    else:
        basetypes = ['deaths','confirmed']
    for dtype in basetypes:
        data = covid_ts[dtype]
        new_dtype = 'new_'+dtype
        data_diff = {}
        n = len(data['dates'])
        for cc in data:
            if cc == 'dates':
                data_diff.update({'dates':data['dates']})
            else:
                data_cc = data[cc] 
                diff_ts = np.zeros(n,dtype=float)
                for t in range(n):
                    if t == 0:
                        diff_ts[t] = data_cc[t]
                    else:
                        diff_ts[t] = data_cc[t]-data_cc[t-1]
                data_diff.update({cc:diff_ts})
        new_covid_ts.update({new_dtype:data_diff})

        new_dtype_smoothed = new_dtype+'_smoothed'
        data_sm = {}
        for cc in data_diff:
            if cc == 'dates':
                data_sm.update({'dates':data_diff['dates']})
            else:
                data_cc = data_diff[cc] 
                sm_ts = np.zeros(n,dtype=float)
                week = 0.
                for t in range(n):
                    week = week + data_cc[t]
                    if t >= 7:
                        week = week - data_cc[t-7]
                        nt = 7.
                    else:
                        nt = float(t+1)
                    sm_ts[t] = week/nt
                data_sm.update({cc:sm_ts})        
        new_covid_ts.update({new_dtype_smoothed:data_sm})

        dtype_smoothed = dtype+'_smoothed'
        data_asm = {}
        for cc in data_sm:
            if cc == 'dates':
                data_asm.update({'dates':data_sm['dates']})
            else:
                data_cc = data_sm[cc] 
                asm_ts = np.zeros(n,dtype=float)
                sum = 0.
                for t in range(n):
                    sum = sum + data_cc[t]
                    asm_ts[t] = sum
                data_asm.update({cc:asm_ts})        
        new_covid_ts.update({dtype_smoothed:data_asm})

        new_dtype_corrected = new_dtype+'_corrected'
        data_cor = {}
        maxfactor = np.exp(0.5) # maximal exponential increase or decrease per day in data

        for cc in data_diff:
            if cc == 'dates':
                data_cor.update({'dates':data_diff['dates']})
            else:
                data_cc = data_diff[cc] 
                data_ccs = data_sm[cc] 
                cor_ts = np.zeros(n,dtype=float)
                week = 0.
                wvar = 0
                cor_ts[0] = data_cc[0]
                for t in range(1,n):
                    wvar = wvar + data_cc[t-1]*data_cc[t-1]
                    nt=min(7,t)
                    if t > 7:
                        wvar = wvar - data_cc[t-8]*data_cc[t-8]
                    mean = data_ccs[t-1]
                    var = wvar/float(nt) - mean*mean
                    if var >= 0.:
                        sigma = np.sqrt(var)
                    elif -var < 0.000001:
                        sigma = 0.
                    else:
                        print('invalid argument var to sqrt',var,'at',dtype,cc,'time',t,'nt mean val var',nt,mean,data_cc[t],var+mean*mean)
                        sigma = 0.
                    if sigma > 0.001:
                        delta = data_cc[t]-mean
                        deltas = data_ccs[t] - mean
                        if np.abs(delta) < 2.*sigma or 7*np.abs(deltas) < 2.*sigma: # no correction
                            cor_ts[t] = data_cc[t]
                        else:                   # do correction
                            if delta > 0.:
                                sign = 1
                            else:
                                sign = -1
                            corr = delta-sign*sigma
                            file.write("%s,\"%s\",%d,%f,%f,%f\n" % (dtype,cc,t,mean,corr,sigma))
                            # print('Correction',dtype,cc,'time',t,'mean',mean,'correction',corr,'sigma',sigma)
                            cor_ts[t] =  data_cc[t] - corr
                            tsum = np.sum(cor_ts[max(0,t-31):t-1])   # redistribute over previous month (could also choose 6-8 weeks)
                            if tsum > 0:
                                for t1 in range(max(0,t-31),t):
                                    cor_ts[t1] = cor_ts[t1]*(1. + corr/tsum)
                    else:
                        cor_ts[t] = data_cc[t]
                data_cor.update({cc:cor_ts})        
        new_covid_ts.update({new_dtype_corrected:data_cor})

        new_dtype_corrected_smoothed = new_dtype+'_corrected_smoothed'
        data_scm = {}
        for cc in data_cor:
            if cc == 'dates':
                data_scm.update({'dates':data_cor['dates']})
            else:
                data_cc = data_cor[cc] 
                scm_ts = np.zeros(n,dtype=float)
                week = 0.
                for t in range(n):
                    week = week + data_cc[t]
                    if t >= 7:
                        week = week - data_cc[t-7]
                        nt = 7.
                    else:
                        nt = float(t+1)
                    scm_ts[t] = week/nt
                data_scm.update({cc:scm_ts})        
        new_covid_ts.update({new_dtype_corrected_smoothed:data_scm})

        dtype_corrected_smoothed = dtype+'_corrected_smoothed'
        data_ascm = {}
        for cc in data_cor:
            if cc == 'dates':
                data_ascm.update({'dates':data_scm['dates']})
            else:
                data_cc = data_scm[cc] 
                ascm_ts = np.zeros(n,dtype=float)
                sum = 0.
                for t in range(n):
                    sum = sum + data_cc[t]
                    ascm_ts[t] = sum
                data_ascm.update({cc:ascm_ts})        
        new_covid_ts.update({dtype_corrected_smoothed:data_ascm})

    file.close()
    return new_covid_ts

# from covid_data_explore-jhu-j
def get_country_data(country_s='World', datatype='confirmed', firstdate=None, lastdate=None):
    if isinstance(country_s,str):
        country = (country_s,'')
    else:                               # single ('country','reg') entry
        country = country_s
    popkeyed = covid_ts[datatype]
    
    dates = popkeyed['dates']
    fmt = '%m/%d/%y'
    xx = [datetime.datetime.strptime(dd,fmt) for dd in dates ]
    if firstdate:
        firstdate_d = datetime.datetime.strptime(firstdate,fmt)
    else:
        firstdate_d = datetime.datetime.strptime(dates[0],fmt)
    if lastdate:
        lastdate_d = datetime.datetime.strptime(lastdate,fmt)
    else:
        lastdate_d = datetime.datetime.strptime(dates[-1],fmt)    
    daystart = (firstdate_d-xx[0]).days
    daystop = (lastdate_d-xx[-1]).days

    try:
        yy = popkeyed[country]
        # print(country)
    except:
            print('country data not found',country)
            return None,None,None
    yyf = [Float(y) for y in yy]
    
    if daystart <0:
        xx0 = [xx[0]+datetime.timedelta(days=i) for i in range(daystart,0)]
        yy0 = [0.]*(-daystart)
    else:
        xx0 = []
        yy0 = []
    if daystop > 0:
        xx1 = [xx[-1]+datetime.timedelta(days=i) for i in range(daystop)]
        yy1 = [0.]*(daystop)
    else:
        xx1 = []
        yy1 = []       
    xx = xx0 + xx + xx1
    xxf = [Float((x-firstdate_d).days) for x in xx ]
    
    yy = yy0 + yyf + yy1
    return xx,xxf,yy 


def get_country_data_nyw(country_s='World', datatype='confirmed', firstdate=None, lastdate=None):
    if isinstance(country_s,str):
        country = (country_s,'')
    else:                               # single ('country','reg') entry
        country = country_s
    popkeyed = covid_ts[datatype]
    
    dates = popkeyed['dates']
    fmt = '%m/%d/%y'
    xx = [datetime.datetime.strptime(dd,fmt) for dd in dates ]
    if firstdate:
        firstdate_d = datetime.datetime.strptime(firstdate,fmt)
    else:
        firstdate_d = datetime.datetime.strptime(dates[0],fmt)
    if lastdate:
        lastdate_d = datetime.datetime.strptime(lastdate,fmt)
    else:
        lastdate_d = datetime.datetime.strptime(dates[-1],fmt)    
    daystart = (firstdate_d-xx[0]).days
    daystop = (lastdate_d-xx[-1]).days
    
    try:
        yy = popkeyed[country]
        # print(country)
    except:
            print('country data not found',country)
            return None,None      
    yyf = [Float(y) for y in yy]

    yy0 = []
    yy1 = []  
    if daystart>len(yyf):
        print('Error: start date does not overlap with available data')
        return None,None
    elif daystart>0:
        yyf = yyf[daystart:]
    elif daystart <0:
        yy0 = [0.]*(-daystart)
        
    if daystop < 0:
        yyf = yyf[:daystop]  
    elif daystop > 0:
        yy1 = [0.]*(daystop)
    yyf = yy0 + yyf + yy1
    xxf = [float(x) for x in range(len(yyf))]
    return xxf,yyf 

def get_data_owid(owid_file,datatype='confirmed',dataaccum = 'cumulative',daysync = 0):
    import numpy as np
    import datetime
    import matplotlib.dates as mdates
    global covid_owid
    with open(owid_file, 'r', newline='') as csvfile:
        myreader = csv.DictReader(csvfile,delimiter=',')
        for row in myreader:
            covid_owid.append(row)
        
    # for key in covid_owid[0].keys():   # to loop through all keys
    
    if datatype == 'confirmed':
        if dataaccum == 'cumulative':
            key = 'total_cases'
        elif dataaccum == 'weekly':
            key = 'new_cases_smoothed'
        else:
            key = 'new_cases'
    elif datatype == 'recovered':
        print('data for recovered cases not available in OWID database')
        key = None
    elif datatype == 'deaths':
        if dataaccum == 'cumulative':
            key = 'total_deaths'
        elif dataaccum == 'weekly':
            key = 'new_deaths_smoothed'
        else:
            key = 'new_deaths'
    elif datatype == 'tests':
        if dataaccum == 'cumulative':  # reporting intervals often sporadic so better to use smoothed weekly
            # key = 'total_tests'
            key = 'new_tests_smoothed'  # will adjust to cumulative below
        elif dataaccum == 'weekly':
            key = 'new_tests_smoothed'
        else:
            key = 'new_tests'          # reporting intervals often sporadic so better to use smoothed weekly
    elif datatype =='stringency':
        key = 'stringency_index'
    elif datatype == 'population':
        # print('data for population changes only slowly if at all in OWID database')
        key = 'population'
    elif datatype == 'population_density':
        # print('data for population density changes only slowly if at all in OWID database')
        key = 'population_density'
    elif datatype == 'gdp_per_capita':
        # print('data for gdp per capita changes only slowly if at all in OWID database')
        key = 'gdp_per_capita'
    elif datatype == 'recovered':
        print('data for recovered cases not available in OWID database')
        key = None
        return 
    else:
        print('data for ', datatype,'not available or not yet translated in OWID database')
        key = None
        return
   
    countries = np.unique(np.array([dd['location'] for dd in covid_owid]))
    dates = np.unique(np.array([dd['date'] for dd in covid_owid]))
    dates.sort()
    fmt = '%Y-%m-%d'
    dates_t = [datetime.datetime.strptime(dd,fmt) for dd in dates ]
    firstdate = dates[daysync]
    lastdate = dates[-1]
    firstdate_t =  dates_t[daysync]
    lastdate_t =  dates_t[-1]

    daystart = 0
    daystop = (lastdate_t-firstdate_t).days
    
    popkeyed = {country: np.zeros(daystop+1,dtype=float) for country in countries} 
    
    for dd in covid_owid:
        country = dd['location']
        day = (datetime.datetime.strptime(dd['date'],fmt)-firstdate_t).days
        popkeyed[country][day] = float(dd[key]) if not dd[key]=='' else 0.0 
        
    # popkeyed = {country: np.transpose(np.array([[dd['date'],dd[key]] for dd in covid_owid if dd['location'] == country])) for country in countries}
    # popkeyed = {country: np.array([float(dd[key]) if not dd[key]=='' else 0.0 for dd in covid_owid if dd['location'] == country]) for country in countries} 

    if datatype == 'tests' and dataaccum == 'cumulative':  # assemble cumulative tests from smooth daily tests
        for country in countries:
            data = popkeyed[country]
            sumdata= np.zeros(len(data))
            sum = 0.0
            for i,d in enumerate(data):
                sum = sum + d
                sumdata[i] = sum
            popkeyed.update({country:sumdata})

    fmt_jhu = '%-m/%-d/%y'
    popkeyed.update({'dates': [date.strftime(fmt_jhu) for date in dates_t[daysync:]]})   # dates are set to strings in jhu date format for compatibility
    return popkeyed

def get_data_owid_key(key, daysync = 0):
    """ data is synchronized to start at beginning of jhu data set"""
    global covid_owid, owid_file
    if not covid_owid:
        with open(owid_file, 'r', newline='') as csvfile:
            myreader = csv.DictReader(csvfile,delimiter=',')
            for row in myreader:
                covid_owid.append(row)
        close(owid_file)
        
    # for key in covid_owid[0].keys():   # to loop through all keys
    if key not in covid_owid[0].keys():
        print('key must be in ',covid_owid[0].keys())
        return None
   
    countries = np.unique(np.array([dd['location'] for dd in covid_owid]))
    dates = np.unique(np.array([dd['date'] for dd in covid_owid]))
    dates.sort()
    fmt = '%Y-%m-%d'
    dates_t = [datetime.datetime.strptime(dd,fmt) for dd in dates ]
    firstdate = dates[daysync]
    lastdate = dates[-1]
    firstdate_t =  dates_t[daysync]
    lastdate_t =  dates_t[-1]

    daystart = 0
    daystop = (lastdate_t-firstdate_t).days
    
    popkeyed = {country: np.zeros(daystop+1,dtype=float) for country in countries} 
    
    for dd in covid_owid:
        country = dd['location']
        day = (datetime.datetime.strptime(dd['date'],fmt)-firstdate_t).days
        popkeyed[country][day] = float(dd[key]) if not dd[key]=='' else 0.0 
        
    # popkeyed = {country: np.transpose(np.array([[dd['date'],dd[key]] for dd in covid_owid if dd['location'] == country])) for country in countries}
    # popkeyed = {country: np.array([float(dd[key]) if not dd[key]=='' else 0.0 for dd in covid_owid if dd['location'] == country]) for country in countries} 

    fmt_jhu = '%-m/%-d/%y'
    popkeyed.update({'dates': [date.strftime(fmt_jhu) for date in dates_t[daysync:]]})   # dates are set to strings in jhu date format for compatibility
    return popkeyed

def truncx(xx,daystart,daystop):
    """truncate array xx to run from daystart to daystop
       do this before trying to extend the arrays if required"""
    daymin = max(daystart,0)
    daymax = min(daystop,(xx[-1]-xx[0]).days)
    return xx[daymin:daymax+1]

def truncy(xx,yy,daystart,daystop):
    """truncate arrays xx and yy to run from daystart to daystop
       do this before trying to extend the arrays if required"""
    daymin = max(daystart,0)
    daymax = min(daystop,(xx[-1]-xx[0]).days)
    return yy[daymin:daymax+1]

def get_WHO_data_acute_beds():
    """ get acute beds data per 100000 (mostly 2014,  äITA, MKD, ROU, RUS 2013, NLD 2012)"""
    import os.path
    from os import path
    who_file = './acute_WHO_2014.csv'   # note that these are acute care beds: ICUs are estimated at only 3-5% of acute care beds
    if path.exists(who_file):
        print('WHO acute file found','dictionary acute_who')
    else:
        print('error',who_file, 'not found')
        return
    icus_data = []
    with open(who_file,'r',newline='') as fp:
        myreader = csv.reader(fp,delimiter=',')
        for i,row in enumerate(myreader):
            terms = []
            for j,elt in enumerate(row):     # assumes three rows : iso_code, country, acute_beds
                if j == 2 and i != 0:
                    terms.append(float(elt))
                else:
                    terms.append(elt)
            icus_data.append(terms)
    # close(who_file)
    # acute_dict = [{elt[1]:elt[2]} for elt in icus_data[1:]]
    acute_dict = {elt[1]:elt[2] for elt in icus_data[1:]}
    iso_dict = {elt[0]:elt[2] for elt in icus_data[1:]}
    acute_dict.update(iso_dict) # combine dictionary to access either with country name or iso code
    return acute_dict

def get_2012_data_ICUs():
    """ get ICU data 2012 from Intensive Care Med (2012) 38:1647–1653 DOI 10.1007/s00134-012-2627-8 """
    import os.path
    from os import path
    ICU_file = './acute care and ICUs.csv'   # note that these are acute care beds: ICUs are estimated at only 3-5% of acute care beds
    if path.exists(ICU_file):
        print('ICU file found','dictionary icus_2012')
    else:
        print('error',ICU_file, 'not found')
        return
    icus_data = []
    with open(ICU_file,'r',newline='') as fp:
        myreader = csv.reader(fp,delimiter=',')
        for i,row in enumerate(myreader):
            if row[0] != '':             # eliminate blank rows in ICU_file (at end)
                terms = []
                for j,elt in enumerate(row):     # assumes nine cols : with data in cols G
                    if j == 0:
                        terms.append(elt)
                    elif j == 6:
                        if i != 0:
                            terms.append(float(elt))
                        else:
                            terms.append(elt)
                icus_data.append(terms)

    # close(ICU_file)
    icu_dict = {elt[0]:elt[1] for elt in icus_data[1:]}
    return icu_dict   

def pwlf_testing(testing,trampday1=50): # reg_testing calculated from testing below : using piecewise linear approximation
    reg_testing={}
    for i,cc in enumerate(countries_common):   # was bcountries in cluster.py
        # testing_cap = np.array([max(t,0.1) for t in testing[cc]])
        testing_cap = testing[cc][trampday1:] # we assume international common starting day 50 of begin of preparation of testing (linear ramp to first recorded data) 
        xxi = range(len(testing_cap))
        xHat=np.linspace(min(xxi), max(xxi), num=len(testing_cap))
        yyf = [Float(y) for y in testing_cap]
        if i<1000:
            my_pwlf = pwlf.PiecewiseLinFit(xxi, yyf)
            res = my_pwlf.fit(2,[0.],[0.1]) # force fit to go through point (0,0.1)
            # breaks = my_pwlf.fit(2,[0.],[0.1])
            slopes = my_pwlf.calc_slopes()
            pred = my_pwlf.predict(xHat)
            yHat = np.concatenate((np.array([0.1]*trampday1),pred))
            yHat = np.array([max(t,0.1) for t in yHat])
            for i,y in enumerate(yHat):
                if i>0 and y<yHat[i-1]:
                    yHat[i]=yHat[i-1]
            reg_testing.update({cc:yHat.copy()})
    return reg_testing 

def regtests(testing,country,trampday1=50):
    """ regularize testing data by ramping up linearly to first reported testing
        from common trampday1
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
def CaCo (Co, Nt, K=2):  # cases_actual / cases_observed given Nt=testing
    K1 = 25*(K-1)/(5.0-K)
    K2 = K1/5
    if Co > 0:
        rt = 1000*Nt/Co
        return (K1+rt)/(K2+rt)
    else:
        return 1

def make_cases_adj_nonlin(testing,cases,K=2):
    cases_adj_nonlin={}
    testing_0p1_c = testing_0p1 = {cc: [0.1 if math.isnan(t) else t for t in testing[cc]] for cc in testing}
    cases_adj_nonlin = {cc:[CaCo(cases[cc][i],regtests(testing_0p1_c,cc)[i],2)*cases[cc][i] for i in range(len(cases[cc]))] for cc in cases if cc != 'dates'}
    return cases_adj_nonlin

#---------------------------------------------- data extraction and processing procedure -----------------------------------------------------------
# ## JHU data

print('getting JHU data...')

base = '../../covid-19-JH/csse_covid_19_data/csse_covid_19_time_series/'
confirmed = get_data(base+'time_series_covid19_confirmed_global.csv')
deaths = get_data(base+'time_series_covid19_deaths_global.csv')
recovered = get_data(base+'time_series_covid19_recovered_global.csv')
covid_ts = {'confirmed':confirmed,'deaths':deaths,'recovered':recovered}

print('expanding JHU data : to new (daily), 7-day rolling (smoothed), reporting glitch (corrected) and combined')
covid_ts = expand_data(covid_ts,'jhu')
print('expansion done.')

countries_jhu = [(row[0],row[1]) for row in confirmed][1:]
print("number of countries listed in JHU database",len(countries_jhu))
i=0
for country in countries_jhu:
    i = i + 1
print('done with JHU data (covid_ts dictionary keys: confirmed, deaths, recovered).')


print('getting owid data...')
daysync = 23      # needs to be same as value in Cluster.py
owid_file = '../../covid-19-owid/public/data/owid-covid-data.csv'
confirmed_owid=get_data_owid(owid_file,datatype='confirmed',dataaccum = 'cumulative',daysync=daysync)
recovered_owid = None                                                         # NB OWID database has no recovered data, substitute with JHU data!
deaths_owid=get_data_owid(owid_file,datatype='deaths',dataaccum = 'cumulative',daysync=daysync)
tests_owid=get_data_owid(owid_file,datatype='tests',dataaccum = 'cumulative',daysync=daysync)
stringency_owid=get_data_owid(owid_file,datatype='stringency',dataaccum = 'daily',daysync=daysync)
population_owid = get_data_owid(owid_file,datatype='population',dataaccum = 'daily',daysync=daysync) # NB use [-2] to get non-zero set of populations from 2nd last time point
population_density_owid = get_data_owid(owid_file,datatype='population_density',dataaccum = 'daily',daysync=daysync)
gdp_per_capita_owid = get_data_owid(owid_file,datatype='gdp_per_capita',dataaccum = 'daily',daysync=daysync)
covid_owid_ts= {'confirmed':confirmed_owid,'deaths':deaths_owid,'recovered':recovered_owid, 'tests': tests_owid , 'stringency': stringency_owid,
                 'population':population_owid,'population_density':population_density_owid,'gdp_per_capita':gdp_per_capita_owid}
countries_owid = [x for x in deaths_owid if x is not 'dates']  

print('expanding OWID data : to new (daily), 7-day rolling (smoothed), reporting glitch (corrected) and combined')
covid_owid_ts = expand_data(covid_owid_ts,'owid')
print("number of countries listed in OWID database",len(countries_owid))
print('done with OWID data (covid_owid_ts dictionary see .keys()) .')

print('mapping country names between JHU and OWID and extracting common countries...')
# jhu equivalents   
jhu_to_owid_str_country=jhu_to_owid_str_country_md(countries_owid)

# owid equivalents
owid_to_jhu_str_country = owid_to_jhu_str_country_md(countries_owid)
countries_jhu_str_total = [cc[0] for cc in countries_jhu if cc[1] == 'Total']

countries_jhu_total= [cc for cc in countries_jhu if cc[1] == 'Total']
countries_jhu_non_total = [cc for cc in countries_jhu if ((cc[0] not in countries_jhu_str_total) and (cc[0] not in ['Diamond Princess', 'MS Zaandam']))]
countries_jhu_4_owid = countries_jhu_non_total + countries_jhu_total
countries_jhu_2_owid=[jhu_to_owid_str_country[cc[0]] for cc in countries_jhu_4_owid ]
countries_owid_to_jhu=[owid_to_jhu_country(cc) for cc in countries_jhu_2_owid]

countries_common_x = [cc for cc in countries_jhu_2_owid if cc not in ['dates','World']] + ['dates','World']
countries_common = [cc for cc in countries_common_x if cc not in ['dates','World']]

print('getting ICU and acute care data icus_2012 and WHO ...')



acute_dict = get_WHO_data_acute_beds()
icu_dict = get_2012_data_ICUs()

print('extracting data sets for common countries both databases...')
# JHU
# raw
total_deaths_jhu = {cc:covid_ts['deaths'][owid_to_jhu_country(cc)] for cc in countries_common}
total_deaths_s_jhu = {cc:covid_ts['deaths_smoothed'][owid_to_jhu_country(cc)] for cc in countries_common}
total_deaths_cs_jhu = {cc:covid_ts['deaths_corrected_smoothed'][owid_to_jhu_country(cc)] for cc in countries_common}

new_deaths_pm_jhu = {cc:covid_ts['new_deaths'][owid_to_jhu_country(cc)]*1000000./population_owid[cc][-2] for cc in countries_common}
new_deaths_pm_jhu.update({'dates':covid_ts['new_deaths']['dates']})  # add dates to dictionary
new_cases_pm_jhu = {cc:covid_ts['new_confirmed'][owid_to_jhu_country(cc)]*1000000./population_owid[cc][-2] for cc in countries_common}
new_cases_pm_jhu.update({'dates':covid_ts['new_confirmed']['dates']})  # add dates to dictionary
# smoothed
new_deaths_spm_jhu = {cc:covid_ts['new_deaths_smoothed'][owid_to_jhu_country(cc)]*1000000./population_owid[cc][-2] for cc in countries_common}
new_deaths_spm_jhu.update({'dates':covid_ts['new_deaths_smoothed']['dates']})  # add dates to dictionary
new_cases_spm_jhu = {cc:covid_ts['new_confirmed_smoothed'][owid_to_jhu_country(cc)]*1000000./population_owid[cc][-2] for cc in countries_common}
new_cases_spm_jhu.update({'dates':covid_ts['new_confirmed_smoothed']['dates']})  # add dates to dictionary
# corrected smoothed
new_deaths_c_spm_jhu = {cc:covid_ts['new_deaths_corrected_smoothed'][owid_to_jhu_country(cc)]*1000000./population_owid[cc][-2] for cc in countries_common}
new_deaths_c_spm_jhu.update({'dates':covid_ts['new_deaths_corrected_smoothed']['dates']})  # add dates to dictionary
new_cases_c_spm_jhu = {cc:covid_ts['new_confirmed_corrected_smoothed'][owid_to_jhu_country(cc)]*1000000./population_owid[cc][-2] for cc in countries_common}
new_cases_c_spm_jhu.update({'dates':covid_ts['new_confirmed_corrected_smoothed']['dates']})  # add dates to dictionary

# OWID
# raw
total_deaths_owid = {cc:covid_owid_ts['deaths'][cc] for cc in countries_common}
total_deaths_s_owid = {cc:covid_owid_ts['deaths_smoothed'][cc] for cc in countries_common}
total_deaths_cs_owid = {cc:covid_owid_ts['deaths_corrected_smoothed'][cc] for cc in countries_common}

new_deaths_pm_owid = {cc:covid_owid_ts['new_deaths'][cc]*1000000./population_owid[cc][-2] for cc in countries_common}   
new_deaths_pm_owid.update({'dates':covid_owid_ts['new_deaths']['dates']})  # add dates to dictionary
new_cases_pm_owid = {cc:covid_owid_ts['new_confirmed'][cc]*1000000./population_owid[cc][-2] for cc in countries_common}
new_cases_pm_owid.update({'dates':covid_owid_ts['new_confirmed']['dates']})  # add dates to dictionary
# smoothed
new_deaths_spm_owid = {cc:covid_owid_ts['new_deaths_smoothed'][cc]*1000000./population_owid[cc][-2] for cc in countries_common}   
new_deaths_spm_owid.update({'dates':covid_owid_ts['new_deaths_smoothed']['dates']})  # add dates to dictionary
new_cases_spm_owid = {cc:covid_owid_ts['new_confirmed_smoothed'][cc]*1000000./population_owid[cc][-2] for cc in countries_common}
new_cases_spm_owid.update({'dates':covid_owid_ts['new_confirmed_smoothed']['dates']})  # add dates to dictionary
# corrected smoothed
new_deaths_c_spm_owid = {cc:covid_owid_ts['new_deaths_corrected_smoothed'][cc]*1000000./population_owid[cc][-2] for cc in countries_common}   
new_deaths_c_spm_owid.update({'dates':covid_owid_ts['new_deaths_corrected_smoothed']['dates']})  # add dates to dictionary
new_cases_c_spm_owid = {cc:covid_owid_ts['new_confirmed_corrected_smoothed'][cc]*1000000./population_owid[cc][-2] for cc in countries_common}
new_cases_c_spm_owid.update({'dates':covid_owid_ts['new_confirmed_corrected_smoothed']['dates']})  # add dates to dictionary

# common big epidemic countries (common to both jhu and owid databases)
mindeaths = 100
mindeathspm = 0.5 
bcountries_1 = [cc for cc in countries_common if (max(total_deaths_cs_jhu[cc])>=mindeaths and max(total_deaths_cs_owid[cc])>=mindeaths)]
bcountries = [cc for cc in bcountries_1 if (max(new_deaths_c_spm_jhu[cc])>=mindeathspm and max(new_deaths_c_spm_owid[cc])>=mindeathspm)]
print('No of big common countries is',len(bcountries))
print('---------------------------------')

print('extracting testing data from OWID database')
testing_x = get_data_owid_key('new_tests_smoothed_per_thousand',daysync) 
testing = {cc:testing_x[cc] for cc in testing_x if cc != 'dates' and cc != 'World'}
print('doing piecewise linear fits to testing data ... reg_testing');
warnings.simplefilter('ignore')

reg_testing=pwlf_testing(testing,trampday1=50)

# print('debugging lengths testing',len(testing['Germany']),'reg_testing',len(reg_testing['Germany']),
#      'new_cases_c_spm_jhu',len(new_cases_c_spm_jhu['Germany']),'new_cases_c_spm_owid',len(new_cases_c_spm_owid['Germany']))

# corrected adjusted (linr: corresponding to pwlf) smoothed data  : corrected for testing limitations
new_cases_c_linr_spm_jhu = {cc:new_cases_c_spm_jhu[cc]/reg_testing[cc] for cc in countries_common}
new_cases_c_linr_spm_jhu.update({'dates':new_cases_c_spm_jhu['dates']})  # add dates to dictionary

new_cases_c_linr_jhu = {cc:new_cases_c_linr_spm_jhu[cc]*population_owid[cc][-2]/1000000. for cc in countries_common}
new_cases_c_linr_jhu.update({'dates':new_cases_c_spm_jhu['dates']})  # add dates to dictionary
covid_ts.update({'confirmed_linr_corrected_smoothed':new_cases_c_linr_jhu})

cases_c_linr_jhu = {cc:np.cumsum(new_cases_c_linr_spm_jhu[cc])*population_owid[cc][-2]/1000000. for cc in countries_common} 
cases_c_linr_jhu.update({'dates':new_cases_c_linr_spm_jhu['dates']})  # add dates to dictionary
covid_ts.update({'confirmed_linr_corrected_smoothed':cases_c_linr_jhu})

new_cases_c_linr_spm_owid = {cc:new_cases_c_spm_owid[cc]/reg_testing[cc] for cc in countries_common}
new_cases_c_linr_spm_owid.update({'dates':new_cases_c_spm_owid['dates']})  # add dates to dictionary

new_cases_c_linr_owid = {cc:new_cases_c_linr_spm_owid[cc]*population_owid[cc][-2]/1000000. for cc in countries_common}
new_cases_c_linr_owid.update({'dates':new_cases_c_spm_owid['dates']})  # add dates to dictionary
covid_owid_ts.update({'confirmed_linr_corrected_smoothed':new_cases_c_linr_owid})

cases_c_linr_owid = {cc:np.cumsum(new_cases_c_linr_spm_owid[cc])*population_owid[cc][-2]/1000000. for cc in countries_common} 
cases_c_linr_owid.update({'dates':new_cases_c_linr_spm_owid['dates']})  # add dates to dictionary
covid_owid_ts.update({'confirmed_linr_corrected_smoothed':cases_c_linr_owid})

print('completed regularization of testing by pwlf and linear adjustment to confirmed cases (linr).')
print('constructing nonlinear adjustment to confirmed cases based on pwlf testing (nonlin and nonlinr ...')

cases_adj_nonlin_jhu = make_cases_adj_nonlin(testing,new_cases_c_spm_jhu,K=2)            # using testing data
new_cases_c_nonlin_spm_jhu = {cc:cases_adj_nonlin_jhu[cc] for cc in countries_common}
new_cases_c_nonlin_spm_jhu.update({'dates':new_cases_c_spm_jhu['dates']})  # add dates to dictionary
new_cases_c_nonlin_jhu = {cc:cases_adj_nonlin_jhu[cc]*population_owid[cc][-2]/1000000. for cc in countries_common} # convert from pm to real pop numbers
new_cases_c_nonlin_jhu.update({'dates':new_cases_c_spm_jhu['dates']})  # add dates to dictionary
covid_ts.update({'new_confirmed_nonlin_corrected_smoothed':new_cases_c_nonlin_jhu})

cases_c_nonlin_jhu = {cc:np.cumsum(new_cases_c_nonlin_spm_jhu[cc])*population_owid[cc][-2]/1000000. for cc in countries_common} # convert from pm to real pop numbers
cases_c_nonlin_jhu.update({'dates':new_cases_c_nonlin_spm_jhu['dates']})  # add dates to dictionary
covid_ts.update({'confirmed_nonlin_corrected_smoothed':cases_c_nonlin_jhu})

cases_adj_nonlinr_jhu = make_cases_adj_nonlin(reg_testing,new_cases_c_spm_jhu,K=2)       # using regularized testing
new_cases_c_nonlinr_spm_jhu = {cc:cases_adj_nonlinr_jhu[cc] for cc in countries_common}
new_cases_c_nonlinr_spm_jhu.update({'dates':new_cases_c_spm_jhu['dates']})  # add dates to dictionary
new_cases_c_nonlinr_jhu = {cc:cases_adj_nonlinr_jhu[cc]*population_owid[cc][-2]/1000000. for cc in countries_common} # convert from pm to real pop numbers
new_cases_c_nonlinr_jhu.update({'dates':new_cases_c_spm_jhu['dates']})  # add dates to dictionary
covid_ts.update({'new_confirmed_nonlinr_corrected_smoothed':new_cases_c_nonlinr_jhu})

cases_c_nonlinr_jhu = {cc:np.cumsum(new_cases_c_nonlinr_spm_jhu[cc])*population_owid[cc][-2]/1000000. for cc in countries_common} # convert from pm to real pop numbers
cases_c_nonlinr_jhu.update({'dates':new_cases_c_nonlinr_spm_jhu['dates']})  # add dates to dictionary
covid_ts.update({'confirmed_nonlinr_corrected_smoothed':cases_c_nonlinr_jhu})

cases_adj_nonlin_owid = make_cases_adj_nonlin(testing,new_cases_c_spm_owid,K=2)            # using testing data
new_cases_c_nonlin_spm_owid = {cc:cases_adj_nonlin_owid[cc] for cc in countries_common}
new_cases_c_nonlin_spm_owid.update({'dates':new_cases_c_spm_owid['dates']})  # add dates to dictionary
new_cases_c_nonlin_owid = {cc:cases_adj_nonlin_owid[cc]*population_owid[cc][-2]/1000000. for cc in countries_common} # convert from pm to real pop numbers
new_cases_c_nonlin_owid.update({'dates':new_cases_c_spm_owid['dates']})  # add dates to dictionary
covid_owid_ts.update({'new_confirmed_nonlin_corrected_smoothed':new_cases_c_nonlin_owid})

cases_c_nonlin_owid = {cc:np.cumsum(new_cases_c_nonlin_spm_owid[cc])*population_owid[cc][-2]/1000000. for cc in countries_common} # convert from pm to real pop numbers
cases_c_nonlin_owid.update({'dates':new_cases_c_nonlin_spm_owid['dates']})  # add dates to dictionary
covid_owid_ts.update({'confirmed_nonlin_corrected_smoothed':cases_c_nonlin_owid})

cases_adj_nonlinr_owid = make_cases_adj_nonlin(reg_testing,new_cases_c_spm_owid,K=2)       # using regularized testing
new_cases_c_nonlinr_spm_owid = {cc:cases_adj_nonlinr_owid[cc] for cc in countries_common}
new_cases_c_nonlinr_spm_owid.update({'dates':new_cases_c_spm_owid['dates']})  # add dates to dictionary
new_cases_c_nonlinr_owid = {cc:cases_adj_nonlinr_owid[cc]*population_owid[cc][-2]/1000000. for cc in countries_common} # convert from pm to real pop numbers
new_cases_c_nonlinr_owid.update({'dates':new_cases_c_spm_owid['dates']})  # add dates to dictionary
covid_owid_ts.update({'new_confirmed_nonlinr_corrected_smoothed':new_cases_c_nonlinr_owid})

cases_c_nonlinr_owid = {cc:np.cumsum(new_cases_c_nonlinr_spm_owid[cc])*population_owid[cc][-2]/1000000. for cc in countries_common} 
cases_c_nonlinr_owid.update({'dates':new_cases_c_nonlinr_spm_owid['dates']})  # add dates to dictionary
covid_owid_ts.update({'confirmed_nonlinr_corrected_smoothed':cases_c_nonlinr_owid})

print('completed nonlinear adjustment to confirmed cases.')

print('Done with data.')
print('---------------------------------')
