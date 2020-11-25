import csv
import numpy as np
import datetime
import warnings
import math
import pwlf
from scipy import stats
from tqdm import tqdm, tqdm_notebook  # progress bars
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from time import time
import pickle as pk

debug = False
# import data_config # not needed here now

# ----------------------------------------- functions for extracting and processing data ---------------------------------
covid_owid = []               # defined globally to allow access to raw data read in for owid
owid_to_jhu_str_country = {}  # defined globally for convenience in country conversions
continent = {}
continents = []
data_days = -1
final_date = "11/14/20" # 14th November earlier 27th October 2020 as cutoff for paper (26th October for JHU, since better sync offset by 1)
daystop = None # will be specified in get_data_owid

def Float(x):
    try:
        rtn = float(x)
    except:
        rtn = float('NaN')
    return rtn

def dic_invert(d):
    inv = {}
    for k, v in d.items():
        if isinstance(v,list):
            for vv in v:
                keys = inv.setdefault(vv, [])
                keys.append(k)                
        else:
            keys = inv.setdefault(v, [])
            keys.append(k)
    for k in inv:
        if len(inv[k]) == 1:
            inv[k] = inv[k][0]
    return inv

def get_data(jhu_file, lastdate=None):
    global data_days
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

    # select data only up to lastdate
    fmt = '%m/%d/%y'
    dbdates = [datetime.datetime.strptime(dd,fmt) for dd in popdat0[1:] ]
    if lastdate:
        lastdate_d = datetime.datetime.strptime(lastdate,fmt)
        # print('DEBUG',dbdates[-1],lastdate_d,lastdate_d-dbdates[-1],(lastdate_d-dbdates[-1]).days)
        if (lastdate_d-dbdates[-1]).days > 0:   # changed from <=
            print('Error: provided last date parameter after end of data in JHU database, using all')
            lastdate_d = datetime.datetime.strptime(popdat0[-1],fmt)
    else:
        lastdate_d = datetime.datetime.strptime(popdat0[-1],fmt)    
    data_days = (lastdate_d-dbdates[0]).days  # -1 corrects the last date to be the equivalent of that for the OWID database, +1 allowing for 'dates' as first elt
    days = data_days+1
    popdat[0] = [pop for pop in popdat0[0:days]] 

    # totals over all countries
    totals = np.zeros(len(popdat[0])-1,dtype=int)  
    # print('debug length of totals is',len(totals))
    for row in popdat[1:]:
        totals = totals + np.array(row[1:days]) 
    totals = list(np.asarray(totals))
    # print(totals)
    popkeyed = {poplist[0]: poplist[1:days] for poplist in popdat} 
    popkeyed.update({'dates':popdat[0][1:days]}) 
    popkeyed.update({('World',''):totals})
    # del popkeyed[('d','a')]

    # assemble totals for countries with multiple regions
    total = np.zeros(len(popkeyed['dates']),dtype=int)      
    poptotkeyed = {}                                                  # need to work with new dictionary, since changing a dictionary in a loop is a problem                            
    for country,tseries in popkeyed.items():                          # also because this enables us to assign states only like Canada, Australia and China to ''
        if country!='dates' and country[1] != '':                     # UK, France, Denmark and Netherlands have both '' (main country) and non '' territories/regions
            if (country[0],'') in popkeyed.keys():
                countrytotal = (country[0],'')                        # lump overseas terriotries together with mother country
                #countrytotal = (country[0]+'_Overseas','')            # sum overseas territories separately
            else:
                countrytotal = (country[0],'')
            # print('country',country,countrytotal)
            if countrytotal in poptotkeyed:
                # print(country,popkeyed[country],poptotkeyed[countrytotal])
                total[:] = np.array(tseries)[:]+np.array(poptotkeyed[countrytotal])[:]
            else:
                total[:] =  np.array(tseries)[:]                        
            poptotkeyed.update({countrytotal:list(total)})
    for countrytotal,tseries in poptotkeyed.items():
        if countrytotal in popkeyed.keys():                           # add to mother country
            total = np.array(tseries)+ np.array(popkeyed[countrytotal])
        else:                                                         # use amalgamation as whole country 
            total = np.array(tseries)
        popkeyed.update({countrytotal:list(total)})
    # remove regions/states to preserve only countries
    countrylist = list(popkeyed.keys())
    for country in countrylist:
        if country != 'dates' and country[1] not in ['',' ']:
            del popkeyed[country]
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

def owid_to_jhu_str_country_md(countries_owid):
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
    global owid_to_jhu_str_country
    cc_j = owid_to_jhu_str_country[cc]
    return (cc_j,'')

def notch_filter(data):
     #pulse = np.array([100 if ((i % 7 == 0 ) or (i % 7 == 2)) else 0 for i in range(259)])
     # fftdat1 = np.fft.rfft(pulse,n=259)
     # fftpow1 = np.square(np.abs(fftdat1))
     #plt.plot(10+np.array(range(len(fftpow1)-10)),fftpow1[10:])
     
     fftdat = np.fft.rfft(data_cc,n=259) # last axis by default
     for k in [37,74,111]: # frequencies 259/7=37 and multiples inside frequency interval 259/2 
         fftdat[k]= (fftdat[k-1]+fftdat[k+1])/2
     nfft = len(fftdat)
     fftpow = np.square(np.abs(fftdat))

     maxarg = np.argmax(fftpow[10:])
     print(ccs,dtype,'maximum frequency component at',maxarg+10,'in vector of length',nfft)
     plt.plot(10+np.array(range(len(fftpow)-10)),fftpow[10:])
     plt.show()

     smoothed = np.fft.irfft(fftdat,n=259)
     plt.plot(data_cc)
     plt.plot(smoothed)
     plt.show()

scountries = ['Australia','Denmark','France','Iran','Italy','Peru','Russia','Sweden','Spain','United Kingdom','United States']
d_countries = ['Afghanistan' 'Albania' 'Argentina' 'Armenia' 'Australia' 'Austria'
 'Azerbaijan' 'Bahamas' 'Belarus' 'Belgium' 'Bolivia'
 'Bosnia and Herzegovina' 'Brazil' 'Bulgaria' 'Canada' 'Chile' 'Colombia'
 'Croatia' 'Czech Republic' 'Denmark' 'Dominican Republic' 'Ecuador'
 'Egypt' 'El Salvador' 'Finland' 'France' 'Germany' 'Greece' 'Guatemala'
 'Honduras' 'Hungary' 'India' 'Iran' 'Iraq' 'Ireland' 'Israel' 'Italy'
 'Kazakhstan' 'Kosovo' 'Kuwait' 'Kyrgyzstan' 'Lebanon' 'Luxembourg'
 'Macedonia' 'Mexico' 'Moldova' 'Morocco' 'Netherlands' 'Norway' 'Oman'
 'Panama' 'Paraguay' 'Peru' 'Philippines' 'Poland' 'Portugal' 'Qatar'
 'Romania' 'Russia' 'Saudi Arabia' 'Serbia' 'Slovenia' 'South Africa'
 'Spain' 'Sweden' 'Switzerland' 'Tunisia' 'Turkey' 'Ukraine'
 'United Arab Emirates' 'United Kingdom' 'United States']    # somewhat longer list of 72 used at one stage in cluster analysis

def win_clus(t,y,clusthresh):
        # An Efficient Method for Detection of Outliers in Tracer Curves Derived from Dynamic Contrast-Enhanced Imaging Linning Ye And Zujun Hou
        # Mathematical Problems in Engineering Volume 2018 ID 5670165 https://doi.org/10.1155/2018/5670165
    if np.sum(y) < 0.001:     # don't need to do analysis
        return 0.
    l = len(t)
    if l != len(y):
        print('Error in win_clus lengths do not match')
        return
    if l %2 != 1:
        print('Error in win_clus window length not odd')
        return 
    m = (l-1)//2
    delta = np.abs((y[1]-y[0])/(t[1]-t[0]))
    clus = [[[t[1]],[y[1]],delta]]                           # first cluster initially containing one point
    nc = 0
    for i in range(2,l):
        cand=[]
        for j,c in enumerate(clus):                           # examine all existing clusters
            tn = c[0][-1]
            yn = c[1][-1]
            delta = np.abs((y[i]-yn)/(t[i]-tn))
            if np.abs(delta-c[2]) < clusthresh:               # point could be entered into this cluster
                cand.append((j,delta,t[i]-tn))
        if cand != []:
            dt = np.array([cd[2] for cd in cand])
            k = np.argmin(dt)
            j = cand[k][0]
            clus[j][0].append(t[i])
            clus[j][1].append(y[i])
            l1 = len(clus[j][0])
            clus[j][2] = (clus[j][2]*l1 + cand[k][1])/(l1+1)  # update average delta
        else:                                                 # new cluster
            delta = np.abs((y[i]-y[i-1])/(t[i]-t[i-1]))
            clus.append([[t[i]],[y[i]],delta])
    cluslens = np.array([len(c[0]) for c in clus])
    clusfitj = np.argmax(cluslens)                            # index of largest cluster
    clusfit = clus[clusfitj]
    clusfitnp = [np.array(clusfit[0]),np.array(clusfit[1]),clusfit[2]]
    # print('In win_clus, nr clus, cluster lengths,clusfit',len(clus),cluslens,clusfitnp)
    tc = clusfit[0]
    yc = clusfit[1]
    
    if t[m] > clusfit[0][0] and t[m]<clusfit[0][-1] and len(yc) >= 3:    
        sl, y0, r, p, se = stats.linregress(tc,yc)                # regression fit to largest cluster data
        yp = y0+t[m]*sl                                           # regression line value at central pt                         
        return yp
    else:
        return y[m]


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
    global debug

    file =open('data_corrections_'+database+'.csv',"w+")
    file.write("dtype,country,reason,day,y[t],yp,delta,sigma\n")

    new_covid_ts = covid_ts.copy()
    if database == 'jhu':
        # basetypes = ['deaths','confirmed','recovered']
        basetypes = ['deaths','confirmed']
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
                # print('debug',cc,'n',n,'len(data_cc)',len(data_cc))
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
        times = np.array(range(0,n),float)
        times7 = np.array(range(0,n,7),float)
        ntimes = np.array(range(0,n),int)
        ntimes7 = np.array(range(0,n,7),int)
        for cc in tqdm_notebook(data_diff, desc='report correction '+dtype ): # loop with progress bar instead of just data_diff                                                                   # 
        # for cc in data_diff:
            if cc == 'dates':
                data_cor.update({'dates':data_diff['dates']})
            else:
                # ccs = cc if database == 'owid' else cc[0] 
                ccs = cc if database == 'owid' else cc    # now only for common countries in string rep
                data_cc = data_diff[cc] 
                if debug and (ccs not in scountries):   # d_countries for longer list
                    data_cor.update({cc:data_cc[:]}) 
                else:
                    float_formatter = "{:.3f}".format
                    np.set_printoptions(formatter={'float_kind':float_formatter})
                    data_cc = data_diff[cc] 
                    data_ccs = data_sm[cc] 
                    n7 = n//7+1
                    data_ccw = np.zeros(n7,dtype=float)
                    for t in ntimes7:
                        t1 = t//7
                        data_ccw[t1] = data_ccs[t]*7
                    data_ccws = data_ccw.copy()                         # copy to hold glitch smoothed weekly data

                    # data_ccw_savgol = savgol_filter(data_ccw, 9, 6)   # Savitzky-Golay filter, window size 5, polynomial order 3 
                    interp = interp1d(ntimes7,data_ccw,kind='linear',fill_value="extrapolate")  # alternative kind = 'cubic'
                    data_ccwsn = interp(ntimes)/7.                      # interpolated back to daily counts

                    data_ccwl = np.log(np.maximum(data_ccw,0.)+1.)  # positive log of data : replaces exponential decay with linear
                    # print('len data_ccwl',len(data_ccwl))
                    data_ccwld = np.zeros(n7,float)
                    data_ccwld[:-1] = [np.abs((data_ccwl[i+1]-data_ccwl[i])/7.) for i in range(0,n7-1)] # forward differences
                    data_ccwld[-1] = 0.
                    if data_ccwld.any() > 0.:
                        clusthresh = np.median(np.array([d for d in data_ccwld if d>0.]))
                    else:
                        clusthresh = 0.                                 # will not be used since win_clus routine returns 0 anyway then
                    # print(ccs,'median difference is',clusthresh)
                    ypredl = data_ccwl.copy()
                    m=3
                    w = 2*m+1
                    ypredl[m:-m] = [win_clus(times7[i:i+w],data_ccwl[i:i+w],clusthresh) for i in range(n7-w+1)]
                    ypred = np.exp(ypredl)-1.
                   
                    for t in range(2,n7):                               # speed up by ignoring correction to first 7 pts with too little data
                        nt = min(4,t)
                        y = data_ccw[t-nt:t]                            # rather than use data_cc we may use the corrected values to avoid glitch errors

                        #nft= float(nt)                                 # linear regression for rolling window - earlier attempt
                        #ne = max(nft-2,1)                              # two points give no deviation
                        #x = times7[t-nt:t]                             # t-nt up to and including t-1  
                        #sl, y0, r, p, se = stats.linregress(x,y)       # regression fit to unsmoothed data
                        #l = np.array(y0+x*sl)                          # unsmoothed regression line pts
                        #sigma =  np.sqrt(np.sum(np.square(y-l)/ne))    # sigma for regression points (standard error)                                   
                        #yp = y0+times[t]*sl                            # predicted value at t from unsmoothed data

                        #yp = data_ccw_savgol[t]                        # Savitzky-Golay filter, did not work well for this data

                        yp  = ypred[t]
                        # ypm = np.mean(y)
                        # sigma = np.std(y)
                        delta = data_ccw[t]-yp  
                        rmax = 2.*(np.exp(7.*0.5)-1.)                 # integral of exp growth at 0.5 per day from 1 for a week = 64.2
                        # rmax = (np.exp(7*0.4)-1.)/0.4                   # integral of exp growth at 0.4 per day from 1 for a week = 30
                        flag = False
                        if yp < 1.:                                     # close to zero : assuming scale is in terms of absolute number of individuals
                            if data_ccw[t] > rmax:
                                flag = True
                                reason = 'explode from 0'
                            elif data_ccw[t] < 0.:
                                flag = True
                                reason = 'negative reported'
                        elif delta>0.:
                            if data_ccw[t]-yp > rmax: 
                                flag = True
                                reason = 'growth too fast'
                        elif delta<=0.:
                            if data_ccw[t] < 0:
                                flag = True
                                reason = 'negative reported'
                            elif data_ccw[t]-yp < -rmax:
                                flag = True
                                reason = 'decay too fast'
                        if flag:               # try correction
                            file.write("%s,\"%s\",\"%s\",%d,%f,%f,%f\n" % (dtype,cc,reason,t,data_ccw[t],yp,delta))
                            data_ccws[t] = yp
                            if True:
                                tmin = max(0,t-4)
                                tsum = np.sum(data_ccws[tmin:t]) 
                                if tsum > 0: # redistribute over previous month proportional to counts                                     
                                    inv_tsum = 1./tsum 
                                    for t1 in range(tmin,t):
                                        data_ccws[t1] = data_ccws[t1]*(1. + delta *inv_tsum)
                                else:
                                    inv_tsum = 1./(t-tmin)# replace by linear ramp       
                                    for t1 in range(tmin,t):
                                        data_ccws[t1] = data_ccws[t1] + delta * inv_tsum                      
                    interp2 = interp1d(ntimes7,data_ccws/7.,kind='linear',fill_value="extrapolate")
                    data_ccwcs = interp2(ntimes)
                    data_cor.update({cc:data_ccwcs})
                    if debug:
                        print(ccs)                
                        fig,axes = plt.subplots(1,1,figsize=(20,10))
                        axes.plot(data_cc,alpha=0.5,label='daily')
                        axes.plot(data_ccs,alpha=0.5,label='rolling7')
                        axes.plot(data_ccwsn,alpha=0.75,label='week_intrp')
                        axes.plot(data_ccwcs,alpha=0.75,label='week_cor_intrp')
                        plt.legend()
                        plt.show()  
                        fig,axes = plt.subplots(1,1,figsize=(20,10))
                        axes.plot(data_ccw,alpha=0.5,label='smw')
                        #axes.plot(data_ccw_savgol,alpha=0.5,label='savgol')
                        axes.plot(ypred,alpha=0.5,label='spred')
                        # axes.plot(data_ccws,alpha=0.75,label='smc')
                        plt.legend()
                        plt.show()      
        new_covid_ts.update({new_dtype_corrected:data_cor})   
        data_smc = data_cor
        new_dtype_corrected_smoothed = new_dtype_corrected + '_smoothed'
        new_covid_ts.update({new_dtype_corrected_smoothed:data_smc})  # with the current glitch scheme the corrected data is already smoothed by interpolation

        dtype_corrected_smoothed = dtype+'_corrected_smoothed'
        data_asmc = {}
        for cc in data_smc:
            if cc == 'dates':
                data_asmc.update({'dates':data_smc['dates']})
            else:
                data_cc = data_smc[cc] 
                asmc_ts = np.zeros(n,dtype=float)
                sum = 0.
                for t in range(n):
                    sum = sum + data_cc[t]
                    asmc_ts[t] = sum
                data_asmc.update({cc:asmc_ts})        
        new_covid_ts.update({dtype_corrected_smoothed:data_asmc})

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

def get_data_owid(owid_file,datatype='confirmed',dataaccum = 'cumulative',daysync = 0,exclude=[]):
    # does not use lastdate, but lastdate should be chosen to have same data length as jhu data from get_data
    # using global parameter data_days set in get_data
    import numpy as np
    import datetime
    import matplotlib.dates as mdates
    global covid_owid,data_days,continents,continent,countries_in_continent,daystop

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
    elif datatype == 'new_tests_smoothed_per_thousand':
        key = 'new_tests_smoothed_per_thousand'
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
   
    countries = np.unique(np.array([dd['location'] for dd in covid_owid if dd['location'] not in exclude]))
    if not continents:
        Middle_East = ['Turkey','Syria','Lebanon','Israel','Palestine','Jordan','Iraq','Iran','Saudi Arabia','Yemen','Oman','United Arab Emirates','Qatar','Bahrain','Kuwait','Egypt','Libya']
        continent = {dd['location']:dd['continent'] for dd in covid_owid if (dd['location'] not in exclude and dd['location'] not in Middle_East)}
        continent.update({cc:'Middle_East' for cc in Middle_East if cc in countries})
        continents = list(set([continent[cc] for cc in countries if cc != 'World']))+['Middle_East']
        continent.update({cont:'World' for cont in continents})
        continent['World']='Planet'
        countries_in_continent = dic_invert(continent)
    #for cont in countries_in_continent.keys():
    #    print('continent members',cont,countries_in_continent[cont])
    # print('continents:',continents)

    dates = np.unique(np.array([dd['date'] for dd in covid_owid if dd['location'] not in exclude]))
    dates.sort()
    fmt = '%Y-%m-%d'
    dates_t = [datetime.datetime.strptime(dd,fmt) for dd in dates ]
    firstdate = dates[daysync]
    lastdate = dates[-1]
    firstdate_t =  dates_t[daysync]
    # print('debug: data_days',data_days,'len dates',len(dates_t),'daysync+data_days-1',daysync+data_days-1)
    lastdate_t =  dates_t[daysync+data_days-1] # data_days is number of days read from jhu data set
    # print('debug:    lastdate_t',   lastdate_t)
    # lastdate_t =  dates_t[-1]

    daystart = 0
    daystop = (lastdate_t-firstdate_t).days # number of day at which data stops
    
    popkeyed = {country: np.zeros(daystop+1,dtype=float) for country in countries}  # indices 0 up to daystop allowed, length daystop+1
    # print('debug','daystop',daystop,'firstdate_t',firstdate_t,'lastdate_t',lastdate_t)
    for dd in covid_owid:
        country = dd['location']
        if country not in exclude:
            day = (datetime.datetime.strptime(dd['date'],fmt)-firstdate_t).days
            if day >= 0 and day <= daystop:
                popkeyed[country][day] = float(dd[key]) if not dd[key]=='' else 0.0 
            # else:
            #    print('day out of range',country,day)

    # totals over all countries in continents+ME
    for cont in continents:
        total = np.zeros(daystop+1,dtype=float)  
        for cc in countries:
            if continent[cc] == cont:
                total = total + popkeyed[cc]
        popkeyed.update({cont:total.copy()})
        
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
    popkeyed.update({'dates': [date.strftime(fmt_jhu) for date in dates_t[daysync:daysync+data_days]]})   # dates are set to strings in jhu date format for compatibility
    return popkeyed

def get_data_owid_key(key, daysync = 0):
    """ data is synchronized to start at beginning of jhu data set"""
    global covid_owid, owid_file, data_days
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
    lastdate_t =  dates_t[daysync+data_days-1]
    # lastdate_t =  dates_t[-1]

    daystart = 0
    daystop = (lastdate_t-firstdate_t).days
    # print('debug data_days daystop+1',data_days,daystop+1)
    popkeyed = {country: np.zeros(daystop+1,dtype=float) for country in countries} 
    
    for dd in covid_owid:
        country = dd['location']
        day = (datetime.datetime.strptime(dd['date'],fmt)-firstdate_t).days
        popkeyed[country][day] = float(dd[key]) if not dd[key]=='' else 0.0 
        
    # popkeyed = {country: np.transpose(np.array([[dd['date'],dd[key]] for dd in covid_owid if dd['location'] == country])) for country in countries}
    # popkeyed = {country: np.array([float(dd[key]) if not dd[key]=='' else 0.0 for dd in covid_owid if dd['location'] == country]) for country in countries} 

    fmt_jhu = '%-m/%-d/%y'
    popkeyed.update({'dates': [date.strftime(fmt_jhu) for date in dates_t[daysync:daysync+data_days-1]]})   # dates are set to strings in jhu date format for compatibility
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
    who_file = '../../data/WHO_and_ICU/acute_WHO_2014.csv'   # note that these are acute care beds: ICUs are estimated at only 3-5% of acute care beds
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
    ICU_file = '../../data/WHO_and_ICU/acute care and ICUs.csv'   # note that these are acute care beds: ICUs are estimated at only 3-5% of acute care beds
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

def get_data_contact_matrices():
    """ get contact matrices for 152 countries from database built by extrapolation in 2017
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5609774/
    """
    global countries_common,translate_contact
    import os.path
    from os import path
    contact_file1 = '../../data/UN/contact_matrices_152_countries/MUestimates_all_locations_1.xlsx' 
    contact_file2 = '../../data/UN/contact_matrices_152_countries/MUestimates_all_locations_2.xlsm' 
    if path.exists(contact_file1) and path.exists(contact_file2):
        print('152 country contact files found 1 A-M and 2 M-Z')
    else:
        print('error',contact_file1,'or',contact_file2, 'not found')
        return
    with open(contact_file1,'rb') as fp:
        dat = pd.read_excel(io=fp,sheet_name=None)
    with open(contact_file2,'rb') as fp:
        dat2 = pd.read_excel(io=fp,sheet_name=None) 
    dat.update(dat2)

    countries_common_contact = [cc for cc in countries_common if translate_contact[cc] in dat]
    print('Of',len(countries_common),'in countries_common',len(countries_common_contact),'have contact matrices')
    # print('Exceptions:',set(countries_common)-set(countries_common_contact))
    contact_dic = {cc:np.array(dat[translate_contact[cc]]) for cc in countries_common_contact}
    contact_dic.update({'Afghanistan':contact_dic['Pakistan'],'Kosovo':contact_dic['Serbia'],
                    'Moldova': contact_dic['Romania'],'Norway':contact_dic['Sweden']}) # to complete cluster country data via close countries
    print('4 country contact matrices set equal to that of neighbour to complete cluster country set')
    print('                   Afghanistan:Pakistan','Kosovo:Serbia','Moldova:Romania','Norway:Sweden')
    countries_common_contact.extend(['Afghanistan','Kosovo','Moldova','Norway'])
    return contact_dic,countries_common_contact

def get_data_age_groups():
    """ get age groups both sexes for countries from UN database
        the country Kosovo was not available and is taken instead from CIA World Fact Book (hand entered from image)
    """
    global countries_common,translate_age
    import os.path
    from os import path
    agedist_file = '../../data/UN/age_structure/WPP2019_POP_F07_1_POPULATION_BY_AGE_BOTH_SEXES_trimmed.xlsx' 

    if path.exists(agedist_file):
        print('UN contact files found 1 and 2')
    else:
        print('error',agedist_file, 'not found')
        return
    with open(agedist_file,'rb') as fp:
        datage = pd.read_excel(io=fp,sheet_name='ESTIMATES')

    headings = list(datage.keys())
    age_array = np.zeros((21,202),np.float)   # hard code dimensions of array must match
    for i,col in enumerate(headings[2:]):
        age_array[i,:] = datage[col][:]*1000  # original data specified in thousands, convert to individuals
    age_array=np.transpose(age_array)
    age_countries = datage['Country']
    age_dic = {age_countries[j]:age_array[j] for j in range(len(age_countries))}  

    countries_common_age = [cc for cc in countries_common if translate_age[cc] in age_dic]
    print('Of',len(countries_common),'in countries_common',len(countries_common_age),'have age structure')
    # print('Exceptions:',set(countries_common)-set(countries_common_age))
    age_group_dic = {cc:np.array(age_dic[translate_age[cc]]) for cc in countries_common_age}
    print('Kosovo age structure digitized from CIA World Fact Book Image 2018 to complete cluster country set in trimmed excel file')
    return age_group_dic,countries_common_age

def pwlf_testing(testing,trampday1=50): # reg_testing calculated from testing below : using piecewise linear approximation
    reg_testing={}
    i = 0
    for cc in tqdm_notebook(countries_common, desc='piecewise linear fit'): # loop with progress bar instead of just loop
    # for cc in countries_common:
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
        i = i+1
    return reg_testing 

def regtests(testing,country,trampday1=50):
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
    testing_0p1_c = testing_0p1 = {cc: [0.1 if math.isnan(t) else t for t in testing[cc]] for cc in testing if cc != 'dates'}
    cases_adj_nonlin = {cc:np.array([CaCo(cases[cc][i],regtests(testing_0p1_c,cc)[i],2)*cases[cc][i] for i in range(len(cases[cc]))]) for cc in cases if cc != 'dates'}
    return cases_adj_nonlin

#---------------------------------------------- data extraction and processing procedure -----------------------------------------------------------
import warnings
warnings.simplefilter('error', RuntimeWarning)   # to replace warnings by errors to allow traceback

print('Getting data:')
# ## JHU data
print('getting JHU data...')

base = '../../covid-19-JH/csse_covid_19_data/csse_covid_19_time_series/'
confirmed = get_data(base+'time_series_covid19_confirmed_global.csv',final_date)
print('jhu data selected from',confirmed['dates'][0],'to',confirmed['dates'][-1])
deaths = get_data(base+'time_series_covid19_deaths_global.csv',final_date)
recovered = get_data(base+'time_series_covid19_recovered_global.csv',final_date)

countries_jhu = [cc for cc in confirmed if cc is not 'dates']

# ## OWID data
print('getting owid data...')

# daysync = 0      # needs to be same as value in Cluster.py
daysync = 23      # needs to be same as value in Cluster.py
owid_file = '../../covid-19-owid/public/data/owid-covid-data.csv'

# get population data first to exclude countries without population data
population_owid = get_data_owid(owid_file,datatype='population',dataaccum = 'daily',daysync=daysync,exclude=['International','Hong Kong','']) # NB use [-2] to get non-zero set of populations from 2nd last time point
countries_nopopulation = [cc for cc in population_owid.keys() if (cc != 'dates' and population_owid[cc][-2] == 0)]
countries_nopopulation = countries_nopopulation + ['International','Hong Kong','']
print('countries without population data excluded:',countries_nopopulation)

confirmed_owid=get_data_owid(owid_file,datatype='confirmed',dataaccum = 'cumulative',daysync=daysync,exclude=countries_nopopulation)
print('owid data selected from',confirmed_owid['dates'][0],'to',confirmed_owid['dates'][-1])
recovered_owid = None                                                         # NB OWID database has no recovered data, substitute with JHU data!
deaths_owid=get_data_owid(owid_file,datatype='deaths',dataaccum = 'cumulative',daysync=daysync,exclude=countries_nopopulation)
tests_owid=get_data_owid(owid_file,datatype='tests',dataaccum = 'cumulative',daysync=daysync,exclude=countries_nopopulation)
stringency_owid=get_data_owid(owid_file,datatype='stringency',dataaccum = 'daily',daysync=daysync,exclude=countries_nopopulation)
population_density_owid = get_data_owid(owid_file,datatype='population_density',dataaccum = 'daily',daysync=daysync,exclude=countries_nopopulation)
gdp_per_capita_owid = get_data_owid(owid_file,datatype='gdp_per_capita',dataaccum = 'daily',daysync=daysync,exclude=countries_nopopulation)

covid_owid_ts= {'confirmed':confirmed_owid,'deaths':deaths_owid,'recovered':recovered_owid, 'tests': tests_owid , 'stringency': stringency_owid,
                 'population':population_owid,'population_density':population_density_owid,'gdp_per_capita':gdp_per_capita_owid}
countries_owid = [cc for cc in deaths_owid if cc is not 'dates']  

print('expanding OWID data : to new (daily), 7-day rolling (smoothed), reporting glitch (corrected) and combined')
covid_owid_ts = expand_data(covid_owid_ts,'owid')
print("number of countries listed in OWID database",len(countries_owid))
print('done with OWID data (covid_owid_ts dictionary see .keys()) .')

# ## WHO & icus_2012
print('getting ICU and acute care data icus_2012 and WHO ...')
acute_dict = get_WHO_data_acute_beds()
icu_dict = get_2012_data_ICUs()

# ## Country mappings and common sets
print('mapping country names between JHU and OWID and extracting common countries...')

# jhu equivalents : lookup dictionary   
jhu_to_owid_str_country=jhu_to_owid_str_country_md(countries_owid)

# owid equivalents : lookup dictionary
owid_to_jhu_str_country = owid_to_jhu_str_country_md(countries_owid)

countries_jhu_overseas= [cc for cc in countries_jhu if '_Overseas' in cc[0]]
countries_jhu_non_special = [cc for cc in countries_jhu if  cc[0] not in ['Diamond Princess', 'MS Zaandam']+countries_nopopulation]
countries_jhu_4_owid = countries_jhu_non_special
countries_jhu_2_owid=[jhu_to_owid_str_country[cc[0]] for cc in countries_jhu_4_owid ]
countries_owid_to_jhu=[owid_to_jhu_country(cc) for cc in countries_jhu_2_owid]

countries_common_x = [cc for cc in countries_jhu_2_owid+continents if cc not in ['dates']] + ['dates'] # World was excluded before, now World and all continents included
countries_common = [cc for cc in countries_common_x if cc not in ['dates']] ## World was excluded before, now World and all continents included

translate_contact = {x:x for x in countries_common}
translate_contact.update({
 'Bolivia':'Bolivia (Plurinational State of',
 'Brunei':'Brunei Darussalam',
 'Cape Verde':'Cabo Verde',
 "Cote d'Ivoire":"Côte d'Ivoire",
 'Democratic Republic of Congo':'Democratic Republic of the Congo',
 'Dominica':'Dominican Republic',
 'Laos':"Lao People's Democratic Republic",
 'Iran':'Iran (Islamic Republic of)',
 'Macedonia':'TFYR of Macedonia',
 'Moldova':'Republic of Moldova',
 'Palestine':'State of Palestine',
 'Syria':'Syrian Arab Republic',  
 'Timor':'Timor-Leste',
 'Russia':'Russian Federation',
 'South Korea':'Republic of Korea',
 'Swaziland': 'Eswatini',
 'Tanzania':'United Republic of Tanzania',
 'United States':'United States of America',
 'United Kingdom':'United Kingdom of Great Britain',
 'Venezuela':'Venezuela (Bolivarian Republic ',
 'Vietnam':'Viet Nam'})

translate_age = {x:x for x in countries_common}
translate_age.update({
 'Bolivia':'Bolivia (Plurinational State of)',
 'Brunei':'Brunei Darussalam',
 'Cape Verde':'Cabo Verde',
 'Taiwan':'China, Taiwan Province of China',
 "Cote d'Ivoire":"Côte d'Ivoire",
 'Czech Republic':'Czechia',
 'Democratic Republic of Congo':'Democratic Republic of the Congo',
 'Dominica':'Dominican Republic',
 'Swaziland': 'Eswatini',
 'Laos':"Lao People's Democratic Republic",
 'Macedonia':'North Macedonia',
 'Moldova':'Republic of Moldova',
 'Iran':'Iran (Islamic Republic of)',
 'Palestine':'State of Palestine',
 'Syria':'Syrian Arab Republic',  
 'Timor':'Timor-Leste',
 'Russia':'Russian Federation',
 'South Korea':'Republic of Korea',
 'Tanzania':'United Republic of Tanzania',
 'United States':'United States of America',
 'United Kingdom':'United Kingdom',
 'Venezuela':'Venezuela (Bolivarian Republic of)',
 'Vietnam':'Viet Nam'})

# 152 Country Contact Data
print('getting 2017 contact matrix data from 152 countries ...')
contact_dic,countries_common_contact = get_data_contact_matrices()

# UN Country Age Group Data
print('getting UN all sex age group data for 2020 ...')
age_group_dic,countries_common_age= get_data_age_groups()

print('extracting data sets for common countries both databases...')

# do summations of data for continents and world for jhu
for cont in continents:
    cont_jhu = (cont,'')
    deaths[cont_jhu] =  np.zeros(daystop+1,dtype=float)
    confirmed[cont_jhu] = np.zeros(daystop+1,dtype=float)
    recovered[cont_jhu] = np.zeros(daystop+1,dtype=float)
    # print(cont,countries_in_continent[cont])
    for cc in countries_in_continent[cont]:
        cc_jhu = owid_to_jhu_country(cc)
        if cc_jhu in countries_jhu and (cont == 'World' or cc not in continents) and cc != 'World':
            deaths[cont_jhu] = deaths[cont_jhu] + deaths[cc_jhu] 
            confirmed[cont_jhu] = confirmed[cont_jhu] + confirmed[cc_jhu]
            recovered[cont_jhu] = recovered[cont_jhu] + recovered[cc_jhu]
# JHU
deaths_jhu = {cc:deaths[owid_to_jhu_country(cc)] for cc in countries_common}
deaths_jhu.update({'dates':deaths['dates']})  # add dates to dictionary
confirmed_jhu = {cc:confirmed[owid_to_jhu_country(cc)] for cc in countries_common}
confirmed_jhu.update({'dates':confirmed['dates']})  # add dates to dictionary
recovered_jhu = {cc:recovered[owid_to_jhu_country(cc)] for cc in countries_common}
recovered_jhu.update({'dates':recovered['dates']})  # add dates to dictionary

covid_ts = {'confirmed':confirmed_jhu,'deaths':deaths_jhu,'recovered':recovered_jhu}
print('expanding JHU data : to new (daily), 7-day rolling (smoothed), reporting glitch (corrected) and combined')
covid_ts = expand_data(covid_ts,'jhu')
print("number of countries listed in JHU database",len(countries_jhu))
print('done with JHU data (covid_ts dictionary keys: confirmed, deaths, recovered).')

# raw
total_deaths_jhu = {cc:covid_ts['deaths'][cc] for cc in countries_common}
total_deaths_s_jhu = {cc:covid_ts['deaths_smoothed'][cc] for cc in countries_common}
total_deaths_cs_jhu = {cc:covid_ts['deaths_corrected_smoothed'][cc] for cc in countries_common}

new_deaths_pm_jhu = {cc:covid_ts['new_deaths'][cc]*1000000./population_owid[cc][-2] for cc in countries_common}
new_deaths_pm_jhu.update({'dates':covid_ts['new_deaths']['dates']})  # add dates to dictionary
new_cases_pm_jhu = {cc:covid_ts['new_confirmed'][cc]*1000000./population_owid[cc][-2] for cc in countries_common}
new_cases_pm_jhu.update({'dates':covid_ts['new_confirmed']['dates']})  # add dates to dictionary
# smoothed
new_deaths_spm_jhu = {cc:covid_ts['new_deaths_smoothed'][cc]*1000000./population_owid[cc][-2] for cc in countries_common}
new_deaths_spm_jhu.update({'dates':covid_ts['new_deaths_smoothed']['dates']})  # add dates to dictionary
new_cases_spm_jhu = {cc:covid_ts['new_confirmed_smoothed'][cc]*1000000./population_owid[cc][-2] for cc in countries_common}
new_cases_spm_jhu.update({'dates':covid_ts['new_confirmed_smoothed']['dates']})  # add dates to dictionary
# corrected smoothed
new_deaths_c_spm_jhu = {cc:covid_ts['new_deaths_corrected_smoothed'][cc]*1000000./population_owid[cc][-2] for cc in countries_common}
new_deaths_c_spm_jhu.update({'dates':covid_ts['new_deaths_corrected_smoothed']['dates']})  # add dates to dictionary
new_cases_c_spm_jhu = {cc:covid_ts['new_confirmed_corrected_smoothed'][cc]*1000000./population_owid[cc][-2] for cc in countries_common}
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

print('extracting testing data from OWID database')
testing_x=get_data_owid(owid_file,datatype='new_tests_smoothed_per_thousand',dataaccum = 'daily',daysync=daysync, exclude=countries_nopopulation)
# testing_x = get_data_owid_key('new_tests_smoothed_per_thousand',daysync) 
testing = {cc:testing_x[cc] for cc in testing_x if cc != 'dates'}
# print("debug len testing_x['Germany'] len testing_x['dates'] ",len(testing_x['Germany']),len(testing_x['dates']))
testing_init_ramp = {cc:regtests(testing,cc,trampday1=50) for cc in testing}  # rampup testing linearly from background 0.01 to first reported value from trampday1
print('doing piecewise linear fits to testing data ... reg_testing');
warnings.simplefilter('ignore')
reg_testing=pwlf_testing(testing_init_ramp,trampday1=50)

# corrected adjusted (linr: corresponding to pwlf) smoothed data  : corrected for testing limitations
new_cases_c_linr_spm_jhu = {cc:new_cases_c_spm_jhu[cc]/reg_testing[cc] for cc in countries_common}
new_cases_c_linr_spm_jhu.update({'dates':new_cases_c_spm_jhu['dates']})  # add dates to dictionary
new_cases_c_linr_jhu = {cc:new_cases_c_linr_spm_jhu[cc]*population_owid[cc][-2]/1000000. for cc in countries_common}
new_cases_c_linr_jhu.update({'dates':new_cases_c_spm_jhu['dates']})  # add dates to dictionary
covid_ts.update({'new_confirmed_linr_corrected_smoothed':new_cases_c_linr_jhu})

cases_c_linr_jhu = {cc:np.cumsum(new_cases_c_linr_spm_jhu[cc])*population_owid[cc][-2]/1000000. for cc in countries_common} 
cases_c_linr_jhu.update({'dates':new_cases_c_linr_spm_jhu['dates']})  # add dates to dictionary
covid_ts.update({'confirmed_linr_corrected_smoothed':cases_c_linr_jhu})

new_cases_c_linr_spm_owid = {cc:new_cases_c_spm_owid[cc]/reg_testing[cc] for cc in countries_common}
new_cases_c_linr_spm_owid.update({'dates':new_cases_c_spm_owid['dates']})  # add dates to dictionary
new_cases_c_linr_owid = {cc:new_cases_c_linr_spm_owid[cc]*population_owid[cc][-2]/1000000. for cc in countries_common}
new_cases_c_linr_owid.update({'dates':new_cases_c_spm_owid['dates']})  # add dates to dictionary
covid_owid_ts.update({'new_confirmed_linr_corrected_smoothed':new_cases_c_linr_owid})

cases_c_linr_owid = {cc:np.cumsum(new_cases_c_linr_spm_owid[cc])*population_owid[cc][-2]/1000000. for cc in countries_common} 
cases_c_linr_owid.update({'dates':new_cases_c_linr_spm_owid['dates']})  # add dates to dictionary
covid_owid_ts.update({'confirmed_linr_corrected_smoothed':cases_c_linr_owid})

print('completed regularization of testing by pwlf and linear adjustment to confirmed cases (linr).')
print('constructing nonlinear adjustment to confirmed cases based on pwlf testing (nonlin and nonlinr ...')

cases_adj_nonlin_jhu = make_cases_adj_nonlin(testing,new_cases_c_spm_jhu,K=2)            # using testing data
new_cases_c_nonlin_spm_jhu = {cc:cases_adj_nonlin_jhu[cc] for cc in countries_common}
new_cases_c_nonlin_spm_jhu.update({'dates':new_cases_c_spm_jhu['dates']})  # add dates to dictionary

#for cc in countries_common:
#    try:
#        temp = cases_adj_nonlin_jhu[cc]*population_owid[cc][-2]/1000000.
#    except:
#        print('Exception at country',cc,'popln',population_owid[cc][-2],cases_adj_nonlin_jhu[cc])

new_cases_c_nonlin_jhu = {cc:cases_adj_nonlin_jhu[cc]*population_owid[cc][-2]/1000000. for cc in countries_common} # convert from pm to real pop numbers
new_cases_c_nonlin_jhu.update({'dates':new_cases_c_spm_jhu['dates']})  # add dates to dictionary
covid_ts.update({'new_confirmed_nonlin_corrected_smoothed':new_cases_c_nonlin_jhu})

cases_c_nonlin_jhu = {cc:np.cumsum(new_cases_c_nonlin_spm_jhu[cc])*population_owid[cc][-2]/1000000. for cc in countries_common} # convert from pm to real pop numbers
cases_c_nonlin_jhu.update({'dates':new_cases_c_nonlin_spm_jhu['dates']})  # add dates to dictionary
covid_ts.update({'confirmed_nonlin_corrected_smoothed':cases_c_nonlin_jhu})

# Note make_cases_adj_nonlin() returns dictionary with owid-style country names
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
