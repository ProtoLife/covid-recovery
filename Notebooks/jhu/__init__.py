import datetime
import re


from .get_data import get_data

def Float(x):
    try:
        rtn = float(x)
    except:
        rtn = float('NaN')
    return rtn 

def get_jhu_ts():
    stmp = __path__[0] # e.g. "../jhu"
    stmp = re.split('jhu',stmp)[0] # e.g. "../"
    base = stmp+'../covid-19-JH/csse_covid_19_data/csse_covid_19_time_series/'
    confirmed = get_data(base+'time_series_covid19_confirmed_global.csv')
    deaths = get_data(base+'time_series_covid19_deaths_global.csv')
    recovered = get_data(base+'time_series_covid19_recovered_global.csv')
    covid_ts = {'confirmed':confirmed,'deaths':deaths,'recovered':recovered}
    return covid_ts


def get_jhu_countries(dat):
    try:
        confirmed = dat['confirmed']
    except:
        print("arg dat must be data returned by get_jhu_ts")
        return
    countries = [(row[0],row[1]) for row in confirmed][1:]
    print("number of countries listed",len(countries_jhu))

def get_module_loc():
    print(__path__)

def get_ave_data(country_s,datatype='confirmed',dataaccum='daily_av_weekly',
                 firstdate=None, lastdate=None):
    countries = []
    if isinstance(country_s,list):
        for country in country_s:
            if isinstance(country,str):
                country = (country,'')
            countries.append(country)
    elif isinstance(country_s,str):
        countries = [(country_s,'')]
    else:                               # single ('country','reg') entry
        countries = [country_s]
    # get the jhu data
    dat = get_jhu_ts()
    popkeyed = dat[datatype]
    # accumulate the results
    res = {}
    # get x coord dates 
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
    res['dates']=xx
    # get y coord for each country:
    i=0
    j=0
    for country in countries:
        try:
            yy = popkeyed[country]
            j = j+1
        except:
            print('country not found',country)
            i = i + 1
            continue
        yyf = [Float(y) for y in yy]
        yy = yy0 + yyf + yy1
        # print('len yy',len(yy))
        # ymax=np.max(np.array(yy))
        yyf = [Float(y) for y in yy]
        if dataaccum == 'daily':
            yy = [0.]*len(yy)
            yy[0] = yyf[0]
            for k in range(1,len(yy)):
                yy[k] = yyf[k]-yyf[k-1]   
        elif dataaccum == 'cum_av_weekly':
            yy = [0.]*len(yy)
            moving_av = 0.
            for k in range(len(yy)):
                if k-7 >= 0:
                    moving_av = moving_av - yyf[k-7]
                moving_av = moving_av + yyf[k]
                yy[k] = moving_av/min(7.0,float(k+1))
        elif dataaccum == 'daily_av_weekly':
            yy = [0.]*len(yyf)
            yy[0] = yyf[0]
            for k in range(1,len(yy)):
                yy[k] = yyf[k]-yyf[k-1]
            yyf = [y for y in yy]
            yy = [0.]*len(yy)
            moving_av = 0.
            for k in range(len(yy)):
                if k-7 >= 0:
                    moving_av = moving_av - yyf[k-7]
                moving_av = moving_av + yyf[k]
                yy[k] = moving_av/min(7.0,float(k+1))
        if country[1] != '':
            country = country[0]+'_'+country[1]
        else:
            country = country[0]
        res[country] = yy
    return res
        
      
