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
debug = False

# ----------------------------------------- functions for extracting and processing data ---------------------------------
covid_owid = []               # defined globally to allow access to raw data read in for owid
owid_to_jhu_str_country = {}  # defined globally for convenience in country conversions
data_days = -1
final_date = "10/09/20" # 9th October 2020 as cutoff for paper (8th October for JHU, since better sync offset by 1)
scountries = ['Australia','Denmark','France','Iran','Italy','Peru','Russia','Sweden','Spain','United Kingdom','United States']
dcountries = ['Afghanistan','Albania','Argentina','Armenia','Australia','Austria',
 'Azerbaijan','Belarus','Belgium','Bolivia','Bosnia and Herzegovina',
 'Brazil','Bulgaria','Canada','Chile','Colombia','Croatia',
 'Czech Republic','Dominican Republic','Ecuador','Egypt','El Salvador',
 'Finland','Germany','Greece','Guatemala','Honduras','Hungary','India',
 'Iran','Iraq','Ireland','Israel','Italy','Kazakhstan','Kosovo','Kuwait',
 'Kyrgyzstan','Lebanon','Luxembourg','Macedonia','Mexico','Moldova',
 'Morocco','Norway','Oman','Pakistan','Panama','Peru','Philippines',
 'Poland','Portugal','Qatar','Romania','Russia','Saudi Arabia','Serbia',
 'Slovenia','South Africa','Spain','Sweden','Switzerland','Tunisia',
 'Turkey','Ukraine','United Arab Emirates','United States']

def Float(x):
def get_data(jhu_file, lastdate=None):
def jhu_to_owid_str_country_md(countries_owid): 
def owid_to_jhu_str_country_md(countries_owid):
def owid_to_jhu_country(cc):
def notch_filter(data):

def win_clus(t,y,clusthresh):
def expand_data(covid_ts,database='jhu'):
 
def get_country_data(country_s='World', datatype='confirmed', firstdate=None, lastdate=None):
def get_country_data_nyw(country_s='World', datatype='confirmed', firstdate=None, lastdate=None):
def get_data_owid(owid_file,datatype='confirmed',dataaccum = 'cumulative',daysync = 0):
def get_data_owid_key(key, daysync = 0):
def truncx(xx,daystart,daystop):
def truncy(xx,yy,daystart,daystop):
def get_WHO_data_acute_beds():
def get_2012_data_ICUs():
def pwlf_testing(testing,trampday1=50): # reg_testing calculated from testing below : using piecewise linear approximation
def regtests(testing,country,trampday1=50):
def CaCo (Co, Nt, K=2):  # cases_actual / cases_observed given Nt=testing
def make_cases_adj_nonlin(testing,cases,K=2):

#---------------------------------------------- data extraction and processing procedure -----------------------------------------------------------
# ## JHU data
base = '../../covid-19-JH/csse_covid_19_data/csse_covid_19_time_series/'
confirmed = get_data(base+'time_series_covid19_confirmed_global.csv',final_date)
deaths = get_data(base+'time_series_covid19_deaths_global.csv',final_date)
recovered = get_data(base+'time_series_covid19_recovered_global.csv',final_date)
covid_ts = {'confirmed':confirmed,'deaths':deaths,'recovered':recovered}
countries_jhu = [cc for cc in confirmed if cc is not 'dates']
covid_ts = expand_data(covid_ts,'jhu')

# ## OWID data
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
countries_owid = [cc for cc in deaths_owid if cc is not 'dates']  
covid_owid_ts = expand_data(covid_owid_ts,'owid')

# ## WHO & icus_2012
acute_dict = get_WHO_data_acute_beds()
icu_dict = get_2012_data_ICUs()


# jhu equivalents   
jhu_to_owid_str_country=jhu_to_owid_str_country_md(countries_owid)
# owid equivalents
owid_to_jhu_str_country = owid_to_jhu_str_country_md(countries_owid)
countries_jhu_overseas= [cc for cc in countries_jhu if '_Overseas' in cc[0]]
countries_jhu_non_special = [cc for cc in countries_jhu if  cc[0] not in ['Diamond Princess', 'MS Zaandam']]
countries_jhu_4_owid = countries_jhu_non_special
countries_jhu_2_owid=[jhu_to_owid_str_country[cc[0]] for cc in countries_jhu_4_owid ]
countries_owid_to_jhu=[owid_to_jhu_country(cc) for cc in countries_jhu_2_owid]
countries_common_x = [cc for cc in countries_jhu_2_owid if cc not in ['dates','World']] + ['dates','World']
countries_common = [cc for cc in countries_common_x if cc not in ['dates','World']]

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
testing_x=get_data_owid(owid_file,datatype='new_tests_smoothed_per_thousand',dataaccum = 'daily',daysync=daysync)
testing = {cc:testing_x[cc] for cc in testing_x if cc != 'dates' and cc != 'World'}
testing_init_ramp = {cc:regtests(testing,cc,trampday1=50) for cc in testing}  # rampup testing linearly from background 0.01 to first reported value from trampday1
reg_testing=pwlf_testing(testing_init_ramp,trampday1=50)

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

