# configuration for data loading
data_loaded = False
cluster_data_loaded = False
report_correct = True
database = 'JHU'
daysync = 23
thresh = 10   # better to use day when #total_deaths (ie cumulative) absolute first reaches 10 or perhaps 30 absolute as sync point & keep entire rest of trace
mindays = 150 # changed from 160 to include more countries on Sep 24
mindeaths = 100
mindeathspm = 0.5 
