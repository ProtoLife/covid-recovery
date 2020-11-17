# configuration for data loading
data_loaded = False
cluster_data_loaded = False
report_correct = True
database = 'JHU'
daysync = 23
thresh = 10   # better to use day when #total_deaths (ie cumulative) absolute first reaches 10 or perhaps 30 absolute as sync point & keep entire rest of trace
mindays = 200 
mindeaths = 200 # was 100
mindeathspm = 0.1  # was 0.5
syncat = 'first major peak' # first daily data peak that is more than 20% of maximum
# syncat = 'first thresh'