
  allnmsdb = ['countries_common', 'countries_common_age', 'countries_common_contact', 'countries_common_x', 'countries_jhu',
 'countries_jhu_2_owid', 'countries_jhu_4_owid', 'countries_jhu_non_special', 'countries_jhu_overseas', 'countries_nopopulation',
 'countries_owid', 'countries_owid_to_jhu', 'countrynms', 'd_countries', 'jhu_to_owid_str_country', 'owid_to_jhu_str_country',
 'scountries', 'covid_owid', 'covid_owid_ts', 'covid_ts', 'covidnms', 'contact_dic', 'age_group_dic', 'deaths', 'deaths_owid',
 'new_deaths_c_spm_jhu', 'new_deaths_c_spm_owid', 'new_deaths_pm_jhu', 'new_deaths_pm_owid', 'new_deaths_spm_jhu',
 'new_deaths_spm_owid', 'total_deaths_cs_jhu', 'total_deaths_cs_owid', 'total_deaths_jhu', 'total_deaths_owid',
 'total_deaths_s_jhu', 'total_deaths_s_owid', 'cases_adj_nonlin_jhu', 'cases_adj_nonlin_owid', 'cases_adj_nonlinr_jhu',
 'cases_adj_nonlinr_owid', 'cases_c_linr_jhu', 'cases_c_linr_owid', 'cases_c_nonlin_jhu', 'cases_c_nonlin_owid',
 'cases_c_nonlinr_jhu', 'cases_c_nonlinr_owid', 'new_cases_c_linr_jhu', 'new_cases_c_linr_owid', 'new_cases_c_linr_spm_jhu',
 'new_cases_c_linr_spm_owid', 'new_cases_c_nonlin_jhu', 'new_cases_c_nonlin_owid', 'new_cases_c_nonlin_spm_jhu',
 'new_cases_c_nonlin_spm_owid', 'new_cases_c_nonlinr_jhu', 'new_cases_c_nonlinr_owid', 'new_cases_c_nonlinr_spm_jhu',
 'new_cases_c_nonlinr_spm_owid', 'new_cases_c_spm_jhu', 'new_cases_c_spm_owid', 'new_cases_pm_jhu', 'new_cases_pm_owid',
 'new_cases_spm_jhu', 'new_cases_spm_owid', 'countries_nopopulation', 'population_density_owid', 'population_owid', 'allnmsdb']
for x in allnmsdb:
    stmp = 'global ' + x
    exec(stmp)