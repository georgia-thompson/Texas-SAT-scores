# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 11:04:20 2025

@author: Georg
"""

import getdatascripts
import texas_schools_utils
import constants
import utils
import numpy as np
import sys


d_averages_by_year = {}
d_performance_scores_by_year = {}
"""
TABLES from texas_schools schema:
| average_sat_scores    |
| box_plot_region_data  |
| counties              |
| districts             |
| group_categories      |
| regions               |
"""

def CalculatePeformanceScore(year, score):
    (q1, q2, q3) = d_performance_scores_by_year[year]
    iqr = q3 - q1
    if(score >= (q3 + 1.5*iqr)):
        return 4
    elif(q3 <= score < q3 + 1.5*iqr):
        return 3
    elif(q2 <= score < q3):
        return 2
    elif((q1 - 1.5*iqr) < score < q2):
        return 1
    elif(score <= (q1 - 1.5*iqr)):
        return 0
    else:
        return -1
    
def FindQuartileNumber(year, total):
    (q1, q2, q3) = d_averages_by_year[year]
    if (total <= q1):
        return (1)
    elif (total <= q2):
        return (2)
    elif (total <=q3):
        return (3)
    elif (total > q3):
        return (4)
    else:
        return np.nan

def IsDuplicatedByYear(df, colname):
    years = df[constants.CLASS].unique()
    for y in years:
        i = df[df[constants.CLASS] == y].index
        duplicated_series = df.loc[i, colname].duplicated()
        bDuplicated_arr = duplicated_series.unique()
        if(True in bDuplicated_arr):
            return True
        
    return False

def GetQuartileValue(df, year):
    return df.loc[df[constants.CLASS] == year, constants.QUARTILE].values[0]

def CalculateRollingScore(x, df_passed):
    # Because it's being called by a groupby the district number should already be unique
    years = sorted(x[constants.CLASS].unique())
    old_out = sys.stdout
    log_file = open("log_file.log", "a")
    sys.stdout = log_file
    print("passing dataframe ", x)
    sys.stdout = old_out
    log_file.close()

    for i in range(3, len(years)):
        # The dataframe passed in is the list of years and Quartiles 
        # sorted by each district num. So the index of a year will 
        # INDEX DISTRICT_NUM YEAR QUARTILE and this index corresponds with 
        # the main dataframe index for those values.
        # Each year will have 1 row since it is grouped and not duplicated
        current_index = x.loc[x[constants.CLASS] == years[i]].index[0]

        cy = GetQuartileValue(x, years[i])
        py = GetQuartileValue(x, years[i-1])
        py_1 = GetQuartileValue(x, years[i-2])
        py_2 = GetQuartileValue(x, years[i-3])
        
        if(np.isnan(cy) or np.isnan(py) or np.isnan(py_1) or 
           np.isnan(py_2)):
            print("I don't have scores for current year ", 
                  years[i], "values are ", cy, py, py_1, py_2)
            score = np.nan
        else:
            score = cy*.5 + py*.25 + py_1*.125 + py_2*.125
        # Apply to main dataframe
        df_passed.loc[current_index, constants.RUNNING_SCORE] = score



tables=[constants.AVERAGES, constants.GROUPS, constants.DISTRICTS, constants.REGIONS, constants.COUNTIES]

d_dfs = getdatascripts.GetDataFromDB(constants.SCHEMA, constants.DB_FILE, 
                                     tables=tables, allTables=False, 
                                     refetch=False)

d_dfs = texas_schools_utils.CleanUp(d_dfs)

df_sat_averages = d_dfs[constants.AVERAGES]
df_group_categories = d_dfs[constants.GROUPS]
df_all_scores = d_dfs[constants.ALL_STUDENTS]

# OUTPUT the cleaned districts csv
getdatascripts.ExportCSV(d_dfs[constants.DF_DISTRICTS_CLEANED], constants.DF_DISTRICTS_CLEANED, 
                         constants.CLM_DTYPES_DISTRICT_NUMBER)
# Create a new districts table 
getdatascripts.CreateCleanedTable(
    d_dfs[constants.DF_DISTRICTS_CLEANED], 
    constants.DF_DISTRICTS_CLEANED,
    constants.DB_FILE, 
    constants.CLM_DTYPES_DISTRICT_NUMBER, 
    constants.SCHEMA)

#OUTPUT the cleaned counties csv
getdatascripts.ExportCSV(
    d_dfs[constants.DF_COUNTIES_CLEANED], 
    constants.DF_COUNTIES_CLEANED, 
    constants.COUNTY)

# Create cleaned table
getdatascripts.CreateCleanedTable(
    d_dfs[constants.DF_COUNTIES_CLEANED], 
    constants.DF_COUNTIES_CLEANED, 
    constants.DB_FILE, 
    constants.COUNTY, 
    constants.SCHEMA)


# Get the averages over all districts per year
df_averages_byyear = df_all_scores.groupby(
    by=[constants.CLASS, constants.DISTRICT_NUM], 
    dropna = True)[[constants.TOTAL_SCORE]].mean().sort_values(constants.CLASS).reset_index()

if(IsDuplicatedByYear(df_averages_byyear, constants.DISTRICT_NUM)):
    sys.exit("Duplicate Districts for a year")
    
years = texas_schools_utils.GetYearsInDataset(df_averages_byyear)

for year in years:
    averages = df_averages_byyear[df_averages_byyear[constants.CLASS] == year][constants.TOTAL_SCORE].to_numpy()
    # In case there are NAN although there should not be!!
    averages = averages[np.isfinite(averages)]
    quartiles = utils.GetQuadrants(averages)
    d_averages_by_year[year] = quartiles
    
    
"""
     Go through the filtered data (df_all_students_scores) and give each row 
     a value of 1, 2, 3, 4 based on quartile data, creating a new column (QUARTILE)
     
""" 
df_all_scores[constants.QUARTILE] = df_all_scores.apply(lambda x: 
                                                         FindQuartileNumber(x[constants.CLASS], x[constants.TOTAL_SCORE]), axis=1)

# Filter out the class and quartile and district to calculate running total 
df_filtered = df_all_scores[[constants.CLASS, constants.DISTRICT_NUM, constants.QUARTILE]]

grouped_data = df_filtered.groupby(by=[constants.DISTRICT_NUM])

grouped_data.apply(lambda x: CalculateRollingScore(x, df_all_scores), include_groups=True)

df_performance_score = df_all_scores[df_all_scores[constants.CLASS] > 2013].reset_index()

# Plot some graphs to find missing scores
#plots = texas_schools_utils.PlotRandomMissingScores(df_performance_score, 5)
# Calculate peformance scores

df_performance_grouped = df_performance_score.groupby(by=[
    constants.CLASS, constants.DISTRICT_NUM], dropna=True)[[constants.RUNNING_SCORE]].mean()

years_performance = df_performance_grouped.index.get_level_values(0).unique().to_list()

for year in years_performance:
    total_scores = df_performance_grouped.loc[year, constants.RUNNING_SCORE].to_numpy()
    quartiles = utils.GetQuadrants(total_scores)
    d_performance_scores_by_year[year] = quartiles
    
df_performance_score[constants.PERFORMANCE] = df_performance_score.apply(
    lambda x: CalculatePeformanceScore(x[constants.CLASS], 
                                       x[constants.RUNNING_SCORE]), axis=1)
df_performance_score = df_performance_score.drop(columns=["index", "SCORE_ID"])

getdatascripts.ExportCSV(df_performance_score, 'all_sat_scores_cleaned', "ID")

getdatascripts.CreateCleanedTable(df_performance_score, 
                                   constants.PERFORMANCE_SCORES, 
                                   constants.DB_FILE, 
                                   "ID", constants.SCHEMA)