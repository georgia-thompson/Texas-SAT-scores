# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 11:32:33 2025

@author: Georg
"""
import random
import utils
from pathlib import Path
import numpy as np
import pandas as pd
import constants
import sys
import os

parent_directory = os.getcwd()

#Clean up group_categories (combine like columns)
   # First change group_categories DF to remove duplicate rows
   # Then change average_sat_scores so taht old group_nums become new group_nums

def CleanUp(d_dfs):
    df_sat_avg = d_dfs[constants.AVERAGES]
    df_groups = d_dfs[constants.GROUPS]
    (df_groups, df_sat_avg)=CleanUpGroupDuplicates(df_groups, df_sat_avg)
    # Add more district information
    df_districts_cleaned = AddDistrictInfo(d_dfs[constants.DISTRICTS], constants.DISTRICTS_DATA_FILE)
    
    # Get Texas County Financial data
    df_counties_cleaned = AddCountyInfo(d_dfs[constants.COUNTIES], 
                                        constants.COUNTIES_DATA_FILE)
    # Grab data for all students
    df_sat_all = df_sat_avg[df_sat_avg[constants.GROUP_NUM] == 1]

    #df_sat_by_group = df_sat_avg[df_sat_avg[constants.GROUP_NUM != 1]]
    df_sat_all = CleanUpMissingSATScores(df_sat_all)
    #df_sat_by_group = CleanUpMissingSATScores(df_sat_by_group)
    d_dfs[constants.ALL_STUDENTS] = df_sat_all
    d_dfs[constants.DF_DISTRICTS_CLEANED] = df_districts_cleaned
    d_dfs[constants.DF_COUNTIES_CLEANED] = df_counties_cleaned
    #d_dfs[constants.DF_BY_GROUP] = df_sat_by_group
    
    return d_dfs

def AddDistrictInfo(districts_df, csv_file):
    new_df = pd.read_csv(csv_file)
    # only keep columns needed 
    new_df = new_df[[constants.CLM_DTYPES_DISTRICT_NUMBER,
                     constants.CLM_DTYPES_TYPE, 
                     constants.CLM_DTYPES_CHARTER]]
    
    # Drop REGION_NUMBER from districts DF since it isn't currently used
    districts_df = districts_df.drop(columns=constants.REGION_NUM)
    districts_df = districts_df.set_index(constants.DISTRICT_NUM)
    new_df = new_df.set_index(constants.CLM_DTYPES_DISTRICT_NUMBER)
    
    joined_df = districts_df.join(new_df, validate='1:1')
    return joined_df

def AddCountyInfo(counties_df, csv_file):
    new_df = pd.read_csv(csv_file)
    #texas_overall_business = new_df.loc[0, constants.CLM_COUNTIES_BUSINESSES]
    #texas_overall_employment = new_df.loc[0, constants.CLM_COUNTIES_EMPLOYMENT]
    texas_overal_weekly_wage = new_df.loc[0, constants.CLM_COUNTIES_WAGE]
    new_df[constants.CLM_COUNTIES_COUNTY_NAME] = new_df[constants.CLM_COUNTIES_COUNTY_NAME].str.upper()
    counties_df = counties_df.set_index(
        constants.CLM_COUNTIES_COUNTY_NAME)
    new_df = new_df.set_index(constants.CLM_COUNTIES_COUNTY_NAME)
    joined = counties_df.join(new_df, validate="1:1").reset_index()
    joined[constants.CLM_COUNTIES_ECONOMIC_DENSITY] = round(joined[constants.CLM_COUNTIES_BUSINESSES]/joined[constants.CLM_COUNTIES_EMPLOYMENT], 3)
    joined[constants.CLM_COUNTIES_REL_WEEKLY_SAL] = round(joined[constants.CLM_COUNTIES_WAGE] / texas_overal_weekly_wage, 3)
    joined[constants.CLM_COUNTIES_ACTIVITY_INDEX] = round(joined[constants.CLM_COUNTIES_ECONOMIC_DENSITY]*joined[constants.CLM_COUNTIES_REL_WEEKLY_SAL], 3)
    return joined.set_index(constants.COUNTY)

def GetYearsInDataset(df):
    return df[constants.CLASS].unique()

def GetDistricts(df, indices):
    districts = []
    if(len(indices) > 0):
        districts = df.loc[indices, constants.DISTRICT_NUM].unique()
    else:
        districts = df[constants.DISTRICT_NUM].unique()
    return list(districts)

def GetIndexNAN(df, column, not_null = False):
    i = None
    if(not_null):
        i = df[df[column].notna()].index
    else:
        i = df[df[column].isna()].index
    return i

def FindUniqueNumber(regions):
    region_num = np.nan
    unique_regions = set(regions)
    if(len(unique_regions) == 1):
        region_num = regions[0]
    return region_num

def CleanUpMissingSATScores(df, add_noise=False):
    years = GetYearsInDataset(df)
    #i_all_nan = GetIndexNAN(df, constants.TOTAL_SCORE)
    #i_not_nan = GetIndexNAN(df, constants.TOTAL_SCORE, not_null=True)
    districts = GetDistricts(df, [])
    
    # pull data for each district missing values
    #d_index_by_district = {}
    dfs_to_append = []
    for d_num in districts:
        # Get the NAN totals and their corresponding years and the not nan totals
        j_index_district = df[df[constants.DISTRICT_NUM] == d_num].index
        # Attempt to find region_number and county_number 
        regions = df.loc[j_index_district, constants.REGION_NUM].to_list()
        counties = df.loc[j_index_district, constants.COUNTY].tolist()
        
        nan_index = GetIndexNAN(df.loc[j_index_district], constants.TOTAL_SCORE)
        
        not_nan_index = GetIndexNAN(df.loc[j_index_district], 
                                    constants.TOTAL_SCORE, not_null=True)
        totals_not_nan = df.loc[not_nan_index, constants.TOTAL_SCORE].to_numpy().reshape(-1, 1)
        years_not_nan = df.loc[not_nan_index, constants.CLASS].to_numpy().reshape(-1,1)
    
        years_nan = df.loc[nan_index, constants.CLASS].to_numpy().reshape(-1,1)

        # if not nan < 3 then drop ALL district data from main dataframe
        if(len(totals_not_nan) < 3):
            df = df.drop(j_index_district)
            continue
        
        # Normalize years to train regression method on
        X_train_normalize = utils.NormalizeValues(years_not_nan, 
                                                  years.max(), years.min())
        
        if(add_noise):
            totals_not_nan = utils.AddNoise(totals_not_nan)
        
        # Normalize totals to train regression method on
        Y_train_normalize = utils.NormalizeValues(totals_not_nan, 
                                                  constants.MAX_SAT_SCORE, 
                                                  constants.MIN_SAT_SCORE)
        res = utils.RunRegressionMethods(X_train_normalize, Y_train_normalize, error_preface=str(d_num))
        if(res is None):
            stdout_old = sys.stdout
            log_file = open("log_file", "a")
            sys.stdout = log_file
            # It was not possible to find a good fit. 
            # Continue on to next district and deal with emptys later
            print(d_num, " Model completely failed")
            log_file.close()
            sys.stdout = stdout_old
            df = df.drop(j_index_district)
            continue
        
        # normalize known years to predict the nan totals
        X_predict_normalize = utils.NormalizeValues(
            years_nan, years.max(), years.min())
        
        # Predict SAT Total
        y = utils.FindMissingValue(X_predict_normalize, res)
        
        # Unnormalize totals
        missing_totals = utils.UnNormalizeValues(y, constants.MAX_SAT_SCORE, 
                                                 constants.MIN_SAT_SCORE)
        
        # Fill in missing total scores
        df.loc[nan_index, constants.TOTAL_SCORE] = missing_totals
        """
            handle case where there weren't some rows for some years
        """
        # Get the years that this district contains. 
        years_district = GetYearsInDataset(df.loc[j_index_district])
        # Check if district is missing any years from entire dataset
        years_missing = list(set(years) - set(years_district))
        if(len(years_missing) > 0):
            region_num = FindUniqueNumber(regions)
            county_num = FindUniqueNumber(counties)
            # the years need to be filled in. 
            x = np.array(years_missing)
            # Normalize years to predict 
            x_norm = utils.NormalizeValues(x, years.max(), years.min())
            # Predict missing totals for these years
            y_norm = utils.FindMissingValue(x_norm, res)
            # convert back to un normalized total
            y = utils.UnNormalizeValues(y_norm, constants.MAX_SAT_SCORE, constants.MIN_SAT_SCORE)
            # Create a dataframe that will be appended later (so I don't mess with indices!)

            append_df = pd.DataFrame(
                {
                    constants.DISTRICT_NUM: np.array([d_num] * len(y)),
                    constants.CLASS: years_missing, 
                    constants.GROUP_NUM: np.ones(len(y)),
                    constants.TOTAL_SCORE: y, 
                    constants.REGION_NUM: np.array([region_num]*len(y)),
                    constants.COUNTY: np.array([county_num]*len(y))
                        
                })
            # Add to list so that I can append later
            dfs_to_append.append(append_df)
    
    for dfs in dfs_to_append:
        # Append to the end of the dataframe
        df = pd.concat([df, dfs], ignore_index=True)
    return df

def FillNanforFailedTotals(df):
    # Add noise and try to run regression methods again
    df = CleanUpMissingSATScores(df, add_noise=True)
    i = GetIndexNAN(df, constants.TOTAL_SCORE)
    if(len(i) > 0):
        df = df.drop(i)
    return df

def ConsolidateGroups(df_groups, df_averages):
    """
    Consolidate Groups as the following
    Race: Minority, Non-Minority, Asian 

    Parameters
    ----------
    df_groups : TYPE
        DESCRIPTION.
    df_averages : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    return None

def CleanUpGroupDuplicates(df_groups, df_averages):
    """
    Clean up the group_categories dataframe and corresponding entries 
    in the average_sat_scores group dataframe. 
    
    Parameters
    ----------
    df_groups : Pandas Dataframe
        The groups_categories table dataframe.
    df_averages : Pandas Dataframe
        The average_sat_scores table dataframe.

    Returns
    -------
    Tuple - Dataframes (group_categories, average_sat_scores)

    """
    # combine english learners
    df_groups = ReassignGroupNum(df_groups, [16, 34, 69], 52, 
                                 new_name="ENGLISH LEARNER")
    df_averages = ReassignGroupNum(df_averages, [16, 34, 69], 52)
    
    # combine the not english learners
    df_groups = ReassignGroupNum(df_groups, [17, 35, 70], 53, 
                                 new_name="NOT ENGLISH LEARNER")
    df_averages = ReassignGroupNum(df_averages, [17, 35, 70], 53)
    
    # combine the missing english learner
    df_groups = ReassignGroupNum(df_groups, [18, 36, 71], 54, 
                                 new_name="MISSING ENGLISH LEARNER")
    df_averages = ReassignGroupNum(df_averages, [17, 35, 70], 54)
    
    # combine dyslexia
    df_groups = ReassignGroupNum(df_groups, [43], 55, 
                                 new_name="DYSLEXIA")
    df_averages = ReassignGroupNum(df_averages, [43], 55)

    df_groups = ReassignGroupNum(df_groups, [44], 56, 
                                 new_name="NOT DYSLEXIA")
    df_averages = ReassignGroupNum(df_averages, [44], 56)
    
    df_groups = ReassignGroupNum(df_groups, [45], 57, 
                                 new_name="MISSING DYSLEXIA")
    df_averages = ReassignGroupNum(df_averages, [45], 57)
    
    # combine CTE (technical concentration)
    df_groups = ReassignGroupNum(df_groups, [67], 19, new_name="CTE")
    df_averages = ReassignGroupNum(df_averages, [67], 19)
    
    df_groups = ReassignGroupNum(df_groups, [68], 20, 
                                 new_name="NOT CTE CONCENTRATOR")
    df_averages = ReassignGroupNum(df_averages, [68], 20)
    

    return(df_groups, df_averages)    

def ReassignGroupNum(df, lst_old_nums, new_num, new_name=None):
    i = df.loc[df[constants.GROUP_NUM].isin(lst_old_nums)].index
    if(new_name is not None):
        df.loc[i, constants.GROUP_NAME] = new_name
        
    lst_values = [new_num]*len(lst_old_nums)
    df[constants.GROUP_NUM] = df[constants.GROUP_NUM].replace(to_replace=lst_old_nums, 
                                              value=lst_values)
    return df
   
def GetDistrictsWithMissingScores(df):
    # Find wehre the scores are NAN
    districts = df[df[constants.RUNNING_SCORE].isna()][constants.DISTRICT_NUM].values

    return list(districts)

def PlotRandomMissingScores(df, num_plots):
    #Just plot 10 random districts to see if there is a linear relationship
    districts = GetDistrictsWithMissingScores(df)
    lst_d = random.sample(districts, num_plots)
    lst_filenames = []
    p = Path("plots")
    Path.mkdir(p, exist_ok=True)
    counter = 0
    for d in lst_d:
        df_district = df[(df[constants.DISTRICT_NUM] == d) & 
                         (df[constants.RUNNING_SCORE].notna())]
        x = df_district[constants.CLASS]
        y = df_district[constants.RUNNING_SCORE]
        str_filename = os.path.join(parent_directory, 'plots', 
                                    'score_linear_' + str(counter) + '.png')
        lst_filenames.append(str_filename)
        utils.PlotLine(x, y, "Scores Per Class", "Class", "Score", str_filename)
        counter += 1
    return lst_filenames
