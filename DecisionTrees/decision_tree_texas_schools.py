# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 12:33:02 2025

@author: Georg
"""
import getdatascripts
import constants
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import os
import logging
import pandas as pd
import random

logger = logging.getLogger(__name__)
logging.basicConfig(filename='decision_tree_info.log', 
                    encoding='utf-8', level=logging.INFO)

import matplotlib.pyplot as plt

NCES_Districts = []
COL_Names = ['DISTRICT_TYPE_City-Large','DISTRICT_TYPE_City-Midsize',
             'DISTRICT_TYPE_City-Small','DISTRICT_TYPE_Rural-Distant',
             'DISTRICT_TYPE_Rural-Fringe','DISTRICT_TYPE_Rural-Remote',
             'DISTRICT_TYPE_Suburb-Large','DISTRICT_TYPE_Suburb-Midsize',
             'DISTRICT_TYPE_Suburb-Small','DISTRICT_TYPE_Town-Distant',
             'DISTRICT_TYPE_Town-Fringe','DISTRICT_TYPE_Town-Remote']

parent_dir = os.getcwd()

def PrepDataForTree(df, year, training=True):

    df = df[df[constants.CLASS] == year]
    df_new = pd.get_dummies(df, dtype=int)
    file_name = 'cleaned_decision_tree_' + str(year)
    
    getdatascripts.ExportCSV(df_new, file_name, '')
    if(training):
        y = df_new[constants.PERFORMANCE]
        X = df_new.drop(columns=[constants.PERFORMANCE,constants.CLASS])
        return (X, y)
    else:
        X = df_new.drop(columns=[constants.CLASS])
        return X

def RunDecisionTreeMethodFit(df, years):
    d_trees = {}
    
    # SHould I split train / test because I am not trying to predict
    for year in years:
        (X, y) = PrepDataForTree(df, year)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        l_max_features = []
        for i in range(1, X.shape[1]):
            l_max_features.append(i+1)
            
        
        tree_clf = DecisionTreeClassifier(random_state=42)
        params = {
            'max_depth': [None, 2, 3, 4, 5, 8], 
            'criterion': ['gini', 'entropy'], 
            'max_features': l_max_features
            }
        grid_search = GridSearchCV(tree_clf, params, scoring='accuracy')
        grid_search.fit(X_train,y_train)
        score = grid_search.best_estimator_.score(X_test, y_test)
        logger.info('Best Params are: %s ', grid_search.best_params_)
        d_trees[year] = (grid_search.best_estimator_, score, grid_search.best_params_)
    return d_trees

def SortNanValuesWithoutFilter(df):
    cols = df.columns
    null_values = []
    for c in cols:
        count = df[c].isna().sum()
        percentage = (count / (len(df[c])))*100
        null_values.append((percentage, c))
    return sorted(null_values, reverse=True)

def SortNANValues(df, filter_by):
    values = df[filter_by].unique()
    d = {}
    for v in values:
        filtered_df = df[df[filter_by] == v]
        null_values = SortNanValuesWithoutFilter(filtered_df)
        d[v] = null_values

    return d
        
def PrintInterestingInfo(trees, target_names):
    for year, model_info in trees.items():
        (model, score) = model_info
        logger.info('LOGGING INFO FOR YEAR %s', year)
        logger.info('SCORE: %s', score)
        importance_array = model.feature_importances_
        feature_array = model.feature_names_in_
        logger.info("The importance of features is:")
        l_importance = []
        for i, j in zip(importance_array, feature_array):
            l_importance.append((i,j))
        
        for x in sorted(l_importance, reverse=True):
            logger.info('%s', x)
            
        font = 14
        if(model.get_depth() > 3):
            font = 11
        file_path = os.path.join(os.getcwd(), 'plots', 'tree_' + str(year) + '.png')
        PlotTree(model, file_path, font, target_names)
    return
        
def PlotTree(tree, file, font, targets):
    plt.figure(figsize=(22,14))
    plt.tight_layout()
    plot_tree(tree, feature_names=tree.feature_names_in_, filled=True, 
              class_names=targets, 
              fontsize=font)
    plt.savefig(file)   
    return

"""
def GetTreeDotData(tree, depth=None):
    dot_data = export_graphviz(tree,
                               out_file=None, 
                               feature_names=tree.feature_names_in_, 
                               class_names=['Poor', 'Acceptable', 'Good'],
                               filled=True, max_depth=depth,
                               rounded=True)
    
    return dot_data
"""
def GetRandomDistrictType(types):
    return random.choices(types)[0]

def GetRandomCharter():
    return random.choice(['Y', 'N'])

def GetRandomActivityIndex(indices, num=1):
    max_index = np.max(indices)
    min_index = np.min(indices)
    l_random_choices = []
    for i in range(0, num):
        n = random.uniform(min_index, max_index)
        l_random_choices.append(n)
    if(num == 1):
        return l_random_choices[0]
    else:
        return l_random_choices

def FindMaxDistrictType(filtered_df):
    max_dtype = filtered_df[constants.CLM_DTYPES_TYPE].value_counts().idxmax()
    dtype_name = 'DISTRICT_TYPE_' + max_dtype
    l_dtypes = []
    for col in COL_Names:
        if(dtype_name == col):
            l_dtypes.append(1)
        else:
            l_dtypes.append(0)  
    return l_dtypes

def CreateDistrictTypeList(value):
    dtype_name = 'DISTRICT_TYPE_' + value
    l_dtypes = []
    for col in COL_Names:
        if(dtype_name == col):
            l_dtypes.append(1)
        else:
            l_dtypes.append(0)  
    return l_dtypes
    
def GetMeanValues(filtered_df, columns):
    means = filtered_df[columns].mean()
    l_means = []
    for i in range(0, len(columns)):
        l_means.append(means.iloc[i])
    return l_means

def GetCharterValue(filtered_df):
    charter = filtered_df['CHARTER'].value_counts().idxmax()[0]
    if(charter == 'Y'):
        return [0, 1]
    else:
        return [1,0]

def PutTogetherRowInArray(year, numerical_list, dtypes_list, charter_list):
     l_final_row = []
     l_final_row.append(year)
     for e in numerical_list:
         l_final_row.append(e)
     for e in dtypes_list:
         l_final_row.append(e)
     for e in charter_list:
         l_final_row.append(e)
        
     return l_final_row
 
def ChangeDatatoBinary(df, mapping, target_column):
    df_new = df.copy(deep = True)
    df_new[target_column] = df_new[target_column].replace(mapping)
    return df_new


# Get the dataframe dictionary for school data (Performance, counties, districts and regions)
d_dfs = getdatascripts.GetDataFromDB(constants.SCHEMA, 
                                     os.path.join(parent_dir, constants.DB_FILE), 
                             tables=[constants.PERFORMANCE_SCORES, 
                                     constants.DF_COUNTIES_CLEANED,
                                     constants.DF_DISTRICTS_CLEANED,
                                     constants.REGIONS],
                             refetch=False, 
                             allTables=False)


#Grab dataframe from dictionary
df_performance = d_dfs[constants.PERFORMANCE_SCORES]
df_districts = d_dfs[constants.DF_DISTRICTS_CLEANED]
df_counties = d_dfs[constants.DF_COUNTIES_CLEANED]
df_regions = d_dfs[constants.REGIONS]

## TODO: Switch joins to function in utils
# Set index for districts and counties as their numbers (these should be unique)
df_districts.set_index(constants.CLM_DTYPES_DISTRICT_NUMBER, inplace=True)
df_counties.set_index(constants.COUNTY, inplace=True)
df_regions.set_index(constants.REGION_NUM, inplace=True)


# Join on DISTRICT_NUMBER to performance dataframe
df_performance = df_performance.join(df_districts, 
                                         on=constants.CLM_DTYPES_DISTRICT_NUMBER, 
                                         validate='m:1')


# Join on COUNTY_NUMBER to performance dataframe
df_performance = df_performance.join(df_counties, 
                                     on=constants.COUNTY, 
                                     validate='m:1')

df_performance = df_performance.join(df_regions, 
                                     on=constants.REGION_NUM,
                                     validate='m:1')


# To prep for the decision tree, I will drop unnecessary columns
df_performance = df_performance[[constants.CLASS, 
                                 constants.CLM_DTYPES_TYPE,
                                 constants.CLM_DTYPES_CHARTER,
                                 constants.CLM_COUNTIES_ECONOMIC_DENSITY,
                                 constants.CLM_COUNTIES_REL_WEEKLY_SAL,
                                 constants.CLM_COUNTIES_ACTIVITY_INDEX,
                                 constants.EXAMINEES,
                                 constants.PARTICIPATION_RATE,
                                 constants.PERFORMANCE
                                 ]]

# Check which columns have too many null values
null_values = SortNanValuesWithoutFilter(df_performance)
#for v in null_values:
    #print(v)
    

# Determine which years are best to use for Decision Tree
null_values = SortNANValues(df_performance, 'CLASS')


cols = [constants.CLM_DTYPES_TYPE,
        constants.CLM_DTYPES_CHARTER,
        constants.CLM_COUNTIES_ECONOMIC_DENSITY,
        constants.CLM_COUNTIES_REL_WEEKLY_SAL,
        constants.CLM_COUNTIES_ACTIVITY_INDEX,
        constants.EXAMINEES,
        constants.PARTICIPATION_RATE]
df_performance = df_performance.dropna(subset=cols)

# If running decision trees on binary outcomes replace here!

binary_mapping = {3: 2, 4: 2}
df_performance_binary = ChangeDatatoBinary(df_performance, binary_mapping, constants.PERFORMANCE)



trees = RunDecisionTreeMethodFit(df_performance,
                                 [2014, 2015, 2016, 2017,
                                  2018, 2019, 2020, 2021, 2022, 
                                  2023])

trees_binary = RunDecisionTreeMethodFit(df_performance_binary,
                                 [2014, 2015, 2016, 2017,
                                  2018, 2019, 2020, 2021, 2022, 
                                  2023])

#PrintInterestingInfo(trees, ['Poor', 'Not Poor'])


NCES_Districts = df_performance[constants.CLM_DTYPES_TYPE].unique().tolist()
"""predict data columns
CLASS,ECONOMIC_DENSITY,RELATIVE_WEEKLY_SALARY,ECONOMIC_ACTIVITY_INDEX,EXNEES_MSKD,PART_RATE,

DISTRICT_TYPE_City-Large,DISTRICT_TYPE_City-Midsize,DISTRICT_TYPE_City-Small,
DISTRICT_TYPE_Rural-Distant,DISTRICT_TYPE_Rural-Fringe,DISTRICT_TYPE_Rural-Remote,
DISTRICT_TYPE_Suburb-Large,DISTRICT_TYPE_Suburb-Midsize,DISTRICT_TYPE_Suburb-Small,
DISTRICT_TYPE_Town-Distant,DISTRICT_TYPE_Town-Fringe,DISTRICT_TYPE_Town-Remote,
CHARTER_N,CHARTER_Y	
"""
"""Predictions for 2014 to be rated > poor: 
  1) If salarty < 0.69 then district type is town - remote
  2) If relative weekly salary > 0,7, then examinees < 23 gives greater than poor
  3) If salary > 0.7, Larger examinees > 87, higher participation rate > 65



df_predict = pd.read_csv('DM/predict_data.csv')
# Case (1) - determine average values for town - remote
df_filtered_2014 = df_performance[df_performance['CLASS'] == 2014]

# get the town - remote items
df_remote_filtered_2014 = df_filtered_2014.loc[df_filtered_2014[constants.CLM_DTYPES_TYPE]==NCES_Districts[3]]

numerical_values = GetMeanValues(df_remote_filtered_2014, ['ECONOMIC_DENSITY','ECONOMIC_ACTIVITY_INDEX', 'EXNEES_MSKD', 'PART_RATE'])
numerical_values.insert(1, 0.5)


dtypes = CreateDistrictTypeList(NCES_Districts[3])
charters = GetCharterValue(df_remote_filtered_2014)
row_value = PutTogetherRowInArray(2014, numerical_values, dtypes, charters)


df_predict.loc[0, :] = row_value

dtypes = CreateDistrictTypeList('City-Midsize')
row_value = PutTogetherRowInArray(2014, numerical_values, dtypes, charters)
df_predict.loc[1, :] = row_value

# Case (2) - determine which values occur for schools with low examinees. 
df_examinees_filtered = df_filtered_2014[df_filtered_2014['EXNEES_MSKD'] < 30]

dtypes = FindMaxDistrictType(df_examinees_filtered)
numerical_values = GetMeanValues(df_examinees_filtered, ['ECONOMIC_DENSITY','ECONOMIC_ACTIVITY_INDEX', 'PART_RATE'])
charters = GetCharterValue(df_examinees_filtered)
numerical_values.insert(1, 0.9)
numerical_values.insert(3, 20)

row_value = PutTogetherRowInArray(2014, numerical_values, dtypes, charters)
df_predict.loc[2, :] = row_value
dtypes = CreateDistrictTypeList('City-Large')
row_value = PutTogetherRowInArray(2014, numerical_values, dtypes, charters)
df_predict.loc[3, :] = row_value

#Case 3 high examinees and high part rate
df_ex_2014 = df_filtered_2014[(df_filtered_2014['EXNEES_MSKD'] > 87)
                                  & (df_filtered_2014['PART_RATE'] > 65)]
numerical_values = GetMeanValues(df_ex_2014, ['ECONOMIC_DENSITY',
                                               'ECONOMIC_ACTIVITY_INDEX'])

numerical_values.insert(1, 0.8)
numerical_values.insert(3, 100)
numerical_values.insert(4, 80)
dtypes = FindMaxDistrictType(df_ex_2014)
charters = GetCharterValue(df_ex_2014)
row_value = PutTogetherRowInArray(2014, numerical_values, dtypes, charters)

df_predict.loc[4,:] = row_value

dtypes = CreateDistrictTypeList('Suburb-Small')
row_value = PutTogetherRowInArray(2014, numerical_values, dtypes, charters)
df_predict.loc[5, :] = row_value

X = df_predict.drop(columns = 'CLASS')
predictions = tree_2014.predict(X)

"""
# Predict for 2017: left branch density < .05, if rural fringe with 
# high activity index (> 0.3) and examinees under 150
# Right branch > 0.06, low part rate (<26) with high relative 
#weekly salary OR Rural -> not poor. 
#OR Right branch but with >26 part rate but low examinees (<25)
# And 