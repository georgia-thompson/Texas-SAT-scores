# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 18:07:27 2025

@author: Georg
"""
import pandas as pd
import sqlite3
import statistics as stat
import seaborn as sns
import matplotlib.pyplot as plt
import operator
import getdatascripts 
import constants
import os

parent_directory = os.path.dirname(os.getcwd())

d_dfs = getdatascripts.GetDataFromDB(constants.SCHEMA, 
                                     os.path.join(parent_directory, constants.DB_FILE), 
                             tables=[constants.AVERAGES,
                                     constants.REGIONS, 
                                     constants.DISTRICTS, 
                                     constants.GROUPS],
                             refetch=False, 
                             allTables=False)

df_avg_sat_scores = d_dfs[constants.AVERAGES]
df_regions = d_dfs[constants.REGIONS]
df_districts = d_dfs[constants.DISTRICTS]
df_groups = d_dfs[constants.GROUPS]

df_districts_avg = df_avg_sat_scores.groupby(by=["DISTRICT_NUMBER", "REGION_NUMBER"], 
                                          dropna=True)[["TOTAL"]].mean().sort_values("TOTAL")

df_districts_avg = df_districts_avg.reset_index()
# drop nan rows
df_districts_avg = df_districts_avg.dropna()

def MapDistrictName(district_num):
    str_name = df_districts[df_districts["DISTRICT_NUMBER"] == 
                            district_num]["DISTRICT_NAME"].head(1).item()
    return str_name

def MapRegionName(region_num):
    str_name = df_regions[df_regions["REGION_NUMBER"] == 
                          region_num]["REGION_NAME"].head(1).item()
    return str_name
    

def GetMedian(data: list) -> float: 
    """
    Gets the mean from a series. If there is not a center, it takes the 
    average of the higher and lower values.
    
    Parameters: 
        data (list): List of data
        
    Returns: 
        float: Median value of data

    """
    return stat.median(data)

def GetQuadrants(lst_data):
    q2 = GetMedian(lst_data)
    upper_part = [item for item in lst_data if item < q2]
    q1 = GetMedian(upper_part)

    lower_part = [item for item in lst_data if item > q2]
    q3 = GetMedian(lower_part)
    return(q1, q2, q3)
    

def FindOutlierValues(df, region=None) -> tuple:
    """
    Calculates the Max and Min values that are still within inliers
    
    Parameters: 
        df (Dataframe): the dataframe to find values for
        
    Returns: 
        float: (Min inlier, Max inlier)

    """
    if(region is not None):
        # Filter by region 
        df = df[df["REGION_NUMBER"] == region]

    q2 = GetMedian(df["TOTAL"].tolist())

    upper_part = df[df["TOTAL"] < q2]["TOTAL"].tolist()
    q1 = GetMedian(upper_part)

    
    lower_part = df[df["TOTAL"] > q2]["TOTAL"].tolist()
    q3 = GetMedian(lower_part)
    
    IQR = q3 - q1
    lower_bound = q1 - IQR*1.5
    upper_bound = q3 + IQR*1.5
    
    return (lower_bound, upper_bound)

def FindDistrictsWithOutliers(df_data, df_districts, bounds: tuple) -> dict:
    """
    Finds the district numbers that have outliers (max or min)
    
    Parameters: 
        df_data (dataframe): the grouped data frame
        df_districts (dataframe): The districts dataframe
        bounds (tuple) : (min bound, max bound)
        
    Returns: 
        dict: {min -> [list of district tuples(number, name)], 
               max->[list of district tuples (number, name)]}

    """
    dct_outlier_districts = {}
    series_districts_min = df_data[df_data["TOTAL"] < bounds[0]]["DISTRICT_NUMBER"].tolist()
    lst_districts_min = []
    for district in series_districts_min:
        """
        str_name = df_districts[df_districts["DISTRICT_NUMBER"] == 
                                district]["DISTRICT_NAME"].head(1).item()
        """
        str_name = MapDistrictName(district)
        name_map = (district, str_name)
        lst_districts_min.append(name_map)
    
    series_districts_max = df_data[df_data["TOTAL"] > bounds[1]]["DISTRICT_NUMBER"].tolist()
    lst_districts_max = []
    for district in series_districts_max:
        MapDistrictName(district)
        """
        str_name = df_districts[df_districts["DISTRICT_NUMBER"] == 
                                district]["DISTRICT_NAME"].head(1).item()
        """
        
        name_map = (district, str_name)
        lst_districts_max.append(name_map)
        
    
    dct_outlier_districts["min"] = lst_districts_min
    dct_outlier_districts["max"] = lst_districts_max
    
    return dct_outlier_districts

def CheckOutlierStatus(total, outlier_values):
    """
    If max outlier -> return 1
    If min outlier -> return -1
    if not outlier -> return 0

    Parameters
    ----------
    total: average score value
    outlier_values: (lower_boundary, upper_boundary)
    """
    
    if(total < outlier_values[0]):
        return ("Below")
    elif(total > outlier_values[1]):
        return ("Above")
    else:
        return ("Within Range")
    

def MapRaceCategory(group_num):
    if(group_num in [6, 7, 9, 10, 11]):
        return "Minority"
    elif(group_num == 12):
        return "White"
    elif(group_num == 8):
        return "Asian"
    
def GetSubsetDataframe(df: pd.DataFrame, lstFilter: list, column: str)-> pd.DataFrame:
    new_df = df[df[column].isin(lstFilter)].reset_index(drop=True)
    return new_df

def ReassignGroupNum(df, lst_old_nums, new_num, new_name):
    i = df.loc[df['GROUP_NUM'].isin(lst_old_nums)].index
    df.loc[i, "GROUP_NAME"] = new_name
    lst_values = [new_num]*len(lst_old_nums)
    df["GROUP_NUM"] = df["GROUP_NUM"].replace(to_replace=lst_old_nums, 
                                              value=lst_values)
    return df


def DropGroupNum(lst_dfs, group_num):
    new_list = []
    for df in lst_dfs:
        i = df[df["GROUP_NUM"] == group_num].index
        df = df.drop(i)
        new_list.append(df)
    return new_list

(lower, upper) = FindOutlierValues(df_districts_avg)
outliers = FindDistrictsWithOutliers(df_districts_avg, df_districts, 
                                     (lower, upper))


# Append Region Name and district name to dataframe

df_districts_avg = df_districts_avg.merge(df_districts[["DISTRICT_NUMBER", 
                                                        "DISTRICT_NAME"]], 
                                          how='inner', on='DISTRICT_NUMBER')

df_districts_avg = df_districts_avg.merge(df_regions, how='inner', on='REGION_NUMBER')

    
# boxplot by region 
plt.figure()
ax = sns.boxplot(data = df_districts_avg, x="REGION_NUMBER", y="TOTAL")

ax.set_xticklabels(df_districts_avg['REGION_NAME'])

plt.xlabel('Region Name')
plt.xticks(rotation=90)
plt.ylabel('SAT Averages')
plt.title('Box Plot of SAT Averages by Region')
plt.savefig("BoxPlot_region.png")


# boxplot by region (add a column of "outlier" with -1 below, 1 above and 
    #0 within range)

df_districts_avg["OUTLIER"] = df_districts_avg.apply(lambda x: 
                                                         CheckOutlierStatus(x["TOTAL"], (lower, upper)), axis=1)

# only chart the districts that have outliers
lst_outlier_districts = []
for key, value in outliers.items():
    lst_outlier_districts.extend(list(map(operator.itemgetter(0), value)))
    

df_outlier_districts = df_districts_avg[df_districts_avg["DISTRICT_NUMBER"].isin(lst_outlier_districts)].reset_index(drop=True)


"""
# bar chart of outlier districts (color regions) with Q1, Q2 and Q3 lines
"""

sns.set_theme(
    context='notebook',
    rc={
        'xtick.labelsize': 22,    # Set x-axis tick label font size
        'legend.fontsize': 18,
        'axes.titlesize': 24
    }
)

df_outlier_districts = df_outlier_districts.sort_values(['REGION_NUMBER']).reset_index(drop=True)
fig = plt.figure(figsize=(24, 18), edgecolor='#FCB714', 
                 linewidth=5, layout="tight")
custom_palette = sns.mpl_palette("tab20", 20)
sns.barplot(df_outlier_districts, x="DISTRICT_NAME", y="TOTAL", 
            hue="REGION_NAME", dodge=False, palette=custom_palette)
(q1, q2, q3) = GetQuadrants(df_districts_avg["TOTAL"].tolist())
plt.axhline(y=q1, color="black")
plt.axhline(y=q3, color="black")
plt.axhline(y=q2, color="black", linestyle="--")
plt.text(1, q1, 'Q1', fontsize=14, va='center', ha='center')
plt.text(1, q2, 'Median', fontsize=14, va='center', ha='center')
plt.text(1, q3, 'Q3', fontsize=14, va='center', ha='center')

plt.legend(loc='upper left', ncol=7, 
           bbox_to_anchor=(0, 1))
plt.xticks(rotation=90)
plt.xlabel("District")
plt.ylabel("Average SAT Score")
plt.title("Average SAT Scores for Outlier Districts")
#plt.tight_layout()


plt.savefig("outlier_districts.png")
        
    
## Look at participation rate for each group category in these districts
# Don't forget to order by region
df_avg_sat_scores = df_avg_sat_scores.merge(df_groups, 
                                          how='inner', on='GROUP_NUM')


df_avg_sat_scores = df_avg_sat_scores.merge(df_districts[["DISTRICT_NUMBER", 
                                                        "DISTRICT_NAME"]], 
                                          how='inner', on='DISTRICT_NUMBER')

df_avg_sat_scores = df_avg_sat_scores.merge(df_regions, how='inner', on='REGION_NUMBER')

str_group_num_name = "GROUP_NUM"

# Some of the group numbers are multiples and need to be fixed in database. For now, fix here.
"""1 | ALL STUDENTS                   |
|         2 | ECONOMICALLY DISADVANTAGED     |
|         3 | NOT ECONOMICALLY DISADVANTAGED |
|         4 | FEMALE                         |
|         5 | MALE                           |
|         6 | AFRICAN AMERICAN               |
|         7 | AMERICAN INDIAN                |
|         8 | ASIAN                          |
|         9 | HISPANIC                       |
|        10 | MULTIRACIAL                    |
|        11 | PACIFIC ISLANDER               |
|        12 | WHITE                          |
|        13 | MISSING ETHNICITY              |
|        14 | MISSING ECON                   |
|        15 | MISSING GENDER                 |
|        16 | BIL/ESL                        |
|        17 | NOT BIL/ESL                    |
|        18 | MISSING BIL/ESL                |
|        19 | CTE                            |
|        20 | NOT CTE                        |
|        21 | MISSING CTE                    |
|        22 | GIFTED                         |
|        23 | NOT GIFTED                     |
|        24 | MISSING GIFTED                 |
|        25 | SPECIAL ED                     |
|        26 | NOT SPECIAL ED                 |
|        27 | MISSING SPECIAL ED             |
|        28 | TITLE1                         |
|        29 | NOT TITLE1                     |
|        30 | MISSING TITLE1                 |
|        31 | AT RISK                        |
|        32 | NOT AT RISK                    |
|        33 | MISSING AT RISK                |
|        34 | LEP/ELL                        |
|        35 | NOT LEP/ELL                    |
|        36 | MISSING LEP/ELL                |
|        37 | IMMIGRANT                      |
|        38 | NOT IMMIGRANT                  |
|        39 | MISSING IMMIGRANT              |
|        40 | MIGRANT                        |
|        41 | NOT MIGRANT                    |
|        42 | MISSING MIGRANT                |
|        43 | DYSLEXIC                       |
|        44 | NOT DYSLEXIC                   |
|        45 | MISSING DYSLEXIC               |
|        46 | HOMELESS                       |
|        47 | NOT HOMELESS                   |
|        48 | MISSING HOMELESS               |
|        49 | AT-RISK                        |
|        50 | NOT AT-RISK                    |
|        51 | MISSING AT-RISK                |
|        52 | ENGLISH LEARNER                |
|        53 | NOT ENGLISH LEARNER            |
|        54 | MISSING ENGLISH LEARNER        |
|        55 | DYSLEXIA                       |
|        56 | NOT DYSLEXIA                   |
|        57 | MISSING DYSLEXIA               |
|        58 | FOSTER CARE                    |
|        59 | NOT FOSTER CARE                |
|        60 | MISSING FOSTER CARE            |
|        61 | MILITARY-CONNECTED             |
|        62 | NOT MILITARY-CONNECTED         |
|        63 | MISSING MILITARY-CONNECTED     |
|        64 | SECTION 504                    |
|        65 | NOT SECTION 504                |
|        66 | MISSING SECTION 504            |
|        67 | CTE CONCENTRATOR               |
|        68 | NOT CTE CONCENTRATOR           |
|        69 | EB/EL                          |
|        70 | NOT EB/EL                      |
|        71 | MISSING EB/EL       
"""
# Change BIL/ESL (16, 17, 18), LEP/ELL (34, 35, 36), EB/EL (69, 70, 71) to ENGLISH LEARNER (52, 53, 54)               |

df_avg_sat_scores = ReassignGroupNum(df_avg_sat_scores, [16, 34, 69], 
                                       52, "ENGLISH LEARNER")
df_avg_sat_scores = ReassignGroupNum(df_avg_sat_scores, [17, 35, 70], 
                                       53, "NOT ENGLISH LEARNER")
df_avg_sat_scores = ReassignGroupNum(df_avg_sat_scores, [18, 36, 71], 
                                       54, "MISSING ENGLISH LEARNER")
df_avg_sat_scores = ReassignGroupNum(df_avg_sat_scores, [49], 31, 'AT RISK')
df_avg_sat_scores = ReassignGroupNum(df_avg_sat_scores, [50], 32, 'NOT AT RISK')

## Create different demographic groups 
df_scores_by_economic = GetSubsetDataframe(df_avg_sat_scores, [1,2,3, 46, 47, 64, 65], str_group_num_name)

df_scores_by_economic = ReassignGroupNum(df_scores_by_economic, [47, 65], 3, 
                                         "NOT ECONOMICALLY DISADVANTAGED")
df_scores_by_race = GetSubsetDataframe(df_avg_sat_scores, [1,6, 7, 8, 9, 10, 11, 12], 
                                       str_group_num_name)

df_scores_by_race["RACE_GROUP"] = df_scores_by_race.apply(lambda x: MapRaceCategory(x["GROUP_NUM"]), 
                                            axis=1)
df_scores_by_sex = GetSubsetDataframe(df_avg_sat_scores, [1,4, 5], str_group_num_name)

df_scores_by_esl = GetSubsetDataframe(df_avg_sat_scores, [1,37, 38, 40, 41, 52, 53], 
                                      str_group_num_name)
df_scores_by_esl = ReassignGroupNum(df_scores_by_esl, [38, 41, 53], 72, "Not ESL, Immigrant, Migrant")

df_scores_by_school_program = GetSubsetDataframe(df_avg_sat_scores, [1,19, 20, 22, 25, 67, 68, 43, 55], 
                                                 str_group_num_name)
df_scores_by_school_program = ReassignGroupNum(df_scores_by_school_program, [67], 19, "CTE")
df_scores_by_school_program = ReassignGroupNum(df_scores_by_school_program, [68], 20, "NOT CTE")
df_scores_by_school_program = ReassignGroupNum(df_scores_by_school_program, [55], 43, "DYSLEXIC")

df_scores_by_atrisk = GetSubsetDataframe(df_avg_sat_scores, [1,28, 31, 46, 58], 
                                                 str_group_num_name)

df_scores_by_military = GetSubsetDataframe(df_avg_sat_scores, [61, 62], 
                                                 str_group_num_name)

sns.reset_defaults()

sns.set_theme(
    rc={
        'xtick.labelsize': 18,    # Set x-axis tick label font size
        'legend.fontsize': 16,
        'axes.titlesize': 20
    }
)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(22, 16)) # 2 rows, 2 columns

sns.lineplot(data=df_scores_by_economic, x="CLASS", y="TOTAL", 
             hue="GROUP_NAME", ax=axes[0, 0])
axes[0,0].set_title("Scores by Economic Circumstances")

sns.lineplot(data=df_scores_by_race, x="CLASS", y="TOTAL", 
             hue="RACE_GROUP", ax=axes[0,1])
axes[0,1].set_title("Scores by Racial Classification")

sns.lineplot(data=df_scores_by_sex, x="CLASS", y="TOTAL", 
             hue="GROUP_NAME", ax=axes[0,2])
axes[0,2].set_title("Scores by Sex")

sns.lineplot(data=df_scores_by_esl, x="CLASS", y="TOTAL", 
             hue="GROUP_NAME", ax=axes[1,0])
axes[1,0].set_title("Scores by Immigrant Status")

sns.lineplot(data=df_scores_by_school_program, x="CLASS", y="TOTAL", 
             hue="GROUP_NAME", ax=axes[1,1])
axes[1,1].set_title("Scores by School Program Placement")

sns.lineplot(data=df_scores_by_atrisk, x="CLASS", y="TOTAL", 
             hue="GROUP_NAME", ax=axes[1,2])
axes[1,2].set_title("At Risk Students")
plt.tight_layout()
plt.savefig("scores_by_categories.png")
"""
plt.figure()
sns.lineplot(data=df_scores_by_military, x="CLASS", y="TOTAL", hue="GROUP_NAME")
plt.show()
"""

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(22, 16)) # 2 rows, 2 columns

sns.histplot(data=df_scores_by_economic, x="TOTAL",
             hue="GROUP_NAME", multiple='stack', ax=axes[0, 0])
axes[0,0].set_title("Histogram by Economic Circumstances")

sns.histplot(data=df_scores_by_race, x="TOTAL", 
             hue="RACE_GROUP",multiple='stack', ax=axes[0,1])
axes[0,1].set_title("Histogram by Racial Classification")

sns.histplot(data=df_scores_by_sex, x="TOTAL", 
             hue="GROUP_NAME",multiple='stack', ax=axes[0,2])
axes[0,2].set_title("Histogram by Sex")

sns.histplot(data=df_scores_by_esl, x="TOTAL", 
             hue="GROUP_NAME",multiple='stack', ax=axes[1,0])
axes[1,0].set_title("Histogram by Immigrant Status")

sns.histplot(data=df_scores_by_school_program, x="TOTAL", 
             hue="GROUP_NAME",multiple='stack', ax=axes[1,1])
axes[1,1].set_title("Histogram by School Program Placement")

sns.histplot(data=df_scores_by_atrisk, x="TOTAL", 
             hue="GROUP_NAME",multiple='stack', ax=axes[1,2])
axes[1,2].set_title("Histogram At Risk Students")
plt.tight_layout()
plt.savefig("histogram_by_categories.png")

"""
Plot on a no - fill bar graph the outlier districts with the 
participation rates for each category normalize SAT scores to be within 
100 (or could do within 1 and normalize both)
"""

"""
Clean up group names and then plot histograms by group name
"""
lst_dataframes_bygroup = DropGroupNum([df_scores_by_atrisk, 
                                df_scores_by_economic, 
                                df_scores_by_esl, 
                                df_scores_by_military, 
                                df_scores_by_race, 
                                df_scores_by_school_program, 
                                df_scores_by_sex], 1)
lst_dataframes_bygroup.append(df_avg_sat_scores[df_avg_sat_scores["GROUP_NUM"] == 1])

df_averages_bygroup = pd.concat(lst_dataframes_bygroup,
                                axis = 0, ignore_index=True)

