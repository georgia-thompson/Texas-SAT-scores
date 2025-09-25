# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 19:16:52 2025

@author: Georg
"""
import getdatascripts
import constants
import utils
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

parent_directory = os.path.dirname(os.getcwd())
bChooseYear = True
year = 2022
filename = os.path.join(parent_directory, 'plots', 'district_type_part_rate.png')

d_dfs = getdatascripts.GetDataFromDB(constants.SCHEMA, 
                                     os.path.join(parent_directory, constants.DB_FILE), 
                             tables=[constants.PERFORMANCE_SCORES,
                                     constants.DF_DISTRICTS_CLEANED],
                             refetch=False, 
                             allTables=False)

df_performance = d_dfs[constants.PERFORMANCE_SCORES]
df_districts = d_dfs[constants.DF_DISTRICTS_CLEANED]

if(bChooseYear):
    df_performance = df_performance[df_performance[constants.CLASS] == year]
    filename = os.path.join(os.path.dirname(filename), 'district_type_part_rate_' + str(year) + '.png')

l_dfs = utils.JoinTables([(df_performance, df_districts)], [constants.CLM_DTYPES_DISTRICT_NUMBER], 
                         new_index=True)

df_joined = l_dfs[0]

df_joined = df_joined.dropna(subset=['DISTRICT_TYPE'])
districts = df_joined['DISTRICT_TYPE'].unique().tolist()

avg_series = df_joined.groupby('DISTRICT_TYPE')['PERFORMANCE'].mean().round(2)
avg_series_sorted = avg_series.sort_values()

df_joined['average'] = df_joined.groupby('DISTRICT_TYPE')['PERFORMANCE'].transform('mean')
sns.set_theme(
    context='notebook',
    rc={
        'xtick.labelsize': 22,    # Set x-axis tick label font size
        'legend.fontsize': 18,
        'axes.titlesize': 24
    }
)

fig = plt.figure(figsize=(24, 18), edgecolor='#FCB714', 
                 linewidth=5, layout="tight")

colors = ["#F53F27", "#F55E27", "#F57D27", "#F58E27", "#F5B027", "#F5C527", "#F5E727", "#F5F527", "#DDF527", "#95F527", "#C2F527", "#49F527"]
custom_palette = sns.color_palette(colors)

sns.barplot(data=df_joined, x='PART_RATE', y='DISTRICT_TYPE', 
            order = avg_series_sorted.index, palette=custom_palette)
plt.xlabel("Participation Rate", fontsize=16)
plt.ylabel("District Type", fontsize=16)
plt.yticks(fontsize=20)
plt.title("The Participation Rate by District Type Sorted Worst Rating down to Highest")

counter=0
for d in avg_series_sorted.index:
    if(d is not None):
        plt.text(3, counter, avg_series_sorted[d], fontsize=26, fontweight='bold', va='center', ha='center')
        counter += 1
        
plt.savefig(filename)
"""
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

"""