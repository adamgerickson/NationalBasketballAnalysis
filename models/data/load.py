#Import Packages
import pandas as pd
import numpy as np
from helper import *

#Load Raw Data from CSV
dfload = pd.read_csv('data/raw/stats.csv')
df = sort_columns_type(dfload)

#-------------------------------------------------------------------
#Fill in team and Position fields for career data

def most_common(l):
    max = 0
    maxitem = None
    for x in set(l):
        count =  l.count(x)
        if count > max:
            max = count
            maxitem = x
    return maxitem

## Getting positions right
for i in (list(set(((((df['name']).values).astype(str)).tolist())))):
    poses = ((df.query('name == @i')['pos']).values).tolist()
    newpos = poses[len(poses) - 2]
    idx = (list(df.query('season == "Career"').query('name == @i').index))[0]
    df.loc[idx , 'pos'] = newpos

#Remove SF-SG position, to get down to 5
for i in list(df.query('pos == "SF-SG"').index):
	df.loc[i , 'pos'] = 'SF'

## Team they played for the most
for i in (list(set(((((df['name']).values).astype(str)).tolist())))):
    teams = ((df.query('name == @i')['team_id']).values).tolist()
    oneteam = most_common(teams)
    idx = (list(df.query('season == "Career"').query('name == @i').index))[0]
    df.loc[idx , 'team_id'] = oneteam

#Helper Functions
def replace(l, X, Y):
    i = 0
    for v in l:
        if v == X:
            l.pop(i)
            l.insert(i, Y)
        i += 1

def pos_to_int(l):
    pos_conv = ['PG', 'SG', 'SF', 'PF', 'C']
    for j in range(len(pos_conv)):
        replace(l, pos_conv[j], int(j+1))
    return l
    

#----------------------------------------------------------------------
### Creat Different subsets

df1 = df.copy()
df2 = df.copy()

dfloadi = pd.read_csv('data/interim/isomap.csv')
dfi = sort_columns_type(dfloadi)

dfloads = pd.read_csv('data/interim/spectral.csv')
dfs = sort_columns_type(dfloads)

dfloadp = pd.read_csv('data/interim/pca.csv')
dfp = sort_columns_type(dfloadp)

dfloadm = pd.read_csv('data/interim/mlle.csv')
dfmlle = sort_columns_type(dfloadm)

#----------------------------------------------------------------------
### Filter Rows (Choosing the players)

df1 = df1.query("g> 100") #played more than 100 games

#----------------------------------------------------------------------
### Filter Columns (choosing the attributes)

shanelist = [   
  #Shane's list for style
 'name', 'season', 'team_id', 'pos', 'pf_per_poss', 
 'mp_per_g', 'pct_fg2_dunk', 'fg3a_pct_fga', 
 'blk_pct', 'stl_pct', 'usg_pct', 'fta_per_poss',
 'orb_pct', 'ast_pct', 'tov_pct', 'fg2_pct_ast',
 'fg3_pct_ast', 'fg3_pct', 'fg3a_per_fga_pct',
 'fg2a_pct_fga', 'fta_per_fga_pct', 'fg_pct_00_03', 
 'fg_pct_03_10', 'fg_pct_10_16', 'fg_pct_16_xx',
 'pct_fg3a_corner', 'pct_fga_00_03', 'pct_fga_03_10',
 'pct_fga_10_16', 'pct_fga_16_xx', 'fg3_pct_corner'
 ]

df1 = df1[shanelist]

df1pos = pos_to_int(list(df['pos']))

#----------------------------------------------------------------------
# Fill in Null values

df1 = df1.fillna(float(0.01))