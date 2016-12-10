#Import Packages
import pandas as pd
import numpy as np
from helper import *

#Load Data from CSV
dfload = pd.read_csv('data/raw/stats.csv')
df = sort_columns_type(dfload)

#----------------------------------------------------------------------
### Creat Different subsets

df1 = df.copy()
df2 = df.copy()

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

#----------------------------------------------------------------------
# Fill in Null values

df1 = df1.fillna(float(0.01))