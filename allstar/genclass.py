# Generate a class field for a given dataset

import numpy as np 
import csv
import pandas as pd 
from sklearn import svm
from sklearn.manifold import mds

allstars = {
    '16': ('LeBron James',
             'Paul George', 
             'Carmelo Anthony', 
             'Dwyane Wade', 
             'Kyle Lowry', 
             'Jimmy Butler', 
             'DeMar DeRozan', 
             'Paul Millsap',
             'Andre Drummond',
             'Chris Bosh',
             'John Wall',
             'Isaiah Thomas',
             'Pau Gasol',
             'Al Horford',
             'Kobe Bryant',
             'Kevin Durant',
             'Kawhi Leanord',
             'Stephen Curry',
             'Russell Westbrook',
             'Draymond Green',
             'James Harden',
             'Chris Paul',
             'Klay Thompson',
             'Anthony Davis',
             'DeMarcus Cousins',
             'LaMarcus Aldridge'),

    '15': ('Carmelo Anthony', 
             'LeBron James',
             'Pau Gasol',
             'John Wall',
             'Kyle Lowry',
             'Chris Bosh',
             'Jimmy Butler',
             'Al Horford',
             'Kyrie Irving',
             'Kyle Korver',
             'Paul Millsap',
             'Jeff Teague',
             'Dwyane Wade',
             'Blake Griffin',
             'Marc Gasol',
             "Kobe Bryant",
             "Anthony Davis",
             "Stephen Curry",
             "LaMarcus Aldridge",
             "DeMarcus Cousins",
             "Tim Duncan",
             "Kevin Durant",
             "James Harden",
             "Damian Lillard",
             "Dirk Nowitzki",
             "Chris Paul",
             "Klay Thompson",
             "Russell Westbrook"),

    '14': ("Carmelo Anthony"
             "Paul George",
             "LeBron James",
             "Kyrie Irving",
             "Dwyane Wade",
             "Chris Bosh",
             "DeMar DeRozan",
             "Roy Hibbert",
             "Joe Johnson",
             "Paul Millsap",
             "Joakim Noah",
             "John Wall",
             "Kevin Durant",
             "Blake Griffin",
             "Kevin Love",
             "Kobe Bryant",
             "Stephen Curry",
             "LaMarcus Aldridge",
             "Anthony Davis",
             "James Harden",
             "Dwight Howard",
             "Damian Lillard",
             "Dirk Nowitzki",
             "Tony Parker",
             "Chris Paul",
             ),

    '13': ("Carmelo Anthony",
             "LeBron James",
             "Kevin Garnett",
             "Rajon Rondo",
             "Dwyane Wade",
             "Chris Bosh",
             "Tyson Chandler",
             "Luol Deng",
             "Paul George",
             "Jrue Holliday",
             "Kyrie Irving",
             "Brook Lopez",
             "Joakim Noah",
             "Kevin Durant",
             "Blake Griffin",
             "Dwight Howard",
             "Kobe Bryant",
             "Chris Paul",
             "LaMarcus Aldridge",
             "Tim Duncan",
             "James Harden",
             "David Lee",
             "Tony Parker",
             "Zach Randolph",
             "Russell Westbrook",
        ),

    '12': ("Carmelo Anthony",
             "LeBron James",
             "Derrick Rose",
             "Dwyane Wade",
             "Dwight Howard",
             "Chris Bosh",
             "Luol Deng",
             "Roy Hibbert",
             "Andre Iguodala",
             "Joe Johnson",
             "Paul Pierce",
             "Rajon Rondo",
             "Deron Williams",
             "Kevin Durant",
             "Blake Griffin",
             "Kobe Bryant",
             "Chris Paul",
             "Andrew Bynum",
             "LaMarcus Aldridge",
             "Marc Gasol",
             "Kevin Love",
             "Steve Nash",
             "Dirk Nowitzki",
             "Tony Parker",
             "Russell Westbrook",
        ),

    '11': ("LeBron James",
             "Amar'e Stoudemire",
             "Dwyane Wade",
             "Derrick Rose",
             "Dwight Howard",
             "Ray Allen",
             "Chris Bosh",
             "Kevin Garnett",
             "Al Horford",
             "Joe Johnson",
             "Paul Pierce",
             "Rajon Rondo",
             "Tim Duncan",
             "Kevin Durant",
             "Carmelo Anthony",
             "Kobe Bryant",
             "Chris Paul",
             "Yao Ming",
             "Manu Ginobili",
             "Pau Gasol",
             "Blake Griffin",
             "Kevin Love",
             "Dirk Nowitzki",
             "Russell Westbrook",
             "Deron Williams",
             ),     
}

# get pandas dataframe from csv
data = pd.read_csv('data_filtered.csv')

# change based on what your class is based on
def genclass(name, year):
    """
    generate class values
    all-star example: take in name and year and return True if player was an
    an all-star in that year

    year comes in as '2015-16', so just take last two chars

    uses global pandas dataframe `data` defined above
    """
    if year[-2:] not in allstars:
        return(0)
    elif name in allstars[year[-2:]]:
        return(1)
    else:
        return(-1)

# create new boolean field 'class': 1 if true, -1 if not
data['allstar'] = pd.Series([0] * len(data['name']), index=data.index)
# print(data['name'][0])

for index, row in data.iterrows():
    data['allstar'][index] = genclass(data['name'][index], data['year'][index])

data.to_csv('data_with_allstars')








