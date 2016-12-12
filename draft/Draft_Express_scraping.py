# Adapted form script used in ASTR 356 By Shane Fenske
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import re
import numpy as np
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

draftexpress_url = "http://www.draftexpress.com"
draft_url_template = "/nba-draft-history/?syear={year}"
player_url_template ="http://www.draftexpress.com/profile{year}/stats/"

draft_df = pd.DataFrame()

column_headers = ['id', 'name','height','weight','dob','age','position','Pk','Draft_Yr','wing_span','standing_reach',
                    'body_fat', 'no_step_vert', 'max_vert', 'gp', 'min', 'pts', 'fg', 'fga', 'fg_perc', '2pt', 
                    '2pta', '2p_perc', '3pt', '3pta', '3p_perc', 'FTM', 'FTA', 'FT_perc', 'off_reb', 'def_reb', 
                    'tot_reb', 'ast', 'stl', 'blks', 'to', 'pf']
for year in range(2002, 2016):  

    url = draftexpress_url + draft_url_template.format(year=year)
    html = urlopen(url)  
    soup = BeautifulSoup(html, 'html5lib')

    # get the td's that include the link to each players individual page
    linked_names = soup.find_all(attrs={"data-title": "Name"})

    # make a list of the urls of each players stats page

    stat_urls = [td.find('a')["href"] + "stats" for td in linked_names]

    player_data = []
    measurements = [np.nan, np.nan, np.nan, np.nan, np.nan]
    for url in stat_urls:
        print(url)
        # TODO make this a for loop
        # url = stat_urls[0]
        url = draftexpress_url + url
        html = urlopen(url)  
        soup = BeautifulSoup(html, 'html5lib')
        try:
            # NAME
            name = soup.find(class_="profiles-left").find(class_="title").get_text()

            # PHYSICAL SECTION
            physical_data = str(soup.find(attrs={"data-title": "PHYSICALS"})).split('\n')[1:]

            (feet, inches) = re.findall(r'\d+',physical_data[0])
            height = int(feet) * 12 + int(inches)
            weight = re.findall(r'\d+',physical_data[1])[0]

            dob_string_numbers = re.findall(r'\d+',physical_data[2])
            dob = dob_string_numbers[0] + '/' + dob_string_numbers[1] + '/' + dob_string_numbers[2]
            age = dob_string_numbers[3]

            # POSITION SECTION
            position_data = str(soup.find(attrs={"data-title": "POSITIONS"})).split('\n')[1]
            position = position_data.split("</b>",1)[1][:2]
            # handle positions G, F, C
            if '<' in position:
                position = position[:1]

            # MISC SECTION - DRAFT YEAR AND PICK
            misc_data = str(soup.find(attrs={"data-title": "MISC"})).split('\n')
            drafted_string = [s for s in misc_data if "Pick" in s][0]
            pick = re.findall(r'\d+',drafted_string)[0]

            # ID - USED FOR MERGING WITH OTHER DATA
            id_string = name[0] + str(year) + str(pick)

            # PRE-DRAFT MEASUREMENTS
            pre_draft_link = soup.find('a', href=True, text='Predraft Measurements')
            if pre_draft_link is not None:
                # most recent set of measurements from the table 
                pre_draft_most_recent = pre_draft_link.find_next_sibling().find_all('tr')[1]
                
                # drop first 5 entries as it is height/weight/year
                measurements = [i.get_text() for i in pre_draft_most_recent.find_all('td')[5:]]

                # fix wingspan                
                if measurements[0] != "NA":
                    (ws_feet, ws_inches) = re.findall(r'[-+]?\d*\.\d+|\d+', measurements[0])
                    measurements[0] = int(ws_feet) * 12 + float(ws_inches)
                else:
                   measurements[0] = np.nan

                # fix standing reach
                if measurements[1] != "NA":                
                    (sr_feet, sr_inches) = re.findall(r'[-+]?\d*\.\d+|\d+', measurements[1])
                    measurements[1] = int(sr_feet) * 12 + float(sr_inches)
                else:
                   measurements[1] = np.nan

                # fix body fat
                if measurements[2] != "NA":                     
                    measurements[2] = float(measurements[2])
                else:
                   measurements[2] = np.nan
                # fix no step vert
                if measurements[3] != "NA":   
                    measurements[3] = float(measurements[3])
                else:
                    measurements[3] = np.nan

                # fix max vert
                if measurements[4] != "NA": 
                    measurements[4] = float(measurements[4])
                else:
                    measurements[4] = np.nan
            else:
                continue

            # STATS - Basic Statistics Per 40 Pace Adjusted
            adj_stats_heading = soup.find('h3', text='Basic Statistics Per 40 Pace Adjusted')
            
            if adj_stats_heading is not None:
                # table of pace adjusted stats 
                adj_stats = adj_stats_heading.find_next_sibling().find_next_sibling()
                
                # get most recent NCAA season
                all_seasons = adj_stats.find_all('tr')[1:]
                ncaa_seasons = [s for s in all_seasons if (s.find(attrs={"data-title": "League"}).get_text() == "NCAA")]
                
                if len(ncaa_seasons) == 0:
                    continue

                season = ncaa_seasons[-1]
                
                stats = [0 if i.get_text().isspace() else float(i.get_text()) for i in season.find_all('td')[3:]]
                 
            else:
                continue

            curr_player = [id_string, name, height,int(weight),dob,int(age),position,int(pick),year] 
            curr_player += measurements
            curr_player += stats 
            player_data.append(curr_player)
            # print(player_data)

        # kyle o'quinn has no stats on his stats page so this allows me to skip him and any others like him
        except IndexError:
            continue
    # break
    # player_data = [[td.getText() for td in data_rows[i].findAll('td')]
    #             for i in range(len(data_rows))]

    # # Turn yearly data into a DatFrame
    year_df = pd.DataFrame(player_data, columns=column_headers)

    draft_df = draft_df.append(year_df, ignore_index=True)

# create and insert the Draft_Yr column
# year_df.insert(0, 'Draft_Yr', year)

draft_df = draft_df.convert_objects(convert_numeric=True)

# Replace NaNs with 0s
draft_df = draft_df.fillna(0)
draft_df.to_csv("pre_draft_data_2002_to_2015.csv")
