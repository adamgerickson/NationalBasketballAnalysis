# -*- coding: utf-8 -*-
import scrapy
from scrapy.selector import Selector
import re


class BBallRefSpider(scrapy.Spider):
    name = "bbref"

    # set the year
    start_urls = [
        'http://www.basketball-reference.com/leagues/NBA_2015.html',
        'http://www.basketball-reference.com/leagues/NBA_2014.html',
        'http://www.basketball-reference.com/leagues/NBA_2013.html',
        'http://www.basketball-reference.com/leagues/NBA_2012.html',
        'http://www.basketball-reference.com/leagues/NBA_2011.html',
        'http://www.basketball-reference.com/leagues/NBA_2010.html',
        'http://www.basketball-reference.com/leagues/NBA_2009.html',
        'http://www.basketball-reference.com/leagues/NBA_2008.html',
        'http://www.basketball-reference.com/leagues/NBA_2007.html',
        'http://www.basketball-reference.com/leagues/NBA_2006.html',
        'http://www.basketball-reference.com/leagues/NBA_2005.html',
        'http://www.basketball-reference.com/leagues/NBA_2004.html',
        'http://www.basketball-reference.com/leagues/NBA_2003.html',
        'http://www.basketball-reference.com/leagues/NBA_2002.html',
        'http://www.basketball-reference.com/leagues/NBA_2001.html',
        'http://www.basketball-reference.com/leagues/NBA_2000.html',
        'http://www.basketball-reference.com/leagues/NBA_2000.html'
    ]

    def parse(self, response):
        east = response.css('#confs_standings_E a::attr(href)').extract()
        west = response.css('#confs_standings_W a::attr(href)').extract()
        urls = east + west

        for url in urls:
            yield scrapy.Request('http://www.basketball-reference.com' + url, callback=self.parse_team) 

    def parse_team(self, response):        
        urls = response.css('#roster td[data-stat="player"] a::attr(href)').extract()
        for url in urls:    #['/players/g/gallida01.html']: #urls:
            yield scrapy.Request('http://www.basketball-reference.com' + url, callback=self.parse_player) 

    def parse_player(self, response): 
            regex = re.compile(r'<!--(.*)-->', re.DOTALL)
            comments = response.xpath('//comment()').re(regex) 

            name = response.css('h1::text').extract_first()

            # PER GAME TABLE
            per_game_seasons = response.css('#all_per_game table tbody tr[class=full_table]')
            career_per_game = response.css('#all_per_game table tfoot tr')[0]
            per_game_seasons.append(career_per_game)


            # PER 100 POSSESSIONS TABLE
            poss_table = [x for x in comments if 'Per 100 Poss Table' in x][0]
            resp = Selector(text = poss_table, type = "html")
            per_poss_seasons = resp.css('#div_per_poss table tbody tr[class=full_table]')
            career_poss = resp.css('#div_per_poss table tfoot tr')[0]
            per_poss_seasons.append(career_poss)

            # ADVANCED TABLE
            adv_table = [x for x in comments if 'Advanced Table' in x][0]
            resp = Selector(text = adv_table, type = "html")
            adv_seasons = resp.css('#div_advanced table tbody tr[class=full_table]')
            career_adv = resp.css('#div_advanced table tfoot tr')[0]
            adv_seasons.append(career_adv)

            # SHOOTING TABLE
            shooting_table = [x for x in comments if 'Shooting Table' in x][0]
            resp = Selector(text = shooting_table, type = "html")
            shooting_seasons = resp.css('#div_shooting table tbody tr[class=full_table]')            
            career_shooting = resp.css('#div_shooting table tfoot tr')[0]
            shooting_seasons.append(career_shooting)


            seasons = zip(per_game_seasons, per_poss_seasons, adv_seasons, shooting_seasons)



            for per_game_season, per_poss_season, adv_season, shooting_season in seasons[-1]: # test if seasons[-1] gives me the current year I am in
                item = {}
                item['name'] = name
                cols = []

                cols += per_game_season.css('td')
                per_game_season_col = per_game_season.css('th')[0]
                cols.append(per_game_season_col)

                cols += per_poss_season.css('td')
                per_poss_season_col = per_poss_season.css('th')[0]
                cols.append(per_poss_season_col)

                cols += adv_season.css('td')
                adv_season_col = adv_season.css('th')[0]
                cols.append(adv_season_col)

                cols += shooting_season.css('td')
                shooting_season_col = shooting_season.css('th')[0]
                cols.append(shooting_season_col)

                for col in cols:
                    label = col.css('::attr(data-stat)').extract()
                    val = col.css('::text').extract()        
                    if label != [''] and val != []:
                        label = label[0]
                        val = val[0]
                        item[label] = val

                yield item
            




