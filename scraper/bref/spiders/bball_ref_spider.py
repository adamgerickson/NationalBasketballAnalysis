# -*- coding: utf-8 -*-
import scrapy
from scrapy.selector import Selector
import re


class BBallRefSpider(scrapy.Spider):
    name = "bbref"
    start_urls = [
        'http://www.basketball-reference.com/leagues/NBA_2017.html',
    ]

    def parse(self, response):
        east = response.css('#confs_standings_E a::attr(href)').extract()
        west = response.css('#confs_standings_W a::attr(href)').extract()
        urls = east + west

        for url in urls:
            yield scrapy.Request('http://www.basketball-reference.com' + url, callback=self.parse_team) 

    def parse_team(self, response):        
        urls = response.css('#roster td[data-stat="player"] a::attr(href)').extract()
        for url in urls:
            yield scrapy.Request('http://www.basketball-reference.com' + url, callback=self.parse_player) 

    def parse_player(self, response):  
            item = {}
            name = response.css('h1::text').extract_first()
            item['name'] = name

            # PER GAME TABLE
            career_per_game = response.css('#all_per_game table tfoot tr')[0]
            cols = career_per_game.css('td')[4:]
            
            for col in cols:
                label = col.css('::attr(data-stat)').extract()
                val = col.css('::text').extract()
                if label != [''] and val != []:
                    label = label[0]
                    val = val[0]
                    item[label] = val
            
            # ALL OTHER TABLES
            # 
            # All of the other tables are loaded initially as comments and then
            # some JS inserts them in DOM. Hence why I parse the comments.
            #

            regex = re.compile(r'<!--(.*)-->', re.DOTALL)
            comments = response.xpath('//comment()').re(regex)
            
            # PER POSS TABLE
            poss_table = [x for x in comments if 'Per 100 Poss Table' in x][0]
            resp = Selector(text = poss_table, type = "html")
            career_poss = resp.css('#div_per_poss table tfoot tr')[0]
            cols = career_poss.css('td')[7:]
            for col in cols:
                label = col.css('::attr(data-stat)').extract()
                val = val = col.css('::text').extract()
                if label != [''] and val != []:
                    label = label[0]
                    val = val[0]
                    item[label] = val

            # ADVANCED TABLE
            adv_table = [x for x in comments if 'Advanced Table' in x][0]
            resp = Selector(text = adv_table, type = "html")
            career_adv = resp.css('#div_advanced table tfoot tr')[0]
            cols = career_adv.css('td')[7:]
            for col in cols:
                label = col.css('::attr(data-stat)').extract()
                val = val = col.css('::text').extract()
                if label != [''] and val != []:
                    print(label, val)
                    label = label[0]
                    val = val[0]
                    item[label] = val

            # SHOOTING TABLE
            shooting_table = [x for x in comments if 'Shooting Table' in x][0]
            resp = Selector(text = shooting_table, type = "html")
            career_shooting = resp.css('#div_shooting table tfoot tr')[0]
            cols = career_shooting.css('td')[7:]
            for col in cols:
                label = col.css('::attr(data-stat)').extract()
                val = val = col.css('::text').extract()
                if label != [''] and val != []:
                    print(label, val)
                    label = label[0]
                    val = val[0]
                    item[label] = val

            yield item



