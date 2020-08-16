import scrapy
from scrapy.http import Request
import re
import json

list_artists = ['Queen']

#inherit from scrapy.Spider
class LyricsSpider(scrapy.Spider):
    #define the name of the spider which we will call from CLI
    name = 'scrape_lyrics'
    custom_settings = {'DOWNLOAD_DELAY': 10.0,}
    #set the range of urls which can be scraped over
    allowed_domain = ['https://www.lyrics.com/']

    def start_requests(self):
        urls = ['https://www.lyrics.com/artist/'+str(artist_name) for artist_name in list_artists]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):

        #step 2: find all their song urls
        songs = ['https://www.lyrics.com' + item for item in response.xpath("//td[@class='tal qx']/strong/a/@href").getall()]
        for song in songs:
            #artist_name_iterative = song.split('/')[-2]
            #yield artist_name_iterative
            yield Request(song)

        #step 3: scrape the lyrics from this page
        lyrics = response.xpath("//pre[@id='lyric-body-text']/text()").getall()
        titles = response.xpath("//h1[@id='lyric-title-text']/text()").getall()
        artists = response.xpath("//h3[@class='lyric-artist']/a/text()").getall()

        #Give the extracted content row wise
        for item in zip(artists,titles,lyrics):
            #create a dictionary to store the scraped info
            scraped_info = {
                'artists' : item[0],
                'titles' : item[1],
                'lyrics' : item[2]
            }

            #yield or give the scraped info to scrapy
            yield scraped_info

        with open('lyrics.json', 'w') as fp:
            json.dump(scraped_info, fp)
