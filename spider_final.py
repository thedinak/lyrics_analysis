import scrapy
from scrapy.http import Request
import re
import json
import jsonlines

list_artists = ['Queen', 'Muse', 'Janelle-Monáe', 'Hot-Chip', 'LCD-Soundsystem','The-Postal-Service', 'Daft-Punk', 'The-Strokes']

#inherit from scrapy.Spider
class LyricsSpider(scrapy.Spider):
    #define the name of the spider which we will call from CLI
    name = 'scrape_lyrics'
    custom_settings = {'DOWNLOAD_DELAY': 1.0,}
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
        lyrics = response.xpath('//pre[@id="lyric-body-text"]/text() | //pre[@id="lyric-body-text"]/a/text()').getall()
        lyrics = ''.join(lyrics)
        title = response.xpath("//h1[@id='lyric-title-text']/text()").getall()
        title = ''.join(title)
        artist = response.xpath("//h3[@class='lyric-artist']/a/text()").getall()
        artist = ', '.join(artist)

            #create a dictionary to store the scraped info
        scraped_info = {
                'artists' : artist,
                'titles' : title,
                'lyrics' : lyrics
            }

            #yield or give the scraped info to scrapy
        yield scraped_info

        with jsonlines.open('lyrics.jl', 'a') as fp:
            fp.write(scraped_info) #this works for pandas
        #with open('lyrics.json', 'a') as fp: # this does not open well in pandas
                #json.dump(scraped_info, fp)
