#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import scrapy
from pymongo import MongoClient
from pymongo import errors
import pandas as pd
from review_parser.items import ReviewParserItem


class ReviewParserPipeline:

    def __init__(self):
        client = MongoClient('mongo', 27017)  # service name in docker-compose.yml
        self.db = client['scrapyDB']
        self.db["kinopoisk"].drop()

    def process_item(self, item: ReviewParserItem, spider):

        result = {'text': item['text'], 'grade': item['grade']}

        try:
            self.db[spider.name].insert_one(result)
        except(errors.WriteError, errors.WriteConcernError) as e:
            print('ERROR inserting the row')
            print(e)

        return item

    def close_spider(self, spider: scrapy.Spider):
        self.extract_csv(spider)
        spider.close(spider, None)

    def extract_csv(self, spider: scrapy.Spider):
        data = pd.DataFrame()
        for review in self.db[spider.name].find({}):
            data = data.append({'text': review['text'], 'grade': review['grade']}, ignore_index=True)
        data.to_csv(spider.name + '.csv')
