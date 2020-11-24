from scrapy.crawler import CrawlerProcess
from scrapy.settings import Settings
from review_parser import settings
from review_parser.spiders.kinopoisk import KinopoiskSpider

if __name__ == '__main__':
    crawler_settings = Settings()
    crawler_settings.setmodule(settings)
    process = CrawlerProcess(settings=crawler_settings)
    process.crawl(KinopoiskSpider)

    process.start()
