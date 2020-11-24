import scrapy
from scrapy.http import HtmlResponse
from review_parser.items import ReviewParserItem
from lxml import html
import re


class KinopoiskSpider(scrapy.Spider):
    name = 'kinopoisk'
    allowed_domains = ['kinopoisk.ru']
    start_urls = ['https://www.kinopoisk.ru/lists/films/']

    def parse(self, response: HtmlResponse):
        categories = response.css('a.film-lists-item::attr(href)').extract()
        for category in categories:
            yield response.follow(category, callback=self.parse_category)

    def parse_category(self, response: HtmlResponse):
        pattern = re.compile(r'''(.*?page)(=\d?)(&.*?)''')
        match = re.match(pattern, response.url)

        if match:
            cur_page = match.group(2).replace('=', '')
            next_page = re.sub(pattern, r'\1={}\3'.format(str(int(cur_page)+1)), response.url)
        else:
            next_page = response.css('a.paginator__page-relative::attr(href)').extract_first()

        if next_page:
            yield response.follow(next_page, callback=self.parse_category)

        movie_links = response.css('a.selection-film-item-meta__link::attr(href)').extract()
        for link in movie_links:
            yield response.follow(link, callback=self.parse_movie)

    def parse_movie(self, response: HtmlResponse):
        menu_links = response.css('a.styles_itemDefault__1qbaT::attr(href)').extract()
        for link in menu_links:
            if 'review' in link:
                yield response.follow(link, callback=self.parse_reviews)
                break

    @staticmethod
    def parse_reviews(response: HtmlResponse):
        reviews = response.css('div.reviewItem').extract()
        for review_item in reviews:
            tree = html.fromstring(review_item)
            text = tree.xpath('//span[@itemprop="reviewBody"]/text()')
            if text:
                text = ''.join(text)
                grade = 0
                if 'response neutral' in review_item:
                    grade = 0.5
                if 'response good' in review_item:
                    grade = 1

                result = {'text': text, 'grade': grade}
                return ReviewParserItem(result)

        return None
