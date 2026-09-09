# webScrapping/crawlers/Crawler.py
from datetime import datetime

class Crawler():
    def __init__(self, url):
        self.url = url
        self.fecha = datetime.now().strftime("%d/%m/%y")

    @staticmethod
    def crawl(): 
        # to be implemented by the child class
        pass
    