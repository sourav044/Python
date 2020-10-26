from newspaper import Article
import pandas as pd
import newspaper
import re

all_domain = {}
article_dataframe = pd.DataFrame()


def article_collection():
    domains = ["https://timesofindia.indiatimes.com/", "https://www.bbc.com/", "https://edition.cnn.com/"]
    article_data_frame = []
    for domain in domains:
        for sub_domain_articles in find_articles_links(domain):
            print(sub_domain_articles)
            article_data_frame.append(fetch_article(sub_domain_articles))
    print(article_data_frame)


def find_articles_links(domain):
    root = newspaper.build(domain)
    parsed_domain = re.findall(r'(https?://[^\"]+)', root.html)

    for article in root.article_urls():
        all_domain[article.url] = len(article.text)
    for sub_domain in root.categories:
        if sub_domain not in all_domain.keys():

            print(sub_domain.url)
            return find_articles_links(sub_domain.url)
            # leaf = newspaper.build(sub_domain)
            # if len(leaf.articles) != 0:
            #    sub_domain.append(sub_domain)
    return


def temp():
    url = 'https://www.japantimes.co.jp/news/2020/10/26/national/yoshihide-suga-carbon-pledge-japan/'
    print(fetch_article(url))


def fetch_article(domain):
    url = 'http://fox13now.com/2013/12/30/new-year-new-laws-obamacare-pot-guns-and-drones/'
    article = Article(url)
    print(domain)
    article = Article(domain)
    return [[article.title, article.authors, article.publish_date, article.text]]



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
