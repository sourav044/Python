import os
import numpy
import math
import gensim
from gensim.test.utils import common_texts
from newspaper import Article
import pandas as pd
import newspaper
import re
import json
import csv
import nltk
from gensim.models import Word2Vec



article_dataframe = pd.DataFrame()
url_extension = ['.cms', '.html']
biased_word = ["oldpeople", "eliminated", "stop", "insult", "extorted", "burning", "flaunt", "bewitched", "blasts", "unbalanced"]
gensim_model = 'word2vec.model'


def article_collection():
    domains = ["https://timesofindia.indiatimes.com/", "https://edition.cnn.com/"]
    domain_list = []
    for domain in domains:
        for sub_domain_articles in find_articles_links(domain):
            domain_list.append(sub_domain_articles)
    with open('urls.txt', 'w') as file:
        file.write(json.dumps(domain_list))


def find_articles_links(domain):
    all_nested_domain = {}
    root = newspaper.build(domain.strip())
    # if there an article
    for article in root.article_urls():
        all_nested_domain[article] = article
        leaf = newspaper.build(article)
        parsed_domain = re.findall(r'(https?://[^\"]+)', leaf.html)
        for pd in parsed_domain:
            if url_check(pd, domain) != "":
                all_nested_domain[url_check(pd, domain)] = 1
    # if there is an sub domain
    for sub_domain in root.categories:
        if sub_domain not in all_nested_domain.keys():
            leaf = newspaper.build(sub_domain.url)
            parsed_domain = re.findall(r'(https?://[^\"]+)', leaf.html)
            for pd in parsed_domain:
                if url_check(pd, domain) != "":
                    all_nested_domain[url_check(pd, domain)] = 1
            all_nested_domain[sub_domain.url] = sub_domain.url

    # Parent domain Check
    all_nested_domain[domain] = domain
    parsed_domain = re.findall(r'(https?://[^\"]+)', root.html)
    for pd in parsed_domain:
        validate_url = url_check(pd.strip(), domain.strip())
        if validate_url != "":
            if validate_url in all_nested_domain.keys():
                all_nested_domain[validate_url.strip()] = 1

    return all_nested_domain.keys()


def url_check(domain, master_domain):
    if len(domain) > len(master_domain) and (domain.find(master_domain) > -1):
        avail_extension = [ext for ext in url_extension if ext in domain]
        if len(avail_extension) > 0:
            return domain[domain.index('https://'): domain.index(avail_extension[0]) + len(avail_extension[0])].strip()
        else:
            return ""
    else:
        return ""


def download_article():
    file1 = open('urls.txt', 'r')
    data = json.load(file1)
    with open('articles.csv', 'w', newline='', encoding='utf-8') as article_file:
        writer = csv.writer(article_file)
        writer.writerow(["title", "publish_date", 'text'])
        for url in list(data):
            data_coll = fetch_article(url)
            if data_coll != "":
                print(data_coll)
                writer.writerow(data_coll)
    file1.close()


def fetch_article(domain):
    try:
        article = Article(domain)
        article.download()
        article.parse()
        # if article.publish_date > '31-Dec-2017':
        if article.text != "" and article.title != "":
            return [article.title, article.publish_date, article.text]
        return ""
    except:
        return ""
    # return


def pre_process_text(title_data):
    title_data.dropna(inplace=True)
    stop_word_file = open("stop_words.txt", "r")
    stop_words = stop_word_file.readlines()
    for index, row in title_data.iteritems():
        l_text = row.lower()
        sentence = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', str(l_text))
        sentence = re.sub(r'[?|!|\'|’|"|#|@|_|:|“|”|-|"|-|-|<|>|{|}.|,|)|(|\|/]', r'', sentence)

        words = [w for w in sentence.split() if w not in stop_words]
        post_sentence = (" ".join(map(str, words)))

        tokens = nltk.word_tokenize(post_sentence)
        words = [w for w in tokens if w not in stop_words]

        title_data.loc[index] = (" ".join(map(str, words)))
    stop_word_file.close()
    return title_data


def tokenize(sentence):
    tokens = []
    for s in sentence.split():
        token = nltk.word_tokenize(s)
        tokens.append(token)
    return tokens

def word_embedding():
    article_coll = pd.read_csv("articles.csv")
    title_data = article_coll.iloc[:, 0]
    title_data = pre_process_text(title_data)
    model = None
    # path = get_tmpfile("word2vec.model")
    # model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
    # model.save("")
    # model.train(model.train([["hello", "world"]], total_examples=1, epochs=1))
    if os.path.isfile(gensim_model):
        print('Loading pre-trained model: {} ...'.format(gensim_model))
        # Loads the model from the disk
        model = gensim.models.Word2Vec.load(gensim_model)
    else:
        print('Training model: {} ...'.format(gensim_model))
        model = gensim.models.Word2Vec(common_texts, size=150, window=5, min_count=1, workers=4)
        model = model.train(model.train([biased_word], total_examples=len(biased_word), epochs=1))
        print('Saving model as {}'.format(gensim_model))
        # Saves the model to the disk
        model.save(gensim_model)
    for index, row in title_data.iteritems():
        for tk in tokenize(row):
            print(tk, cosine_distance(model, tk, biased_word, len(tk)))


def cosine_distance(model, word, target_list, num):
    cosine_dict ={}
    word_list = []
    a = model[word]
    for item in target_list:
        if item != word:
            b = model[item]
            cos_sim = numpy.dot(a, b)/(numpy.linalg.norm(a)*numpy.linalg.norm(b))
            cosine_dict[item] = cos_sim
    dist_sort = sorted(cosine_dict.items(), key=lambda dist: dist[1], reverse=True)
    for item in dist_sort:
        word_list.append((item[0], item[1]))
    return word_list[0:num]


if __name__ == '__main__':
    ## 1. Article collection from the URL
     # article_collection()
    ## 2.  Download the article
    # download_article()
    ## 3. Word embeddings to calculate the 100 most similar words to each of those words
    word_embedding()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
