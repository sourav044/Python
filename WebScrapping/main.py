import os
import gensim
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
biased_word = ["old", "eliminated", "flaunt", "blasts", "unbalanced", "sympathy", "weak", "worst", "insulted", "tumbles"]
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


def pre_process_text(title_data):
    title_data.dropna(inplace=True)
    stop_word_file = open("stop_words.txt", "r")
    stop_words = stop_word_file.readlines()
    for index, row in title_data.iteritems():
        l_text = row.lower()
        sentence = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', str(l_text))
        post_sentence = re.sub(r'[?|!|\'|’|"|#|@|_|:|“|”|-|"|-|-|<|>|{|}.|,|)|(|\|/]', r'', sentence)

        tokens = nltk.word_tokenize(post_sentence)
        words = [w for w in tokens if w not in stop_words]
        post_sentence = (" ".join(map(str, words)))

        title_data.loc[index] = post_sentence
    stop_word_file.close()
    return title_data


def tokenize(sentence):
    tokens = []
    for index, row in sentence.iteritems():
        for s in row.split():
            token = nltk.word_tokenize(s)
            tokens.append(token)
    return tokens


# Computes n=max_count similar words for each word in 'seeds'
def get_similarity_by_word(model, seeds, max_count):
    for seed in seeds:
        similar_words = model.wv.most_similar(seed, topn=max_count)
        print("The {} most similar bias words for the given bias word {}  are: ".format(max_count, seed.upper()))
        print(similar_words)
        df = pd.DataFrame(similar_words, columns=['word', 'cosine_distance'])
        df.to_csv("similar_words_for_{}.csv".format(seed.upper()), encoding='utf-8', index=False)
        print('Result saved as: similar_words_for_{}.csv in directory'.format(seed.upper()))
        print('\n')


def word_embedding():
    article_coll = pd.read_csv("articles.csv")
    title_data = article_coll.iloc[:, 0]
    title_data = pre_process_text(title_data)
    token_title_data = tokenize(title_data)
    model = None
    if os.path.isfile(gensim_model):
        print('Loading pre-trained model: {} ...'.format(gensim_model))
        # Loads the model from the disk
        model = gensim.models.Word2Vec.load(gensim_model)
    else:
        print('Training model: {} ...'.format(gensim_model))
        model = gensim.models.Word2Vec(token_title_data, size=150, window=5, min_count=1, workers=4)
        print('Saving model as {}'.format(gensim_model))
        # Saves the model to the disk
        model.save(gensim_model)
        words = list(model.wv.vocab)
        print('Vocabulary size: %d' % len(words))
        filename = 'embedding_word2vec.txt'
        model.wv.save_word2vec_format(filename, binary=False)
    get_similarity_by_word(model, biased_word, max_count=100)


if __name__ == '__main__':
    ## 1. Article collection from the URL
     #article_collection()
    ## 2.  Download the article
    # download_article()
    ## 3. Word embeddings to calculate the 100 most similar words to each of those words
    word_embedding()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
