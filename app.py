from flask import Flask
from scrapy.crawler import CrawlerRunner
from twisted.internet import reactor
from scrapy.settings import Settings
from review_parser import settings
from review_parser.spiders.kinopoisk import KinopoiskSpider
from scrapy.utils.log import configure_logging

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import dill

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/fetch_data/')
def fetch_data():
    configure_logging()
    crawler_settings = Settings()
    crawler_settings.setmodule(settings)
    runner = CrawlerRunner(settings=crawler_settings)
    runner.crawl(KinopoiskSpider)
    d = runner.join()
    d.addBoth(lambda _: reactor.stop())

    reactor.run()

    return 'All reviews fetched!<BR>Now you can manually train the model by visiting /fit_model/'


@app.route('/fit_model/')
def fit_model():
    data = pd.read_csv('kinopoisk.csv')

    label_encoder = LabelEncoder()
    data_labelled = data.copy()
    data_labelled['grade'] = label_encoder.fit_transform(data['grade'])

    pipeline = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                              alpha=5e-4, random_state=100,
                              max_iter=150, tol=None)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(data_labelled, data_labelled['grade'], random_state=0,
                                                        test_size=0.25)
    pipeline.fit(X_train['text'].values, y_train.values)
    preds = pipeline.predict(X_test['text'].values)
    acc = np.mean(preds == y_test.values)
    full_metrics = metrics.classification_report(y_test.values, preds, target_names=('negative', 'neutral', 'positive'))

    with open("model.dill", "wb") as f:
        dill.dump(pipeline, f)

    return 'Model is ready!<BR>' \
           'Model accuracy score on test subset is {}<BR>{}<BR>' \
           'Now you can make predictions for your reviews on /get_review_type/'\
        .format(str(acc), full_metrics.replace('\n', '<BR>'))


if __name__ == '__main__':
    app.run()
