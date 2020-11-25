from flask import Flask
from flask import render_template
from flask import request
import subprocess

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
review_type = ['Negative', 'Neutral', 'Positive']

@app.route('/')
def index_view():
    return '<h1>Review Analyzer</h1><BR><BR>' \
           'To analyze your review, please proceed to /check_review/<BR>' \
           'To get new training data from kinopoisk.ru go to /fetch_data/<BR>' \
           'To train model on newly-fetched data visit /fit_model/'


@app.route('/fetch_data/')
def fetch_data():

    spider_name = "kinopoisk"
    subprocess.check_output(['scrapy', 'crawl', spider_name, "-o", "output.json"])
    # with open("output.json") as items_file:
    #     return items_file.read()

    return 'All reviews fetched!<BR>Now you can manually train the model by visiting /fit_model/'


@app.route('/fit_model/')
def fit_model():
    data = pd.read_csv('kinopoisk.csv')

    label_encoder = LabelEncoder()
    data_labelled = data.copy()
    data_labelled['grade'] = label_encoder.fit_transform(data['grade'])

    pipeline = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer(use_idf=True)),
        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                              alpha=1e-9, random_state=0,
                              max_iter=1000, tol=None)),
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
           'Now you can make predictions for your reviews on /check_review/'\
        .format(str(acc), full_metrics.replace('\n', '<BR>'))


@app.route('/check_review/', methods=['GET', 'POST'])
def check_review():
    if request.method == 'GET':
        return render_template('check_model.html')
    if request.method == 'POST':
        with open("model.dill", "rb") as f:
            model = dill.load(f)
        result = model.predict([request.form['review']])
        return render_template('result.html', result=review_type[int(result)])


if __name__ == '__main__':
    app.run()
