"""
fuser -k port_number/tcp
to kill process at port port_number
"""

import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask import Flask
from flask import render_template, request, jsonify
import plotly.graph_objects as gob
# from sklearn.externals import joblib
from sqlalchemy import create_engine
import pickle
import re


def replace_urls(string_input: str, replace_by: str = "URL"):
    """
    Replace url's in a string by replace_by
    :param string_input: string input
    :param replace_by: string, what we want to replace the url with
    :return: string, with urls replaced by replaced_by
    """
    return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', replace_by,
                  string_input)


def remove_punctuation(text):
    return re.sub(r"[^a-zA-Z0-9]", " ", text)


def tokenize(text):
    # lowercase
    text = text.lower()

    # remove punctuation
    text = remove_punctuation(text)

    # replace url's
    text = replace_urls(text)

    # remove numbers, replace with space (they don't really add much I think)
    text = re.sub("\d", " ", text)

    # tokenize into words
    tokens = word_tokenize(text)

    # lemmatize and remove stopwords
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word, pos="v") for word in tokens]

    return tokens


app = Flask(__name__)

# load data
engine = create_engine('sqlite:///data/disaster_response.db')
df = pd.read_sql_table('disaster_response', engine)

# load model
with open("./models/classifier.pkl", "rb") as file:
    model = pickle.load(file)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                gob.Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
