"""
fuser -k port_number/tcp
to kill process at port port_number
"""

import json
import plotly

from flask import Flask
from flask import render_template, request
import plotly.graph_objects as gob
from src.models.train_classifier import replace_urls, remove_punctuation, tokenize
from src.data import load_save_data
from src.data.process_data import load_data
from src.models.train_classifier import select_inputs_labels


app = Flask(__name__)

df = load_data("data/raw/disaster_messages.csv", "data/raw/disaster_categories.csv")
model = load_save_data.pickle_load("./models/classifier.pkl")
processed_df = load_save_data.load_data_from_database("disaster_response.db")
_, labels = select_inputs_labels(processed_df)

# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():

    # extract data needed for visuals
    genre_counts = df.groupby("genre").count()["message"]
    genre_names = list(genre_counts.index)
    sum_by_categories = labels.sum()

    # create visuals
    graphs = [
        {
            "data": [
                gob.Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            "layout": {
                "title": "Distribution of Message Genres",
                "yaxis": {
                    "title": "Count"
                },
                "xaxis": {
                    "title": "Genre"
                }
            }
        },
        {
            "data": [
                gob.Bar(
                    x=sum_by_categories.index.values,
                    y=sum_by_categories.values
                )
            ],

            "layout": {
                "title": "Distribution of message categories",
                "yaxis": {
                    "title": "count"
                },
                "xaxis": {
                    "title": "category"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route("/go")
def go():
    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(labels.columns, classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        "go.html",
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()
