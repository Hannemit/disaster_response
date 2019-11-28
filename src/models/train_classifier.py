import sys

import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import os

from src.data import load_save_data

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

# these are required for the tokenize function to work. Download them once at the start here.
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


def select_inputs_labels(data_frame):
    """
    Given the input dataframe, split it into our input data and our labels
    :param data_frame:
    :return:
    """
    inputs = data_frame["message"]
    labels = data_frame.drop(columns=["id", "message", "original", "genre"], axis=1)
    return inputs, labels


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


def tokenize(text: str):
    """
    tokenize some input text. We convert to lower case, remove punctuation, replace urls, remove stop words,
    etc..
    :param text: string, some text we want to tokenize
    :return: a list of strings, the tokens in the original text
    """

    # lowercase
    text = text.lower()

    # remove punctuation
    text = remove_punctuation(text)

    # replace url's
    text = replace_urls(text)

    # remove numbers, replace with space (they don't really add much)
    text = re.sub("\d", " ", text)

    # tokenize into words
    tokens = word_tokenize(text)

    # lemmatize and remove stopwords
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word, pos="v") for word in tokens]

    return tokens


def build_model(do_gridsearch: bool = True):
    """
    Create a pipeline and define parameters to search over. GridSearchCV will find the best set of parameter
    using cross validation.
    If we do not want to use a grid search (it may take long), just set do_gridsearch to False
    :param do_gridsearch: boolean, if True then return a model which will perform a grid search, otherwise the model
            just consistent of the steps in the pipeline.
    :return: model
    """
    pipeline = Pipeline([
        ("count_vec", CountVectorizer(tokenizer=tokenize)),
        ("tfidf", TfidfTransformer()),
        ("classifier", MultiOutputClassifier(RandomForestClassifier(n_estimators=100))),
    ])

    parameters = {"count_vec__max_df": [0.95, 0.99, 1.0],
                  #"count_vec__min_df": [0.005, 0.01, 1],
                  "classifier__estimator__n_estimators": [50, 100],
                  "classifier__estimator__max_features": ["sqrt", "log2"]
                  }

    if do_gridsearch:
        model = GridSearchCV(pipeline, parameters, cv=5, n_jobs=4, verbose=2)
    else:
        model = pipeline
    return model


def evaluate_model(model, inputs_test, labels_test, category_names):
    """
    Given our model and some input test data with known labels, we evaluate how well the model performs.
    We return a dataframe with the precision, recall and F1 score for each of our output categories.
    :param model: sklearn estimator, the trained model which has a .predict() function
    :param inputs_test: test input, should be e.g. a pandas series of strings
    :param labels_test: pandas dataframe, the known outputs for our inputs, should have same number of rows as
    inputs_test and has multiple column as we're predicting multiple categories
    :param category_names: list of strings, names of our output categories
    :return: pandas dataframe with scores for every output category, each category is a row.
    """
    y_hat = model.predict(inputs_test)

    score_df = pd.DataFrame({"category": category_names, "precision": np.nan, "recall": np.nan, "F1 score": np.nan})

    for ii, col_name in enumerate(category_names):
        pre, rec, score, support = precision_recall_fscore_support(labels_test.iloc[:, ii], y_hat[:, ii], average="weighted")
        score_df.loc[score_df["category"] == col_name, "precision"] = pre
        score_df.loc[score_df["category"] == col_name, "recall"] = rec
        score_df.loc[score_df["category"] == col_name, "F1 score"] = score

    print(score_df)
    print(score_df.mean())


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(f"Loading data...\n    DATABASE: {database_filepath}")
        print(f"current directory {os.getcwd()}")

        df = load_save_data.load_data_from_database(database_filepath)
        inputs, labels = select_inputs_labels(df)
        category_names = labels.columns

        inputs_train, inputs_test, labels_train, labels_test = train_test_split(inputs, labels, test_size=0.2)
        
        print("Building model...")
        model = build_model(do_gridsearch=False)
        
        print("Training model...")
        model.fit(inputs_train, labels_train)

        # print(f"Parameters used are {model.best_params_}")
        
        print("Evaluating model...")
        evaluate_model(model, inputs_test, labels_test, category_names)

        print(f"Saving model...\n    MODEL: {model_filepath}")
        load_save_data.pickle_dump(model, model_filepath)

        print("Trained model saved!")

    else:
        print("Please provide the filepath of the disaster messages database as the first argument and the filepath "
              "of the pickle file to save the model to as the second argument. \n\nExample: "
              "python src/models/train_classifier.py disaster_response.db models/classifier.pkl")


if __name__ == '__main__':
    main()
