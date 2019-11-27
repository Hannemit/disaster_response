import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

import pickle

from collections import Counter


def load_data(database_filepath, data_base_name=None):
    assert database_filepath[-3:] == ".db", "include the .db extension in the database filepath"
    if data_base_name is None:
        data_base_name = database_filepath[:-3]
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table(f"{data_base_name}", engine)
    inputs = df["message"]
    labels = df.drop(columns=["id", "message", "original", "genre"], axis=1)

    return inputs, labels


def process_labels(labels):
    """
    Do some pre-processing on the labels. They're supposed to all be zeros and ones, but some values might
    be no 0 and 1, we replace these values by the mode of the column.
    Also, drop all columns that only have a single value (only zeros or only ones)
    :param labels: pandas dataframe, (n_samples x n_outputs)
    :return: pandas dataframe, (n_samples x n_outputs) with only zeros and ones
    """
    labels[labels > 1] = np.nan
    labels[labels < 0] = np.nan

    for column in labels.columns:
        labels[column].fillna(labels[column].mode()[0], inplace=True)
        labels[column] = labels[column].astype(int)

    columns_only_zeros = labels.columns[labels.sum(axis=0) == 0].values
    columns_only_ones = labels.columns[labels.sum(axis=0) == len(labels)].values
    labels = labels.drop(columns=np.concatenate((columns_only_ones, columns_only_zeros)), axis=1)
    category_names = labels.columns.values

    return labels, category_names


def remove_punctuation(text):
    return re.sub(r"[^a-zA-Z0-9]", " ", text)


def tokenize(text):
    # lowercase
    text = text.lower()

    # remove punctuation
    text = remove_punctuation(text)

    # tokenize into words
    tokens = word_tokenize(text)

    # lemmatize and remove stopwords
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word, pos="v") for word in tokens]

    return tokens


def build_model():
    pipeline = Pipeline([
        ("count_vec", CountVectorizer(tokenizer=tokenize)),
        ("tfidf", TfidfTransformer()),
        ("classifier", MultiOutputClassifier(RandomForestClassifier())),
    ])

    parameters = {"count_vec__max_df": [0.95, 0.99, 1.0],
                  "count_vec__min_df": [0.005, 0.01, 1],
                  # "classifier__estimator__n_estimators": [10, 50, 100],
                  "classifier__estimator__max_features": ["sqrt", "log2"]}
    model = GridSearchCV(pipeline, parameters)
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


def save_model(model, model_filepath):
    assert model_filepath[-4:] == ".pkl", "include the .pkl extension in the model filepath"
    with open(f"{model_filepath}", "wb") as output_file:
        pickle.dump(model, output_file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(f"Loading data...\n    DATABASE: {database_filepath}")
        inputs, labels = load_data(database_filepath)
        labels, category_names = process_labels(labels)

        inputs_train, inputs_test, labels_train, labels_test = train_test_split(inputs, labels, test_size=0.2)
        
        print("Building model...")
        model = build_model()
        
        print("Training model...")
        model.fit(inputs_train, labels_train)
        
        print("Evaluating model...")
        evaluate_model(model, inputs_test, labels_test, category_names)

        print(f"Saving model...\n    MODEL: {model_filepath}")
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print("Please provide the filepath of the disaster messages database "
              "as the first argument and the filepath of the pickle file to "
              "save the model to as the second argument. \n\nExample: python "
              "train_classifier.py ../data/DisasterResponse.db classifier.pkl")


if __name__ == '__main__':
    main()
