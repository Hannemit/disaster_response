import sys
import pandas as pd
from src.data import load_save_data
import numpy as np


def load_data(messages_filepath: str, categories_filepath: str):
    """
    Read in the datasets and merge them into a single dataframe
    :param messages_filepath: string, path to messages file (should be csv)
    :param categories_filepath: string, path to categories file (should be csv)
    :return: pandas dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge the two datasets
    df = pd.merge(categories, messages, on="id", how="outer")
    return df


def remove_non_informative_categories(categories_df):
    """
    Remove columns which have a single label for the entire input dataset, i.e. categories that have only zeros
    or only ones.
    :param categories_df: pandas dataframe, the category labels
    :return: pandas dataframe with possibly fewer columns
    """
    columns_only_zeros = categories_df.columns[categories_df.sum(axis=0) == 0].values
    columns_only_ones = categories_df.columns[categories_df.sum(axis=0) == len(categories_df)].values
    categories_df = categories_df.drop(columns=np.concatenate((columns_only_ones, columns_only_zeros)), axis=1)
    return categories_df


def create_categories_columns(categories_column):
    """
    Clean the categories column, create many columns out of it with 0's and 1's
    :param categories_column: pandas series, a column from the original categories dataframe.
    :return: pandas dataframe, cleaned categories dataframe containing 0's and 1's.
    """
    # create a dataframe, we now have 36 columns
    categories = categories_column.str.split(";", expand=True)
    # assert len(categories.columns) == 36, f"Need 36 columns, not {len(categories.columns)}, {categories.colunns}"

    # use the first row to extract the new column names
    row = categories.iloc[0]
    category_col_names = [value[:-2] for value in row]
    assert "related" in category_col_names
    assert "hospitals" in category_col_names
    categories.columns = category_col_names

    # convert the values in categories to 0's and 1's. If the original value is not 0 or 1, replace it by the col mode
    for column in categories:
        category_values = categories[column].str[-1]  # get series with last characters, ideally all 0 or 1
        category_values[(category_values != "0") & (category_values != "1")] = np.nan
        categories[column] = category_values

        if categories[column].isnull().sum() == len(categories[column]):
            raise RuntimeError(f"column did not have any 0's or 1's...{categories[column]}")

        # replace nans by mode, and cast as integers
        categories[column].fillna(categories[column].mode()[0], inplace=True)
        categories[column] = categories[column].astype(int)

    categories = remove_non_informative_categories(categories)
    assert categories.isnull().sum().sum() == 0, "should not have nans in dataframe"

    return categories


def remove_duplicates(data_frame):
    """
    Remove duplicate rows from dataframe
    :param data_frame: pandas dataframe
    :return: pandas dataframe without duplicate rows
    """
    num_dups = sum(data_frame.duplicated())
    if num_dups == 0:
        return data_frame
    else:
        df = data_frame.drop_duplicates(keep="first")
        assert sum(df.duplicated()) == 0, "Still duplicates present"
        return df


def clean_data(df):
    """
    Clean our dataframe, this mainly means cleaning the categories column
    :param df: pandas dataframe
    :return: pandas dataframe, cleaned.
    """
    cleaned_categories = create_categories_columns(df["categories"])

    # replace old categories with the cleaned one (which itself is a whole dataframe), then remove duplicates
    df = df.drop(columns=["categories"], axis=1)
    df = pd.concat([df, cleaned_categories], sort=False, axis=1)
    df = remove_duplicates(df)

    return df


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(f"Loading data...\nMESSAGES: {messages_filepath}\nCATEGORIES: {categories_filepath}")
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)
        
        print(f"Saving data...\nDATABASE: {database_filepath}")
        load_save_data.save_data_to_database(df, database_filepath)
        
        print("Cleaned data saved to database!")
    
    else:
        print("Please provide the filepaths of the messages and categories datasets as the first and second argument "
              "respectively, as well as the filepath of the database to save the cleaned data to as the third argument."
              "\n\nExample\n: python src/data/process_data.py data/raw/disaster_messages.csv "
              "data/raw/disaster_categories.csv disaster_response.db")


if __name__ == '__main__':
    main()
