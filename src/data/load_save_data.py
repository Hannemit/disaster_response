from sqlalchemy import create_engine
import pickle
import pandas as pd

DATABASE_LOCATION = "data/processed/"


def save_data_to_database(df, database_filename: str, table_name=None):
    """
    save a dataframe to sqlite database
    :param df: pandas dataframe
    :param database_filename: string, name of the database, must include the .db
    :param table_name: string, name of table. If None then the table name is taken to be the same as the
                database name
    :return:
    """
    assert database_filename[-3:] == ".db", "database filename must end in '.db'"
    if table_name is None:
        table_name = database_filename[:-3]
    engine = create_engine(f"sqlite:///{DATABASE_LOCATION}{database_filename}")
    df.to_sql(f"{table_name}", engine, index=False)


def load_data_from_database(database_filepath: str, table_name=None):
    """
    load from a sqlite database, return as pandas dataframe
    :param database_filepath: name of database, must include the .db at the end
    :param table_name: string or None, name of the data base table. If None, it's assumed that it's the same
                as the data base name
    :return: pandas dataframe
    """
    assert database_filepath[-3:] == ".db", "include the .db extension in the database filepath"
    if table_name is None:
        table_name = database_filepath[:-3]
    engine = create_engine(f"sqlite:///{DATABASE_LOCATION}{database_filepath}")
    df = pd.read_sql_table(f"{table_name}", engine)

    return df


def pickle_dump(model, model_filepath: str):
    """
    use pickle to save a model
    :param model: the model we want to save
    :param model_filepath: string, path of where we want to save the model
    :return:
    """
    assert model_filepath[-4:] == ".pkl", "include the .pkl extension in the model filepath"
    with open(f"{model_filepath}", "wb") as output_file:
        pickle.dump(model, output_file)


def pickle_load(model_filepath: str):
    """
    load a pickled model
    :param model_filepath: string, path to where model was pickled
    :return: the model
    """
    try:
        with open(model_filepath, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"No model found at {model_filepath}, make sure to create and save the model first..")
