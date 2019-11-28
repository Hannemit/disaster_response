from sqlalchemy import create_engine
import pickle
import pandas as pd

DATABASE_LOCATION = "data/processed/"


def save_data_to_database(df, database_filename):
    assert database_filename[-3:] == ".db", "database filename must end in '.db'"
    engine = create_engine(f"sqlite:///{DATABASE_LOCATION}{database_filename}")
    df.to_sql(f"{database_filename[:-3]}", engine, index=False)


def load_data_from_database(database_filepath, data_base_name=None):
    assert database_filepath[-3:] == ".db", "include the .db extension in the database filepath"
    if data_base_name is None:
        data_base_name = database_filepath[:-3]
    engine = create_engine(f"sqlite:///{DATABASE_LOCATION}{database_filepath}")
    df = pd.read_sql_table(f"{data_base_name}", engine)

    return df


def pickle_dump(model, model_filepath):
    assert model_filepath[-4:] == ".pkl", "include the .pkl extension in the model filepath"
    with open(f"{model_filepath}", "wb") as output_file:
        pickle.dump(model, output_file)


def pickle_load(model_filepath):
    try:
        with open(model_filepath, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"No model found at {model_filepath}, make sure to create and save the model first..")
