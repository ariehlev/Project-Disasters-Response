import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load Messages and Categories data from filepaths

    INPUT
    messages_filepath -- str, path to file
    categories_filepath -- str, path to file

    OUTPUT
    df - pandas DataFrame, combined Messages and Categories data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")
    return df


def clean_data(df):
    """Clean data included in the DataFrame and transform categories part

    INPUT
    df -- pandas DataFrame, combined Messages and Categories data

    OUTPUT
    df -- pandas DataFrame, clean combined Messages and Categories data
    """
    categories = df['categories'].str.split(";", expand=True)
    row = categories.loc[0,]
    category_colnames = row.apply(lambda x: x.split("-")[0])
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)

    df.drop(['categories'], axis=1, inplace = True)

    df = pd.concat([df, categories], axis=1)

    df.drop_duplicates(inplace=True)

    df = df[(df.iloc[:, 4:] != 2).all(axis=1)]

    return df


def save_data(df, database_filename):
    """Saves DataFrame (df) to SQLite database path

    INPUT
    df -- pandas DataFrame, cleancombined Messages and Categories data
    database_filename -- str, Path to SQLite database
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisastersTable', engine, if_exists = 'replace', index=False)


def main():
    """Main function which will:
    - Load the data
    - Clean it
    - Save it in a Database
    """
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
