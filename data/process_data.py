import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # Import Data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Drop Duplicates
    categories = categories.drop_duplicates(subset=['id'])
    messages = messages.drop_duplicates(subset=['id'])

    # Merge Data
    df = messages.merge(categories, how='inner', on=['id'])

    # Drop Duplicates
    df = df.drop_duplicates()

    # create a dataframe of the 36 individual category columns
    categories = categories.categories.str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.loc[0, :]

    # Extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2])

    # Rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str)
        categories[column] = categories[column].apply(lambda x: x[-1:])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    return df


def clean_data(df):
    # drop duplicates
    df = df.drop_duplicates()

    # Drop Nulls
    df = df.dropna(subset=['aid_related', 'id'])

    # Drop 'original' Column
    df = df.drop(columns=['original'])

    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(str(database_filename)))
    # engine = create_engine('sqlite:///{}'.format(str(database_filepath)))

    df.to_sql('DisasterMessages', engine, chunksize=5, index=False, if_exists='replace')


# df2.to_sql('cat', engine, chunksize= 5, if_exists='replace')
#
# df2 = df[0:999]


def main():
    if len(sys.argv) == 4:
        # messages_filepath = 'data/disaster_messages.csv'
        # categories_filepath = 'data/disaster_categories.csv'
        # database_filepath = 'data/DisasterResponse.db'

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
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
