import sys
import os

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
import pickle


def load_data(database_filepath):
    """
    Loads the data from the Database

    INPUT:
        database_filepath -- str,  path to SQLite database
    OUTPUT:
        X -- pandas df, containins features
        Y -- pandas df, containins labels
        category_names -- list, contains category names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisastersTable', con=engine)
    X = df['message']
    Y = df[df.columns[4:]]
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    """
    Tokenize the text

    INPUT:
        text -- str, message to be tokenized
    OUTPUT:
        words -- list, tokens extracted from text
    """
    # Normalizing text
    normalized_text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenizing text
    words = word_tokenize(normalized_text)
    words = [w for w in words if w not in stopwords.words("english")]
    # Stemming text
    words = [PorterStemmer().stem(w) for w in words]
    words = [WordNetLemmatizer().lemmatize(w) for w in words]

    return words


def build_model():
    """
    Builds Pipeline

    Output:
        cv_model -- ML Pipeline, processes text messages and applies classifier.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {'clf__estimator__n_estimators': [50, 100],
              'clf__estimator__min_samples_split': [2, 3],}

    cv_model = GridSearchCV(pipeline, param_grid=parameters)

    return cv_model


def evaluate_model(pipeline, X_test, Y_test, target_names):
    """
    Evaluates Model

    INPUT:
        pipeline -- Scikit ML Pipeline
        X_test -- Test features
        Y_test -- Test labels
        taget_names -- label names
    """
    y_pred = pipeline.predict(X_test)

    print(classification_report(Y_test.values, y_pred, target_names=target_names))


def save_model(model, model_filepath):
     """
    Save model as a pickle file

    INPUT:
        model -- GridSearchCV or Scikit ML Pipeline object
        model_filepath -- str, path to save .pkl file on

    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Main function:
        1) Extract the data from the SQLite database
        2) Train the Machine Learning model
        3) Evalueate model performance
        4) Save the trained model as Pickle file
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
