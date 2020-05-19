import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score

def load_data(database_filepath):
    """
    Loads data from database
    Args:
        database_filepath: path to database
    Returns:
        (DataFrame) X: feature
        (DataFrame) Y: labels
    """
    # reading the cleaned data set and initializing the features and labels
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disastertab', con=engine)
    X = df.iloc[:,1]
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Tokenizes a given text.
    Args:
        text: text string
    Returns:
        (str[]): array of clean tokens
    """
    # Cleaning the text by forming tokens and lemmatizing them
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Builds classification model """
    # Builiding of a pipeline, with set of transforment and machine learning model is implemented
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()), ('clf', MultiOutputClassifier(KNeighborsClassifier()))])
    #Preparation for GridSearch
    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),'tfidf__norm':['l1','l2'], 'clf__estimator__n_neighbors': [5,10],
    'clf__estimator__weights': ['uniform', 'distance']}

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=-1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model against a test dataset
    Args:
        model: Trained model
        X_test: Test features
        Y_test: Test labels
        category_names: String array of category names
    """
    
    y_pred = model.predict(X_test)
    #Printing Classificiation Report
    print(classification_report(Y_test ,y_pred, target_names=category_names))
    print("Accuracy Score are\n")
    for col in range(36):
        print("Accuracy score for " + Y_test.columns[col], accuracy_score(Y_test.values[:,col],y_pred[:,col]))


def save_model(model, model_filepath):
    """
    Save the model to a Python pickle
    Args:
        model: Trained model
        model_filepath: Path where to save the model
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
