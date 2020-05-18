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
    engine = create_engine('sqlite:///data.db')
    df = pd.read_sql_table('df', con=engine)
    X = df.iloc[:,1]
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X,Y,category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),

        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])
    parameters = {'tfidf__norm': ['l1','l2'],'tfidf__sublinear_tf': [True, False],
              'vect__ngram_range': ((1, 1), (1, 2)), 'clf__estimator__min_samples_split': [2, 4],
              'clf__estimator__n_neighbors': [4,5,10],'clf__estimator__weights':['uniform', 'distance']
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred= model.predict(X_test)
    print(classification_report(Y_test,Y_pred,target_names=category_names))
    print("Accuracy scores for each category are:")
    for col in range(36):
    print("Accuracy score for " + y_test.columns[col], accuracy_score(y_test.values[:,col],y_pred[:,col]))
    
def save_model(model, model_filepath):
    pickle.dump(model_filepath, open(filename, 'wb'))


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