# Disaster Response Pipeline Project

## Summary of Projects
A data set containing real messages that were sent during disaster events is present using that, A web app is build where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

## Files Description
1- **disaster_messages.csv and disaster_categories.csv**
The dataset we use in this project, `disaster_messages.csv` contains the messages with traslation and genre of the message. `disaster_categories.csv` contains the categories of each message.

2- **data/process_data.py**
This python file is used to clean the data and load it. Basically an ETL pipeline formation takes place in this file.

3- **models/train_classifier.py**
In this python file the clean files are extracted from Sqlite Database and then the classifiers are trained, I have used KMeansClassifier for training and GridSearchCv for tuning the hyperparameters. Due to Computational Restraints, two paramenters could be used, users can try using more.

4- **run.py**
This file deploys to the website, where we can see visualizations and enter our own message for classification.

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Github Repository Link
https://github.com/chandiwalaaadhar/DisasterPipeline
