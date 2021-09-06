# Project-Disasters-Response

Link to repository: https://github.com/ariehlev/Project-Disasters-Response
 
 ## Summary
 In this project we have a data set containing real messages that were sent during disaster events. The aim was to create a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.
The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 
 
 
 ## How to run the Python scripts and the web app
 The Python scripts should be run in the following order:
 1. Make sure you are in the root of this repository
 2. To run ETL pipeline that cleans data and stores in database: "python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisastersDB.db"
 3. To run ML pipeline that trains classifier and saves: "python models/train_classifier.py data/DisastersDB.db models/disastermodel.pkl"
 4. Run the following command in the app's directory to run the web app: python run.py
 5. Finally, go to http://0.0.0.0:3001/

## Files in this repository
- app
  - run.py: script to run the web app
  - templates: folder that contains HTML templates
- data
  - disaster_categories.csv: Disaster Categories data set.
  - disaster_messages.csv: Disaster Messages data set.
  - process_data.py: script that reads in the data, cleans it, and stores it in a SQLite database
- models
  - train_classifier.py: script that trains the Machine Learning model
- App screenshots: contains screenshots of the web app
