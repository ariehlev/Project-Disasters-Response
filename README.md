# Project-Disasters-Response
 
 ## How to run the Python scripts and the web app
 The Python scripts should be run in the following order:
 1. Make sure you are in the root of this repository
 2. To run ETL pipeline that cleans data and stores in database: "python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisastersDB.db"
 3. To run ML pipeline that trains classifier and saves: "python models/train_classifier.py data/DisastersDB.db models/disastermodel.pkl"
 4. Run the following command in the app's directory to run the web app: python run.py
 5. Finally, go to http://0.0.0.0:3001/

## Files in this repository
