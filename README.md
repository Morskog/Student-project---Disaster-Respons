# Student-project---Disaster-Respons
In this student project we get a real dataset from Figure Eight where we would make a model that classifies disaster messages. We will make a maskinlearing model that classifies this messages into one of 36 categories. In the end we made a APP where the classification of the messages can be classified in one or more of the categories. The messages came in three different genres and 36 categories.


 

# Usage
The project consists of three different areas:
	-	Write a ETL pipeline to process_data, who should do:
	
 	 	o	Loads the messages and categories datasets
		
  		o	Merge the two datasets
	
  		o	Clean the data
	
  		o	Store the datasets in a SQLite database
	
	-	Write a machine learning pipeline to train_classifier:
	
  		o	Loads the data from the SQLite database
		
  		o	Splits the data into training and test sets
		
  		o	Build a machine learning pipeline 
		
  		o	Train and tune the model
		
  		o	Show at test set with the model
		
  		o	Export the model as a pickle file
		
	-	Make a web app:
	
  		o	The app shows the classifications results
 
# The repository
The repository consist of three folders and one readme file:
	
	-	data folders consist of four files
	
		o	Process_data.py
			* The model that load and cleans the data

		o	disaster_messages.csv
			* Input data

		o	disaster_categories.csv
			* Input data

		o	DisasterResponse.db
			* The database with the cleaned data

	-	Models folders consist of two files:
		o	Train_classifier.py
			* The trained model
		o	classifier.pkl
			* The saved model
		
	-	APP folders consist of one folder and one file

		o	run.py
			* The code that start and run the APP

		o	Templates folder with two html files
		
# To run the APP:
First step:
	-	Run the process_data.py, disaster_messages.csv, disaster_categories.csv and DisasterRespons.db in the data folder

Second step:
	- 	Run the train_classifier.py in the models fail, start the DisasterRespns.db in the data file and the classifier.pkl in 			the models file
	
Third step:
	- 	Go to http://0.0.0.0:3001/
	
# Thanks to:
Udacity lessons and coworkers
