# Disaster Response Pipeline Project
In this project, we create a pipeline to classify messages about disasters. Every time a disaster occurs, lots of
messages come in, some are relevant (e.g. "please help us, we need food and water"), but
some might just be news messages reporting about the disaster. Some messages may indicate that 
people need food, other can be about needing medical care, being trapped in a fire, needing a hospital, etc.. 

In our pipeline, we take in raw messages, change them into features and use a random forest
to classify the messages. There are about 35 different categories (water, food, hospital, fire, earthquake, etc..) and
for every message we predict whether or not every category is relevant. This way, messages can
quickly be prioritized by e.g. only looking at messages which are classified as including 'hospital'.  

After training the model and launching the web app, type in a message and it will be classified for you.

### Requirements:

##### Create virtual environment
Create a virtual environment using (linux)  
`conda create -n disaster_response python=3.7.5`
 
 or, for windows  
 `conda create -n disaster_response python=3.7.5 numpy==1.17.3 scipy==1.3.1`

#### install other requirements
First, activate the virtual environment  

`conda activate disaster_response`

then, install the requirements  

`pip install requirements.txt`


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database  
        `python src/data/process_data.py src/data/disaster_messages.csv src/data/disaster_categories.csv disaster_response.db`
    - To run ML pipeline that trains classifier and saves  
        `python src/models/train_classifier.py disaster_response.db models/classifier.pkl`

2. Run the following command (also in the project root directory) to run the web app.  
    `python run.py`

3. Go to http://0.0.0.0:3001/


#### Trouble shooting

run  
`fuser -k *port_number*/tcp` 

if the port is already in use (e.g. fuser -k 3001/tcp)

If using pycharm and the go.html and master.html files are not found, try marking
the templates directory as a template folder (by right clicking and choosing `Mark Directory as`)
