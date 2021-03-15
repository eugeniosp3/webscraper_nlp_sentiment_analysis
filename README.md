
-- This project is a scrape of a software review site. 

ClassificationModels.ipynb and ClassificationModels.py are the files holding the information which manages the machine learning models to be used in classification. Also uses multiple scoring methods to evaluate model performance. 

Xxxxxx.csv are files which are used in different parts of the code in both model training and data exporting and analysis before classification.
soft_scrape - New Frame.pdf is the document which summarizes the plan for scraping the site we chose for reviews. 

tables schema shows the information for how our tables are created and structure within the MySQL database. 

text_dash.twb is the current state of the tableau dashboard creation process as we identify business oportunities for using this data as a viable method of making new transcations. 

The Main Notebook.ipynb is the jupyter notebook file used to create the models and analyze our data. 

The clean_data.py file is the functionalized code which will process data files from our scraped data database. 

The scraper.py file is the functionalized code for our scraper which uses a python mysql.connector object to connect to a local MySQL database. 

wordcloud11.png is an example of a wordcloud derivative from our code based on the words included in our merged_columns column in our dataframe used for model training. 
