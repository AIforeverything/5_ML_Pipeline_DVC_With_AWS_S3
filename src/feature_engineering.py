import os
import logging
import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml

# loggging all the steps
# making the logs directory
logs_dir= "logs"
os.makedirs(logs_dir,exist_ok=True)

# logging configuration
# 1. making the object of logging with some name (here 'data_ingestion') and with level
logger= logging.getLogger('feature_engineering')
logger.setLevel("DEBUG")

# 2. making the console handler to display the message on the console/ stream
console_handler= logging.StreamHandler()
console_handler.setLevel("DEBUG")

#3. making file handler with file
log_file_path= os.path.join(logs_dir,'feature_engineering.log')
file_handler= logging.FileHandler(filename=log_file_path)
file_handler.setLevel("DEBUG")

#4. making the format for messages
format= logging.Formatter("%(asctime)s -%(name)s -%(levelname)s -%(message)s")

#5.adding the format to console and file handlers
console_handler.setFormatter(format)
file_handler.setFormatter(format)

# 6. attaching both the  consolde and file handlers to logger object
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# function to load parameters from the yaml file
def load_params(params_path:str)->dict:
    """This function loads parameters from the yaml file."""
    try:
        with open(params_path,'r') as file:
            params= yaml.safe_load(file)
        logger.debug('Parameters are received from: %s',params_path)    
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('Yaml error: %s',e)
        raise
    except Exception as e:
        logger.error("Unexcepted error was occurred: %s", e)
        raise

# function to load data from the given url
def load_data(file_path:str)->pd.DataFrame:
    """This function loads the data from the given path into a pandas dataframe."""
    try:
        df= pd.read_csv(file_path)
        logger.debug('Data is loaded from the given url %s',file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file: %s",e) 
        raise
    except Exception as e:
        logger.error("Unexpected error has occurred: %s",e)   
        raise 

def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """Apply TfIdf to the data."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)

        X_train = train_data['text'].values
        y_train = train_data['target'].values
        X_test = test_data['text'].values
        y_test = test_data['target'].values

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        logger.debug('tfidf applied and data transformed')
        return train_df, test_df
    except Exception as e:
        logger.error('Error during tfidf transformation: %s', e)
        raise

def save_data(df:pd.DataFrame,file_path:str)->None:
    """This function is used to save the dataframe as csv in the given directory.""" 
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        df.to_csv(file_path,index=False)
        logger.debug("File saved to path: %s",file_path)  
    except Exception as e:
        logger.error("Error occurred while saving the file: %s",e)  
        raise 

def main():
    try:
        # loading params
        params= load_params('params.yaml')
        features= params["feature_engineering"]["max_features"]
        #loading data
        train_data= pd.read_csv("./data/interim/train_preprocessed.csv")
        train_data.dropna(inplace=True)
        test_data= pd.read_csv("./data/interim/test_preprocessed.csv")
        test_data.dropna(inplace=True)
        # applying tfidf
        train_df,test_df = apply_tfidf(train_data,test_data,features)

        save_data(train_df,os.path.join("./data","processed","train_tfidf.csv"))
        save_data(test_df,os.path.join("./data","processed","test_tfidf.csv"))

        logger.debug("data is saved after TFIDF")
    except Exception as e:
        logger.error("Error occurred while applying TFIDF : %s",e)   
        print(f"Error: {e}") 

if __name__=="__main__":
    main()

