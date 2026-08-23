import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
from sklearn.model_selection import train_test_split
import logging
import yaml

# loggging all the steps
# making the logs directory
logs_dir= "logs"
os.makedirs(logs_dir,exist_ok=True)

# logging configuration
# 1. making the object of logging with some name (here 'data_ingestion') and with level
logger= logging.getLogger('data_ingestion')
logger.setLevel("DEBUG")

# 2. making the console handler to display the message on the console/ stream
console_handler= logging.StreamHandler()
console_handler.setLevel("DEBUG")

#3. making file handler with file
log_file_path= os.path.join(logs_dir,'data_ingestion.log')
file_handler= logging.FileHandler(filename=log_file_path)
file_handler.setLevel("DEBUG")

#4. making the format for messages
format= logging.Formatter("%(asctime)s -%(name)s -%(level)s -%(levelname)s -%(message)s")

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
        logger.error("Unexcepted error was occurred: %s", params_path)
        raise

# function to load data from the given url
def load_data(file_url:str)->pd.DataFrame:
    """This function loads the data from the given url into a pandas dataframe."""
    try:
        df= pd.read_csv(file_url)
        logger.debug('Data is loaded from the given url: %s',file_url)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file: %s",e) 
        raise
    except Exception as e:
        logger.error("Unexpected error has occurred: %s",e)   
        raise

    