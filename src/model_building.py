import os
import pandas as pd 
import numpy as np 
import pickle
from sklearn.ensemble import RandomForestClassifier
import logging 
import yaml

# loggging all the steps
# making the logs directory
logs_dir= "logs"
os.makedirs(logs_dir,exist_ok=True)

# logging configuration
# 1. making the object of logging with some name (here 'data_ingestion') and with level
logger= logging.getLogger('model_building')
logger.setLevel("DEBUG")

# 2. making the console handler to display the message on the console/ stream
console_handler= logging.StreamHandler()
console_handler.setLevel("DEBUG")

#3. making file handler with file
log_file_path= os.path.join(logs_dir,'model_building.log')
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
    """This function loads the data from the given path into a pandas dataframe.
    :param file_path: Path to the CSV file
    :return: Loaded DataFrame
    """
    try:
        df= pd.read_csv(file_path)
        logger.debug('Data is loaded from the given %s',file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file: %s",e) 
        raise
    except Exception as e:
        logger.error("Unexpected error has occurred: %s",e)   
        raise 

def train_model(X_train:np.ndarray,y_train:np.ndarray,params:dict)->RandomForestClassifier:
    """
    Train the RandomForest model.
    
    :param X_train: Training features
    :param y_train: Training labels
    :param params: Dictionary of hyperparameters
    :return: Trained RandomForestClassifier
    """

    try:
        # checking the shape of X_train, y_train
        if X_train.shape[0] != y_train.shape[0]:
            logger.debug("The number of samples in X_train is not equal to y_train") 
            raise ValueError("The number of samples in X_train is not equal to y_train")
        
        #training the model
        logger.debug("Initializing Random Forest Classifier with paramaters: %s",params)
        clf= RandomForestClassifier(n_estimators=params['model_building']["n_estimators"],random_state=params['model_building']["random_state"])
        clf.fit(X_train,y_train)
        logger.debug("Model training is completed.")

        return clf

    except ValueError as e:
        logger.error("Value error during model training: %s",e)
        raise

    except Exception as e:
        logger.error("Error during model training: %s",e)
        raise

def save_model(model:RandomForestClassifier,file_path:str)->None:
    """
    Save the trained model to a file.
    
    :param model: Trained model object
    :param file_path: Path to save the model file
    """

    try: 
        # ensuring the existency of directory
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        # saving the model
        with open(file_path,'wb') as file:
            pickle.dump(model,file)
        logger.debug("model is saved to %s",file_path) 
    except FileNotFoundError as e:
        logger.error("File path is not found: %s",e)  
        raise
    except Exception as e:
        logger.error("Error was occurred while saving the model: %s",e)  
        raise   

def main():
    try:
        params = load_params("params.yaml")

        train_data = load_data("./data/processed/train_tfidf.csv")

        # Separate features and target: pandas slicing is through iloc
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        logger.debug("X_train shape: %s", X_train.shape)
        logger.debug("y_train shape: %s", y_train.shape)

        clf = train_model(X_train, y_train, params)

        model_save_path = 'models/model.pkl'
        save_model(clf, model_save_path)

        logger.debug("Model is saved to path: %s", model_save_path)

    except Exception as e:
        logger.error("Failed to complete the model building process: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()          

