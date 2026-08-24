import os
import numpy as np 
import pandas as pd 
import pickle 
import yaml
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score,f1_score
import json
import logging
from dvclive import live

# loggging all the steps
# making the logs directory
logs_dir= "logs"
os.makedirs(logs_dir,exist_ok=True)

# logging configuration
# 1. making the object of logging with some name (here 'data_ingestion') and with level
logger= logging.getLogger('model_evaluation')
logger.setLevel("DEBUG")

# 2. making the console handler to display the message on the console/ stream
console_handler= logging.StreamHandler()
console_handler.setLevel("DEBUG")

#3. making file handler with file
log_file_path= os.path.join(logs_dir,'model_evaluation.log')
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

def load_model(file_path:str)->pickle:
    """Loads the trained model from the path""" 
    try:
        with open(file_path,"rb") as m:
            model=pickle.load(m)
        logger.debug("Model is loaded from the path: %s",file_path)  
        return model
    except FileNotFoundError as e:
        logger.error("File was not found : %s",e)  
        raise
    except Exception as e:
        logger.error("Error occurred while loading the model: %s",e)
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

def evaluate_model(clf,X_test:np.ndarray,y_test:np.ndarray)->dict:
    """Evaluating the model and returning the evaluation metrics as dictionary."""  
    try: 
        y_pred= clf.predict(X_test) 
        y_pred_proba= clf.predict_proba(X_test)[:,1]

        accuracy= accuracy_score(y_test,y_pred)
        precision= precision_score(y_test,y_pred)
        recall= recall_score(y_test,y_pred) 
        aoc_score= roc_auc_score(y_test,y_pred_proba)
        f1= f1_score(y_test,y_pred)

        metrics_dict={
            "accuracy":accuracy,
            "precision":precision,
            "recall":recall,
            "aoc_score":aoc_score,
            "f1_score":f1
        }

        logger.debug("Model evaluation metrics were calculated.")
        return metrics_dict
    except Exception as e:
        logger.error("Error during model evaluation: %s",e)
        raise

def save_metrics(metrics:dict,file_path:str)->None:
    """Saving the metrics as a json file."""
    try: 
        # ensuring directory exists 
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,'w') as file:
            json.dump(metrics,file,indent=5)
        logger.debug("Metrics saved to :%s",file_path)
    except Exception as e:
        logger.error("Error was occurred while saving the file: %s",e) 
        raise

def main():
    try:
        params= load_params('params.yaml')  
        clf= load_model('models/model.pkl')   
        test_data= load_data('./data/processed/test_tfidf.csv') 
        X_test= test_data.iloc[:,:-1].values  
        y_test= test_data.iloc[:,-1].values

        metrics= evaluate_model(clf,X_test,y_test)



        save_metrics(metrics,'reports/metrics.json') 

    except Exception as e:
        logger.error('Failed to complete the model evaluation: %s',e)
        raise        

if __name__=='__main__':
    main()




