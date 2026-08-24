import os
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
import string
import yaml

import nltk
# importing the string module to handle special characters
from nltk.corpus import stopwords
# importing porter stemmer for text stemming
from nltk.stem.porter import PorterStemmer

nltk.download('punkt_tab') # to run nltk.word_tokenize(text1)
nltk.download("stopwords")
nltk.download('punkt')

#logging
# logging directory
log_dir= 'logs'
os.makedirs(log_dir,exist_ok=True)

# logging configuration

# making the object of logging with some name (here 'data_ingestion') and with level
logger= logging.getLogger('data_preprocessing')
logger.setLevel("DEBUG")

console_handler= logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path= os.path.join(log_dir,'data_preprocessing.log')
file_handler= logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter= logging.Formatter("%(asctime)s-%(name)s-%(levelname)s-%(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text)->str:
    """ 
    This function is used to transform the given text.
    """
    # convert to lowercase
    text=text.lower()
    # tokenize the text
    ps= PorterStemmer()
    text= nltk.word_tokenize(text)
    # remove non alpha numeric characters
    text=[word for word in text if word.isalnum()]
    # remove stopwords and punctuations
    text=[word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    # stem the words
    text= [ps.stem(word) for word in text]
    # join the tokens back into a string
    return " ".join(text)


def preprocess_df(df,text_column='text',target_column='target'):
    """
    Preprocesses the DataFrame by encoding the target column, removing duplicates, 
    and transforming the text column.
    """
    try:
        # encoding the target column
        logger.debug("Data preprocessing is started")
        encoder= LabelEncoder()
        df[target_column]=encoder.fit_transform(df[target_column])
        
        # Apply text transformation to the specified text column
        df.loc[:,text_column]=df[text_column].apply(transform_text)
        logger.debug("Text column is transformed.")
        return df

    except KeyError as e:
        logger.error("column was not found: %s",e)
        raise
    except Exception as e:
        logger.error("Error occurred during preprocessing the data: %s",e) 
        raise  

def main(text_column='text',target_column='target')->None:
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        # loading data and preprocessing
        input_data_path= './data'
        logger.debug("Loading the data from %s",input_data_path)
        train_df=preprocess_df(pd.read_csv(input_data_path+'/raw/train.csv'),text_column,target_column)
        test_df=preprocess_df(pd.read_csv(input_data_path+'/raw/test.csv'),text_column,target_column)
        logger.debug("Data is loaded and preprocessed successfully.")

        
        # making interim folder to save processed data
        save_data_path= os.path.join(input_data_path,'interim')
        os.makedirs(save_data_path,exist_ok=True)

        # saving data
        train_df.to_csv(os.path.join(save_data_path,'train_preprocessed.csv'),index=False)
        test_df.to_csv(os.path.join(save_data_path,'test_preprocessed.csv'),index=False)
        logger.debug("Processed data is saved to: %s",save_data_path)

    except FileNotFoundError as e:
        logger.error("file not found: %s",e)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error("No data: %s",e)  
        raise
    except Exception as e:
        logger.error("Error was occurred while processing and saving the data: %s",e)  
        raise    

if __name__=='__main__':
    main()

     
     



