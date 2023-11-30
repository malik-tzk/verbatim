import re
import string
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize

def clean_text(text, rmv_num=False, rmv_punct=False, lower=False):
# Cleans a string according to some conditions

    # rmv HTML tags:
    text = re.sub(re.compile('<.*?>'), '', text)

    # rmv various non-text items:
    text = text.replace('\n', ' ')
    text = text.replace("\\", "")
    text = text.replace("\ufeff", "")
    text = text.replace('\t', ' ')

    #rmv double spaces
    text = re.sub(' +', ' ',text)

    #rmv useless special caracters
    text = re.sub(r'[#$%&*+@~]', '', text)

    # rmv signature at the end:
    signature_pattern = r'\s*THE END\s*\nHope you have enjoyed the reading!\nCome back to http://english-e-reader\.net/ to find more fascinating and exciting stories!'
    text = re.sub(signature_pattern, '', text, flags=re.IGNORECASE)
    text = text.replace('- THE END - Hope you have enjoyed the reading! Come back to http://english-e-reader.net/ to find more fascinating and exciting stories!', '')

    #rmv numbers (if option selected)
    if rmv_num == True:
        text = ''.join([c for c in text if c not in string.digits])

    #rmv punctuation (if option selected)
    if rmv_punct == True:
        text = ''.join([c for c in text if c not in string.punctuation])

    if lower == True:
        text = text.lower()

    return text.strip()



def clean_raw(df, rmv_num=False, rmv_punct=False, lower=False, raw_text='raw_text'):
    #Get a df with a raw text column and create a clean text column
    df['clean_text'] = df[raw_text].apply(lambda x: clean_text(x, rmv_num, rmv_punct, lower))
    return df


def split_text_into_extracts(text: str, max_words=250)-> list:
    """
    Takes a string of text and returns a list of multiple strings(=extracts), each one with less than max_words
    """
    #to split the text into sentences, using nltk tokenizer :
    sentences = sent_tokenize(text)

    result = []
    current_sentence = ""

    for sentence in sentences:
        # Tokenize the sentence into words
        words = sentence.split()

        # Check if adding the current sentence exceeds the word limit
        if len(current_sentence.split()) + len(words) <= max_words:
            current_sentence += sentence + " "
        else:
            result.append(current_sentence.strip())
            current_sentence = sentence + " "

    # Add the last remaining sentence
    result.append(str(current_sentence).strip())

    return result

def create_sampled_df(df: pd.DataFrame, max_word:int=250, min_word:int=50)->pd.DataFrame :
    """
    Take a df with a 'clean_text' column
    Create a column that contains a list of extracts, each of a size up to the number given in max_word
    Return a df with columns : 'source','source_label','normalized_label','extracts'
    """

    # Create a column that contains a list of extracts
    # extract = string containing sentences which add up to a certain nb of words
    df['extracts']= df['clean_text'].apply(lambda x: split_text_into_extracts(x, max_word,))

    #Spread the list over into different rows
    df = df.explode('extracts')

    #Rmv extracts out of the boundaries <= min_word >= max_word
    df['count'] = df['extracts'].apply(lambda x: len(x.split()))
    df = df[(df['count'] >= min_word) & (df['count'] <= max_word)]

    #Clean indexes :
    df.reset_index(drop=True, inplace=True)

    return df[['source', 'source_label', 'normalized_label',
       'extracts']]

def get_dfs_classes (df:pd.DataFrame, column:str='normalized_label',shuffled:bool=True)->list :
    """
    Gets a DataFrame and a column name
    Returns a list of DataFrames sorted by the column class
    Each DataFrame can be suffled if desired
    """
    df_list = []
    # get all the classes in the df in the right order
    classes = np.sort(df[column].unique())

    #Create a DF for each class
    for label in classes:
        mask_label = df[column]==label
        df_i = df[mask_label]

        if shuffled == True : #Suffles
            df_i = df_i.sample(frac=1).reset_index(drop=True)

        df_list.append(df_i)

    return df_list

def create_train_test_df(list_df:list ,
                                  sample_sizes:list)->pd.DataFrame:
    """
    Takes a list of Dataframe and a list of rows sizes for each DF and creates 2 DF :
    One with the number of row selected for each class = used for training
    One with the rest = used for testing
    """

    if len(list_df) != len(sample_sizes) :
        return 'Error len of list of dataframes and samples sizes are not the same'

    #extract the rows from each dataset
    list_df_balanced_train = []
    list_df_test = []

    for i in range(len(list_df)) :

        #Get the right df and the corresponding number of row :
        df = list_df[i]
        n_rows_to_extract = sample_sizes[i]

        #Extract the rows for each final df :
        df_sample = df.iloc[:n_rows_to_extract]
        df_rest = df.iloc[n_rows_to_extract:]
        list_df_balanced_train.append(df_sample)
        list_df_test.append(df_rest)

    #Concatenate each dataframe in the lists
    df_train = pd.concat(list_df_balanced_train)
    df_test = pd.concat(list_df_test)

    #  Clean indexes:
    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)


    return df_train, df_test

def custom_train_test_df(df:pd.DataFrame, sample_sizes:list, column:str='normalized_label',suffled:bool=True):
    """
    from 1 df returns 2 df :
    one customed with sample_size for each class = used for training
    one with the rest = used for training
    combines get_dfs_classes and create_train_test_df
    """
    list_df = get_dfs_classes(df,column,suffled)
    df_train, df_test = create_train_test_df(list_df,sample_sizes)
    return df_train, df_test



if __name__ == "__main__":
    pass
