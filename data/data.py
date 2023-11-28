import re
import string
import pandas as pd
from nltk.tokenize import sent_tokenize

def clean_text(text, rmv_num=False, rmv_punct=False, lower=False):
# Cleans a string according to some conditions

    # rmv HTML tags:
    text = re.sub(re.compile('<.*?>'), '', text)

    # rmv various non-text items:
    text = text.replace('\n', ' ')
    text = text.replace("\n\n", " ")
    text = text.replace("\\", "")
    text = text.replace("\ufeff", "")

    #rmv double spaces
    text = re.sub(' +', ' ',text)

    #rmv useless special caracters
    text = re.sub(r'[#$%&*+@~]', '', text)

    # rmv signature at the end:
    signature_pattern = r'\s*THE END\s*\nHope you have enjoyed the reading!\nCome back to http://english-e-reader\.net/ to find more fascinating and exciting stories!'
    text = re.sub(signature_pattern, '', text, flags=re.IGNORECASE)

    #rmv numbers (if option selected)
    if rmv_num == True:
        text = ''.join([c for c in text if c not in string.digits])

    #rmv punctuation (if option selected)
    if rmv_punct == True:
        text = ''.join([c for c in text if c not in string.punctuation])

    if lower == True:
        text = text.lower()


    return text.strip()



def clean_raw(df, raw_text='raw_text'):
    #Get a df with a raw text column and create a clean text column
    df['clean_text'] = df[raw_text].apply(clean_text)
    return df


def split_text_into_extracts(text, max_words=250):
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
    result.append(current_sentence.strip())

    return result

def extract_clean_samples_from_df(df: pd.DataFrame, words_by_extracts:int=250)->pd.DataFrame :
    """
    Take a df with a 'clean_text' column
    Create a column that contains a list of extracts, each of a size up to the number given in words_by_extracts
    Return a df with columns : 'source','source_label','normalized_label','extracts'
    """

    # Create a column that contains a list of extracts
    # extract = string containing sentences which add up to a certain nb of words
    df['extracts']= df['clean_text'].apply(split_text_into_extracts,
                                                      args=(words_by_extracts,))
    #Spread the list over into different rows
    df = df.explode('extracts')

    #Clean data so we only get back what we want :
    df.drop(columns = ['raw_text','clean_text','Unnamed: 0'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df



if __name__ == "__main__":
    print(clean_text('hemmp guys23...!',rmv_punct=True))
