import re
import string
import pandas as pd 


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



if __name__ == "__main__":
    print(clean_text('hemmp guys23...!',rmv_punct=True))
