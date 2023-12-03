import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


from data.data import clean_raw, create_sampled_df

def preprocessing(df, min_word=160, max_word=210, rmv_num=False, lower=False, test_size=0.2):

    df_cleaned = clean_raw(df, rmv_num, lower)
    df_sampled = create_sampled_df(df_cleaned, max_word, min_word)

    # Define X et target
    X = df_sampled[['extracts']]
    y = df_sampled['normalized_label']

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Turn X_train into a list of sequences for tokenization
    X_train= X_train['extracts'].values.tolist()

    # Instantiate tokenizer
    tk = Tokenizer()

    # Fit X_train on texts and tokenize
    tk.fit_on_texts(X_train)
    vocab_size = len(tk.word_index)
    X_train_token = tk.texts_to_sequences(X_train)

    # Turn X_test into a list and tokenize
    X_test_token = tk.texts_to_sequences(X_test['extracts'].values.tolist())

    # Pad X
    X_train_padded = pad_sequences(X_train_token, dtype='float32', padding='post', maxlen=max_word+1)
    X_test_padded = pad_sequences(X_test_token, dtype='float32', padding='post', maxlen=max_word+1)

    # Instanciate label encoder
    encoder = OneHotEncoder()

    # Encode Y_train
    y_train_enc = encoder.fit_transform(pd.DataFrame(y_train))
    y_train_enc = y_train_enc.todense()

    # Encode Y_test
    y_test_enc = encoder.transform(pd.DataFrame(y_test))
    y_test_enc = y_test_enc.todense()

    return vocab_size, X_train_padded, y_train_enc, X_test_padded, y_test_enc
