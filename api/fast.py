from fastapi import FastAPI
import pickle
import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from data.data import clean_raw, create_sampled_df

app = FastAPI()
print('loading model')


# First api
# app.state.model = pickle.load(open('api/api_naive_bayes', 'rb'))
# print('clodo loaded')

# Second api
app.state.model_nb = pickle.load(open('/api/api_nb_opti', 'rb'))
print('nb loaded')

# app.state.model_nba = pickle.load(open('/api/api_nba_opti_2', 'rb'))
# print('nba loaded')

app.state.model_gru = pickle.load(open('/api/best_gru', 'rb'))

with open('/api/best_gru_tk', 'rb') as handle:
    gru_tk = pickle.load(handle)

# Third api


@app.get('/')
def index():
    return {'ok': True}

@app.get("/ping")
def ping():
    return {"ping": "pong"}

# http://127.0.0.1:8000/predict?text=I love cats


# First api
# @app.get("/predict")
# def model_prediction(text):
#     result = app.state.model.predict([text])
#     print(result)
#     return {'naive_bayes_level':int(result[0])}


#Second api
# @app.get("/simple_predict")
# def model_prediction(text):
#     result = app.state.model_nb.predict([text])
#     print(result)
#     return {'estimated_level':int(result[0])}


# @app.get("/composite_predict")
# def composite_model_predict(text):
#     result = app.state.model_nb.predict([text])
#     if (int(result[0]) == 0) or (int(result[0]) == 1):
#         sub_result = app.state.model_nba.predict([text])
#         return {'estimated_level':int(sub_result[0])}
#     else:
#         return {'estimated_level':int(result[0])}


# Third api
@app.get("/ml_predict")
def ml_model_prediction(text):
    result = app.state.model_nb.predict([text])
    print(result)
    return {'estimated_level':int(result[0])}

@app.get("/dl_predict")
def dl_model_prediction(text):
    df = pd.DataFrame({'raw_text': [text], 'source':1, 'source_label':1, 'normalized_label':1})
    df = clean_raw(df)
    df = create_sampled_df(df, min_word=160, max_word=210)
    X = df[['extracts']]
    X_token = gru_tk.texts_to_sequences(X['extracts'].values.tolist())
    X_padded = pad_sequences(X_token, dtype='float32', padding='post', maxlen=211)

    results = app.state.model_gru.model.predict(X_padded)
    result = results.mean(axis=0)
    return {'estimated_level': int(result.argmax())}
