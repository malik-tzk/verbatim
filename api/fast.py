from fastapi import FastAPI
import pickle

app = FastAPI()
print('loading model')
app.state.model = pickle.load(open('api/api_naive_bayes', 'rb'))

@app.get('/')
def index():
    return {'ok': True}

# http://127.0.0.1:8000/predict?text=I love cats



@app.get("/predict")
def model_prediction(text):
    result = app.state.model.predict([text])
    print(result)
    return {'naive_bayes_level':int(result[0]),
            'type': str(type(result))}
