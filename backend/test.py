import pandas as pd
from keras.layers import Embedding, LSTM, Dense  
from keras.models import Sequential, load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load data
data = pd.read_csv('./IMDB.csv', nrows=5000)

def map_sentiment(x):
  if x == 'positive':
      return 1
  elif x == 'negative':
      return 0
      
data['sentiment'] = data['sentiment'].apply(map_sentiment)

# Preprocess data
tokenizer = Tokenizer(num_words=5000, split=' ')
tokenizer.fit_on_texts(data['review'].values)
X = tokenizer.texts_to_sequences(data['review'].values)
X = pad_sequences(X)

# Load model
model = load_model('./sentiment_model.h5')


@app.route('/predict', methods=['POST'])
def predict():
  text = request.json['text']

  seq = tokenizer.texts_to_sequences([text])
  
  seq = pad_sequences(seq, maxlen=X.shape[1])
  pred = model.predict(seq)

  if pred > 0.5:
    sentiment = 'positive'
  else:
    sentiment = 'negative'

  result = {'prediction': sentiment}
  
  return jsonify(result)

@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*') 
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
  return response

if __name__ == '__main__':
  app.run(debug=True)