import pandas as pd
import numpy as np
import nltk
import os
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords")
chachedWords = stopwords.words('english')
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
# !pip install -U gensim
import gensim
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.layers import Dense, Embedding, GlobalMaxPooling1D, CuDNNGRU, Dropout, BatchNormalization, Activation
from keras.utils import to_categorical
from keras import backend as K
from flask import Flask,render_template,url_for,request,jsonify


app = Flask(__name__)

model = None
graph = None
tokenizer = Tokenizer()

def load_model():
    global model
    global graph
    model = keras.models.load_model("twitter_model_trained.h5")
    graph = K.get_session().graph


load_model()

@app.route('/')
def home():
	return render_template('index.html')

def tweet_cleaner(tweet):
    tweet = re.sub(r"@\w*", " ", str(tweet).lower()).strip() #removing username
    tweet = re.sub(r'https?://[A-Za-z0-9./]+', " ", str(tweet).lower()).strip() #removing links
    tweet = re.sub(r'[^a-zA-Z]', " ", str(tweet).lower()).strip() #removing sp_char
    tw = []
    
    for text in tweet.split():
        if text not in chachedWords:
            tw.append(text)
    
    return " ".join(tw)

@app.route('/predict',methods=['POST'])
def predict():
    data = {"success": False}
    if request.method == 'POST':
        print(request)
        message = request.form['message']
        text = [message]
        tweet = tweet_cleaner(text)
        
        global graph
        with graph.as_default():

        # Use the model to make a prediction
            sentence_encoding = tokenizer.texts_to_sequences([tweet])
            padded_sentence = sequence.pad_sequences(sentence_encoding, maxlen=32)

            my_prediction = model.predict(np.array(padded_sentence))
            sentiment = []
        # indicate that the request was a success
            data["success"] = True
            if my_prediction[0][0] == max(my_prediction[0]):
                data["Positive"] = str(my_prediction)
            elif my_prediction[0][1] == max(my_prediction[0]):
                data["Negative"] = str(my_prediction)
            elif my_prediction[0][2] == max(my_prediction[0]):
                data["Neutral"] = str(my_prediction)
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
