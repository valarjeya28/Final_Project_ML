import os
import csv
import numpy as np
import re
import pandas as pd
from flask import Flask,render_template,url_for,request,jsonify

import keras
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, GlobalMaxPooling1D, CuDNNGRU, Dropout, BatchNormalization, Activation
from keras.utils import to_categorical

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

model = None
graph = None


# Loading a keras model with flask
# https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html
def load_model():
    global model
    global graph
    model = keras.models.load_model("twitter_model_trained.h5")
    graph = K.get_session().graph


load_model()
tokenizer = Tokenizer()

@app.route("/")
def home():
    """Return the homepage."""
    return render_template("index.html")

@app.route('/', methods=['GET', 'POST'])
def upload_file():

    data = {"success": False}
    while data == {"success": False}:
        if request.method == 'POST':
            if request.files.get('file'):
                # read the file
                file = request.files['file']

                # read the filename
                filename = file.filename

                # create a path to the uploads folder
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                # Save the file to the uploads folder
                file.save(filepath)
                
                # Access the data 
                csv_input = pd.read_csv(os.path.join(file.filename))
                tweets = csv_input["Tweet"].apply(lambda x: x.lower())
                tweets = csv_input["Tweet"].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

                for idx,row in csv_input.iterrows():
                    #print(idx,row)
                    row[0] = row[0].replace('rt',' ')
                print(tweets)

                # # Make the prediction
                # global graph
                # with graph.as_default():
                #     sentence_encoding = tokenizer.texts_to_sequences(tweets)
                #     padded_sentence = sequence.pad_sequences(sentence_encoding, maxlen=32)
                #     prediction = model.predict(np.array(padded_sentence))
                #     if prediction[0][0] == max(prediction[0]):
                #         data["Positive"] = str(prediction)
                #     elif prediction[0][1] == max(prediction[0]):
                #         data["Neutral"] = str(prediction)
                #     elif prediction[0][2] == max(prediction[0]):
                #         data["Negative"] = str(prediction)
                #     data = {'success': True}
                # return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)