from flask import Flask, render_template
from flask_restful import Api, Resource, reqparse, abort
import numpy as np
import autokeras
import keras
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)
api = Api(app)

model = keras.models.load_model('./Model/English Suicide Prevention Model')
post_args = reqparse.RequestParser()
ps = PorterStemmer()
post_args.add_argument("tweet", type=str, help="Need Tweet predict", required=True)
text = {}

@app.route("/")
def index():
	return render_template("index.html")

def preprocessing(args):
    args["tweet"] = args["tweet"].lower()
    args["tweet"] = args["tweet"].replace(r'http\S+', '').replace(r'www\S+', '').replace(r'rt', '')
    args["tweet"] = args["tweet"].replace('\d+', '')
    args["tweet"] = args["tweet"].replace('[^\w\s]', '')
    args["tweet"] = args["tweet"].strip()
    args["tweet"] = nltk.word_tokenize(args["tweet"])
    stop_words = stopwords.words('english')
    args["tweet"] = [item for item in args["tweet"] if item not in stop_words]
    args["tweet"] = [ps.stem(item) for item in args["tweet"]]
    lemmatizer = WordNetLemmatizer()
    args["tweet"] = [lemmatizer.lemmatize(item) for item in args["tweet"]]
    return args

class Posts(Resource):
    def post(self):
        args = post_args.parse_args()
        args = preprocessing(args)
        test_set = tf.data.Dataset.from_tensor_slices(np.reshape(args["tweet"][0], -1))
        y = model.predict(test_set)
        return {"probability" : float(y)}
api.add_resource(Posts, "/Posts")

if __name__ == '__main__':
    app.run(debug=True)