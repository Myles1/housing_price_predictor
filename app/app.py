from flask import Flask, render_template, request, jsonify
from src.build_model import TextClassifier, get_data
import cPickle as pickle
import pandas as pd
import numpy as np
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    # return render_template('doc_predictor.html')
    return render_template('index.html')



@app.route('/submit', methods=['POST'])
def predict_doc():
    print("Test")
    user_doc = np.array([request.json['user_doc']])
    return jsonify({
            'doc' : model.predict(user_doc)[0]
        })

@app.route('/js/<path:path>')
def static_file(path):
    return app.send_static_file('js/' + path)




if __name__ == '__main__':
    with open('static/model.pkl') as f:
        model = pickle.load(f)

    app.run(host='0.0.0.0', port=8080, debug=True)
