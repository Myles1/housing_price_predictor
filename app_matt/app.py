from __future__ import division
from math import sqrt
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)
model = load_model()


@app.route('/', methods=['GET'])
def index():
    return render_template('quadratic.html')


@app.route('/predict', methods=['POST'])
def predict():
    user_data = request.json
    address = user_data['address']
    
    try:
        price = model.predict(address)[0]
    except IndexError:
        price = "Error no predictions"
    
    return jsonify({'price': price})  

def load_model():
    

    return model


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
