from __future__ import division
from math import sqrt
from flask import Flask, render_template, request, jsonify, make_response
import pickle
from src.model_create import RealEstatePredictor
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_data = request.json
    address = user_data['address']

    try:
        price = model.predict_all(address)
    except IndexError:
        price = "Error no predictions"


    import datetime
    from io import BytesIO
    import random

    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.dates import DateFormatter

    fig=Figure()
    ax=fig.add_subplot(111)
    print("Predicting Address: {}".format(address))
    ax.plot(model.predict_all(address))

    canvas=FigureCanvas(fig)
    png_output = BytesIO()
    canvas.print_png(png_output)
    response=make_response(png_output.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response




    price = [list(x) for x in price]



    return jsonify({'price': str(price)})

@app.route("/test.png")
def simple():
    if 'address' in request.args:
        address = request.args['address']
    else:
        address = '351  N 137'

    import datetime
    from io import BytesIO
    import random

    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.dates import DateFormatter

    fig=Figure()
    ax=fig.add_subplot(111)
    print("Predicting Address: {}".format(address))
    ax.plot(model.predict_all(address))

    canvas=FigureCanvas(fig)
    png_output = BytesIO()
    canvas.print_png(png_output)
    response=make_response(png_output.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response

def load_model():
    with open('static/model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
