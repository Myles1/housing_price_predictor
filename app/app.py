from flask import Flask, render_template, request, jsonify, make_response
from src.model_create import RealEstatePredictor
import pickle
import pandas as pd
import numpy as np
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    # return render_template('doc_predictor.html')
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def predict_doc():
    user_doc = request.json['user_doc']

    y_hat = model.predict(user_doc)
    print("y_hat: {}".format(y_hat))
    return jsonify({
            'doc' : str(y_hat[0])
        })

@app.route('/js/<path:path>')
def static_file(path):
    return app.send_static_file('js/' + path)


@app.route('/api/chart_data')
def chart_data():
    CHART_DATA = [
        { "y": '2006', "a": 100, "b": 90 },
        { "y": '2007', "a": 75,  "b": 65 },
        { "y": '2008', "a": 50,  "b": 40 },
        { "y": '2009', "a": 75,  "b": 65 },
        { "y": '2010', "a": 50,  "b": 40 },
        { "y": '2011', "a": 75,  "b": 65 },
        { "y": '2012', "a": 100, "b": 90 }
    ]

    # CHART_DATA = str(model.predict_all('351  N 137'))
    return jsonify({ "data": CHART_DATA })




@app.route("/test.png")
def simple():
    import datetime
    from io import BytesIO
    import random

    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.dates import DateFormatter

    fig=Figure()
    ax=fig.add_subplot(111)
    # x=[]
    # y=[]
    # now=datetime.datetime.now()
    # delta=datetime.timedelta(days=1)
    # for i in range(10):
    #     x.append(now)
    #     now+=delta
    #     y.append(random.randint(0, 1000))


    y = model.predict_all('351  N 137')
    print(y)
    x = list(range(1, len(y)+1))
    ax.plot(x, y)


    #ax.set_ylim(ymin=0)

    #ax.plot_date(x, y, '-')
    #ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    canvas=FigureCanvas(fig)
    png_output = BytesIO()
    canvas.print_png(png_output)
    response=make_response(png_output.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response



# def lookup_address(address, df):
#     '''Find if address appears in df['address'].
#     If so, returns df[~'address']
#     '''
#     print("We're returning a DataFrame")
#     print("address: {}".format(address))
#     return df[df['address'].str.contains(address, na=False)]




if __name__ == '__main__':
    with open('static/model.pkl', 'rb') as f:
        model = pickle.load(f)
    #address_lookup_df = pd.read_pickle('static/df.pkl')
    #print(address_lookup_df.info())

    app.run(host='0.0.0.0', port=8080, debug=True)
