from __future__ import division
from math import sqrt
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('quadratic.html')


@app.route('/solve', methods=['POST'])
def solve():
    user_data = request.json
    a, b, c = user_data['a'], user_data['b'], user_data['c']
    root_1, root_2 = _solve_quadratic(a, b, c)
    return jsonify({'root_1': root_1, 'root_2': root_2})  


def _solve_quadratic(a, b, c):
    disc = b*b - 4*a*c
    root_1 = (-b + sqrt(disc))/(2*a)
    root_2 = (-b - sqrt(disc))/(2*a)
    return root_1, root_2


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
