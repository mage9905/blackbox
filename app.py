from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from model import BlackBoxEngine

app = Flask(__name__)
CORS(app)
engine = BlackBoxEngine()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    result = engine.predict(data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)