from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from model import BlackBoxEngine
import os
from pathlib import Path

# 首次运行下载训练数据
if not Path("data/team_stats.csv").exists():
    print("📦 下载训练数据...")
    try:
        from download_data import download_matches, compute_team_stats
        download_matches()
        compute_team_stats()
    except Exception as e:
        print(f"数据下载失败: {e}")

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

@app.route('/health')
def health():
    return jsonify({"status": "ok", "teams": len(engine.team_stats)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
