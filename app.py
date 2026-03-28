from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import requests
import time
import pickle
import threading
import math
from pathlib import Path

app = Flask(__name__)
CORS(app)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
MODEL_PATH = DATA_DIR / "model.pkl"

model_stats = {}
training_status = {"status": "idle", "message": "", "progress": 0}
api_key = os.environ.get("API_FOOTBALL_KEY")  # 仅用于预测

# ========== Dixon-Coles 核心 ==========
def dixon_coles_tau(x, y, lam, mu, rho=-0.13):
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    elif x == 0 and y == 1:
        return 1 + lam * rho
    elif x == 1 and y == 0:
        return 1 + mu * rho
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1

def poisson_prob(lam, k):
    if lam == 0:
        return 1 if k == 0 else 0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)

def dixon_coles_predict(lam, mu, rho=-0.13, max_goals=5):
    probs = {}
    total = 0.0
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            p_pois = poisson_prob(lam, x) * poisson_prob(mu, y)
            tau = dixon_coles_tau(x, y, lam, mu, rho)
            prob = p_pois * tau
            probs[f"{x}-{y}"] = prob
            total += prob
    for k in probs:
        probs[k] /= total
    return probs


# ========== 训练（只用 football.json，不调 API）==========
def train_from_football_json():
    global model_stats, training_status
    
    training_status = {"status": "running", "message": "开始拉取历史数据...", "progress": 0}
    
    leagues = {
        "en.1": "英超", "en.2": "英冠", "en.3": "英甲",
        "fr.1": "法甲", "fr.2": "法乙",
        "de.1": "德甲", "de.2": "德乙",
        "nl.1": "荷甲", "nl.2": "荷乙",
        "es.1": "西甲", "it.1": "意甲", "pt.1": "葡超",
        "no.1": "挪威超", "se.1": "瑞典超",
        "jp.1": "日职联", "us.1": "美职联", "kr.1": "韩K联"
    }
    
    seasons = [f"{y}-{str(y+1)[-2:]}" for y in range(2015, 2025)]
    all_matches = []
    
    total_tasks = len(leagues) * len(seasons)
    completed = 0
    
    for league_code, league_name in leagues.items():
        for season in seasons:
            url = f"https://raw.githubusercontent.com/openfootball/football.json/master/{season}/{league_code}.json"
            training_status["message"] = f"拉取 {league_name} {season}..."
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    for m in data.get('matches', []):
                        score = m.get('score', {})
                        ft = score.get('ft', [None, None])
                        all_matches.append({
                            "home_team": m.get('team1', ''),
                            "away_team": m.get('team2', ''),
                            "home_goals": ft[0] if ft[0] is not None else 0,
                            "away_goals": ft[1] if ft[1] is not None else 0,
                            "date": m.get('date', '')
                        })
            except:
                pass
            
            completed += 1
            training_status["progress"] = int(completed / total_tasks * 100)
            time.sleep(0.05)
    
    if not all_matches:
        training_status = {"status": "failed", "message": "未拉取到任何数据", "progress": 0}
        return
    
    df = pd.DataFrame(all_matches)
    training_status["message"] = f"已拉取 {len(df)} 场比赛，正在训练..."
    
    # 时间加权
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    max_date = df['date'].max()
    df['weight'] = df['date'].apply(lambda d: 1.0 if pd.isna(d) else 0.5 ** ((max_date - d).days / 70))
    
    stats = {}
    teams = set(df['home_team']) | set(df['away_team'])
    
    for i, team in enumerate(teams):
        training_status["message"] = f"训练中... {i+1}/{len(teams)}"
        training_status["progress"] = 80 + int(i / len(teams) * 20)
        
        home_games = df[df['home_team'] == team]
        away_games = df[df['away_team'] == team]
        
        stats[team] = {
            'home_goals_avg': round((home_games['home_goals'] * home_games['weight']).sum() / max(home_games['weight'].sum(), 1), 2),
            'home_conceded_avg': round((home_games['away_goals'] * home_games['weight']).sum() / max(home_games['weight'].sum(), 1), 2),
            'away_goals_avg': round((away_games['away_goals'] * away_games['weight']).sum() / max(away_games['weight'].sum(), 1), 2),
            'away_conceded_avg': round((away_games['home_goals'] * away_games['weight']).sum() / max(away_games['weight'].sum(), 1), 2),
            'games': len(home_games) + len(away_games)
        }
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(stats, f)
    
    model_stats = stats
    training_status = {"status": "success", "message": f"训练完成！共 {len(stats)} 支球队", "progress": 100}


# ========== 加载模型 ==========
def load_model():
    global model_stats
    if MODEL_PATH.exists():
        with open(MODEL_PATH, 'rb') as f:
            model_stats = pickle.load(f)
        print(f"✅ 已加载模型，共 {len(model_stats)} 支球队")
        return True
    return False

load_model()


# ========== API 获取实时数据（仅预测用）==========
def get_realtime_data(team_name):
    """调用 API 获取伤停信息，仅用于预测"""
    if not api_key:
        return None
    try:
        headers = {"x-apisports-key": api_key}
        r = requests.get(f"https://v3.football.api-sports.io/teams?search={team_name}", headers=headers, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data.get("response"):
            return None
        team_id = data["response"][0]["team"]["id"]
        
        injuries = []
        try:
            r2 = requests.get(f"https://v3.football.api-sports.io/injuries?team={team_id}&season=2024", headers=headers, timeout=10)
            if r2.status_code == 200:
                for inj in r2.json().get("response", []):
                    injuries.append(inj.get("player", {}).get("name", "Unknown"))
        except:
            pass
        
        return {"injuries": injuries, "adjustment": 0.85 if len(injuries) > 0 else 1.0}
    except:
        return None


# ========== 预测 ==========
def predict_match(home, away, injuries_input=""):
    if home not in model_stats or away not in model_stats:
        return {"error": f"球队 '{home}' 或 '{away}' 不在训练数据中，请先训练模型"}
    
    home_s = model_stats[home]
    away_s = model_stats[away]
    
    lam = home_s['home_goals_avg'] * away_s['away_conceded_avg']
    mu = away_s['away_goals_avg'] * home_s['home_conceded_avg']
    
    # API 实时数据修正（仅伤停）
    home_realtime = get_realtime_data(home)
    away_realtime = get_realtime_data(away)
    
    injuries_info = {"home": [], "away": []}
    if home_realtime:
        lam *= home_realtime["adjustment"]
        injuries_info["home"] = home_realtime["injuries"]
    if away_realtime:
        mu *= away_realtime["adjustment"]
        injuries_info["away"] = away_realtime["injuries"]
    
    # 用户输入的伤停
    if injuries_input:
        lam *= 0.9
        mu *= 0.9
    
    probs = dixon_coles_predict(lam, mu, -0.13)
    
    import random
    r = random.random()
    cum = 0
    selected = "0-0"
    for score, prob in probs.items():
        cum += prob
        if r <= cum:
            selected = score
            break
    
    hg, ag = map(int, selected.split('-'))
    
    if hg > ag:
        result = "主胜"
        detail = "净胜1球" if hg - ag == 1 else "至少净胜2球"
    elif hg < ag:
        result = "客胜"
        detail = "净胜1球" if ag - hg == 1 else "至少净胜2球"
    else:
        result = "平局"
        detail = ""
    
    top_scores = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
    
    return {
        "比赛": f"{home} vs {away}",
        "预测结果": result,
        "细分": detail,
        "最可能比分": selected,
        "比分概率": [(s, f"{p:.2%}") for s, p in top_scores],
        "实时数据": {
            "主队伤停": injuries_info["home"],
            "客队伤停": injuries_info["away"]
        }
    }


# ========== Flask 路由 ==========
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/training_status')
def get_training_status():
    return jsonify(training_status)

@app.route('/train', methods=['POST'])
def train():
    if training_status["status"] == "running":
        return jsonify({"error": "训练已在运行中"})
    threading.Thread(target=train_from_football_json).start()
    return jsonify({"status": "started"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    home = data.get('home')
    away = data.get('away')
    injuries = data.get('injuries', '')
    
    if not model_stats:
        return jsonify({"error": "模型未训练，请先点击「开始训练」"})
    
    result = predict_match(home, away, injuries)
    return jsonify(result)

@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "teams": len(model_stats),
        "trained": MODEL_PATH.exists(),
        "api_key_configured": bool(api_key)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
