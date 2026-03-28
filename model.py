import random
import numpy as np
import pandas as pd
from pathlib import Path
import os

class BlackBoxEngine:
    def __init__(self):
        self.team_stats = {}
        self.api_key = os.environ.get("API_FOOTBALL_KEY")
        self._load_training_data()
    
    def _load_training_data(self):
        stats_file = Path("data/team_stats.csv")
        if stats_file.exists():
            df = pd.read_csv(stats_file)
            for _, row in df.iterrows():
                self.team_stats[row["team"]] = {
                    "goals_avg": row["goals_avg"],
                    "conceded_avg": row["conceded_avg"],
                    "source": "训练数据"
                }
            print(f"✅ 已加载 {len(self.team_stats)} 支球队")
        else:
            teams = ["曼联", "利物浦", "阿森纳", "切尔西", "曼城", "热刺"]
            for team in teams:
                self.team_stats[team] = {"goals_avg": 1.6, "conceded_avg": 1.3, "source": "默认"}
    
    def _get_api_stats(self, team_name):
        if not self.api_key:
            return None
        try:
            import requests
            headers = {"x-apisports-key": self.api_key}
            r = requests.get(f"https://v3.football.api-sports.io/teams?search={team_name}", headers=headers, timeout=10)
            if r.status_code != 200:
                return None
            data = r.json()
            if not data.get("response"):
                return None
            team_id = data["response"][0]["team"]["id"]
            r2 = requests.get(f"https://v3.football.api-sports.io/teams/statistics?team={team_id}&season=2024&league=39", headers=headers, timeout=10)
            if r2.status_code != 200:
                return None
            s = r2.json().get("response", {})
            goals_for = s.get("goals", {}).get("for", {}).get("total", {}).get("total", 0)
            goals_against = s.get("goals", {}).get("against", {}).get("total", {}).get("total", 0)
            games = s.get("fixtures", {}).get("played", {}).get("total", 1)
            return {"goals_avg": round(goals_for / games, 2), "conceded_avg": round(goals_against / games, 2), "source": "API"}
        except:
            return None
    
    def get_team_stats(self, team_name, use_api=False):
        if use_api and self.api_key:
            api_stats = self._get_api_stats(team_name)
            if api_stats:
                return api_stats
        if team_name in self.team_stats:
            return self.team_stats[team_name]
        for name in self.team_stats:
            if team_name in name or name in team_name:
                return self.team_stats[name]
        return {"goals_avg": 1.5, "conceded_avg": 1.3, "source": "默认"}
    
    def predict(self, match_info):
        home = match_info.get('home', '主队')
        away = match_info.get('away', '客队')
        injuries = match_info.get('injuries', '')
        home_stats = self.get_team_stats(home, use_api=True)
        away_stats = self.get_team_stats(away, use_api=True)
        lam = home_stats["goals_avg"] * away_stats["conceded_avg"]
        mu = away_stats["goals_avg"] * home_stats["conceded_avg"]
        if injuries:
            lam *= 0.9
            mu *= 0.9
        home_goals = min(np.random.poisson(lam), 5)
        away_goals = min(np.random.poisson(mu), 5)
        if home_goals > away_goals:
            result = "主胜"
            detail = "净胜1球" if home_goals - away_goals == 1 else "至少净胜2球"
        elif home_goals < away_goals:
            result = "客胜"
            detail = "净胜1球" if away_goals - home_goals == 1 else "至少净胜2球"
        else:
            result = "平局"
            detail = ""
        return {
            "比赛": f"{home} vs {away}",
            "预测结果": result,
            "细分": detail,
            "最可能比分": [f"{home_goals}-{away_goals}"],
            "数据来源": {"home": home_stats["source"], "away": away_stats["source"]}
        }
