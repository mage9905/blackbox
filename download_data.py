import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def download_matches():
    print("📥 下载比赛数据...")
    url = "https://raw.githubusercontent.com/jokecamp/FootballData/master/International_Football_Results/results.csv"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with open(DATA_DIR / "matches.csv", 'wb') as f:
                f.write(r.content)
            print(f"✅ 已下载")
            return True
    except Exception as e:
        print(f"下载失败: {e}")
    print("⚠️ 使用示例数据")
    create_sample_data()
    return False

def create_sample_data():
    import random
    teams = ["曼联", "利物浦", "阿森纳", "切尔西", "曼城", "热刺", "皇马", "巴萨", "拜仁", "多特"]
    matches = []
    for _ in range(500):
        home = random.choice(teams)
        away = random.choice([t for t in teams if t != home])
        matches.append({"home_team": home, "away_team": away, "home_goals": random.randint(0,4), "away_goals": random.randint(0,3)})
    pd.DataFrame(matches).to_csv(DATA_DIR / "matches.csv", index=False)
    print(f"✅ 已创建 500 场示例数据")

def compute_team_stats():
    print("📊 计算球队统计...")
    df = pd.read_csv(DATA_DIR / "matches.csv")
    teams = set(df["home_team"]) | set(df["away_team"])
    stats = []
    for team in teams:
        home_games = df[df["home_team"] == team]
        away_games = df[df["away_team"] == team]
        total_goals = home_games["home_goals"].sum() + away_games["away_goals"].sum()
        total_conceded = home_games["away_goals"].sum() + away_games["home_goals"].sum()
        total_games = len(home_games) + len(away_games)
        if total_games > 0:
            stats.append({"team": team, "goals_avg": round(total_goals/total_games,2), "conceded_avg": round(total_conceded/total_games,2)})
    pd.DataFrame(stats).to_csv(DATA_DIR / "team_stats.csv", index=False)
    print(f"✅ 已计算 {len(stats)} 支球队统计")

if __name__ == "__main__":
    download_matches()
    compute_team_stats()
    print("🎯 训练数据准备完成")
