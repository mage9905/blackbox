import requests
import pandas as pd
import os
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def download_matches():
    print("📥 下载真实比赛数据...")
    
    # 使用多个代理源，总有一个能用
    proxies = [
        "",  # 直连
        "https://ghproxy.com/",
        "https://ghproxy.net/",
        "https://mirror.ghproxy.com/",
        "https://github.moeyy.xyz/",
    ]
    
    for proxy in proxies:
        url = f"{proxy}https://raw.githubusercontent.com/jokecamp/FootballData/master/International_Football_Results/results.csv"
        try:
            print(f"尝试: {proxy or '直连'}")
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                with open(DATA_DIR / "matches.csv", 'wb') as f:
                    f.write(r.content)
                print(f"✅ 已下载 {len(r.content)} 字节真实数据")
                return True
        except Exception as e:
            print(f"失败: {e}")
            continue
    
    print("❌ 所有代理均失败，无法下载真实数据")
    return False

def compute_team_stats():
    print("📊 计算球队统计...")
    df = pd.read_csv(DATA_DIR / "matches.csv")
    
    # 处理列名（CSV 可能不同）
    if 'home_team' not in df.columns:
        df.columns = ['date', 'home_team', 'away_team', 'home_score', 'away_score', 'tournament', 'city', 'country', 'neutral']
        df = df.rename(columns={'home_score': 'home_goals', 'away_score': 'away_goals'})
    
    teams = set(df["home_team"]) | set(df["away_team"])
    stats = []
    for team in teams:
        home_games = df[df["home_team"] == team]
        away_games = df[df["away_team"] == team]
        total_goals = home_games["home_goals"].sum() + away_games["away_goals"].sum()
        total_conceded = home_games["away_goals"].sum() + away_games["home_goals"].sum()
        total_games = len(home_games) + len(away_games)
        if total_games > 0:
            stats.append({
                "team": team,
                "goals_avg": round(total_goals / total_games, 2),
                "conceded_avg": round(total_conceded / total_games, 2)
            })
    
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(DATA_DIR / "team_stats.csv", index=False)
    print(f"✅ 已计算 {len(stats)} 支球队统计")

if __name__ == "__main__":
    if download_matches():
        compute_team_stats()
        print("🎯 真实数据准备完成")
    else:
        print("❌ 部署失败：无法下载真实数据")
        exit(1)
