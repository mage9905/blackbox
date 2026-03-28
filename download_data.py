import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def download_matches():
    print("📥 下载真实比赛数据...")
    
    # FiveThirtyEight 足球数据（国内一般能访问）
    url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/soccer-spi/raw_data/spi_matches.csv"
    
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            size = len(r.content)
            print(f"下载大小: {size} 字节")
            
            if size < 100000:
                print("文件太小，可能不是真实数据")
                return False
                
            with open(DATA_DIR / "matches.csv", 'wb') as f:
                f.write(r.content)
            print("✅ 已下载真实数据")
            return True
    except Exception as e:
        print(f"下载失败: {e}")
        return False

def compute_team_stats():
    print("📊 计算球队统计...")
    
    df = pd.read_csv(DATA_DIR / "matches.csv")
    print(f"原始数据: {len(df)} 行")
    
    # 映射列名
    if 'team1' in df.columns:
        df = df.rename(columns={'team1': 'home_team', 'team2': 'away_team', 'score1': 'home_goals', 'score2': 'away_goals'})
    
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
        print("❌ 无法下载真实数据")
