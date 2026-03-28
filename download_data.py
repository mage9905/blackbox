import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def compute_team_stats():
    print("📊 读取 matches.csv...")
    
    matches_file = Path("matches.csv")
    if not matches_file.exists():
        print("❌ 找不到 matches.csv")
        return False
    
    df = pd.read_csv(matches_file)
    print(f"✅ 已加载 {len(df)} 场比赛")
    
    # 处理可能的列名差异
    if 'HomeTeam' in df.columns:
        df = df.rename(columns={'HomeTeam': 'home_team', 'AwayTeam': 'away_team', 'FTHG': 'home_goals', 'FTAG': 'away_goals'})
    
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
    return True

if __name__ == "__main__":
    compute_team_stats()
