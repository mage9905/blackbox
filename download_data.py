import pandas as pd
from pathlib import Path
from underdata.league import League

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def fetch_matches():
    print("📥 通过 underdata 拉取比赛数据...")
    all_matches = []
    
    # 拉取英超近 3 个赛季
    for season in [2021, 2022, 2023]:
        try:
            league = League(league_name="Premier League", season=season)
            # 获取赛季所有比赛
            matches = league.get_fixtures()
            for m in matches:
                all_matches.append({
                    "home_team": m.get("home_team"),
                    "away_team": m.get("away_team"),
                    "home_goals": m.get("home_goals"),
                    "away_goals": m.get("away_goals"),
                    "season": season
                })
            print(f"✅ 已拉取 {season} 赛季数据")
        except Exception as e:
            print(f"⚠️ {season} 赛季拉取失败: {e}")
    
    if not all_matches:
        print("❌ 未拉取到任何数据")
        return False
    
    df = pd.DataFrame(all_matches)
    df.to_csv(DATA_DIR / "matches.csv", index=False)
    print(f"✅ 已保存 {len(df)} 场比赛到 data/matches.csv")
    return True

def compute_team_stats():
    print("📊 计算球队统计...")
    df = pd.read_csv(DATA_DIR / "matches.csv")
    print(f"✅ 已加载 {len(df)} 场比赛")
    
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
    if fetch_matches():
        compute_team_stats()
        print("🎯 真实数据准备完成")
    else:
        print("❌ 数据拉取失败")
