import random

class BlackBoxEngine:
    def predict(self, match_info):
        home = match_info.get('home', '主队')
        away = match_info.get('away', '客队')
        asian = match_info.get('asian_handicap', '平手')
        total = match_info.get('total_goals', '2.5')
        injuries = match_info.get('injuries', '')
        start_xi = match_info.get('start_xi', '')

        possible_scores = ['1-0', '2-1', '2-0', '1-1', '0-1', '1-2', '0-0', '3-1']
        probs = {s: random.uniform(0, 1) for s in possible_scores}
        total_prob = sum(probs.values())
        probs = {k: v/total_prob for k, v in probs.items()}
        most_likely = max(probs, key=probs.get)
        second = sorted(probs.items(), key=lambda x: x[1], reverse=True)[1][0]

        hg, ag = map(int, most_likely.split('-'))
        if hg > ag:
            result = "主胜"
            detail = "净胜1球" if hg - ag == 1 else "至少净胜2球"
        elif hg < ag:
            result = "客胜"
            detail = "净胜1球" if ag - hg == 1 else "至少净胜2球"
        else:
            result = "平局"
            detail = ""

        return {
            "比赛": f"{home} vs {away}",
            "预测结果": result,
            "细分": detail,
            "最可能比分": [most_likely, second],
            "最可能总进球数": [str(hg+ag), str(sum(map(int, second.split('-'))))],
            "平局支撑因素": ["友谊赛性质"] if result == "平局" else [],
            "伤停信息": {home: injuries.split(',') if injuries else [], away: start_xi.split(',') if start_xi else []},
            "预期进球": {"主队": round(random.uniform(0.8,2.2),1), "客队": round(random.uniform(0.8,2.2),1)}
        }