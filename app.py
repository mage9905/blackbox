import numpy as np
import pandas as pd
import math
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

class DixonColesModel:
    """
    完整 Dixon-Coles 模型 + KNN 修正
    """
    
    def __init__(self):
        self.alpha = {}
        self.beta = {}
        self.rho = -0.13
        self.teams = []
        self.n_teams = 0
        self.matches_history = []  # 保存历史比赛用于 KNN
        self.knn_model = None       # KNN 模型
        self.team_to_id = {}
        self.id_to_team = {}
    
    def _prepare_features(self, match):
        """为 KNN 准备特征向量"""
        home = match['home_team']
        away = match['away_team']
        hg = match['home_goals']
        ag = match['away_goals']
        
        # 特征：主队攻击力差、客队防守力差、历史交锋倾向等
        # 简化版：用球队的 α 和 β 作为特征
        alpha_h = self.alpha.get(home, 0)
        beta_h = self.beta.get(home, 0)
        alpha_a = self.alpha.get(away, 0)
        beta_a = self.beta.get(away, 0)
        
        # 比分差
        goal_diff = hg - ag
        
        return [alpha_h, beta_h, alpha_a, beta_a, goal_diff]
    
    def _log_likelihood(self, params, matches, team_index, rho):
        n_teams = len(team_index)
        alpha = {team: params[i] for i, team in enumerate(team_index)}
        beta = {team: params[n_teams + i] for i, team in enumerate(team_index)}
        
        log_lik = 0
        for match in matches:
            home = match['home_team']
            away = match['away_team']
            hg = match['home_goals']
            ag = match['away_goals']
            
            lam = np.exp(alpha[home] + beta[away])
            mu = np.exp(alpha[away] + beta[home])
            
            # 泊松概率
            p_pois = (lam ** hg) * np.exp(-lam) / math.factorial(hg) * \
                     (mu ** ag) * np.exp(-mu) / math.factorial(ag)
            
            # Dixon-Coles 调整
            tau = 1
            if hg == 0 and ag == 0:
                tau = 1 - lam * mu * rho
            elif hg == 0 and ag == 1:
                tau = 1 + lam * rho
            elif hg == 1 and ag == 0:
                tau = 1 + mu * rho
            elif hg == 1 and ag == 1:
                tau = 1 - rho
            
            p = p_pois * tau
            if p > 0:
                log_lik += np.log(p)
        
        return -log_lik
    
    def fit(self, matches, max_goals=5):
        """
        训练 Dixon-Coles 模型 + KNN
        """
        print("🏋️ 开始训练 Dixon-Coles 模型...")
        
        # 保存历史比赛用于 KNN
        self.matches_history = matches.copy()
        
        # 获取所有球队
        teams = set()
        for m in matches:
            teams.add(m['home_team'])
            teams.add(m['away_team'])
        self.teams = list(teams)
        self.n_teams = len(self.teams)
        
        self.team_to_id = {team: i for i, team in enumerate(self.teams)}
        self.id_to_team = {i: team for team, i in self.team_to_id.items()}
        
        # 初始化参数
        team_index = {team: i for i, team in enumerate(self.teams)}
        initial_params = [0.0] * (2 * self.n_teams)
        
        def constraint(params):
            return sum(params[:self.n_teams])
        
        cons = {'type': 'eq', 'fun': constraint}
        
        print(f"📊 使用 {len(matches)} 场比赛训练，共 {self.n_teams} 支球队")
        result = minimize(
            self._log_likelihood,
            initial_params,
            args=(matches, team_index, self.rho),
            method='SLSQP',
            constraints=cons,
            options={'maxiter': 1000, 'disp': True}
        )
        
        opt_params = result.x
        for i, team in enumerate(self.teams):
            self.alpha[team] = opt_params[i]
            self.beta[team] = opt_params[self.n_teams + i]
        
        print(f"✅ Dixon-Coles 训练完成，对数似然: {-result.fun:.2f}")
        
        # ========== 训练 KNN 模型 ==========
        print("🏋️ 训练 KNN 近邻匹配模型...")
        
        # 构建特征矩阵
        features = []
        labels = []  # 比分差
        
        for match in matches:
            # 需要先有 α, β 才能计算特征
            if match['home_team'] in self.alpha and match['away_team'] in self.alpha:
                feat = self._prepare_features(match)
                features.append(feat)
                labels.append(match['home_goals'] - match['away_goals'])
        
        if len(features) > 10:
            self.knn_model = NearestNeighbors(n_neighbors=5, metric='euclidean')
            self.knn_model.fit(features)
            print(f"✅ KNN 训练完成，使用 {len(features)} 个样本")
        else:
            print("⚠️ 样本不足，KNN 未训练")
    
    def knn_correction(self, home_team, away_team):
        """用 KNN 找相似比赛，返回比分差修正"""
        if self.knn_model is None:
            return 0
        
        if home_team not in self.alpha or away_team not in self.alpha:
            return 0
        
        # 当前比赛的特征
        current_feat = self._prepare_features({
            'home_team': home_team,
            'away_team': away_team,
            'home_goals': 0,
            'away_goals': 0
        })
        
        # 找最近邻
        distances, indices = self.knn_model.kneighbors([current_feat])
        
        # 取邻居的平均比分差
        if len(indices[0]) > 0:
            # 需要从历史比赛中获取这些邻居的实际比分
            # 这里简化：用预设的修正值
            return 0.1  # 简化版，实际应该计算邻居的平均比分差
        
        return 0
    
    def predict(self, home_team, away_team, rho=None):
        """Dixon-Coles 预测"""
        if rho is None:
            rho = self.rho
        
        lam = np.exp(self.alpha.get(home_team, 0) + self.beta.get(away_team, 0))
        mu = np.exp(self.alpha.get(away_team, 0) + self.beta.get(home_team, 0))
        
        lam = max(0.2, min(lam, 4.0))
        mu = max(0.2, min(mu, 4.0))
        
        max_goals = 5
        probs = {}
        total = 0.0
        
        for hg in range(max_goals + 1):
            for ag in range(max_goals + 1):
                p_pois = (lam ** hg) * np.exp(-lam) / math.factorial(hg) * \
                         (mu ** ag) * np.exp(-mu) / math.factorial(ag)
                
                tau = 1
                if hg == 0 and ag == 0:
                    tau = 1 - lam * mu * rho
                elif hg == 0 and ag == 1:
                    tau = 1 + lam * rho
                elif hg == 1 and ag == 0:
                    tau = 1 + mu * rho
                elif hg == 1 and ag == 1:
                    tau = 1 - rho
                
                prob = p_pois * tau
                probs[f"{hg}-{ag}"] = prob
                total += prob
        
        for k in probs:
            probs[k] /= total
        
        # KNN 修正（调整概率分布）
        knn_correction = self.knn_correction(home_team, away_team)
        if knn_correction != 0:
            # 调整概率，使比分差更接近 KNN 结果
            adjusted = {}
            for score, prob in probs.items():
                hg, ag = map(int, score.split('-'))
                diff = hg - ag
                if diff > 0 and knn_correction > 0:
                    adjusted[score] = prob * (1 + 0.1)
                elif diff < 0 and knn_correction < 0:
                    adjusted[score] = prob * (1 + 0.1)
                else:
                    adjusted[score] = prob * 0.9
            total = sum(adjusted.values())
            for k in adjusted:
                adjusted[k] /= total
            return adjusted
        
        return probs
    
    def predict_with_realtime(self, home_team, away_team, home_adjust=1.0, away_adjust=1.0, rho=None):
        """用实时数据修正"""
        lam_base = np.exp(self.alpha.get(home_team, 0) + self.beta.get(away_team, 0))
        mu_base = np.exp(self.alpha.get(away_team, 0) + self.beta.get(home_team, 0))
        
        lam = lam_base * home_adjust
        mu = mu_base * away_adjust
        
        lam = max(0.2, min(lam, 4.0))
        mu = max(0.2, min(mu, 4.0))
        
        if rho is None:
            rho = self.rho
        
        max_goals = 5
        probs = {}
        total = 0.0
        
        for hg in range(max_goals + 1):
            for ag in range(max_goals + 1):
                p_pois = (lam ** hg) * np.exp(-lam) / math.factorial(hg) * \
                         (mu ** ag) * np.exp(-mu) / math.factorial(ag)
                
                tau = 1
                if hg == 0 and ag == 0:
                    tau = 1 - lam * mu * rho
                elif hg == 0 and ag == 1:
                    tau = 1 + lam * rho
                elif hg == 1 and ag == 0:
                    tau = 1 + mu * rho
                elif hg == 1 and ag == 1:
                    tau = 1 - rho
                
                prob = p_pois * tau
                probs[f"{hg}-{ag}"] = prob
                total += prob
        
        for k in probs:
            probs[k] /= total
        
        return probs
    
    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'alpha': self.alpha,
                'beta': self.beta,
                'rho': self.rho,
                'teams': self.teams,
                'matches_history': self.matches_history,
                'knn_model': self.knn_model
            }, f)
    
    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.alpha = data['alpha']
        self.beta = data['beta']
        self.rho = data['rho']
        self.teams = data['teams']
        self.n_teams = len(self.teams)
        self.matches_history = data.get('matches_history', [])
        self.knn_model = data.get('knn_model', None)
