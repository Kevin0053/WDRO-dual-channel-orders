import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from tqdm import tqdm  # 显示进度条

# 设置随机种子
np.random.seed(42)

# 参数设置
N_values = np.arange(30, 601, 30)  # N 从 30 到 600，步长 30
num_realization = 10**6            # 真实分布下的模拟样本数

# 固定参数
c_d = 1.0   # 固定产能单位成本
c_e = 0.5   # 弹性产能单位成本
h = 0.2     # 固定产能持有成本
p1 = 5.0    # 平台订单单位利润
p2 = 4.5    # 线下订单单位利润
rho = 1.0   # SLA惩罚权重

# 最大需求值
bar_D1 = 100.0
bar_D2 = 80.0

# 存储结果
results = []

def generate_samples(N):
    """生成正态分布的需求样本，并截断到合理范围"""
    D1 = np.random.normal(loc=bar_D1 * 0.5, scale=10, size=N)
    D2 = np.random.normal(loc=bar_D2 * 0.5, scale=10, size=N)
    D1 = np.clip(D1, 0, bar_D1)
    D2 = np.clip(D2, 0, bar_D2)
    return np.column_stack((D1, D2))

def solve_stochastic_model(samples):
    """基于正态分布的随机优化模型，求解最优决策变量"""
    M = len(samples)
    model = gp.Model("Stochastic_Model")
    model.setParam("OutputFlag", 0)
    model.setParam("MIPGap", 0.01)
    model.setParam("TimeLimit", 60)
    
    # 决策变量
    bar_xd = model.addVar(lb=0, name="bar_xd")
    xe = model.addVar(lb=0, name="xe")
    
    # 辅助变量 t1_i 和 t2_i（用于处理 max 函数）
    t1 = model.addVars(M, lb=0, name="t1")
    t2 = model.addVars(M, lb=0, name="t2")
    
    # 目标函数：最小化总成本
    obj = c_d * bar_xd + c_e * xe
    model.setObjective(obj, GRB.MINIMIZE)
    
    # 约束：对于每个样本 i，约束收入 - 成本 >= -惩罚项
    for i in range(M):
        D1, D2 = samples[i]
        model.addConstr(t1[i] >= D1 - bar_xd - xe)
        model.addConstr(t2[i] >= D2 - bar_xd - xe)
        model.addConstr(p1 * D1 + p2 * D2 - obj >= -rho * (t1[i] + t2[i]))
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        return bar_xd.X, xe.X
    else:
        raise ValueError("Stochastic model did not find optimal solution")

def compute_profit(samples, bar_xd, xe):
    """计算利润（收入 - 成本）"""
    profits = []
    for D in samples:
        D1, D2 = D
        revenue = p1 * D1 + p2 * D2
        
        # 成本计算
        cost_fixed = c_d * bar_xd
        cost_flexible = c_e * xe
        shortage1 = max(D1 - bar_xd - xe, 0)
        shortage2 = max(D2 - bar_xd - xe, 0)
        cost_penalty = rho * (shortage1 + shortage2)
        
        total_cost = cost_fixed + cost_flexible + cost_penalty
        profit = revenue - total_cost
        profits.append(profit)
    return np.mean(profits)

def solve_dro_model(samples, epsilon):
    """求解 DRO 模型"""
    N = len(samples)
    model = gp.Model("DRO_Model")
    
    # 设置求解器参数
    model.setParam("MIPGap", 0.01)
    model.setParam("TimeLimit", 60)
    model.setParam("Threads", 4)
    model.setParam("OutputFlag", 0)
    
    # 决策变量
    bar_xd = model.addVar(lb=0, name="bar_xd")
    xe = model.addVar(lb=0, name="xe")
    
    # 辅助变量 t1_i 和 t2_i（用于处理 max 函数）
    t1 = model.addVars(N, lb=0, name="t1")
    t2 = model.addVars(N, lb=0, name="t2")
    
    # 双变量 lambda
    lambda_var = model.addVar(lb=0, name="lambda")
    
    # 目标函数
    obj = c_d * bar_xd + c_e * xe + (1/N) * gp.quicksum(t1[i] + t2[i] for i in range(N)) + lambda_var * epsilon
    model.setObjective(obj, GRB.MINIMIZE)
    
    # 约束：对于每个样本 i，约束收入 - 成本 >= -惩罚项
    for i in range(N):
        D1, D2 = samples[i]
        model.addConstr(t1[i] >= D1 - bar_xd - xe)
        model.addConstr(t2[i] >= D2 - bar_xd - xe)
        model.addConstr(p1 * D1 + p2 * D2 - (c_d * bar_xd + c_e * xe) >= -rho * (t1[i] + t2[i]))
    
    # L1 范数约束
    for i in range(N):
        model.addConstr(lambda_var >= t1[i])
        model.addConstr(lambda_var >= t2[i])
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        return {
            'bar_xd': bar_xd.X,
            'xe': xe.X,
            'objective': model.ObjVal
        }
    else:
        raise ValueError("Model did not find optimal solution")

# 第一步：基于 M = 10^6 样本计算真实最优利润 Π_true
print("=== 计算真实最优利润 ===")
true_samples = generate_samples(num_realization)
bar_xd_true, xe_true = solve_stochastic_model(true_samples)
profit_true = compute_profit(true_samples, bar_xd_true, xe_true)
print(f"真实最优利润 Π_true: {profit_true:.2f}")

# 第二步：对每个 N 运行鲁棒模型，计算利润和绩效损失
for N in tqdm(N_values, desc="Running Experiments"):
    print(f"\n=== N = {N} ===")
    samples = generate_samples(N)
    epsilon_current = 1 / np.sqrt(N)
    
    # 求解鲁棒优化模型
    robust_result = solve_dro_model(samples, epsilon_current)
    bar_xd_robust = robust_result['bar_xd']
    xe_robust = robust_result['xe']
    
    # 计算鲁棒策略下的利润（在真实分布下）
    profit_robust = compute_profit(true_samples, bar_xd_robust, xe_robust)
    
    # 计算绩效损失
    performance_loss = (profit_true - profit_robust) / profit_true
    
    # 存储结果
    results.append({
        'N': N,
        'epsilon': epsilon_current,
        'profit_robust': profit_robust,
        'performance_loss': performance_loss
    })

# 可视化
results_df = pd.DataFrame(results)

plt.figure(figsize=(12, 6))

# 折线图 1: 真实利润 vs 鲁棒利润
plt.subplot(1, 2, 1)
plt.plot(results_df['N'], [profit_true]*len(results_df), label='True Profit', color='blue', linestyle='--', linewidth=2)
plt.plot(results_df['N'], results_df['profit_robust'], label='Robust Profit', color='orange', marker='s')
plt.title('Profit Comparison (True vs Robust)')
plt.xlabel('Sample Size N')
plt.ylabel('Profit')
plt.grid(True)
plt.legend()

# 折线图 2: 绩效损失
plt.subplot(1, 2, 2)
plt.plot(results_df['N'], results_df['performance_loss'], label='Performance Loss', color='red', marker='^')
plt.title('Performance Loss Ratio')
plt.xlabel('Sample Size N')
plt.ylabel('Loss Ratio')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('performance_comparison.png')
plt.show()

# 保存结果
results_df.to_csv('performance_comparison_results.csv', index=False)