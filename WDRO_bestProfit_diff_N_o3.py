"""
WDRO Best-Profit Experiment (o3 Version)
========================================
本脚本实现基于 Wasserstein 距离的分布鲁棒优化 (WDRO) 双渠道产能投资模型（32 段仿射函数版本）。
功能：
1. 针对不同样本规模 N，动态计算 ε = 1 / N。
2. 多次随机模拟生成样本并求解 WDRO 最优投资与分配方案。
3. 输出各 N 下的最优决策变量与最坏情况保证利润，并可视化分析。
输出：
- 结果数据： results/bestProfit_03/bestProfit_diff_N_results.xlsx
- 分析图片： results/bestProfit_03/bestProfit_diff_N_plots.png
"""

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import os

# ---------- 全局设置 ---------- #
np.random.seed(42)
plt.rcParams['font.family'] = 'SimHei'  # 如中文字体不可用，可注释

# 输出目录
OUTPUT_DIR = os.path.join('results', 'bestProfit_03')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- 基础与固定参数 ---------- #
base_params = {
    'p1': 6.0,          # 渠道1 单位销售价
    'p2': 7.0,          # 渠道2 单位销售价
    'bar_D1': 100.0,    # 渠道1 最大需求
    'bar_D2': 80.0      # 渠道2 最大需求
}

fixed_params = {
    'c_d': 1.0,   # 总固定产能单位成本
    'c_e': 1.2,   # 总弹性产能单位成本
    'c_d1': 1.0,  # 渠道1固定产能单位成本
    'c_d2': 1.0,  # 渠道2固定产能单位成本
    'c_e1': 1.0,  # 渠道1弹性产能单位成本
    'c_e2': 1.0,  # 渠道2弹性产能单位成本
    'h': 0.5,     # 固定产能持有成本
    'rho1': 1.0,  # 渠道1 SLA惩罚权重
    'rho2': 1.0,  # 渠道2 SLA惩罚权重
    't1': 7.0,    # 渠道1需求违约惩罚成本
    't2': 7.5,    # 渠道2需求违约惩罚成本
    'mu1': 0.1,   # 渠道1 SLA置信水平
    'mu2': 0.1    # 渠道2 SLA置信水平
}

# ---------- 仿射函数系数 ---------- #

def get_affine_coefficients(fp):
    """返回 (a_k, b_k, c_k) 共 32 段。"""
    p1, p2 = base_params['p1'], base_params['p2']
    c_d1, c_d2, c_e1, c_e2 = fp['c_d1'], fp['c_d2'], fp['c_e1'], fp['c_e2']
    h, t1, t2 = fp['h'], fp['t1'], fp['t2']
    rho1, rho2, mu1, mu2 = fp['rho1'], fp['rho2'], fp['mu1'], fp['mu2']

    a_k, b_k, c_k = {}, {}, {}

    # 基础 8 段
    f_a = {
        1: np.array([-p1, -p2]),
        2: np.array([-p1, -p2]),
        3: np.array([-p1, t2]),
        4: np.array([-p1, t2]),
        5: np.array([t1, -p2]),
        6: np.array([t1, -p2]),
        7: np.array([t1, t2]),
        8: np.array([t1, t2])
    }
    f_b = {
        1: np.array([0, 0, c_d1, c_d2, c_e1, c_e2]),
        2: np.array([h, 0, c_d1 - h, c_d2 - h, c_e1, c_e2]),
        3: np.array([0, 0, c_d1, c_d2 - p2 - t2, c_e1, c_e2 - p2 - t2]),
        4: np.array([h, 0, c_d1 - h, c_d2 - p2 - t2 - h, c_e1, c_e2 - p2 - t2]),
        5: np.array([0, 0, c_d1 - p1 - t1, c_d2, c_e1 - p1 - t1, c_e2]),
        6: np.array([h, 0, c_d1 - p1 - t1 - h, c_d2 - h, c_e1 - p1 - t1, c_e2]),
        7: np.array([0, 0, c_d1 - p1 - t1, c_d2 - p2 - t2, c_e1 - p1 - t1, c_e2 - p2 - t2]),
        8: np.array([h, 0, c_d1 - p1 - t1 - h, c_d2 - p2 - t2 - h, c_e1 - p1 - t1, c_e2 - p2 - t2])
    }

    idx = 1
    for k in range(1, 9):  # Normal + Normal
        a_k[idx], b_k[idx], c_k[idx] = f_a[k], f_b[k], np.array([rho1, rho2]); idx += 1
    for k in range(1, 9):  # Tail ch1 + Normal ch2
        a_k[idx] = f_a[k] + np.array([rho1/mu1, 0])
        b_k[idx] = f_b[k] + np.array([0, 0, -rho1/mu1, 0, -rho1/mu1, 0])
        c_k[idx] = np.array([rho1*(1-1/mu1), rho2]); idx += 1
    for k in range(1, 9):  # Normal ch1 + Tail ch2
        a_k[idx] = f_a[k] + np.array([0, rho2/mu2])
        b_k[idx] = f_b[k] + np.array([0, 0, 0, -rho2/mu2, 0, -rho2/mu2])
        c_k[idx] = np.array([rho1, rho2*(1-1/mu2)]); idx += 1
    for k in range(1, 9):  # Tail ch1 + Tail ch2
        a_k[idx] = f_a[k] + np.array([rho1/mu1, rho2/mu2])
        b_k[idx] = f_b[k] + np.array([0, 0, -rho1/mu1, -rho2/mu2, -rho1/mu1, -rho2/mu2])
        c_k[idx] = np.array([rho1*(1-1/mu1), rho2*(1-1/mu2)]); idx += 1
    return a_k, b_k, c_k

# ---------- 需求样本 ---------- #

def generate_samples(N):
    D1 = np.random.normal(loc=base_params['bar_D1']*0.5, scale=10, size=N)
    D2 = np.random.normal(loc=base_params['bar_D2']*0.5, scale=10, size=N)
    D1 = np.clip(D1, 0, base_params['bar_D1'])
    D2 = np.clip(D2, 0, base_params['bar_D2'])
    return np.column_stack((D1, D2))

# ---------- WDRO 求解 ---------- #

C = np.array([[1,0],[0,1],[-1,0],[0,-1]])
d = np.array([base_params['bar_D1'], base_params['bar_D2'], 0.0, 0.0])

def solve_dro_model(samples, epsilon, fp):
    N = len(samples)
    model = gp.Model('WDRO_bestProfit_o3'); model.Params.OutputFlag = 0
    model.Params.TimeLimit = 120; model.Params.MIPGap = 0.01; model.Params.Threads = 4

    # 决策变量
    x = model.addVars(6, lb=0, name='x')  # [bar_xd, xe, xd1, xd2, xe1, xe2]
    tau = model.addVars(2, lb=-GRB.INFINITY, name='tau')
    lambda_var = model.addVar(lb=0, name='lambda')
    s = model.addVars(N, lb=0, name='s')

    gamma = {(i,k): model.addVars(4, lb=0, name=f'g_{i}_{k}')
             for i in range(N) for k in range(1,33)}

    a_k, b_k, c_k = get_affine_coefficients(fp)

    # 目标函数
    # Profit = - (cost term); maximize worst-case guaranteed profit
    profit_expr = -1 * (lambda_var * epsilon + (1 / N) * gp.quicksum(s[i] for i in range(N))
                       + fp['c_d'] * x[0] + fp['c_e'] * x[1])
    model.setObjective(profit_expr, GRB.MAXIMIZE)

    # 约束
    for i in range(N):
        Cxi = C @ samples[i]
        for k in range(1,33):
            expr = (a_k[k] @ samples[i] +
                    gp.quicksum(b_k[k][j]*x[j] for j in range(6)) +
                    gp.quicksum(c_k[k][j]*tau[j] for j in range(2)) +
                    gp.quicksum(gamma[(i,k)][m]*(d[m]-Cxi[m]) for m in range(4)))
            model.addConstr(expr <= s[i])

    for i in range(N):
        for k in range(1,33):
            diff = C.T @ [gamma[(i,k)][m] for m in range(4)] - a_k[k]
            model.addConstr(diff[0] <= lambda_var); model.addConstr(-diff[0] <= lambda_var)
            model.addConstr(diff[1] <= lambda_var); model.addConstr(-diff[1] <= lambda_var)

    # 容量约束
    model.addConstr(x[2]+x[3] <= x[0])
    model.addConstr(x[4]+x[5] <= x[1])

    model.optimize()

    if model.status == GRB.OPTIMAL:
        # Since we maximized profit directly, ObjVal already equals the worst-case guaranteed profit
        worst_case_profit = model.ObjVal
        return {
            'bar_xd': x[0].X, 'xe': x[1].X,
            'xd1': x[2].X, 'xd2': x[3].X,
            'xe1': x[4].X, 'xe2': x[5].X,
            'profit': worst_case_profit,
            'objective': model.ObjVal
        }
    return None

# ---------- 仿真流程 ---------- #

N_values = np.arange(50, 401, 50)
num_simulations = 3

records = []
for N in N_values:
    eps = 1.0 / N
    for sim in range(num_simulations):
        samples = generate_samples(N)
        res = solve_dro_model(samples, eps, fixed_params)
        if res:
            res.update({'N': N, 'epsilon': eps, 'sim': sim+1})
            records.append(res)
        else:
            print(f'N={N}, sim={sim+1}: No feasible solution')

results_df = pd.DataFrame(records)
results_df.to_excel(os.path.join(OUTPUT_DIR, 'o3_bestProfit_diff_N_results_raw.xlsx'), index=False)

summary = results_df.groupby('N').mean(numeric_only=True).reset_index()
summary.to_excel(os.path.join(OUTPUT_DIR, 'o3_bestProfit_diff_N_results.xlsx'), index=False)

# ---------- 可视化 ---------- #
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(4, 1, figsize=(12, 20))
fig.suptitle('WDRO Worst-case Guaranteed Profit and Decisions vs Sample Size (ε=1/N)', fontsize=16)

axes[0].plot(summary['N'], summary['profit'], 'o-', label='Worst-case Profit')
axes[0].set_ylabel('Profit'); axes[0].legend(); axes[0].grid(True)

axes[1].plot(summary['N'], summary['bar_xd'], 's-', label='bar_xd (Fixed capacity)')
axes[1].plot(summary['N'], summary['xe'], '^-', label='xe (Flexible option)')
axes[1].set_ylabel('Capacity'); axes[1].legend(); axes[1].grid(True)

axes[2].plot(summary['N'], summary['xd1'], 'o--', label='xd1')
axes[2].plot(summary['N'], summary['xd2'], 's--', label='xd2')
axes[2].set_ylabel('Capacity'); axes[2].legend(); axes[2].grid(True)

axes[3].plot(summary['N'], summary['xe1'], 'o:', label='xe1')
axes[3].plot(summary['N'], summary['xe2'], 's:', label='xe2')
axes[3].set_ylabel('Capacity'); axes[3].set_xlabel('Sample Size N'); axes[3].legend(); axes[3].grid(True)

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(os.path.join(OUTPUT_DIR, 'o3_bestProfit_diff_N_plots.png'), dpi=300)
plt.show()
