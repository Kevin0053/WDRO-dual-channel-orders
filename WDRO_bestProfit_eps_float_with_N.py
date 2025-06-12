"""
本代码实现了一个基于Wasserstein距离的分布鲁棒优化(WDRO)模型,用于求解双渠道订单分配问题。

主要功能:
1. 针对不同样本规模N,动态计算Wasserstein半径ε(epsilon)
2. 对每个样本规模进行多次模拟,求解最优决策变量
3. 考虑了固定产能和弹性产能的分配
4. 包含了SLA(服务水平协议)的约束条件

关键参数:
- 固定参数包括各类成本(固定产能、弹性产能、持有成本等)
- 基础参数包括单位利润和最大需求等
- 使用仿射函数系数来构建优化模型的目标函数

输出结果:
1. 不同样本规模下的最优决策变量
2. 目标函数值(总利润)
3. 结果可视化和数据保存

"""

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import os

# 设置随机种子
np.random.seed(42)

# 参数设置
N_values = np.arange(50, 401, 50)  # 样本规模从50到400，间隔50
num_simulations = 5        # 每个样本规模的模拟次数

# 固定参数
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
    'mu1': 0.1,   # 渠道1 SLA置信水平参数
    'mu2': 0.1    # 渠道2 SLA置信水平参数
}

# 基础参数（用于需求生成）
base_params = {
    'p1': 6.0,    # 平台订单单位价格
    'p2': 7.0,    # 线下订单单位价格
    'bar_D1': 100.0,  # 平台最大需求
    'bar_D2': 80.0    # 线下最大需求
}

# 定义仿射函数系数
def get_affine_coefficients():
    p1 = base_params['p1']
    p2 = base_params['p2']
    c_d1 = fixed_params['c_d1']
    c_d2 = fixed_params['c_d2']
    c_e1 = fixed_params['c_e1']
    c_e2 = fixed_params['c_e2']
    h = fixed_params['h']
    t1 = fixed_params['t1']
    t2 = fixed_params['t2']
    rho1 = fixed_params['rho1']
    rho2 = fixed_params['rho2']
    mu1 = fixed_params['mu1']
    mu2 = fixed_params['mu2']

    a_k = {}
    b_k = {}
    c_k = {}

    # 8 base scenarios for f_k
    f_k_a = {
        1: np.array([-p1, -p2]),
        2: np.array([-p1, -p2]),
        3: np.array([-p1, t2]),
        4: np.array([-p1, t2]),
        5: np.array([t1, -p2]),
        6: np.array([t1, -p2]),
        7: np.array([t1, t2]),
        8: np.array([t1, t2])
    }
    f_k_b = {
        1: np.array([0, 0, c_d1, c_d2, c_e1, c_e2]),
        2: np.array([h, 0, c_d1 - h, c_d2 - h, c_e1, c_e2]),
        3: np.array([0, 0, c_d1, c_d2 - p2 - t2, c_e1, c_e2 - p2 - t2]),
        4: np.array([h, 0, c_d1 - h, c_d2 - p2 - t2 - h, c_e1, c_e2 - p2 - t2]),
        5: np.array([0, 0, c_d1 - p1 - t1, c_d2, c_e1 - p1 - t1, c_e2]),
        6: np.array([h, 0, c_d1 - p1 - t1 - h, c_d2 - h, c_e1 - p1 - t1, c_e2]),
        7: np.array([0, 0, c_d1 - p1 - t1, c_d2 - p2 - t2, c_e1 - p1 - t1, c_e2 - p2 - t2]),
        8: np.array([h, 0, c_d1 - p1 - t1 - h, c_d2 - p2 - t2 - h, c_e1 - p1 - t1, c_e2 - p2 - t2])
    }
    
    # 32 pieces in total
    idx = 1
    # Type 1: Normal + Normal
    for k in range(1, 9):
        a_k[idx] = f_k_a[k]
        b_k[idx] = f_k_b[k]
        c_k[idx] = np.array([rho1, rho2])
        idx += 1

    # Type 2: Tail ch1 + Normal ch2
    for k in range(1, 9):
        a_k[idx] = f_k_a[k] + np.array([rho1/mu1, 0])
        b_k[idx] = f_k_b[k] + np.array([0, 0, -rho1/mu1, 0, -rho1/mu1, 0])
        c_k[idx] = np.array([rho1*(1-1/mu1), rho2])
        idx += 1

    # Type 3: Normal ch1 + Tail ch2
    for k in range(1, 9):
        a_k[idx] = f_k_a[k] + np.array([0, rho2/mu2])
        b_k[idx] = f_k_b[k] + np.array([0, 0, 0, -rho2/mu2, 0, -rho2/mu2])
        c_k[idx] = np.array([rho1, rho2*(1-1/mu2)])
        idx += 1
    
    # Type 4: Tail ch1 + Tail ch2
    for k in range(1, 9):
        a_k[idx] = f_k_a[k] + np.array([rho1/mu1, rho2/mu2])
        b_k[idx] = f_k_b[k] + np.array([0, 0, -rho1/mu1, -rho2/mu2, -rho1/mu1, -rho2/mu2])
        c_k[idx] = np.array([rho1*(1-1/mu1), rho2*(1-1/mu2)])
        idx += 1

    return a_k, b_k, c_k


def generate_samples(N):
    """生成正态分布的需求样本，并截断到合理范围"""
    D1 = np.random.normal(loc=base_params['bar_D1'] * 0.5, scale=10, size=N)
    D2 = np.random.normal(loc=base_params['bar_D2'] * 0.5, scale=10, size=N)
    
    # 截断到非负且不超过最大需求
    D1 = np.clip(D1, 0, base_params['bar_D1'])
    D2 = np.clip(D2, 0, base_params['bar_D2'])
    
    return np.column_stack((D1, D2))

def solve_dro_model(samples, c_d, c_e, c_d1, c_d2, c_e1, c_e2, h, rho1, rho2, t1, t2, mu1, mu2, epsilon):
    """求解 DRO 模型"""
    N = len(samples)
    model = gp.Model("DRO_Model")
    
    # 设置求解器参数
    model.setParam("MIPGap", 0.01)
    model.setParam("TimeLimit", 60)
    model.setParam("Threads", 4)
    
    # 定义约束矩阵 C
    C = np.array([
        [1, 0],   # D1 <= bar_D1
        [0, 1],   # D2 <= bar_D2
        [-1, 0],  # -D1 <= 0 (D1 >= 0)
        [0, -1]   # -D2 <= 0 (D2 >= 0)
    ])
    
    # 决策变量
    x_vars = model.addVars(6, lb=0, name="x")  # x = [bar_xd, xe, xd1, xd2, xe1, xe2]
    tau = model.addVars(2, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="tau")
    
    # 辅助变量
    lambda_var = model.addVar(lb=0, name="lambda")
    s = model.addVars(N, lb=0, name="s")
    
    # 对偶变量 gamma_ik
    gamma = {}
    for i in range(N):
        for k in range(1, 33):  # 32个分段
            gamma[(i, k)] = model.addVars(4, lb=0, name=f"gamma_{i}_{k}")
    
    # 获取仿射函数系数
    a_k, b_k, c_k = get_affine_coefficients()
    
    # 最大需求值
    d = np.array([base_params['bar_D1'], base_params['bar_D2'], 0.0, 0.0])
    
    # 目标函数 - 根据式(3)
    obj = lambda_var * epsilon + (1/N) * gp.quicksum(s[i] for i in range(N)) + \
          c_d * x_vars[0] + c_e * x_vars[1]
    model.setObjective(obj, GRB.MINIMIZE)
    
    # 约束条件
    for i in range(N):
        # 32个分段约束
        for k in range(1, 33):
            a_term = a_k[k] @ samples[i]
            b_term = gp.quicksum(b_k[k][j] * x_vars[j] for j in range(6))
            c_term = gp.quicksum(c_k[k][j] * tau[j] for j in range(2))
            Cxi = C @ samples[i]
            gamma_term = sum(gamma[(i, k)][m] * (d[m] - Cxi[m]) for m in range(4))
            model.addConstr(a_term + b_term + c_term + gamma_term <= s[i])
            
    # L1 范数对偶约束
    for i in range(N):
        for k in range(1, 33):
            C_T_gamma = C.T @ [gamma[(i, k)][m] for m in range(4)]
            diff = C_T_gamma - a_k[k]
            for m in range(len(diff)):
                model.addConstr(diff[m] <= lambda_var)
                model.addConstr(-diff[m] <= lambda_var)
    
    # 产能约束
    model.addConstr(x_vars[2] + x_vars[3] <= x_vars[0], "fixed_capacity")  # 固定产能约束
    model.addConstr(x_vars[4] + x_vars[5] <= x_vars[1], "flexible_capacity")  # 弹性产能约束
    
    # 求解
    model.optimize()
    
    # 记录结果
    if model.status == GRB.OPTIMAL:
        # model.ObjVal 是最小化的最坏情况总成本
        # worst_case_loss = model.ObjVal - (c_d * x_vars[0].X + c_e * x_vars[1].X)
        # profit = -worst_case_loss
        worst_case_loss = model.ObjVal - (fixed_params['c_d'] * x_vars[0].X + fixed_params['c_e'] * x_vars[1].X)
        
        result = {
            'bar_xd': x_vars[0].X,
            'xe': x_vars[1].X,
            'xd1': x_vars[2].X,
            'xd2': x_vars[3].X,
            'xe1': x_vars[4].X,
            'xe2': x_vars[5].X,
            'profit': worst_case_loss,  # 此处的'profit'存储的是计算出的最坏情况利润
            'objective': model.ObjVal,   # 保留原始目标值
            'epsilon': epsilon
        }
        return result
    else:
        raise ValueError(f"DRO Model did not find optimal solution for N={N}")

# 在保存结果之前，确保文件夹存在
output_dir = os.path.join('results', '32_dro_profit_analysis')
os.makedirs(output_dir, exist_ok=True)


# 运行不同样本规模的仿真
results = []
for N in N_values:
    print(f"\n处理样本规模 N = {N}")
    epsilon_current = 1 / np.sqrt(N)  # 动态计算 ε
    
    # 对每个样本规模进行多次模拟
    for sim in range(num_simulations):
        print(f"  模拟 {sim+1}/{num_simulations}")
        samples = generate_samples(N)
        result = solve_dro_model(samples, 
                               fixed_params['c_d'], fixed_params['c_e'],
                               fixed_params['c_d1'], fixed_params['c_d2'],
                               fixed_params['c_e1'], fixed_params['c_e2'],
                               fixed_params['h'], fixed_params['rho1'],
                               fixed_params['rho2'], fixed_params['t1'], 
                               fixed_params['t2'], fixed_params['mu1'],
                               fixed_params['mu2'], epsilon_current)
        result['N'] = N
        result['simulation'] = sim+1
        results.append(result)

# 数据处理和可视化
results_df = pd.DataFrame(results)

# Group by epsilon and calculate statistics
summary = results_df.groupby('epsilon').agg(['mean', 'std'])

# Flatten MultiIndex columns and reset index
summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
summary = summary.reset_index()


# Save results
output_dir = 'results/best_profit_analysis_with_N_final'
os.makedirs(output_dir, exist_ok=True)
excel_path = os.path.join(output_dir, 'dro_best_profit_summary_with_N.xlsx')
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name='Raw_Data', index=False)
    summary.to_excel(writer, sheet_name='Summary_Statistics', index=False)


# 可视化分析
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(3, 1, figsize=(12, 18))
fig.suptitle('DRO Model Performance vs. Wasserstein Radius ($\\epsilon$)', fontsize=16)

# --- Subplot 1: Profit ---
axes[0].errorbar(summary['epsilon'], summary['profit_mean'], yerr=summary['profit_std'],
                 fmt='-o', capsize=5, color='b', label='Worst-case Profit')
axes[0].set_xlabel('Wasserstein Radius ($\\epsilon$)', fontsize=12)
axes[0].set_ylabel('Worst-case Guaranteed Profit', fontsize=12)
axes[0].set_title('Profit vs. Epsilon', fontsize=14)
axes[0].legend()

# --- Subplot 2: Capacities ---
axes[1].errorbar(summary['epsilon'], summary['bar_xd_mean'], yerr=summary['bar_xd_std'],
                 fmt='-s', capsize=5, color='r', label='Total Fixed Capacity (bar_xd)')
axes[1].errorbar(summary['epsilon'], summary['xe_mean'], yerr=summary['xe_std'],
                 fmt='-^', capsize=5, color='g', label='Total Flexible Capacity (xe)')
axes[1].set_xlabel('Wasserstein Radius ($\\epsilon$)', fontsize=12)
axes[1].set_ylabel('Capacity Level', fontsize=12)
axes[1].set_title('Capacity Decisions vs. Epsilon', fontsize=14)
axes[1].legend()

# --- Subplot 3: Capacity Allocation ---
axes[2].plot(summary['epsilon'], summary['xd1_mean'], 'o-', label='Fixed Capacity Ch1 (xd1)')
axes[2].plot(summary['epsilon'], summary['xd2_mean'], 's-', label='Fixed Capacity Ch2 (xd2)')
axes[2].plot(summary['epsilon'], summary['xe1_mean'], '^--', label='Flexible Capacity Ch1 (xe1)')
axes[2].plot(summary['epsilon'], summary['xe2_mean'], 'v--', label='Flexible Capacity Ch2 (xe2)')
axes[2].set_xlabel('Wasserstein Radius ($\\epsilon$)', fontsize=12)
axes[2].set_ylabel('Allocated Capacity', fontsize=12)
axes[2].set_title('Capacity Allocation vs. Epsilon', fontsize=14)
axes[2].legend()


plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig(os.path.join(output_dir, '32_dro_simulation_results_plot.png'), dpi=300)
plt.show()