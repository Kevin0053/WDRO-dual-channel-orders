"""
这是一个用于评估样本外最坏情况性能的代码。

主要功能:
1. 使用不同样本量N生成训练和测试数据
2. 对每个N进行多次模拟,计算最优决策和目标值
3. 分析Wasserstein距离半径ε如何随N变化影响结果
4. 评估模型在样本外数据上的表现

关键参数:
- num_simulations: 每个N的模拟次数(5次)
- N_values: 样本量从50到400,步长50
- test_size: 验证集比例0.2
- fixed_params: 包含各种成本参数
- base_params: 需求分布的基础参数

主要步骤:
1. 设置随机种子确保可重复性
2. 定义参数和超参数
3. 实现Wasserstein半径计算函数
4. 实现样本生成函数
5. 实现求解器和评估函数
"""

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os

# 设置随机种子
np.random.seed(42)

# 参数设置
num_simulations = 1  # 每个 N 运行 1 次
N_values = np.arange(50, 401, 50)  # 样本数量从 50 到 400，步长 50
test_size = 0.2  # 验证集比例

# 固定参数
fixed_params = {
    'c_d': 1.0,  # 固定产能单位成本
    'c_e': 1.2,  # 弹性产能单位成本
    'h': 0.2,    # 固定产能持有成本
    'rho': 1.0,  # SLA惩罚权重
    't1': 7.0,   # 渠道1需求违约惩罚成本
    't2': 7.5,   # 渠道2需求违约惩罚成本
    'mu': 0.1    # SLA置信水平参数
}

# 基础参数（用于需求生成）
base_params = {
    'p1': 6.0,
    'p2': 7.0,
    'bar_D1': 100.0,
    'bar_D2': 80.0
}

# 计算Wasserstein距离半径
def calculate_epsilon(N):
    """
    计算Wasserstein距离半径，随样本数量N变化
    epsilon = 1/sqrt(N)
    """
    return 1.0 / np.sqrt(N)

# 生成正态分布样本（并截断以确保非负）
def generate_samples(N):
    D1 = np.random.normal(loc=base_params['bar_D1'] * 0.5, scale=10, size=N)
    D2 = np.random.normal(loc=base_params['bar_D2'] * 0.5, scale=10, size=N)
    D1 = np.clip(D1, 0, base_params['bar_D1'])
    D2 = np.clip(D2, 0, base_params['bar_D2'])
    return np.column_stack((D1, D2))

# 定义仿射函数系数
def get_affine_coefficients():
    # SLA-Normal情况下的系数
    a_k_N = {
        1: np.array([-base_params['p1'], -base_params['p2']]),  # 基础利润项
        2: np.array([-base_params['p1'], -base_params['p2']]),  # 基础利润项 + 持有成本
        3: np.array([-base_params['p1'], 0.0]),  # 渠道1产能过剩
        4: np.array([0.0, -base_params['p2']]),  # 渠道2产能过剩
        5: np.array([-base_params['p1'], 0.0]),  # 渠道1产能过剩 + 持有成本
        6: np.array([0.0, -base_params['p2']]),  # 渠道2产能过剩 + 持有成本
        7: np.array([fixed_params['t1'], 0.0]),  # 渠道1短缺惩罚
        8: np.array([0.0, fixed_params['t2']]),  # 渠道2短缺惩罚
        9: np.array([-base_params['p1'], -base_params['p2']]),  # 双渠道产能过剩 + 持有成本
        10: np.array([-base_params['p1'], -base_params['p2']]),  # 双渠道产能过剩
        11: np.array([fixed_params['t1'], fixed_params['t2']])  # 双渠道短缺惩罚
    }

    b_k_N = {
        1: np.array([0.0, 0.0, fixed_params['c_d'], fixed_params['c_d'], fixed_params['c_e'], fixed_params['c_e']]),
        2: np.array([fixed_params['h'], 0.0, fixed_params['c_d']-fixed_params['h'], fixed_params['c_d']-fixed_params['h'], fixed_params['c_e'], fixed_params['c_e']]),
        3: np.array([0.0, 0.0, base_params['p1'], 0.0, base_params['p1'], 0.0]),
        4: np.array([0.0, 0.0, 0.0, base_params['p2'], 0.0, base_params['p2']]),
        5: np.array([fixed_params['h'], 0.0, base_params['p1']-fixed_params['h'], 0.0, base_params['p1'], 0.0]),
        6: np.array([fixed_params['h'], 0.0, 0.0, base_params['p2']-fixed_params['h'], 0.0, base_params['p2']]),
        7: np.array([0.0, 0.0, -fixed_params['t1'], 0.0, -fixed_params['t1'], 0.0]),
        8: np.array([0.0, 0.0, 0.0, -fixed_params['t2'], 0.0, -fixed_params['t2']]),
        9: np.array([fixed_params['h'], 0.0, base_params['p1']-fixed_params['h'], base_params['p2']-fixed_params['h'], base_params['p1'], base_params['p2']]),
        10: np.array([0.0, 0.0, base_params['p1'], base_params['p2'], base_params['p1'], base_params['p2']]),
        11: np.array([0.0, 0.0, -fixed_params['t1'], -fixed_params['t2'], -fixed_params['t1'], -fixed_params['t2']])
    }

    c_k_N = {k: fixed_params['rho'] for k in range(1, 12)}

    # SLA-Tail情况下的系数
    a_k_T = {
        k: a_k_N[k] + np.array([fixed_params['rho']/fixed_params['mu'], fixed_params['rho']/fixed_params['mu']])
        for k in range(1, 12)
    }

    b_k_T = {
        k: b_k_N[k] - np.array([0.0, 0.0, fixed_params['rho']/fixed_params['mu'], fixed_params['rho']/fixed_params['mu'], 
                               fixed_params['rho']/fixed_params['mu'], fixed_params['rho']/fixed_params['mu']])
        for k in range(1, 12)
    }

    c_k_T = {k: fixed_params['rho']*(1-1/fixed_params['mu']) for k in range(1, 12)}

    return a_k_N, b_k_N, c_k_N, a_k_T, b_k_T, c_k_T

# 求解 DRO 模型（鲁棒优化）
def solve_dro_model(samples, c_d, c_e, h, rho, t1, t2, mu, epsilon):
    N = len(samples)
    model = gp.Model("DRO_Model")
    
    # 设置求解器参数
    model.setParam("MIPGap", 0.01)
    model.setParam("TimeLimit", 60)
    model.setParam("Threads", 4)
    
    # 定义约束矩阵 C
    C = np.array([
        [1, 0],   # 固定产能约束：xd1 + xd2 <= bar_xd
        [0, 1],   # 弹性产能约束：xe1 + xe2 <= xe
        [-1, 0],  # xd1 >= -D1
        [0, -1]   # xd2 >= -D2
    ])
    
    # 决策变量
    x_vars = model.addVars(6, lb=0, name="x")  # x = [bar_xd, xe, xd1, xd2, xe1, xe2]
    tau = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="tau")
    
    # 辅助变量
    lambda_var = model.addVar(lb=0, name="lambda")
    s = model.addVars(N, lb=0, name="s")
    
    # 对偶变量 gamma_ik
    gamma = {}
    for i in range(N):
        for k in range(1, 23):  # 22个分段
            gamma[(i, k)] = model.addVars(4, lb=0, name=f"gamma_{i}_{k}")
    
    # 获取仿射函数系数
    a_k_N, b_k_N, c_k_N, a_k_T, b_k_T, c_k_T = get_affine_coefficients()
    
    # 最大需求值
    d = np.array([base_params['bar_D1'], base_params['bar_D2'], 0.0, 0.0])

    # 目标函数
    obj = lambda_var * epsilon + (1/N) * gp.quicksum(s[i] for i in range(N)) + c_d * x_vars[0] + c_e * x_vars[1]
    model.setObjective(obj, GRB.MINIMIZE)
    
    # 约束条件
    for i in range(N):
        # SLA-Normal情况下的约束
        for k in range(1, 12):
            a_term = a_k_N[k] @ samples[i]
            b_term = sum(b_k_N[k][j] * x_vars[j] for j in range(6))
            c_term = c_k_N[k] * tau
            Cxi = C @ samples[i]
            gamma_term = sum(gamma[(i, k)][m] * (d[m] - Cxi[m]) for m in range(4))
            model.addConstr(a_term + b_term + c_term + gamma_term <= s[i])
        
        # SLA-Tail情况下的约束
        for k in range(1, 12):
            a_term = a_k_T[k] @ samples[i]
            b_term = sum(b_k_T[k][j] * x_vars[j] for j in range(6))
            c_term = c_k_T[k] * tau
            Cxi = C @ samples[i]
            gamma_term = sum(gamma[(i, k+11)][m] * (d[m] - Cxi[m]) for m in range(4))
            model.addConstr(a_term + b_term + c_term + gamma_term <= s[i])
    
    # L1 范数对偶约束
    for i in range(N):
        # SLA-Normal情况下的约束
        for k in range(1, 12):
            C_T_gamma = C.T @ [gamma[(i, k)][m] for m in range(4)]
            diff = C_T_gamma - a_k_N[k]
            for m in range(len(diff)):
                model.addConstr(diff[m] <= lambda_var)
                model.addConstr(-diff[m] <= lambda_var)
        
        # SLA-Tail情况下的约束
        for k in range(1, 12):
            C_T_gamma = C.T @ [gamma[(i, k+11)][m] for m in range(4)]
            diff = C_T_gamma - a_k_T[k]
            for m in range(len(diff)):
                model.addConstr(diff[m] <= lambda_var)
                model.addConstr(-diff[m] <= lambda_var)
    
    # 其他约束
    model.addConstr(x_vars[2] + x_vars[3] <= x_vars[0], "fixed_capacity")
    model.addConstr(x_vars[4] + x_vars[5] <= x_vars[1], "flexible_capacity")
    
    # 求解
    model.optimize()
    
    # 记录结果
    if model.status == GRB.OPTIMAL:
        result = {
            'bar_xd': x_vars[0].X,
            'xe': x_vars[1].X,
            'xd1': x_vars[2].X,
            'xd2': x_vars[3].X,
            'xe1': x_vars[4].X,
            'xe2': x_vars[5].X,
            'objective': model.ObjVal,
            'epsilon': epsilon  # 记录使用的epsilon值
        }
        return result
    else:
        raise ValueError(f"DRO Model did not find optimal solution for N={N}")

# 求解 SAA 模型（样本平均近似）
def solve_saa_model(samples, c_d, c_e, h, rho, t1, t2, mu):
    N = len(samples)
    model = gp.Model("SAA_Model")
    
    # 设置求解器参数
    model.setParam("MIPGap", 0.01)
    model.setParam("TimeLimit", 60)
    model.setParam("Threads", 4)
    
    # 决策变量
    x_vars = model.addVars(6, lb=0, name="x")  # x = [bar_xd, xe, xd1, xd2, xe1, xe2]
    tau = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="tau")
    
    # 短缺变量
    shortage1 = model.addVars(N, lb=0, name="shortage1")
    shortage2 = model.addVars(N, lb=0, name="shortage2")
    total_shortage = model.addVars(N, lb=0, name="total_shortage")
    cvar_aux = model.addVars(N, lb=0, name="cvar_aux")
    
    # 约束：shortage >= D - x_d1 - x_e1
    for i in range(N):
        D1, D2 = samples[i]
        model.addConstr(shortage1[i] >= D1 - (x_vars[2] + x_vars[4]))
        model.addConstr(shortage2[i] >= D2 - (x_vars[3] + x_vars[5]))
        model.addConstr(total_shortage[i] >= D1 + D2 - (x_vars[0] + x_vars[1]))
        # CVaR约束
        model.addConstr(cvar_aux[i] >= D1 + D2 - (x_vars[2] + x_vars[3] + x_vars[4] + x_vars[5]) - tau)
    
    # 目标函数：基于样本的平均利润最大化（等价于成本最小化）
    profit = 0
    for i in range(N):
        D1, D2 = samples[i]
        profit += (base_params['p1'] * D1 + base_params['p2'] * D2) - \
                  (c_d * x_vars[0] + c_e * x_vars[1] + h * x_vars[0] + \
                   rho * (shortage1[i] + shortage2[i]) + t1 * x_vars[2] + t2 * x_vars[3])
    
    # 添加CVaR项到目标函数
    cvar_term = tau + (1/(mu * N)) * gp.quicksum(cvar_aux[i] for i in range(N))
    
    model.setObjective(profit / N - rho * cvar_term, GRB.MAXIMIZE)
    
    # 约束条件
    model.addConstr(x_vars[2] + x_vars[3] <= x_vars[0], "fixed_capacity")
    model.addConstr(x_vars[4] + x_vars[5] <= x_vars[1], "flexible_capacity")
    
    # 求解
    model.optimize()
    
    # 记录结果
    if model.status == GRB.OPTIMAL:
        result = {
            'bar_xd': x_vars[0].X,
            'xe': x_vars[1].X,
            'xd1': x_vars[2].X,
            'xd2': x_vars[3].X,
            'xe1': x_vars[4].X,
            'xe2': x_vars[5].X,
            'objective': model.ObjVal
        }
        return result
    else:
        raise ValueError(f"SAA Model did not find optimal solution for N={N}")

# 计算验证集下的利润
def compute_profit(samples, bar_xd, xe, c_d, c_e, h, rho, t1, t2):
    total_profit = 0
    for D in samples:
        D1, D2 = D
        x_d1 = min(bar_xd, D1)  # 固定产能满足需求
        x_e1 = min(xe, D1 - x_d1)  # 弹性产能满足剩余需求
        shortage1 = max(D1 - x_d1 - x_e1, 0)
        shortage2 = max(D2 - x_d1 - x_e1, 0)
        total_shortage = max(D1 + D2 - bar_xd - xe, 0)
        total_profit += (base_params['p1'] * D1 + base_params['p2'] * D2) - \
                        (c_d * bar_xd + c_e * xe + h * bar_xd + \
                         rho * (shortage1 + shortage2) + t1 * x_d1 + t2 * x_e1)
    return total_profit / len(samples)

# 运行样本数量对比
results = []
for N in N_values:
    sim_results = []
    for sim in range(num_simulations):
        # 生成样本
        samples = generate_samples(N)
        
        # 划分训练集和验证集（k=5 fold）
        kf = KFold(n_splits=5, shuffle=True, random_state=sim)
        fold_profits = []
        for train_index, val_index in kf.split(samples):
            train_samples = samples[train_index]
            val_samples = samples[val_index]
            
            # 计算epsilon
            epsilon = calculate_epsilon(len(train_samples))
            
            # 求解 DRO 模型
            dro_result = solve_dro_model(train_samples, fixed_params['c_d'], fixed_params['c_e'], 
                                       fixed_params['h'], fixed_params['rho'], fixed_params['t1'], 
                                       fixed_params['t2'], fixed_params['mu'], epsilon)
            
            # 求解 SAA 模型
            saa_result = solve_saa_model(train_samples, fixed_params['c_d'], fixed_params['c_e'], 
                                       fixed_params['h'], fixed_params['rho'], fixed_params['t1'], 
                                       fixed_params['t2'], fixed_params['mu'])
            
            # 计算验证集下的利润
            dro_profit = compute_profit(val_samples, dro_result['bar_xd'], dro_result['xe'], 
                                      fixed_params['c_d'], fixed_params['c_e'], fixed_params['h'], 
                                      fixed_params['rho'], fixed_params['t1'], fixed_params['t2'])
            saa_profit = compute_profit(val_samples, saa_result['bar_xd'], saa_result['xe'], 
                                      fixed_params['c_d'], fixed_params['c_e'], fixed_params['h'], 
                                      fixed_params['rho'], fixed_params['t1'], fixed_params['t2'])
            
            fold_profits.append({
                'dro_bar_xd': dro_result['bar_xd'],
                'dro_xe': dro_result['xe'],
                'dro_xd1': dro_result['xd1'],
                'dro_xd2': dro_result['xd2'],
                'dro_xe1': dro_result['xe1'],
                'dro_xe2': dro_result['xe2'],
                'dro_profit': dro_profit,
                'dro_objective': dro_result['objective'],
                'dro_epsilon': dro_result['epsilon'],
                'saa_bar_xd': saa_result['bar_xd'],
                'saa_xe': saa_result['xe'],
                'saa_profit': saa_profit,
                'saa_objective': saa_result['objective']
            })
        
        # 平均每个 fold 的结果
        avg_fold = {key: np.mean([f[key] for f in fold_profits]) for key in fold_profits[0]}
        sim_results.append(avg_fold)
    
    # 计算统计量
    df = pd.DataFrame(sim_results)
    results.append({
        'N': N,
        'avg_dro_bar_xd': df['dro_bar_xd'].mean(),
        'avg_dro_xe': df['dro_xe'].mean(),
        'avg_dro_xd1': df['dro_xd1'].mean(),
        'avg_dro_xd2': df['dro_xd2'].mean(),
        'avg_dro_xe1': df['dro_xe1'].mean(),
        'avg_dro_xe2': df['dro_xe2'].mean(),
        'avg_dro_profit': df['dro_profit'].mean(),
        'avg_dro_objective': df['dro_objective'].mean(),
        'avg_dro_epsilon': df['dro_epsilon'].mean(),
        'avg_saa_bar_xd': df['saa_bar_xd'].mean(),
        'avg_saa_xe': df['saa_xe'].mean(),
        'avg_saa_profit': df['saa_profit'].mean(),
        'avg_saa_objective': df['saa_objective'].mean()
    })

# 在保存结果之前，确保文件夹存在
os.makedirs('results/Pics', exist_ok=True)
os.makedirs('results/Data', exist_ok=True)

# 保存结果到 Excel
results_df = pd.DataFrame(results)          # results 是一个已经存在的数据列表或字典
# 创建 ExcelWriter 对象
with pd.ExcelWriter(os.path.join('results', 'Data', 'dro_vs_saa_comparison.xlsx'), mode='a') as writer:
    # 将 DataFrame 写入指定的工作表
    results_df.to_excel(writer, sheet_name='Sheet2', index=False)

# 可视化分析
plt.figure(figsize=(15, 20))

# 绘制 bar_xd
plt.subplot(6, 1, 1)
plt.plot(results_df['N'], results_df['avg_dro_bar_xd'], 
         marker='o', linestyle='-', color='blue', 
         label='DRO bar_xd')
plt.plot(results_df['N'], results_df['avg_saa_bar_xd'], 
         marker='s', linestyle='--', color='red', 
         label='SAA bar_xd')
plt.xlabel('Sample Size (N)', fontsize=12)
plt.ylabel('Average bar_xd', fontsize=12)
plt.title('Comparison of bar_xd: DRO vs SAA', fontsize=14)
plt.grid(True)
plt.legend()

# 绘制 xe
plt.subplot(6, 1, 2)
plt.plot(results_df['N'], results_df['avg_dro_xe'], 
         marker='o', linestyle='-', color='blue', 
         label='DRO xe')
plt.plot(results_df['N'], results_df['avg_saa_xe'], 
         marker='s', linestyle='--', color='red', 
         label='SAA xe')
plt.xlabel('Sample Size (N)', fontsize=12)
plt.ylabel('Average xe', fontsize=12)
plt.title('Comparison of xe: DRO vs SAA', fontsize=14)
plt.grid(True)
plt.legend()

# 绘制 profit
plt.subplot(6, 1, 3)
plt.plot(results_df['N'], results_df['avg_dro_profit'], 
         marker='o', linestyle='-', color='blue', 
         label='DRO Profit')
plt.plot(results_df['N'], results_df['avg_saa_profit'], 
         marker='s', linestyle='--', color='red', 
         label='SAA Profit')
plt.xlabel('Sample Size (N)', fontsize=12)
plt.ylabel('Average Profit', fontsize=12)
plt.title('Comparison of Profit: DRO vs SAA', fontsize=14)
plt.grid(True)
plt.legend()

# 绘制最坏情况目标函数值
plt.subplot(6, 1, 4)
plt.plot(results_df['N'], results_df['avg_dro_objective'], 
         marker='o', linestyle='-', color='blue', 
         label='DRO Worst-case Objective ($\\widehat{J}_N$)')
plt.plot(results_df['N'], -results_df['avg_saa_objective'], 
         marker='s', linestyle='--', color='red', 
         label='SAA Expected Objective (Negative Profit)')
plt.xlabel('Sample Size (N)', fontsize=12)
plt.ylabel('Objective Value', fontsize=12)
plt.title('Comparison of Worst-case Objective: DRO vs SAA', fontsize=14)
plt.grid(True)
plt.legend()

# 绘制 epsilon 随 N 的变化
plt.subplot(6, 1, 5)
plt.plot(results_df['N'], results_df['avg_dro_epsilon'], 
         marker='o', linestyle='-', color='green', 
         label='Wasserstein Radius ($\\epsilon$)')
plt.xlabel('Sample Size (N)', fontsize=12)
plt.ylabel('$\\epsilon$', fontsize=12)
plt.title('Wasserstein Radius vs Sample Size', fontsize=14)
plt.grid(True)
plt.legend()

# 绘制产能使用情况对比
plt.subplot(6, 1, 6)
plt.plot(results_df['N'], results_df['avg_dro_xd1'], 
         marker='o', linestyle='-', color='blue', 
         label='DRO x_d^1 (Fixed Capacity Channel 1)')
plt.plot(results_df['N'], results_df['avg_dro_xd2'], 
         marker='s', linestyle='-', color='red', 
         label='DRO x_d^2 (Fixed Capacity Channel 2)')
plt.plot(results_df['N'], results_df['avg_dro_xe1'], 
         marker='^', linestyle='--', color='green', 
         label='DRO x_e^1 (Flexible Capacity Channel 1)')
plt.plot(results_df['N'], results_df['avg_dro_xe2'], 
         marker='v', linestyle='--', color='purple', 
         label='DRO x_e^2 (Flexible Capacity Channel 2)')
plt.xlabel('Sample Size (N)', fontsize=12)
plt.ylabel('Capacity Allocation', fontsize=12)
plt.title('Capacity Allocation Comparison Across Channels', fontsize=14)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join('results', 'Pics', 'dro_vs_saa_worst_expectation_analysis.png'), dpi=300)
plt.show()