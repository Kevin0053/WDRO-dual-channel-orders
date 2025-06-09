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
    'c_e':1.1,   # 总弹性产能单位成本
    'c_d1': 1.0,  # 渠道1固定产能单位成本
    'c_d2': 1.0,  # 渠道2固定产能单位成本
    'c_e1': 1.0,  # 渠道1弹性产能单位成本
    'c_e2': 1.0,  # 渠道2弹性产能单位成本
    'h': 0.2,     # 固定产能持有成本
    'rho': 2.0,   # SLA惩罚权重
    't1': 10.0,   # 渠道1需求违约惩罚成本
    't2': 10.5,   # 渠道2需求违约惩罚成本
    'mu': 0.05    # SLA置信水平参数
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
    # 定义SLA参数
    mu_1 = 0.05  # 渠道1置信水平参数
    mu_2 = 0.05  # 渠道2置信水平参数
    rho_1 = 2.0  # 渠道1SLA惩罚权重
    rho_2 = 2.0  # 渠道2SLA惩罚权重

    # 第一类：SLA-Normal (ℓ_k^(1) = f_k + g_{1,1} + g_{2,1})
    a_k_N = {
        1: np.array([-base_params['p1'], -base_params['p2']]),
        2: np.array([-base_params['p1'], -base_params['p2']]),
        3: np.array([-base_params['p1'], fixed_params['t2']]),
        4: np.array([-base_params['p1'], fixed_params['t2']]),
        5: np.array([fixed_params['t1'], -base_params['p2']]),
        6: np.array([fixed_params['t1'], -base_params['p2']]),
        7: np.array([fixed_params['t1'], fixed_params['t2']]),
        8: np.array([fixed_params['t1'], fixed_params['t2']])
    }

    b_k_N = {
        1: np.array([0.0, 0.0, fixed_params['c_d1'], fixed_params['c_d2'], fixed_params['c_e1'], fixed_params['c_e2']]),
        2: np.array([fixed_params['h'], 0.0, fixed_params['c_d1'], fixed_params['c_d2'], fixed_params['c_e1'], fixed_params['c_e2']]),
        3: np.array([0.0, 0.0, fixed_params['c_d1'], fixed_params['c_d2']-base_params['p2']+fixed_params['t2'], fixed_params['c_e1'], fixed_params['c_e2']-base_params['p2']+fixed_params['t2']]),
        4: np.array([fixed_params['h'], 0.0, fixed_params['c_d1'], fixed_params['c_d2']-base_params['p2']+fixed_params['t2'], fixed_params['c_e1'], fixed_params['c_e2']-base_params['p2']+fixed_params['t2']]),
        5: np.array([0.0, 0.0, fixed_params['c_d1']-base_params['p1']+fixed_params['t1'], fixed_params['c_d2'], fixed_params['c_e1']-base_params['p1']+fixed_params['t1'], fixed_params['c_e2']]),
        6: np.array([fixed_params['h'], 0.0, fixed_params['c_d1']-base_params['p1']+fixed_params['t1'], fixed_params['c_d2'], fixed_params['c_e1']-base_params['p1']+fixed_params['t1'], fixed_params['c_e2']]),
        7: np.array([0.0, 0.0, fixed_params['c_d1']-base_params['p1']+fixed_params['t1'], fixed_params['c_d2']-base_params['p2']+fixed_params['t2'], fixed_params['c_e1']-base_params['p1']+fixed_params['t1'], fixed_params['c_e2']-base_params['p2']+fixed_params['t2']]),
        8: np.array([fixed_params['h'], 0.0, fixed_params['c_d1']-base_params['p1']+fixed_params['t1'], fixed_params['c_d2']-base_params['p2']+fixed_params['t2'], fixed_params['c_e1']-base_params['p1']+fixed_params['t1'], fixed_params['c_e2']-base_params['p2']+fixed_params['t2']])
    }

    c_k_N = {k: np.array([rho_1, rho_2]) for k in range(1, 9)}

    # 第二类：SLA-Tail (ℓ_k^(2) = f_k + g_{1,2} + g_{2,1})
    a_k_T1 = {
        k: a_k_N[k] + np.array([rho_1/mu_1, 0.0])
        for k in range(1, 9)
    }

    b_k_T1 = {
        k: b_k_N[k] - np.array([0.0, 0.0, rho_1/mu_1, 0.0, rho_1/mu_1, 0.0])
        for k in range(1, 9)
    }

    c_k_T1 = {
        k: np.array([rho_1*(1-1/mu_1), rho_2])
        for k in range(1, 9)
    }

    # 第三类：SLA-Tail (ℓ_k^(3) = f_k + g_{1,1} + g_{2,2})
    a_k_T2 = {
        k: a_k_N[k] + np.array([0.0, rho_2/mu_2])
        for k in range(1, 9)
    }

    b_k_T2 = {
        k: b_k_N[k] - np.array([0.0, 0.0, 0.0, rho_2/mu_2, 0.0, rho_2/mu_2])
        for k in range(1, 9)
    }

    c_k_T2 = {
        k: np.array([rho_1, rho_2*(1-1/mu_2)])
        for k in range(1, 9)
    }

    # 第四类：SLA-Tail (ℓ_k^(4) = f_k + g_{1,2} + g_{2,2})
    a_k_T3 = {
        k: a_k_N[k] + np.array([rho_1/mu_1, rho_2/mu_2])
        for k in range(1, 9)
    }

    b_k_T3 = {
        k: b_k_N[k] - np.array([0.0, 0.0, rho_1/mu_1, rho_2/mu_2, rho_1/mu_1, rho_2/mu_2])
        for k in range(1, 9)
    }

    c_k_T3 = {
        k: np.array([rho_1*(1-1/mu_1), rho_2*(1-1/mu_2)])
        for k in range(1, 9)
    }

    return a_k_N, b_k_N, c_k_N, a_k_T1, b_k_T1, c_k_T1, a_k_T2, b_k_T2, c_k_T2, a_k_T3, b_k_T3, c_k_T3

# 支撑集约束
C = np.array([
    [1, 0],   # D1 >= 0
    [0, 1],   # D2 >= 0
    [-1, 0],  # D1 <= bar_D1
    [0, -1]   # D2 <= bar_D2
])

# 最大需求值
d = np.array([base_params['bar_D1'], base_params['bar_D2'], 0.0, 0.0])

def generate_samples(N):
    """生成正态分布的需求样本，并截断到合理范围"""
    D1 = np.random.normal(loc=base_params['bar_D1'] * 0.5, scale=10, size=N)
    D2 = np.random.normal(loc=base_params['bar_D2'] * 0.5, scale=10, size=N)
    
    # 截断到非负且不超过最大需求
    D1 = np.clip(D1, 0, base_params['bar_D1'])
    D2 = np.clip(D2, 0, base_params['bar_D2'])
    
    return np.column_stack((D1, D2))

def solve_dro_model(samples, epsilon):
    """求解 DRO 模型"""
    N = len(samples)
    model = gp.Model("DRO_Model")
    
    # 设置求解器参数
    model.setParam("MIPGap", 0.01)
    model.setParam("TimeLimit", 60)
    model.setParam("Threads", 4)
    
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
    a_k_N, b_k_N, c_k_N, a_k_T1, b_k_T1, c_k_T1, a_k_T2, b_k_T2, c_k_T2, a_k_T3, b_k_T3, c_k_T3 = get_affine_coefficients()
    
    # 目标函数
    obj = lambda_var * epsilon + (1/N) * gp.quicksum(s[i] for i in range(N)) + \
          fixed_params['c_d'] * x_vars[0] + fixed_params['c_e'] * x_vars[1]
    model.setObjective(obj, GRB.MINIMIZE)
    
    # 约束条件
    for i in range(N):
        # 第一类：SLA-Normal (ℓ_k^(1))
        for k in range(1, 9):
            a_term = a_k_N[k] @ samples[i]
            b_term = sum(b_k_N[k][j] * x_vars[j] for j in range(6))
            c_term = c_k_N[k] @ tau
            Cxi = C @ samples[i]
            gamma_term = sum(gamma[(i, k)][m] * (d[m] - Cxi[m]) for m in range(4))
            model.addConstr(a_term + b_term + c_term + gamma_term <= s[i])
        
        # 第二类：SLA-Tail (ℓ_k^(2))
        for k in range(1, 9):
            a_term = a_k_T1[k] @ samples[i]
            b_term = sum(b_k_T1[k][j] * x_vars[j] for j in range(6))
            c_term = c_k_T1[k] @ tau
            Cxi = C @ samples[i]
            gamma_term = sum(gamma[(i, k+8)][m] * (d[m] - Cxi[m]) for m in range(4))
            model.addConstr(a_term + b_term + c_term + gamma_term <= s[i])
            
        # 第三类：SLA-Tail (ℓ_k^(3))
        for k in range(1, 9):
            a_term = a_k_T2[k] @ samples[i]
            b_term = sum(b_k_T2[k][j] * x_vars[j] for j in range(6))
            c_term = c_k_T2[k] @ tau
            Cxi = C @ samples[i]
            gamma_term = sum(gamma[(i, k+16)][m] * (d[m] - Cxi[m]) for m in range(4))
            model.addConstr(a_term + b_term + c_term + gamma_term <= s[i])
            
        # 第四类：SLA-Tail (ℓ_k^(4))
        for k in range(1, 9):
            a_term = a_k_T3[k] @ samples[i]
            b_term = sum(b_k_T3[k][j] * x_vars[j] for j in range(6))
            c_term = c_k_T3[k] @ tau
            Cxi = C @ samples[i]
            gamma_term = sum(gamma[(i, k+24)][m] * (d[m] - Cxi[m]) for m in range(4))
            model.addConstr(a_term + b_term + c_term + gamma_term <= s[i])
    
    # L1 范数对偶约束
    for i in range(N):
        # 第一类：SLA-Normal (ℓ_k^(1))
        for k in range(1, 9):
            C_T_gamma = C.T @ [gamma[(i, k)][m] for m in range(4)]
            diff = C_T_gamma - a_k_N[k]
            for m in range(len(diff)):
                model.addConstr(diff[m] <= lambda_var)
                model.addConstr(-diff[m] <= lambda_var)
        
        # 第二类：SLA-Tail (ℓ_k^(2))
        for k in range(1, 9):
            C_T_gamma = C.T @ [gamma[(i, k+8)][m] for m in range(4)]
            diff = C_T_gamma - a_k_T1[k]
            for m in range(len(diff)):
                model.addConstr(diff[m] <= lambda_var)
                model.addConstr(-diff[m] <= lambda_var)
                
        # 第三类：SLA-Tail (ℓ_k^(3))
        for k in range(1, 9):
            C_T_gamma = C.T @ [gamma[(i, k+16)][m] for m in range(4)]
            diff = C_T_gamma - a_k_T2[k]
            for m in range(len(diff)):
                model.addConstr(diff[m] <= lambda_var)
                model.addConstr(-diff[m] <= lambda_var)
                
        # 第四类：SLA-Tail (ℓ_k^(4))
        for k in range(1, 9):
            C_T_gamma = C.T @ [gamma[(i, k+24)][m] for m in range(4)]
            diff = C_T_gamma - a_k_T3[k]
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
            'epsilon': epsilon
        }
        return result
    else:
        raise ValueError(f"DRO Model did not find optimal solution for N={N}")

# 在保存结果之前，确保文件夹存在
os.makedirs('results/Pics', exist_ok=True)
os.makedirs('results/Data', exist_ok=True)

# 运行不同样本规模的仿真
results = []
for N in N_values:
    print(f"\n处理样本规模 N = {N}")
    epsilon_current = 1 / np.sqrt(N)  # 动态计算 ε
    
    # 对每个样本规模进行多次模拟
    for sim in range(num_simulations):
        print(f"  模拟 {sim+1}/{num_simulations}")
        samples = generate_samples(N)
        result = solve_dro_model(samples, epsilon_current)
        result['N'] = N
        result['simulation'] = sim+1
        results.append(result)

# 数据处理和可视化
results_df = pd.DataFrame(results)

# 计算每个样本规模的平均值和标准差
summary = results_df.groupby('N').agg({
    'objective': ['mean', 'std'],
    'bar_xd': ['mean', 'std'],
    'xe': ['mean', 'std'],
    'xd1': ['mean', 'std'],
    'xd2': ['mean', 'std'],
    'xe1': ['mean', 'std'],
    'xe2': ['mean', 'std'],
    'epsilon': 'mean'
}).reset_index()

# 打印汇总统计
print("\n=== 不同样本规模下的结果统计 ===")
for _, row in summary.iterrows():
    print(f"\n样本规模 N = {row['N']}")
    print(f"平均利润: {row[('objective', 'mean')]:.2f} (±{row[('objective', 'std')]:.2f})")
    print(f"平均固定产能: {row[('bar_xd', 'mean')]:.2f} (±{row[('bar_xd', 'std')]:.2f})")
    print(f"平均弹性产能: {row[('xe', 'mean')]:.2f} (±{row[('xe', 'std')]:.2f})")
    print(f"Epsilon: {row[('epsilon', 'mean')]:.4f}")


# 保存结果到 Excel
results_df = pd.DataFrame(results)          # results 是一个已经存在的数据列表或字典
# 创建 ExcelWriter 对象
with pd.ExcelWriter(os.path.join('results', 'Data', 'modified_dro_simulation_results.xlsx'), mode='a') as writer:
    # 将 DataFrame 写入指定的工作表
    results_df.to_excel(writer, sheet_name='Sheet5', index=False)

# 📊 Visualization
plt.figure(figsize=(15, 20))

# --- Subplot 1: 利润随样本规模的变化 ---
plt.subplot(5, 1, 1)
plt.errorbar(summary['N'], summary[('objective', 'mean')], 
             yerr=summary[('objective', 'std')], 
             fmt='o-', color='blue', alpha=0.7)
plt.title('Profit vs Sample Size')
plt.xlabel('Sample Size (N)')
plt.ylabel('Profit')
plt.grid(True)

# --- Subplot 2: 固定产能随样本规模的变化 ---
plt.subplot(5, 1, 2)
plt.errorbar(summary['N'], summary[('bar_xd', 'mean')], 
             yerr=summary[('bar_xd', 'std')], 
             fmt='o-', color='red', alpha=0.7, label='Fixed Capacity')
plt.title('Fixed Capacity vs Sample Size')
plt.xlabel('Sample Size (N)')
plt.ylabel('Fixed Capacity')
plt.grid(True)
plt.legend()

# --- Subplot 3: 弹性产能随样本规模的变化 ---
plt.subplot(5, 1, 3)
plt.errorbar(summary['N'], summary[('xe', 'mean')], 
             yerr=summary[('xe', 'std')], 
             fmt='o-', color='green', alpha=0.7, label='Flexible Capacity')
plt.title('Flexible Capacity vs Sample Size')
plt.xlabel('Sample Size (N)')
plt.ylabel('Flexible Capacity')
plt.grid(True)
plt.legend()

# --- Subplot 4: 渠道1产能分配随样本规模的变化 ---
plt.subplot(5, 1, 4)
plt.errorbar(summary['N'], summary[('xd1', 'mean')], 
             yerr=summary[('xd1', 'std')], 
             fmt='o-', color='blue', alpha=0.7, label='Fixed Capacity (Channel 1)')
plt.errorbar(summary['N'], summary[('xe1', 'mean')], 
             yerr=summary[('xe1', 'std')], 
             fmt='s-', color='red', alpha=0.7, label='Flexible Capacity (Channel 1)')
plt.title('Channel 1 Capacity Allocation vs Sample Size')
plt.xlabel('Sample Size (N)')
plt.ylabel('Capacity')
plt.grid(True)
plt.legend()

# --- Subplot 5: 渠道2产能分配随样本规模的变化 ---
plt.subplot(5, 1, 5)
plt.errorbar(summary['N'], summary[('xd2', 'mean')], 
             yerr=summary[('xd2', 'std')], 
             fmt='o-', color='blue', alpha=0.7, label='Fixed Capacity (Channel 2)')
plt.errorbar(summary['N'], summary[('xe2', 'mean'), 
             yerr=summary[('xe2', 'std')], 
             fmt='s-', color='red', alpha=0.7, label='Flexible Capacity (Channel 2)')
plt.title('Channel 2 Capacity Allocation vs Sample Size')
plt.xlabel('Sample Size (N)')
plt.ylabel('Capacity')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join('results', 'Pics', 'modified_dro_simulation_plot.png'), dpi=300)
plt.show()