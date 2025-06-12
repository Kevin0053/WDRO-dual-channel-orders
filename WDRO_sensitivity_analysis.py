"""
本程序用于进行WDRO模型的参数灵敏度分析。
主要分析以下参数对最优决策的影响：
1. 固定产能成本 (c_d, c_d1, c_d2)
2. 弹性产能成本 (c_e, c_e1, c_e2)
3. 持有成本 (h)
4. SLA惩罚权重 (rho1, rho2)
5. 违约惩罚成本 (t1, t2)
6. SLA置信水平 (mu1, mu2)
"""

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import os

# 创建必要的文件夹
output_data_dir = 'results/Data/32_sensitivity_analysis'
output_pics_dir = 'results/Pics/Sens/32_sensitivity_analysis'
os.makedirs(output_data_dir, exist_ok=True)
os.makedirs(output_pics_dir, exist_ok=True)

# 基础参数设置
base_params = {
    'p1': 6.0,    # 平台订单单位价格
    'p2': 7.0,    # 线下订单单位价格
    'bar_D1': 100.0,  # 平台最大需求
    'bar_D2': 80.0    # 线下最大需求
}

# 基准参数设置
base_fixed_params = {
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

# 参数变化范围设置
param_ranges = {
    'c_d': np.linspace(0.5, 3.0, 10),
    'c_e': np.linspace(0.5, 3.0, 10),
    'h': np.linspace(0.1, 1.0, 10),
    'rho1': np.linspace(0.5, 4.0, 10),
    'rho2': np.linspace(0.5, 4.0, 10),
    't1': np.linspace(1.0, 10.0, 10),
    't2': np.linspace(1.0, 10.0, 10),
    'mu1': np.linspace(0.01, 0.2, 10),
    'mu2': np.linspace(0.01, 0.2, 10)
}

def get_affine_coefficients(fixed_params):
    """定义仿射函数系数"""
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

def solve_dro_model(samples, epsilon, fixed_params):
    """求解 DRO 模型"""
    N = len(samples)
    model = gp.Model("DRO_Model_Sens")
    
    # 设置求解器参数
    model.setParam("MIPGap", 0.01)
    model.setParam("TimeLimit", 120)
    model.setParam("Threads", 4)
    model.setParam("OutputFlag", 0) # 关闭冗长输出
    
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
    a_k, b_k, c_k = get_affine_coefficients(fixed_params)
    
    # 目标函数
    obj = lambda_var * epsilon + (1/N) * gp.quicksum(s[i] for i in range(N)) + \
          fixed_params['c_d'] * x_vars[0] + fixed_params['c_e'] * x_vars[1]
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
            model.addConstr(diff[0] <= lambda_var)
            model.addConstr(-diff[0] <= lambda_var)
            model.addConstr(diff[1] <= lambda_var)
            model.addConstr(-diff[1] <= lambda_var)
    
    # 其他约束
    model.addConstr(x_vars[2] + x_vars[3] <= x_vars[0], "fixed_capacity")
    model.addConstr(x_vars[4] + x_vars[5] <= x_vars[1], "flexible_capacity")
    
    # 求解
    model.optimize()
    
    # 记录结果
    if model.status == GRB.OPTIMAL:
        worst_case_loss = model.ObjVal - (fixed_params['c_d'] * x_vars[0].X + fixed_params['c_e'] * x_vars[1].X)
        result = {
            'bar_xd': x_vars[0].X,
            'xe': x_vars[1].X,
            'xd1': x_vars[2].X,
            'xd2': x_vars[3].X,
            'xe1': x_vars[4].X,
            'xe2': x_vars[5].X,
            'profit': worst_case_loss,
            'total_cost_objective': model.ObjVal
        }
        return result
    else:
        # 如果模型无解或未达到最优，可以返回一个标记，例如None或者包含错误信息的字典
        return None

# 支撑集约束
C = np.array([
    [1, 0],   # D1 <= bar_D1
    [0, 1],   # D2 <= bar_D2
    [-1, 0],  # -D1 <= 0 (D1 >= 0)
    [0, -1]   # -D2 <= 0 (D2 >= 0)
])

# 最大需求值
d = np.array([base_params['bar_D1'], base_params['bar_D2'], 0.0, 0.0])

def run_sensitivity_analysis(param_name, param_values):
    """运行单个参数的灵敏度分析"""
    results = []
    N = 100  # 固定样本数量
    epsilon = 1 / np.sqrt(N)
    
    for value in param_values:
        # 复制基准参数
        current_params = base_fixed_params.copy()
        current_params[param_name] = value
        
        # 生成样本并求解模型
        samples = generate_samples(N)
        try:
            result = solve_dro_model(samples, epsilon, current_params)
            if result:
                result[param_name] = value
                results.append(result)
        except Exception as e:
            print(f"Error in solving for {param_name}={value}: {str(e)}")
            continue
    
    return pd.DataFrame(results)

def plot_sensitivity_results(df, param_name, save_path):
    """绘制灵敏度分析结果"""
    if df.empty:
        print(f"No data to plot for {param_name}.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))
    fig.suptitle(f'Sensitivity Analysis for {param_name}', fontsize=16)

    # 绘制利润
    axes[0].plot(df[param_name], df['profit'], 'b-o', label='Worst-case Profit')
    axes[0].set_title(f'Worst-case Profit vs {param_name}')
    axes[0].set_xlabel(param_name)
    axes[0].set_ylabel('Worst-case Guaranteed Profit')
    axes[0].grid(True)
    axes[0].legend()
    
    # 绘制总产能分配
    axes[1].plot(df[param_name], df['bar_xd'], 'r-s', label='Total Fixed Capacity (bar_xd)')
    axes[1].plot(df[param_name], df['xe'], 'g-^', label='Total Flexible Capacity (xe)')
    axes[1].set_title(f'Total Capacity vs {param_name}')
    axes[1].set_xlabel(param_name)
    axes[1].set_ylabel('Capacity')
    axes[1].grid(True)
    axes[1].legend()

    # 绘制渠道产能分配
    axes[2].plot(df[param_name], df['xd1'], 'o--', label='Fixed Cap Ch1 (xd1)')
    axes[2].plot(df[param_name], df['xd2'], 's--', label='Fixed Cap Ch2 (xd2)')
    axes[2].plot(df[param_name], df['xe1'], 'o:', label='Flex Cap Ch1 (xe1)')
    axes[2].plot(df[param_name], df['xe2'], 's:', label='Flex Cap Ch2 (xe2)')
    axes[2].set_title(f'Capacity Allocation vs {param_name}')
    axes[2].set_xlabel(param_name)
    axes[2].set_ylabel('Capacity')
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    # 存储所有结果
    all_results = {}
    
    # 对每个参数进行灵敏度分析
    for param_name, param_values in param_ranges.items():
        print(f"\nAnalyzing sensitivity to {param_name}...")
        
        # 运行分析
        results_df = run_sensitivity_analysis(param_name, param_values)
        
        if not results_df.empty:
            # 保存结果
            all_results[param_name] = results_df
            
            # 绘制单个参数的灵敏度分析图
            save_path = os.path.join(output_pics_dir, f'32_sensitivity_{param_name}.png')
            plot_sensitivity_results(results_df, param_name, save_path)
    
    # 保存结果到Excel，每个参数一个sheet
    excel_path = os.path.join(output_data_dir, '32_sensitivity_analysis_results.xlsx')
    with pd.ExcelWriter(excel_path) as writer:
        for param_name, results_df in all_results.items():
            results_df.to_excel(writer, sheet_name=f'Sens_{param_name}', index=False)
    
    print("\nSensitivity analysis completed.")
    print(f"Results saved in '{excel_path}'")
    print(f"Plots saved in '{output_pics_dir}/'")

if __name__ == "__main__":
    main() 