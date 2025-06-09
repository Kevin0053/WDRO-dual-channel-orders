"""
本程序用于进行WDRO模型的参数灵敏度分析。
主要分析以下参数对最优决策的影响：
1. 固定产能成本 (c_d, c_d1, c_d2)
2. 弹性产能成本 (c_e, c_e1, c_e2)
3. 持有成本 (h)
4. SLA惩罚权重 (rho)
5. 违约惩罚成本 (t1, t2)
6. SLA置信水平 (mu)
"""

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import os

# 创建必要的文件夹
os.makedirs('results/Data', exist_ok=True)
os.makedirs('results/Pics/Sens', exist_ok=True)

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
    'c_e': 1.1,   # 总弹性产能单位成本
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

# 参数变化范围设置
param_ranges = {
    'c_d': np.linspace(0.5, 2.0, 20),    # 固定产能成本范围
    'c_e': np.linspace(0.5, 2.0, 20),    # 弹性产能成本范围
    'c_d1': np.linspace(0.5, 2.0, 20),   # 渠道1固定产能成本范围
    'c_d2': np.linspace(0.5, 2.0, 20),   # 渠道2固定产能成本范围
    'c_e1': np.linspace(0.5, 2.0, 20),   # 渠道1弹性产能成本范围
    'c_e2': np.linspace(0.5, 2.0, 20),   # 渠道2弹性产能成本范围
    'h': np.linspace(0.1, 0.5, 20),      # 持有成本范围
    'rho': np.linspace(1.0, 5.0, 20),    # SLA惩罚权重范围
    't1': np.linspace(5.0, 15.0, 20),    # 渠道1违约惩罚成本范围
    't2': np.linspace(5.0, 15.0, 20),    # 渠道2违约惩罚成本范围
    'mu': np.linspace(0.01, 0.2, 20)     # SLA置信水平范围
}

def get_affine_coefficients(fixed_params):
    """定义仿射函数系数"""
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
        1: np.array([0.0, 0.0, fixed_params['c_d1'], fixed_params['c_d2'], fixed_params['c_e1'], fixed_params['c_e2']]),
        2: np.array([fixed_params['h'], 0.0, fixed_params['c_d1']-fixed_params['h'], fixed_params['c_d2']-fixed_params['h'], fixed_params['c_e1'], fixed_params['c_e2']]),
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
    a_k_N, b_k_N, c_k_N, a_k_T, b_k_T, c_k_T = get_affine_coefficients(fixed_params)
    
    # 目标函数
    obj = lambda_var * epsilon + (1/N) * gp.quicksum(s[i] for i in range(N)) + \
          fixed_params['c_d'] * x_vars[0] + fixed_params['c_e'] * x_vars[1]
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
            'epsilon': epsilon
        }
        return result
    else:
        raise ValueError(f"DRO Model did not find optimal solution for N={N}")

# 支撑集约束
C = np.array([
    [1, 0],   # D1 >= 0
    [0, 1],   # D2 >= 0
    [-1, 0],  # D1 <= bar_D1
    [0, -1]   # D2 <= bar_D2
])

# 最大需求值
d = np.array([base_params['bar_D1'], base_params['bar_D2'], 0.0, 0.0])

def run_sensitivity_analysis(param_name, param_values):
    """运行单个参数的灵敏度分析"""
    results = []
    N = 200  # 固定样本数量
    epsilon = 1 / np.sqrt(N)
    
    for value in param_values:
        # 复制基准参数
        current_params = base_fixed_params.copy()
        current_params[param_name] = value
        
        # 生成样本并求解模型
        samples = generate_samples(N)
        try:
            result = solve_dro_model(samples, epsilon, current_params)
            result[param_name] = value
            
            # 添加其他参数的基准值
            for other_param, base_value in base_fixed_params.items():
                if other_param != param_name:
                    result[f'base_{other_param}'] = base_value
            
            results.append(result)
        except Exception as e:
            print(f"Error in solving for {param_name}={value}: {str(e)}")
            continue
    
    return pd.DataFrame(results)

def plot_sensitivity_results(df, param_name, save_path):
    """绘制灵敏度分析结果"""
    plt.figure(figsize=(10, 6))
    
    # 绘制利润
    plt.subplot(2, 1, 1)
    plt.plot(df[param_name], df['objective'], 'b-', label='Profit')
    plt.title(f'Profit vs {param_name}')
    plt.xlabel(param_name)
    plt.ylabel('Profit')
    plt.grid(True)
    plt.legend()
    
    # 绘制产能分配
    plt.subplot(2, 1, 2)
    plt.plot(df[param_name], df['bar_xd'], 'r-', label='Fixed Capacity')
    plt.plot(df[param_name], df['xe'], 'g-', label='Flexible Capacity')
    plt.title(f'Capacity Allocation vs {param_name}')
    plt.xlabel(param_name)
    plt.ylabel('Capacity')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_comparison_chart(all_results, param_names, save_path):
    """绘制不同参数对利润影响的对比图"""
    plt.figure(figsize=(12, 8))
    
    for param_name in param_names:
        df = all_results[param_name]
        plt.plot(df[param_name], df['objective'], label=param_name)
    
    plt.title('Profit Comparison Across Different Parameters')
    plt.xlabel('Parameter Value')
    plt.ylabel('Profit')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    # 存储所有结果
    all_results = {}
    summary_data = []
    
    # 对每个参数进行灵敏度分析
    for param_name, param_values in param_ranges.items():
        print(f"\nAnalyzing sensitivity to {param_name}...")
        
        # 运行分析
        results_df = run_sensitivity_analysis(param_name, param_values)
        
        # 保存结果
        all_results[param_name] = results_df
        
        # 计算平均利润
        avg_profit = results_df['objective'].mean()
        summary_data.append({
            'Parameter': param_name,
            'Average_Profit': avg_profit,
            'Min_Profit': results_df['objective'].min(),
            'Max_Profit': results_df['objective'].max(),
            'Base_Value': base_fixed_params[param_name]
        })
        
        # 绘制单个参数的灵敏度分析图
        save_path = os.path.join('results', 'Pics', 'Sens', f'sensitivity_{param_name}.png')
        plot_sensitivity_results(results_df, param_name, save_path)
    
    # 绘制对比图
    # 1. 固定产能成本对比
    fixed_capacity_params = ['c_d', 'c_d1', 'c_d2']
    save_path = os.path.join('results', 'Pics', 'Sens', 'comparison_fixed_capacity.png')
    plot_comparison_chart(all_results, fixed_capacity_params, save_path)
    
    # 2. 弹性产能成本对比
    flexible_capacity_params = ['c_e', 'c_e1', 'c_e2']
    save_path = os.path.join('results', 'Pics', 'Sens', 'comparison_flexible_capacity.png')
    plot_comparison_chart(all_results, flexible_capacity_params, save_path)
    
    # 保存结果到Excel，每个参数一个sheet
    with pd.ExcelWriter(os.path.join('results', 'Data', 'sensitivity_analysis_results.xlsx')) as writer:
        # 保存汇总数据
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # 保存每个参数的详细数据
        for param_name, results_df in all_results.items():
            # 重新排列列，使基准参数值更容易查看
            cols = [param_name, 'objective', 'bar_xd', 'xe', 'xd1', 'xd2', 'xe1', 'xe2', 'epsilon']
            base_cols = [col for col in results_df.columns if col.startswith('base_')]
            other_cols = [col for col in results_df.columns if col not in cols + base_cols]
            
            # 重新排列列顺序
            ordered_cols = cols + base_cols + other_cols
            results_df = results_df[ordered_cols]
            
            # 保存到Excel
            results_df.to_excel(writer, sheet_name=f'Sens_{param_name}', index=False)
    
    print("\nSensitivity analysis completed.")
    print("Results saved in 'results/Data/sensitivity_analysis_results.xlsx'")
    print("Plots saved in 'results/Pics/Sens/")

if __name__ == "__main__":
    main() 