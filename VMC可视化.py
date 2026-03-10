import numpy as np
import matplotlib.pyplot as plt
import time

# ---------- simulation parameters ----------

n_walkers = 200
n_steps = 500
step_size = 1.0

alpha = 0.8
learning_rate = 0.05

dim = 3

grad_tol = 1e-3
alpha_tol = 1e-4

max_iter = 60  # 保险上限防止死循环

# ---------- 基础函数 ----------

def psi(r, alpha):
    r_norm = np.linalg.norm(r)
    return np.exp(-alpha * r_norm)

def local_energy(r, alpha):
    r_norm = np.linalg.norm(r)
    return -0.5 * alpha**2 + (alpha - 1.0) / r_norm

# ---------- 定义两种不同的随机行走策略 ----------

def uniform_proposal(r, step_size, dim):
    """均匀分布提议"""
    return r + step_size * (np.random.uniform(-0.5, 0.5, size=dim))

def normal_proposal(r, step_size, dim):
    """正态分布提议"""
    return r + step_size * np.random.normal(0, 1, size=dim)

# ---------- 运行VMC优化的函数 ----------

def run_vmc(proposal_func, proposal_name, walkers_initial):
    """
    运行VMC优化
    
    参数:
    proposal_func: 提议分布函数
    proposal_name: 提议分布名称
    walkers_initial: 初始walker位置
    
    返回:
    history: 包含优化历史的字典
    total_time: 总运行时间
    """
    
    # 复制初始walker，避免影响另一个模拟
    walkers = walkers_initial.copy()
    alpha_current = alpha
    
    history = {
        'iterations': [],
        'energy': [],
        'energy_error': [],
        'alpha': [],
        'grad': [],
        'accept_ratio': [],
        'time': []  # 记录每次迭代的时间
    }
    
    prev_energy = None
    start_time_total = time.time()
    
    for iteration in range(max_iter):
        iter_start_time = time.time()
        
        E_list = []
        O_list = []
        accepted = 0
        
        for step in range(n_steps):
            for i in range(n_walkers):
                r_old = walkers[i]
                
                # 使用指定的提议分布
                proposal = proposal_func(r_old, step_size, dim)
                
                p_old = psi(r_old, alpha_current)**2
                p_new = psi(proposal, alpha_current)**2
                
                A = min(1.0, p_new / p_old)
                
                if np.random.rand() < A:
                    walkers[i] = proposal
                    accepted += 1
                
                r = walkers[i]
                r_norm = np.linalg.norm(r)
                
                E = local_energy(r, alpha_current)
                O = -r_norm
                
                E_list.append(E)
                O_list.append(O)
        
        E_list = np.array(E_list)
        O_list = np.array(O_list)
        
        E_mean = np.mean(E_list)
        variance = np.var(E_list)
        error = np.sqrt(variance / len(E_list))
        
        O_mean = np.mean(O_list)
        EO_mean = np.mean(E_list * O_list)
        
        grad = 2 * (EO_mean - E_mean * O_mean)
        
        old_alpha = alpha_current
        alpha_current -= learning_rate * grad
        
        delta_alpha = abs(alpha_current - old_alpha)
        acc_ratio = accepted / (n_walkers * n_steps)
        
        iter_time = time.time() - iter_start_time
        
        # 存储历史数据
        history['iterations'].append(iteration)
        history['energy'].append(E_mean)
        history['energy_error'].append(error)
        history['alpha'].append(alpha_current)
        history['grad'].append(grad)
        history['accept_ratio'].append(acc_ratio)
        history['time'].append(iter_time)
        
        print(
            f"{proposal_name:10s} | iter {iteration:3d} | "
            f"E = {E_mean:.6f} ± {error:.6f} | "
            f"alpha = {alpha_current:.5f} | "
            f"accept = {acc_ratio:.3f} | "
            f"time = {iter_time:.3f}s"
        )
        
        # 收敛检查
        if prev_energy is not None:
            energy_change = abs(E_mean - prev_energy)
            if (abs(grad) < grad_tol and delta_alpha < alpha_tol and energy_change < error):
                print(f"{proposal_name} converged after {iteration + 1} iterations.")
                break
        
        prev_energy = E_mean
    
    total_time = time.time() - start_time_total
    history['total_time'] = total_time
    
    return history

# ---------- 主程序：对比两种分布 ----------

print("="*80)
print("VMC OPTIMIZATION: UNIFORM vs NORMAL PROPOSAL DISTRIBUTION COMPARISON")
print("="*80)

# 使用相同的初始walkers确保公平对比
np.random.seed(42)  # 设置随机种子以确保可重复性
initial_walkers = np.random.randn(n_walkers, dim)

# 运行均匀分布
print("\n" + "-"*40)
print("Running with UNIFORM proposal distribution")
print("-"*40)
np.random.seed(42)  # 重置随机种子
history_uniform = run_vmc(uniform_proposal, "UNIFORM", initial_walkers)

# 运行正态分布
print("\n" + "-"*40)
print("Running with NORMAL proposal distribution")
print("-"*40)
np.random.seed(42)  # 重置随机种子
history_normal = run_vmc(normal_proposal, "NORMAL", initial_walkers)

# ---------- 创建对比图 ----------

plt.style.use('default')
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('Comparison of Uniform vs Normal Proposal Distributions in VMC', fontsize=16)

# 颜色设置
color_uniform = 'blue'
color_normal = 'red'

# 1. 能量收敛对比
ax1 = axes[0, 0]
ax1.errorbar(history_uniform['iterations'], history_uniform['energy'], 
             yerr=history_uniform['energy_error'], fmt='o-', capsize=3, 
             markersize=4, linewidth=1.5, color=color_uniform, alpha=0.7, 
             label=f'Uniform (final: {history_uniform["energy"][-1]:.4f})')
ax1.errorbar(history_normal['iterations'], history_normal['energy'], 
             yerr=history_normal['energy_error'], fmt='s-', capsize=3, 
             markersize=4, linewidth=1.5, color=color_normal, alpha=0.7,
             label=f'Normal (final: {history_normal["energy"][-1]:.4f})')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Energy')
ax1.set_title('Energy Convergence')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Alpha收敛对比
ax2 = axes[0, 1]
ax2.plot(history_uniform['iterations'], history_uniform['alpha'], 
         'o-', color=color_uniform, linewidth=1.5, markersize=4, alpha=0.7,
         label=f'Uniform (final: {history_uniform["alpha"][-1]:.4f})')
ax2.plot(history_normal['iterations'], history_normal['alpha'], 
         's-', color=color_normal, linewidth=1.5, markersize=4, alpha=0.7,
         label=f'Normal (final: {history_normal["alpha"][-1]:.4f})')
ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Exact value (α=1.0)')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Alpha')
ax2.set_title('Parameter Alpha Convergence')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 梯度对比
ax3 = axes[1, 0]
ax3.plot(history_uniform['iterations'], history_uniform['grad'], 
         'o-', color=color_uniform, linewidth=1.5, markersize=4, alpha=0.7,
         label='Uniform')
ax3.plot(history_normal['iterations'], history_normal['grad'], 
         's-', color=color_normal, linewidth=1.5, markersize=4, alpha=0.7,
         label='Normal')
ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Gradient')
ax3.set_title('Gradient vs Iteration')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 接受率对比
ax4 = axes[1, 1]
ax4.plot(history_uniform['iterations'], history_uniform['accept_ratio'], 
         'o-', color=color_uniform, linewidth=1.5, markersize=4, alpha=0.7,
         label=f'Uniform (mean: {np.mean(history_uniform["accept_ratio"]):.3f})')
ax4.plot(history_normal['iterations'], history_normal['accept_ratio'], 
         's-', color=color_normal, linewidth=1.5, markersize=4, alpha=0.7,
         label=f'Normal (mean: {np.mean(history_normal["accept_ratio"]):.3f})')
ax4.set_xlabel('Iteration')
ax4.set_ylabel('Acceptance Ratio')
ax4.set_title('Acceptance Ratio Comparison')
ax4.set_ylim(0, 1)
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. 累积时间对比
ax5 = axes[2, 0]
time_uniform_cumsum = np.cumsum(history_uniform['time'])
time_normal_cumsum = np.cumsum(history_normal['time'])
ax5.plot(history_uniform['iterations'], time_uniform_cumsum, 
         'o-', color=color_uniform, linewidth=1.5, markersize=4, alpha=0.7,
         label=f'Uniform (total: {time_uniform_cumsum[-1]:.2f}s)')
ax5.plot(history_normal['iterations'], time_normal_cumsum, 
         's-', color=color_normal, linewidth=1.5, markersize=4, alpha=0.7,
         label=f'Normal (total: {time_normal_cumsum[-1]:.2f}s)')
ax5.set_xlabel('Iteration')
ax5.set_ylabel('Cumulative Time (s)')
ax5.set_title('Computational Time Comparison')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. 每步时间对比
ax6 = axes[2, 1]
ax6.plot(history_uniform['iterations'], history_uniform['time'], 
         'o-', color=color_uniform, linewidth=1.5, markersize=4, alpha=0.7,
         label=f'Uniform (mean: {np.mean(history_uniform["time"]):.3f}s)')
ax6.plot(history_normal['iterations'], history_normal['time'], 
         's-', color=color_normal, linewidth=1.5, markersize=4, alpha=0.7,
         label=f'Normal (mean: {np.mean(history_normal["time"]):.3f}s)')
ax6.set_xlabel('Iteration')
ax6.set_ylabel('Time per Iteration (s)')
ax6.set_title('Iteration Time Comparison')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()

# ---------- 创建统计汇总表 ----------

print("\n" + "="*80)
print("STATISTICAL COMPARISON SUMMARY")
print("="*80)

print(f"\n{'Metric':<30} {'Uniform':<20} {'Normal':<20}")
print("-"*70)

metrics = [
    ("Final Energy", f"{history_uniform['energy'][-1]:.6f} ± {history_uniform['energy_error'][-1]:.6f}", 
                    f"{history_normal['energy'][-1]:.6f} ± {history_normal['energy_error'][-1]:.6f}"),
    ("Final Alpha", f"{history_uniform['alpha'][-1]:.6f}", f"{history_normal['alpha'][-1]:.6f}"),
    ("Iterations to Converge", f"{len(history_uniform['iterations'])}", f"{len(history_normal['iterations'])}"),
    ("Total Time (s)", f"{history_uniform['total_time']:.3f}", f"{history_normal['total_time']:.3f}"),
    ("Mean Time/Iteration (s)", f"{np.mean(history_uniform['time']):.4f}", f"{np.mean(history_normal['time']):.4f}"),
    ("Mean Acceptance Ratio", f"{np.mean(history_uniform['accept_ratio']):.4f}", f"{np.mean(history_normal['accept_ratio']):.4f}"),
    ("Final Gradient", f"{history_uniform['grad'][-1]:.6f}", f"{history_normal['grad'][-1]:.6f}"),
]

for metric, uniform_val, normal_val in metrics:
    print(f"{metric:<30} {uniform_val:<20} {normal_val:<20}")

# ---------- 创建收敛速度对比图 ----------

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Convergence Speed Comparison', fontsize=14)

# 能量 vs 时间
ax21 = axes2[0]
ax21.plot(time_uniform_cumsum, history_uniform['energy'], 
          'o-', color=color_uniform, linewidth=1.5, markersize=4, alpha=0.7,
          label='Uniform')
ax21.plot(time_normal_cumsum, history_normal['energy'], 
          's-', color=color_normal, linewidth=1.5, markersize=4, alpha=0.7,
          label='Normal')
ax21.set_xlabel('Cumulative Time (s)')
ax21.set_ylabel('Energy')
ax21.set_title('Energy vs Computational Time')
ax21.legend()
ax21.grid(True, alpha=0.3)

# Alpha vs 时间
ax22 = axes2[1]
ax22.plot(time_uniform_cumsum, history_uniform['alpha'], 
          'o-', color=color_uniform, linewidth=1.5, markersize=4, alpha=0.7,
          label='Uniform')
ax22.plot(time_normal_cumsum, history_normal['alpha'], 
          's-', color=color_normal, linewidth=1.5, markersize=4, alpha=0.7,
          label='Normal')
ax22.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Exact α=1.0')
ax22.set_xlabel('Cumulative Time (s)')
ax22.set_ylabel('Alpha')
ax22.set_title('Alpha vs Computational Time')
ax22.legend()
ax22.grid(True, alpha=0.3)

plt.tight_layout()

# ---------- 显示图表 ----------

plt.show()

# ---------- 结论 ----------

print("\n" + "="*80)
print("CONCLUSIONS")
print("="*80)

# 计算收敛速度
uniform_time_to_converge = time_uniform_cumsum[-1]
normal_time_to_converge = time_normal_cumsum[-1]

if uniform_time_to_converge < normal_time_to_converge:
    faster = "Uniform"
    ratio = normal_time_to_converge / uniform_time_to_converge
else:
    faster = "Normal"
    ratio = uniform_time_to_converge / normal_time_to_converge

print(f"\n{faster} distribution is {ratio:.2f}x faster in total computation time.")

# 比较接受率
uniform_accept_mean = np.mean(history_uniform['accept_ratio'])
normal_accept_mean = np.mean(history_normal['accept_ratio'])

if uniform_accept_mean > normal_accept_mean:
    print(f"Uniform distribution has {((uniform_accept_mean/normal_accept_mean - 1)*100):.1f}% higher acceptance rate.")
else:
    print(f"Normal distribution has {((normal_accept_mean/uniform_accept_mean - 1)*100):.1f}% higher acceptance rate.")

# 比较能量精度
uniform_energy_error = history_uniform['energy_error'][-1]
normal_energy_error = history_normal['energy_error'][-1]

if uniform_energy_error < normal_energy_error:
    print(f"Uniform distribution gives {((normal_energy_error/uniform_energy_error - 1)*100):.1f}% more precise energy estimate.")
else:
    print(f"Normal distribution gives {((uniform_energy_error/normal_energy_error - 1)*100):.1f}% more precise energy estimate.")

print("="*80)