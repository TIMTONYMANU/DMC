import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 初始化 DMC 参数
# ==========================================
N_target = 2000      # 目标 Walker 数量
dt = 0.05            # 虚时间步长
steps = 2000         # 总模拟步数
alpha = 1.0 / dt     # 能量反馈系数

# 初始状态：所有 Walkers 都在 x=0
walkers = np.random.uniform(-3, 3, N_target)
E_R = 0.0            # 初始参考能量

# 记录数据
energy_history = []
snapshots = []

# ==========================================
# 2. DMC 主循环 (二阶 Trotter-Suzuki)
# ==========================================
print("开始二阶 DMC 模拟...")
for step in range(steps):
    
    # ---- 步骤 1: 保存旧位置 (用于对称化权重) ----
    old_positions = walkers.copy()
    
    # ---- 步骤 2: 半势能步 (e^{-dt/2 * V}) ----
    # 先应用半个势能步的权重因子，在下一步与扩散结合
    # 实际上我们用最终对称化权重一次性处理，所以这里只扩散，权重最后统一算
    
    # ---- 步骤 3: 全扩散步 (e^{-dt * T}) ----
    walkers += np.random.normal(0, np.sqrt(dt), len(walkers))
    
    # ---- 步骤 4: 计算对称化权重 ----
    # 使用扩散前和扩散后位置的平均势能
    V_old = 0.5 * old_positions**2
    V_new = 0.5 * walkers**2
    V_avg = (V_old + V_new) / 2.0
    
    # 权重 W = exp[-dt * (V_avg - E_R)]
    W = np.exp(-(V_avg - E_R) * dt)
    
    # ---- 步骤 5: 生与死 (Branching) ----
    u = np.random.uniform(0, 1, len(walkers))
    copies = np.floor(W + u).astype(int)
    
    # 生成新一代 Walkers
    walkers = np.repeat(walkers, copies)
    
    # ---- 步骤 6: 更新参考能量 E_R ----
    N_curr = len(walkers)
    # 能量估计用平均势能 + 动能修正？这里简单用平均势能作为估计
    # 更精确的做法是用 "成长估计" 或 "混合估计"，这里简化
    E_est = np.mean(0.5 * walkers**2)  # 这只是势能平均，但基态时势能=动能=0.25，总和0.5
    E_R = E_est - alpha * np.log(N_curr / N_target)
    
    # 记录数据
    energy_history.append(E_R)
    if step in [0, 10,100,  1999]:
        snapshots.append((step, walkers.copy()))

print("模拟完成！")

# ==========================================
# 3. 数据可视化
# ==========================================
plt.figure(figsize=(15, 5))

# 图1：能量收敛
plt.subplot(1, 3, 1)
plt.plot(energy_history, color='blue', alpha=0.7)
plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Exact $E_0=0.5$')
plt.title("Energy Convergence ($E_R$,  Trotter)")
plt.xlabel("Imaginary Time Steps")
plt.ylabel("Energy")
plt.ylim(0.2, 0.8)
plt.legend()
plt.grid(True)

# 图2：波函数演化
plt.subplot(1, 3, 2)
colors = ['lightgray', 'skyblue', 'royalblue', 'darkblue']
for i, (step, w) in enumerate(snapshots):
    plt.hist(w, bins=50, density=True, histtype='step', 
             linewidth=2, color=colors[i], label=f'Step {step}')
plt.title("Wavefunction Evolution ")
plt.xlabel("Position (x)")
plt.ylabel("Walker Density")
plt.legend()
plt.grid(True)

# 图3：最终状态 vs 解析解
plt.subplot(1, 3, 3)
plt.hist(walkers, bins=60, density=True, color='lightblue', edgecolor='black', label='DMC Walkers ')

x_grid = np.linspace(-4, 4, 100)
psi_exact = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x_grid**2)
plt.plot(x_grid, psi_exact, color='red', linewidth=2.5, label='Analytical $\Psi_0(x)$')

plt.title("Final State vs Analytical ")
plt.xlabel("Position (x)")
plt.ylabel("Wavefunction $\Psi(x)$")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()