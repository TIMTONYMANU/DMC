import numpy as np
import matplotlib.pyplot as plt

# ---------- simulation parameters ----------
n_walkers = 200
n_steps = 500
step_size = 0.5  # reduced step size for stability
alpha = 0.8
learning_rate = 0.01  # reduced learning rate
dim = 3
max_iter = 100
eps = 1e-8  # prevent division by zero

# ---------- store results for each iteration ----------
iterations = []
energies = []
alphas = []
gradients = []
accept_ratios = []

# ---------- initial walkers ----------
walkers = np.random.randn(n_walkers, dim) * 2.0  # wider initial distribution

# ---------- wave function ----------
def psi(r, alpha):
    r_norm = np.linalg.norm(r)
    return np.exp(-alpha * r_norm**2)

# ---------- correct local energy ----------
def local_energy(r, alpha):
    r_norm = np.linalg.norm(r)
    # prevent division by zero
    r_norm = max(r_norm, eps)
    
    # Kinetic term: -1/2 ∇²ψ/ψ
    # For ψ = exp(-αr²)
    # ∇²ψ/ψ = 4α²r² - 6α
    kinetic = -0.5 * (4 * alpha**2 * r_norm**2 - 6 * alpha)
    
    # Potential term: -1/r
    potential = -1.0 / r_norm
    
    return kinetic + potential

# ---------- local operator O = d/dα ln|ψ|^2 ----------
def local_operator(r, alpha):
    r_norm = np.linalg.norm(r)
    return -r_norm**2

print("Starting optimization...")
print("=" * 80)

# ---------- optimization loop ----------
for iteration in range(max_iter):
    
    E_list = []
    O_list = []
    accepted = 0
    
    # ----- Metropolis sampling -----
    for step in range(n_steps):
        for i in range(n_walkers):
            
            r_old = walkers[i].copy()
            
            # proposal move (using normal distribution)
            proposal = r_old + step_size * np.random.randn(dim)
            
            # compute probability density ratio
            p_old = psi(r_old, alpha)**2
            p_new = psi(proposal, alpha)**2
            
            # Metropolis acceptance criterion
            if p_old > 0:
                A = min(1.0, p_new / p_old)
            else:
                A = 1.0
            
            if np.random.rand() < A:
                walkers[i] = proposal
                accepted += 1
            
            # compute energy and local operator for current configuration
            r = walkers[i]
            E = local_energy(r, alpha)
            O = local_operator(r, alpha)
            
            E_list.append(E)
            O_list.append(O)
    
    # ----- compute statistics -----
    E_list = np.array(E_list)
    O_list = np.array(O_list)
    
    # check for NaN or infinity
    if np.any(np.isnan(E_list)) or np.any(np.isinf(E_list)):
        print(f"\nIteration {iteration}: Detected NaN or infinity!")
        print(f"Energy statistics: min={np.min(E_list):.3f}, max={np.max(E_list):.3f}")
        break
    
    # compute mean and error
    E_mean = np.mean(E_list)
    E_error = np.std(E_list) / np.sqrt(len(E_list))
    
    O_mean = np.mean(O_list)
    EO_mean = np.mean(E_list * O_list)
    
    # compute gradient (derivative of energy with respect to alpha)
    # dE/dα = 2(⟨E·O⟩ - ⟨E⟩⟨O⟩)
    grad = 2 * (EO_mean - E_mean * O_mean)
    
    # gradient clipping to prevent too large updates
    grad = np.clip(grad, -1.0, 1.0)
    
    # update alpha
    old_alpha = alpha
    alpha -= learning_rate * grad
    
    # ensure alpha stays positive
    alpha = max(alpha, 0.01)
    
    # compute acceptance ratio
    acc_ratio = accepted / (n_walkers * n_steps)
    
    # store results
    iterations.append(iteration)
    energies.append(E_mean)
    alphas.append(alpha)
    gradients.append(grad)
    accept_ratios.append(acc_ratio)
    
    # print progress
    print(f"Iter {iteration:3d} | "
          f"E = {E_mean:10.6f} ± {E_error:8.6f} | "
          f"α = {alpha:8.5f} | "
          f"∇ = {grad:10.6f} | "
          f"accept = {acc_ratio:.3f}")
    
    # check convergence
    if iteration > 0 and abs(grad) < 1e-3:
        print(f"\nGradient small, possibly converged")

print("=" * 80)
print(f"Optimization completed, {len(iterations)} iterations")

# ---------- plot results ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Energy vs iteration
ax1 = axes[0, 0]
ax1.errorbar(iterations, energies, yerr=E_error, fmt='o-', 
             color='blue', capsize=3, markersize=4)
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('Energy E', fontsize=12)
ax1.set_title('Energy vs Iteration', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=-0.5, color='r', linestyle='--', alpha=0.7, label='Exact ground state (-0.5)')
ax1.legend()

# Parameter alpha vs iteration
ax2 = axes[0, 1]
ax2.plot(iterations, alphas, 'ro-', linewidth=2, markersize=4)
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('Parameter α', fontsize=12)
ax2.set_title('Parameter α vs Iteration', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0.5, color='b', linestyle='--', alpha=0.7, label='Exact value (0.5)')
ax2.legend()

# Gradient vs iteration
ax3 = axes[1, 0]
ax3.plot(iterations, gradients, 'go-', linewidth=2, markersize=4)
ax3.set_xlabel('Iteration', fontsize=12)
ax3.set_ylabel('Gradient ∇E', fontsize=12)
ax3.set_title('Gradient vs Iteration', fontsize=14)
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)

# Acceptance ratio vs iteration
ax4 = axes[1, 1]
ax4.plot(iterations, accept_ratios, 'mo-', linewidth=2, markersize=4)
ax4.set_xlabel('Iteration', fontsize=12)
ax4.set_ylabel('Acceptance Ratio', fontsize=12)
ax4.set_title('Metropolis Acceptance Ratio vs Iteration', fontsize=14)
ax4.grid(True, alpha=0.3)
ax4.set_ylim([0, 1])

plt.tight_layout()
plt.show()

# ---------- final results analysis ----------
print("\n" + "="*50)
print("Final Results Analysis:")
print(f"Final energy: {energies[-1]:.6f} ± {E_error:.6f}")
print(f"Theoretical ground state energy: -0.5")
print(f"Energy error: {energies[-1] + 0.5:.6f}")

print(f"\nFinal alpha: {alphas[-1]:.6f}")
print(f"Theoretical optimal alpha: 0.5")
print(f"Alpha error: {alphas[-1] - 0.5:.6f}")