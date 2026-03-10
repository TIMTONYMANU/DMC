import numpy as np

# ---------- simulation parameters ----------

n_walkers = 200
n_steps = 500
step_size = 1.0

alpha = 0.8
learning_rate = 0.05

dim = 3

grad_tol = 1e-3
alpha_tol = 1e-4

max_iter = 1000   # 保险上限防止死循环

# ---------- initial walkers ----------

walkers = np.random.randn(n_walkers, dim)


# ---------- wavefunction ----------

def psi(r, alpha):
    r_norm = np.linalg.norm(r)
    return np.exp(-alpha * r_norm)


# ---------- local energy ----------

def local_energy(r, alpha):
    r_norm = np.linalg.norm(r)
    return -0.5 * alpha**2 + (alpha - 1.0) / r_norm


prev_energy = None

# ---------- optimization loop ----------

for iteration in range(max_iter):

    E_list = []
    O_list = []

    accepted = 0

    for step in range(n_steps):

        for i in range(n_walkers):

            r_old = walkers[i]

            proposal = r_old + step_size * (np.random.normal(size=dim) - 0.5)

            p_old = psi(r_old, alpha)**2
            p_new = psi(proposal, alpha)**2

            A = min(1.0, p_new / p_old)

            if np.random.rand() < A:
                walkers[i] = proposal
                accepted += 1

            r = walkers[i]

            r_norm = np.linalg.norm(r)

            E = local_energy(r, alpha)
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

    old_alpha = alpha
    alpha -= learning_rate * grad

    delta_alpha = abs(alpha - old_alpha)

    acc_ratio = accepted / (n_walkers * n_steps)

    print(
        f"iter {iteration:3d} | "
        f"E = {E_mean:.6f} ± {error:.6f} | "
        f"alpha = {alpha:.5f} | "
        f"grad = {grad:.5f} | "
        f"accept = {acc_ratio:.3f}"
    )

    # ---------- convergence check ----------

    if prev_energy is not None:

        energy_change = abs(E_mean - prev_energy)

        if (
            abs(grad) < grad_tol
            and delta_alpha < alpha_tol
            and energy_change < error
        ):
            print("\nConverged.")
            break

    prev_energy = E_mean