import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 0. ПАПКА ДЛЯ РЕЗУЛЬТАТІВ
# ==========================================
if not os.path.exists("results"):
    os.makedirs("results")

# ==========================================
# 1. ЕТАЛОННА ФУНКЦІЯ
# ==========================================
def f_true(x):
    return (x**2)/3000 + x/100


# ==========================================
# 2. МЕТОД НЬЮТОНА
# ==========================================
def divided_diff_table(x, y):
    n = len(y)
    coef = np.zeros((n, n))
    coef[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j] - x[i])
    return coef

def newton_poly(coef, x_nodes, x):
    n = len(x_nodes) - 1
    p = coef[0][n]
    for k in range(1, n+1):
        p = coef[0][n-k] + (x - x_nodes[n-k]) * p
    return p


# ==========================================
# 3. МЕТОД ЛАГРАНЖА
# ==========================================
def lagrange_poly(x_nodes, y_nodes, x):
    result = 0.0
    for i in range(len(x_nodes)):
        term = y_nodes[i]
        for j in range(len(x_nodes)):
            if i != j:
                term *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += term
    return result


# ==========================================
# 4. ОСНОВНА ЧАСТИНА
# ==========================================
def basic_part():

    a, b = 1000, 16000
    x_plot = np.linspace(a, b, 500)
    y_true = f_true(x_plot)

    n_values = [5, 10, 20]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Основна частина: Інтерполяція методом Ньютона", fontsize=16)

    for i, n in enumerate(n_values):

        x_nodes = np.linspace(a, b, n)
        y_nodes = f_true(x_nodes)

        diff = divided_diff_table(x_nodes, y_nodes)
        y_interp = np.array([newton_poly(diff, x_nodes, xi) for xi in x_plot])
        error = np.abs(y_true - y_interp)

        # Графік інтерполяції
        axes[0, i].plot(x_plot, y_true, 'k--', label='Еталон')
        axes[0, i].plot(x_plot, y_interp, 'b-', label='Ньютон')
        axes[0, i].scatter(x_nodes, y_nodes, color='red')
        axes[0, i].set_title(f'n = {n}')
        axes[0, i].legend()
        axes[0, i].grid()

        # Графік похибки
        axes[1, i].plot(x_plot, error, 'r-')
        axes[1, i].set_yscale('log')
        axes[1, i].set_title("Похибка (лог шкала)")
        axes[1, i].grid()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("results/basic_part.png")


# ==========================================
# 5. ДОСЛІДНИЦЬКА ЧАСТИНА
# ==========================================
def research_part():

    a, b = 1000, 16000
    x_plot = np.linspace(a, b, 500)
    y_true = f_true(x_plot)

    # --------------------------------------------------
    # 1. Фіксований інтервал, різна кількість вузлів
    # --------------------------------------------------
    plt.figure("Дослідження 1", figsize=(10,6))

    for n in [5, 10, 15]:
        x_nodes = np.linspace(a, b, n)
        y_nodes = f_true(x_nodes)
        diff = divided_diff_table(x_nodes, y_nodes)
        y_interp = [newton_poly(diff, x_nodes, xi) for xi in x_plot]
        plt.plot(x_plot, y_interp, label=f'n={n}')

    plt.plot(x_plot, y_true, 'k--', label='Еталон')
    plt.title("Фіксований інтервал, різна кількість вузлів")
    plt.legend()
    plt.grid()
    plt.savefig("results/research1.png")

    # --------------------------------------------------
    # 2. Фіксований крок, змінний інтервал
    # --------------------------------------------------
    plt.figure("Дослідження 2", figsize=(10,6))

    h = 2000
    intervals = [(1000,6000),(1000,10000),(1000,16000)]

    for start, end in intervals:
        x_nodes = np.arange(start, end+h, h)
        y_nodes = f_true(x_nodes)
        diff = divided_diff_table(x_nodes, y_nodes)

        x_local = np.linspace(start, end, 300)
        y_interp = [newton_poly(diff, x_nodes, xi) for xi in x_local]
        error = np.abs(f_true(x_local) - y_interp)

        plt.plot(x_local, error,
                 label=f"[{start},{end}], вузлів={len(x_nodes)}")

    plt.yscale('log')
    plt.title("Фіксований крок, змінний інтервал")
    plt.legend()
    plt.grid()
    plt.savefig("results/research2.png")

    # --------------------------------------------------
    # 3. Ефект Рунге
    # --------------------------------------------------
    n = 20
    x_nodes = np.linspace(a, b, n)
    y_nodes = f_true(x_nodes)
    diff = divided_diff_table(x_nodes, y_nodes)

    y_interp = np.array([newton_poly(diff, x_nodes, xi) for xi in x_plot])
    error = np.abs(y_true - y_interp)

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,10))

    ax1.plot(x_plot, y_true, 'k--')
    ax1.plot(x_plot, y_interp, 'r-')
    ax1.scatter(x_nodes, y_nodes, color='black')
    ax1.set_title("Ефект Рунге")
    ax1.grid()

    ax2.plot(x_plot, error, 'r-')
    ax2.set_yscale('log')
    ax2.set_title("Зростання похибки на краях")
    ax2.grid()

    plt.savefig("results/research3.png")

    # --------------------------------------------------
    # 4. Порівняння з Лагранжем
    # --------------------------------------------------
    n = 8
    x_nodes = np.linspace(a, b, n)
    y_nodes = f_true(x_nodes)

    diff = divided_diff_table(x_nodes, y_nodes)

    y_newton = np.array([newton_poly(diff, x_nodes, xi) for xi in x_plot])
    y_lagrange = np.array([lagrange_poly(x_nodes, y_nodes, xi) for xi in x_plot])

    diff_methods = np.abs(y_newton - y_lagrange)

    fig2, (ax1, ax2) = plt.subplots(2,1, figsize=(10,10))

    ax1.plot(x_plot, y_newton, 'b-', label='Ньютон')
    ax1.plot(x_plot, y_lagrange, 'r--', label='Лагранж')
    ax1.set_title("Порівняння методів")
    ax1.legend()
    ax1.grid()

    ax2.plot(x_plot, diff_methods, 'g-')
    ax2.set_title("Машинна похибка")
    ax2.grid()

    plt.savefig("results/research4.png")


# ==========================================
# ЗАПУСК
# ==========================================
if __name__ == "__main__":
    basic_part()
    research_part()
    plt.show()