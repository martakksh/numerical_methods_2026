import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)

a, b = 0, 24

def simpson_method(f, a, b, N):
    if N % 2 != 0: N += 1
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    I = (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])
    return I

I_true = simpson_method(f, a, b, 100000)

# Графік 1: Функція навантаження
x_plot = np.linspace(a, b, 1000)
plt.figure(figsize=(10, 5))
plt.plot(x_plot, f(x_plot), label='f(x) - навантаження')
plt.title("Графік функції навантаження на сервер")
plt.xlabel("Час, x (год)")
plt.ylabel("Навантаження, f(x)")
plt.grid(True)
plt.legend()
plt.show()

# Графік 2: Похибка 
N_values = np.arange(10, 1001, 10)
errors = [abs(simpson_method(f, a, b, n) - I_true) for n in N_values]

plt.figure(figsize=(10, 5))
plt.semilogy(N_values, errors, color='red')
plt.title("Залежність похибки від кількості розбиттів N")
plt.xlabel("N")
plt.ylabel("Похибка |I(N) - I0|")
plt.grid(True, which="both")
plt.show()

# Розрахункові дані
eps_targets = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
print("\nДослідження адаптивного алгоритму:")
print(f"{'Точність (eps)':<15} | {'Результат':<18} | {'К-сть обчислень f(x)':<20} | {'Фактична похибка':<15}")
print("-" * 75)

# Модифікована функція для підрахунку викликів
def adaptive_simpson_counted(f, a, b, eps):
    mid = (a + b) / 2
    h = b - a
    # 3 точки на першому кроці
    I1 = (h / 6) * (f(a) + 4 * f(mid) + f(b))
    
    m1, m2 = (a + mid) / 2, (mid + b) / 2
    # +2 нові точки (m1, m2), точки a, mid, b вже відомі
    I2 = (h / 12) * (f(a) + 4 * f(m1) + f(mid)) + (h / 12) * (f(mid) + 4 * f(m2) + f(b))
    
    if abs(I1 - I2) <= 15 * eps:
        return I2, 5  # Початкові 3 точки + 2 нові
    else:
        res_left, calls_left = adaptive_simpson_counted(f, a, mid, eps / 2)
        res_right, calls_right = adaptive_simpson_counted(f, mid, b, eps / 2)
        return res_left + res_right, calls_left + calls_right - 1 # -1 бо середня точка спільна

for target in eps_targets:
    res, count = adaptive_simpson_counted(f, a, b, target)
    actual_err = abs(res - I_true)
    print(f"{target:<15e} | {res:<18.12f} | {count:<20} | {actual_err:<15.2e}")