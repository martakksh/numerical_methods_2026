import numpy as np
import matplotlib.pyplot as plt

def f(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def df_exact(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

def central_diff(t, h):
    return (f(t + h) - f(t - h)) / (2 * h)

t0 = 1.0
y_exact = df_exact(t0)

# 1. Пошук оптимального h0 (регуляризація) - 20 точок як на фото
h_range = np.logspace(-20, 3, 20)
errors_h = []
for h_val in h_range:
    errors_h.append(abs(central_diff(t0, h_val) - y_exact))

best_idx = np.argmin(errors_h)
h0 = h_range[best_idx]
r0 = errors_h[best_idx]

# 2. Обчислення за пунктами 3-7 методички (h = 10^-3)
h_lab = 1e-3
y_h = central_diff(t0, h_lab)
y_2h = central_diff(t0, 2 * h_lab)
y_4h = central_diff(t0, 4 * h_lab)

r1 = abs(y_h - y_exact)

# Метод Рунге-Ромберга (p=2 для центральної різниці)
y_rr = y_h + (y_h - y_2h) / 3
r2 = abs(y_rr - y_exact)

# Метод Ейткена
denom = 2 * y_2h - (y_4h + y_h)
y_aitken = (y_2h**2 - y_4h * y_h) / denom
r3 = abs(y_aitken - y_exact)
p_aitken = np.log(abs((y_4h - y_2h) / (y_2h - y_h))) / np.log(2)

# ВИВІД У КОНСОЛЬ
print("-" * 30)
print(f"Точне значення похідної: {y_exact:.10f}")
print("-" * 30)
print(f"Оптимальний крок h0: {h0:.2e}")
print(f"Мінімальна похибка R0: {r0:.2e}")
print("-" * 30)
print(f"Чисельна похідна (h=1e-3): {y_h:.10f}")
print(f"Похибка R1: {r1:.10f}")
print("-" * 30)
print(f"Уточнення за Рунге-Ромбергом: {y_rr:.10f}")
print(f"Похибка R2 (після уточнення): {r2:.10f}")
print("-" * 30)
print(f"Уточнення за Ейткеном: {y_aitken:.10f}")
print(f"Похибка R3: {r3:.10f}")
print(f"Оцінка порядку точності p: {p_aitken:.4f}")
print("-" * 30)

if y_rr < -1.8:
    print("Рекомендація: Висока швидкість висихання. Активувати полив.")
else:
    print("Рекомендація: Швидкість висихання в межах норми.")

# ВІЗУАЛІЗАЦІЯ
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Графік 1: Модель вологості M(t)
t_vals = np.linspace(0, 20, 500)
ax1.plot(t_vals, f(t_vals), label='M(t) Вологість', color='darkcyan', lw=2)
ax1.scatter([t0], [f(t0)], color='red', zorder=5, label=f'Точка t0={t0}')
ax1.set_title('Soil Moisture Model M(t)')
ax1.set_xlabel('Час t')
ax1.set_ylabel('Вологість M')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Графік 2: Залежність похибки від кроку h (як на фото)
ax2.loglog(h_range, errors_h, 'b-o', markersize=6, label='Похибка R(h)')
ax2.scatter([h0], [r0], color='red', s=100, edgecolors='black', zorder=10, label=f'Оптимум h={h0:.1e}')
ax2.set_title('Залежність похибки від кроку h')
ax2.set_xlabel('Крок h (log scale)')
ax2.set_ylabel('Похибка R (log scale)')
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.show()