import numpy as np
import matplotlib.pyplot as plt


# ВХІДНІ ДАНІ ТА АНАЛІТИЧНИЙ РОЗВ'ЯЗОК 

def f(x, y):
    """Права частина диференціального рівняння dy/dx = f(x, y)"""
    return y - x**2

def y_exact(x):
    """Точний (аналітичний) розв'язок рівняння (Пункт 1)"""
    return x**2 + 2*x + 2 - np.exp(x)

# Задані параметри інтегрування
x0, xn = 0.0, 2.0
y0 = 1.0
h_fixed = 0.1
epsilon = 1e-4  # Точність для автоматичного кроку


# ЧАСТИНА 1. Метод прогнозу та корекції Адамса 2-го порядку


def adams_pc_fixed(f, x0, xn, y0, h):
    """Метод Адамса 2-го порядку з фіксованим кроком (Пункт 2)"""
    N = int((xn - x0) / h)
    x = np.linspace(x0, xn, N + 1)
    y = np.zeros(N + 1)
    
    # Стартуємо за допомогою РК2 (покращений Ейлера) для першого кроку
    y[0] = y0
    k1 = f(x[0], y[0])
    k2 = f(x[0] + h, y[0] + h * k1)
    y[1] = y[0] + 0.5 * h * (k1 + k2)
    
    # Основний цикл Адамса (Предиктор-Коректор)
    for n in range(1, N):
        # 1. Прогноз (Предиктор)
        y_pred = y[n] + (h / 2.0) * (3.0 * f(x[n], y[n]) - f(x[n-1], y[n-1]))
        
        # 2. Корекція (Коректор - 2 ітерації)
        y_corr = y_pred
        for _ in range(2):
            y_corr = y[n] + (h / 2.0) * (f(x[n+1], y_corr) + f(x[n], y[n]))
            
        y[n+1] = y_corr
        
    return x, y

def adams_pc_auto(f, x0, xn, y0, eps):
    """Метод Адамса з автоматичним вибором кроку (Пункт 5)"""
    x_points = [x0]
    y_points = [y0]
    h_points = [0.05]
    
    x = x0
    y = y0
    h = 0.05
    
    # Перший крок через РК2
    y_1 = y + 0.5 * h * (f(x, y) + f(x + h, y + h * f(x, y)))
    x_points.append(x + h)
    y_points.append(y_1)
    h_points.append(h)
    
    curr_idx = 1
    while x_points[curr_idx] < xn:
        xc, xp = x_points[curr_idx], x_points[curr_idx-1]
        yc, yp = y_points[curr_idx], y_points[curr_idx-1]
        
        if xc + h > xn:
            h = xn - xc
            
        y_pred = yc + (h / 2.0) * (3.0 * f(xc, yc) - f(xp, yp))
        y_corr = yc + (h / 2.0) * (f(xc + h, y_pred) + f(xc, yc))
        
        # Оцінка локальної похибки коректора R2 = |y_corr - y_pred| / 6
        R_local = abs(y_corr - y_pred) / 6.0
        
        if R_local > eps:
            h /= 2.0  # Зменшуємо крок
            y_points[curr_idx] = y_points[curr_idx-1] + 0.5 * h * (f(xp, yp) + f(xp+h, yc)) 
            x_points[curr_idx] = xp + h
            continue
        elif R_local < eps / 8.0:
            next_h = h * 2.0  # Збільшуємо крок
        else:
            next_h = h
            
        x_points.append(xc + h)
        y_points.append(y_corr)
        h_points.append(h)
        h = next_h
        curr_idx += 1
        
    return np.array(x_points), np.array(y_points), np.array(h_points)



# ЧАСТИНА 2. Метод Рунге-Кутта 4-го порядку (РК4)


def rk4_fixed(f, x0, xn, y0, h):
    """Метод Рунге-Кутта 4-го порядку з фіксованим кроком (Пункт 6)"""
    N = int((xn - x0) / h)
    x = np.linspace(x0, xn, N + 1)
    y = np.zeros(N + 1)
    y[0] = y0
    
    for n in range(N):
        k1 = f(x[n], y[n])
        k2 = f(x[n] + h/2.0, y[n] + h*k1/2.0)
        k3 = f(x[n] + h/2.0, y[n] + h*k2/2.0)
        k4 = f(x[n] + h, y[n] + h*k3)
        y[n+1] = y[n] + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
    return x, y

def rk4_auto(f, x0, xn, y0, eps):
    """Метод РК4 з автоматичним вибором кроку за Рунге (Пункт 9)"""
    x_points, y_points, h_points = [x0], [y0], [0.1]
    x, y, h = x0, y0, 0.1
    
    while x < xn:
        if x + h > xn:
            h = xn - x
            
        # Обчислення з кроком h
        k1 = f(x, y)
        k2 = f(x + h/2, y + h*k1/2)
        k3 = f(x + h/2, y + h*k2/2)
        k4 = f(x + h, y + h*k3)
        y_h = y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Два кроки по h/2
        h2 = h / 2.0
        k1_1 = f(x, y)
        k2_1 = f(x + h2/2, y + h2*k1_1/2)
        k3_1 = f(x + h2/2, y + h2*k2_1/2)
        k4_1 = f(x + h2, y + h2*k3_1)
        y_h2_1 = y + (h2/6.0) * (k1_1 + 2*k2_1 + 2*k3_1 + k4_1)
        
        k1_2 = f(x + h2, y_h2_1)
        k2_2 = f(x + h2 + h2/2, y_h2_1 + h2*k1_2/2)
        k3_2 = f(x + h2 + h2/2, y_h2_1 + h2*k2_2/2)
        k4_2 = f(x + h2 + h2, y_h2_1 + h2*k3_2)
        y_h2_2 = y_h2_1 + (h2/6.0) * (k1_2 + 2*k2_2 + 2*k3_2 + k4_2)
        
        # Оцінка похибки за методом Рунге (Пункт 8)
        err = (16.0 / 15.0) * abs(y_h - y_h2_2)
        
        if err > eps:
            h /= 2.0
            continue
        else:
            x += h
            y = y_h2_2
            x_points.append(x)
            y_points.append(y)
            h_points.append(h)
            
            if err < eps / 32.0:
                h *= 2.0
                
    return np.array(x_points), np.array(y_points), np.array(h_points)



# ВИКОНАННЯ ОБЧИСЛЕНЬ ТА ФОРМУВАННЯ ДАНИХ

# --- Обчислення для Частини 1 ---
x_adams, y_adams = adams_pc_fixed(f, x0, xn, y0, h_fixed)
err_adams_exact = abs(y_adams - y_exact(x_adams))

err_adams_theor = np.zeros_like(err_adams_exact)
for i in range(1, len(x_adams)-1):
    ypred = y_adams[i] + (h_fixed/2.0)*(3*f(x_adams[i], y_adams[i]) - f(x_adams[i-1], y_adams[i-1]))
    ycorr = y_adams[i] + (h_fixed/2.0)*(f(x_adams[i+1], y_adams[i+1]) + f(x_adams[i], y_adams[i]))
    err_adams_theor[i+1] = abs(ycorr - ypred) / 6.0

x_a_auto, y_a_auto, h_a_auto = adams_pc_auto(f, x0, xn, y0, epsilon)

# --- Обчислення для Частини 2 ---
x_rk4, y_rk4 = rk4_fixed(f, x0, xn, y0, h_fixed)
err_rk4_exact = abs(y_rk4 - y_exact(x_rk4))

err_rk4_runge = np.zeros_like(err_rk4_exact)
for i in range(1, len(x_rk4)):
    xc, yc = x_rk4[i-1], y_rk4[i-1]
    k1 = f(xc, yc); k2 = f(xc+h_fixed/2, yc+h_fixed*k1/2); k3 = f(xc+h_fixed/2, yc+h_fixed*k2/2); k4 = f(xc+h_fixed, yc+h_fixed*k3)
    yh = yc + (h_fixed/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    
    h2 = h_fixed/2.0
    k1_1 = f(xc, yc); k2_1 = f(xc+h2/2, yc+h2*k1_1/2); k3_1 = f(xc+h2/2, yc+h2*k2_1/2); k4_1 = f(xc+h2, yc+h2*k1_1)
    yh2_1 = yc + (h2/6.0)*(k1_1 + 2*k2_1 + 2*k3_1 + k4_1)
    k1_2 = f(xc+h2, yh2_1); k2_2 = f(xc+h2+h2/2, yh2_1+h2*k1_2/2); k3_2 = f(xc+h2+h2/2, yh2_1+h2*k2_2/2); k4_2 = f(xc+h2+h2, yh2_1+h2*k3_2)
    yh2_2 = yh2_1 + (h2/6.0)*(k1_2 + 2*k2_2 + 2*k3_2 + k4_2)
    err_rk4_runge[i] = (16.0 / 15.0) * abs(yh - yh2_2)

x_r_auto, y_r_auto, h_r_auto = rk4_auto(f, x0, xn, y0, epsilon)



# ВИВЕДЕННЯ ТАБЛИЦЬ У КОНСОЛЬ (Новий блок)

def print_table(title, x_arr, y_arr, h_arr, is_auto=True):
    print("\n" + "="*80)
    print(f" {title.upper()} (Точність eps = {epsilon})")
    print("="*80)
    print(f"{'№':<4} | {'x':<8} | {'y (чисельний)':<15} | {'y (точний)':<15} | {'Крок h':<8} | {'Похибка':<12}")
    print("-"*80)
    for idx in range(len(x_arr)):
        x_val = x_arr[idx]
        y_val = y_arr[idx]
        y_ex  = y_exact(x_val)
        h_val = h_arr[idx] if is_auto else h_fixed
        err   = abs(y_val - y_ex)
        print(f"{idx:<4} | {x_val:<8.4f} | {y_val:<15.7f} | {y_ex:<15.7f} | {h_val:<8.4f} | {err:<12.4e}")
    print("="*80 + "\n")

# Виводимо таблиці результатів з автоматичним вибором кроку
print_table("Таблиця результатів: Частина 1 (Метод Адамса 2-го порядку з автокроком)", x_a_auto, y_a_auto, h_a_auto)
print_table("Таблиця результатів: Частина 2 (Метод Рунге-Кутта 4-го порядку з автокроком)", x_r_auto, y_r_auto, h_r_auto)





#  Графіки до Ч.1 (Метод Адамса)

fig1 = plt.figure(num="ЕКРАН 1: Частина 1 (Метод Адамса)", figsize=(11, 9))

# Графік 1: Локальна похибка через точний розв'язок
plt.subplot(3, 1, 1)
plt.plot(x_adams, err_adams_exact, 'r-o', label=r'$\epsilon(x_n) = |y_{exact} - y_{num}|$')
plt.title('Графік 1: Точна локальна похибка (Пункт 3)')
plt.xlabel('x')
plt.ylabel('Похибка')
plt.grid(True)
plt.legend()

# Графік 2: Локальна похибка через вираз оцінки похибки
plt.subplot(3, 1, 2)
plt.plot(x_adams, err_adams_theor, 'b--s', label=r'Оцінка $R_2 = |y_{corr} - y_{pred}| / 6$')
plt.title('Графік 2: Оціночна похибка за теоретичною формулою (Пункт 4)')
plt.xlabel('x')
plt.ylabel('Похибка R_2')
plt.grid(True)
plt.legend()

# Графік 3: Автоматичний вибір кроку
plt.subplot(3, 1, 3)
plt.step(x_a_auto, h_a_auto, where='post', color='purple', label='Крок h(x)')
plt.title('Графік 3: Залежність величини автоматичного кроку від X (Пункт 5)')
plt.xlabel('x')
plt.ylabel('Величина кроку h')
plt.grid(True)
plt.legend()

plt.tight_layout()


# ЕКРАН 2: Графіки до Ч.2 (Метод Рунге-Кутта)

fig2 = plt.figure(num="ЕКРАН 2: Частина 2 (Метод Рунге-Кутта 4)", figsize=(11, 9))

# Графік 4: Точна локальна похибка РК4
plt.subplot(3, 1, 1)
plt.plot(x_rk4, err_rk4_exact, 'g-^', label=r'$\epsilon(x_n) = |y_{exact} - y_{RK4}|$')
plt.title('Графік 4: Точна локальна похибка РК4 (Пункт 7)')
plt.xlabel('x')
plt.ylabel('Похибка')
plt.grid(True)
plt.legend()

# Графік 5: Оцінка похибки за методом Рунге
plt.subplot(3, 1, 2)
plt.plot(x_rk4, err_rk4_runge, 'm--d', label=r'Оцінка за Рунге $\frac{16}{15}|y_h - y_{h/2}|$')
plt.title('Графік 5: Локальна похибка за методом Рунге (Пункт 8)')
plt.xlabel('x')
plt.ylabel('Похибка за Рунге')
plt.grid(True)
plt.legend()

# Графік 6: Автоматичний вибір кроку для РК4
plt.subplot(3, 1, 3)
plt.step(x_r_auto, h_r_auto, where='post', color='orange', label='Крок h(x)')
plt.title('Графік 6: Залежність автоматичного кроку від X для РК4 (Пункт 9)')
plt.xlabel('x')
plt.ylabel('Величина кроку h')
plt.grid(True)
plt.legend()

plt.tight_layout()

# Відображення обох вікон
plt.show()
