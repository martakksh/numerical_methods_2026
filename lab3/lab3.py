import csv
import matplotlib.pyplot as plt

def read_csv(filename):
    """Зчитує дані з CSV файлу."""
    x, y = [], []
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader) 
        for row in reader:
            if row:
                x.append(float(row[0]))
                y.append(float(row[1]))
    return x, y

def form_matrix(x, m):
    """Формування матриці A для методу найменших квадратів."""
    A = [[0.0] * (m + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(m + 1):
            A[i][j] = sum(xi ** (i + j) for xi in x)
    return A

def form_vector(x, y, m):
    """Формування вектора вільних членів b."""
    b = [0.0] * (m + 1)
    for i in range(m + 1):
        b[i] = sum(y[k] * (x[k] ** i) for k in range(len(x)))
    return b

def gauss_solve(A, b):
    """Розв'язок СЛАР методом Гауса."""
    n = len(b)
    A = [row[:] for row in A]
    b = b[:]
    for k in range(n - 1):
        max_row = k
        for i in range(k + 1, n):
            if abs(A[i][k]) > abs(A[max_row][k]): 
                max_row = i
        A[k], A[max_row] = A[max_row], A[k]
        b[k], b[max_row] = b[max_row], b[k]
        for i in range(k + 1, n):
            if A[k][k] == 0: 
                continue
            factor = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]
    x_sol = [0.0] * n
    for i in range(n - 1, -1, -1):
        suma = sum(A[i][j] * x_sol[j] for j in range(i + 1, n))
        x_sol[i] = (b[i] - suma) / A[i][i] if A[i][i] != 0 else 0
    return x_sol

def polynomial(x, coef):
    """Обчислення значень полінома."""
    return [sum(coef[i] * (xi ** i) for i in range(len(coef))) for xi in x]

def calculate_variance(y_true, y_approx):
    """Обчислення дисперсії."""
    n = len(y_true)
    return sum((y_true[i] - y_approx[i]) ** 2 for i in range(n)) / n

def main():
    # 1. Створення файлу з даними
    csv_filename = "temperatures.csv"
    csv_data = "Month,Temp\n1,-2\n2,0\n3,5\n4,10\n5,15\n6,20\n7,23\n8,22\n9,17\n10,10\n11,5\n12,0\n13,-10\n14,3\n15,7\n16,13\n17,19\n18,20\n19,22\n20,21\n21,18\n22,15\n23,10\n24,3"
    with open(csv_filename, "w", encoding='utf-8') as f:
        f.write(csv_data)

    x, y = read_csv(csv_filename)

    # 2. Розрахунки для m = 1 до 10
    max_degree = 10
    results = {}
    
    print(f"{'Степінь (m)':<12} | {'Дисперсія':<15}")
    print("-" * 40)

    for m in range(1, max_degree + 1):
        A = form_matrix(x, m)
        b_vec = form_vector(x, y, m)
        coef = gauss_solve(A, b_vec)
        y_approx = polynomial(x, coef)
        var = calculate_variance(y, y_approx)
        
        results[m] = {
            'coef': coef,
            'y_approx': y_approx,
            'variance': var
        }
        print(f"{m:<12} | {var:<15.6f}")

    best_m = min(results, key=lambda k: results[k]['variance'])
    print("-" * 40)
    print(f"Найкраща точність у m={best_m}")

    # --- ВІЗУАЛІЗАЦІЯ ---

    # Вікно 1: Порівняння апроксимацій (m=1,3,6,9)
    fig1, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig1.suptitle('Порівняння апроксимацій різних степенів', fontsize=14)
    
    selected_m = [1, 3, 6, 9]
    for idx, m in enumerate(selected_m):
        row, col = idx // 2, idx % 2
        axs[row, col].scatter(x, y, color='blue', s=15, alpha=0.6, label='Дані')
        axs[row, col].plot(x, results[m]['y_approx'], color='red', linewidth=2, label=f'm={m}')
        axs[row, col].set_title(f"Степінь {m} (Var: {results[m]['variance']:.2f})")
        axs[row, col].grid(True, alpha=0.3)
        axs[row, col].legend()

    # Вікно 2: Зменшення дисперсії від 1 до 10
    plt.figure(figsize=(12, 6))
    m_values = list(range(1, max_degree + 1))
    variances = [results[m]['variance'] for m in m_values]
    
    plt.subplot(1, 2, 1)
    plt.plot(m_values, variances, 'bo-', linewidth=2, markersize=8)
    plt.plot(m_values, variances, 'r--', alpha=0.5)
    plt.scatter([best_m], [results[best_m]['variance']], color='red', s=100, zorder=5, 
                label=f'Мінімум (m={best_m})')
    plt.title('Зменшення дисперсії із ростом ступеня m\n(лінійний масштаб)')
    plt.xlabel('Степінь полінома (m)')
    plt.ylabel('Дисперсія')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Підпис значень дисперсії на графіку
    for i, var in enumerate(variances):
        plt.annotate(f'{var:.1f}', (m_values[i], var), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8)

    plt.subplot(1, 2, 2)
    plt.semilogy(m_values, variances, 'go-', linewidth=2, markersize=8)
    plt.semilogy(m_values, variances, 'r--', alpha=0.5)
    plt.scatter([best_m], [results[best_m]['variance']], color='red', s=100, zorder=5,
                label=f'Мінімум (m={best_m})')
    plt.title('Зменшення дисперсії із ростом ступеня m\n(логарифмічний масштаб)')
    plt.xlabel('Степінь полінома (m)')
    plt.ylabel('Дисперсія (log scale)')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend()
    
    plt.tight_layout()

    # Вікно 3: Таблиця значень дисперсії
    plt.figure(figsize=(10, 6))
    plt.axis('off')
    plt.title('Значення дисперсії для різних степенів m', fontsize=14, pad=20)
    
    # Створення таблиці
    table_data = [[f"m={m}", f"{results[m]['variance']:.6f}"] for m in m_values]
    table_data.append(["Найкраще", f"m={best_m} (var={results[best_m]['variance']:.6f})"])
    
    table = plt.table(cellText=table_data, 
                     colLabels=['Степінь', 'Дисперсія'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.2, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Виділення найкращого результату
    table[(best_m, 0)].set_facecolor('#90EE90')
    table[(best_m, 1)].set_facecolor('#90EE90')

    # Вікно 4: Всі апроксимації на одному графіку
    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, color='black', s=30, alpha=0.8, label='Фактичні дані', zorder=10)
    
    # Використовуємо різні кольори для різних степенів
    colors = plt.cm.viridis(np.linspace(0, 1, max_degree))
    for m, color in zip(m_values, colors):
        if m in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            alpha = 0.5 if m != best_m else 1.0
            linewidth = 3 if m == best_m else 1.5
            plt.plot(x, results[m]['y_approx'], color=color, alpha=alpha, 
                    linewidth=linewidth, label=f'm={m} (var={results[m]["variance"]:.1f})')
    
    plt.title('Порівняння всіх апроксимацій (m=1..10)', fontsize=14)
    plt.xlabel('Місяць')
    plt.ylabel('Температура (°C)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()

    # Вікно 5: Прогноз
    plt.figure(figsize=(10, 5))
    x_future = [25, 26, 27]
    
    # Використовуємо найкращу модель для прогнозу
    y_future = polynomial(x_future, results[best_m]['coef'])
    
    plt.scatter(x, y, color='blue', s=50, alpha=0.6, label='Фактичні дані')
    plt.plot(x, results[best_m]['y_approx'], '--', color='red', linewidth=2, 
            label=f'Апроксимація (m={best_m})')
    plt.plot(x_future, y_future, 'go-', markersize=10, linewidth=2, 
            label=f'Прогноз на наступні місяці')
    
    # Додавання підписів до точок прогнозу
    for i, (xi, yi) in enumerate(zip(x_future, y_future)):
        plt.annotate(f'{yi:.1f}°C', (xi, yi), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9, fontweight='bold')
    
    plt.title(f'Прогноз температури (модель m={best_m})')
    plt.xlabel('Місяць')
    plt.ylabel('Температура (°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Додаткова інформація про якість моделі
    plt.text(0.02, 0.98, f'Дисперсія моделі: {results[best_m]["variance"]:.2f}', 
            transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import numpy as np  # Додаємо для роботи з кольорами
    main()

