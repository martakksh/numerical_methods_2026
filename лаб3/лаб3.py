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
            if abs(A[i][k]) > abs(A[max_row][k]): max_row = i
        A[k], A[max_row] = A[max_row], A[k]
        b[k], b[max_row] = b[max_row], b[k]
        for i in range(k + 1, n):
            if A[k][k] == 0: continue
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

    # 2. Розрахунки для m = 1, 2, 3, 4
    max_degree = 4
    results = {}
    
    print(f"{'Степінь (m)':<12} | {'Дисперсія':<15}")
    print("-" * 30)

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
        print(f"{m:<12} | {var:<15.4f}")

    best_m = min(results, key=lambda k: results[k]['variance'])
    print("-" * 30)
    print(f"Найкраща точність у m={best_m}")

    # --- ВІЗУАЛІЗАЦІЯ ---

    # Вікно 1: Порівняння апроксимацій (m=1..4)
    fig1, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig1.suptitle('Порівняння апроксимацій (m=1..4)', fontsize=14)
    for i, m in enumerate(range(1, max_degree + 1)):
        row, col = i // 2, i % 2
        axs[row, col].scatter(x, y, color='blue', s=15, alpha=0.6)
        axs[row, col].plot(x, results[m]['y_approx'], color='red', label=f'm={m}')
        axs[row, col].set_title(f"Степінь {m} (Var: {results[m]['variance']:.2f})")
        axs[row, col].grid(True, alpha=0.3)
        axs[row, col].legend()

    # Вікно 2: Дисперсія
    plt.figure(figsize=(8, 4))
    variances = [results[m]['variance'] for m in range(1, 5)]
    plt.plot(range(1, 5), variances, 'o-', color='green')
    plt.title('Зменшення дисперсії із ростом ступеня m')
    plt.xlabel('m')
    plt.ylabel('Дисперсія')
    plt.grid(True)

    # Вікно 3: Прогноз (Розумний підхід)
    plt.figure(figsize=(10, 5))
    x_future = [25, 26, 27]
    # Використовуємо m=1 для прогнозу, щоб уникнути різкого падіння, 
    # але m=4 для відображення існуючого тренду
    y_future = polynomial(x_future, results[1]['coef']) 
    
    plt.scatter(x, y, color='blue', label='Фактичні дані')
    plt.plot(x, results[best_m]['y_approx'], '--', color='red', label=f'Апроксимація (m={best_m})')
    plt.plot(x_future, y_future, 'go-', label='Прогноз (лінійний тренд)')
    
    plt.title('Адекватний прогноз температури')
    plt.xlabel('Місяць')
    plt.ylabel('Температура (°C)')
    plt.legend()
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()
