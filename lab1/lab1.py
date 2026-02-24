import requests
import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------------------------Data Get--------------------------------------------------------------------#

def get_elevation_data():
    url = f" https://api.open-elevation.com/api/v1/lookup?locations=48.164214,%2024.536044|48.164983,%2024.534836|48.165605,%2024.534068|48.166228,%2024.532915|48.166777,%2024.531927|48.167326,%2024.530884|48.167011,%2024.530061|48.166053,%2024.528039|48.166655,%2024.526064|48.166497,%2024.523574|48.166128,%2024.520214|48.165416,%2024.517170|48.164546,%2024.514640|48.163412,%2024.512980|48.162331,%2024.511715|48.162015,%2024.509462|48.162147,%2024.506932|48.161751,%2024.504244|48.161197,%2024.501793|48.160580,%2024.500537|48.160250,%2024.500106"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()['results']
    except:
        print("Failed to get data from API")
        return 0;
    
#-----------------------------------------------------Math--------------------------------------------------------------------# 
def dot_distance(lat1, long1 ,lat2, long2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(long2 - long1)
    
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def spline_coef(x,y, show_prints=False):
    intnervalsN = len(x) - 1
    h = np.diff(x)
    
    alpha_arr = np.zeros(intnervalsN)
    beta_arr = np.zeros(intnervalsN)
    c_sec = np.zeros(intnervalsN+1)
    alpha_arr[0] = 0
    beta_arr[0] = 0
    
    #--------Forward---------------#
    if show_prints:
        print("\n--- Прогоночні коефіцієнти ---")
        print(f"{'i':<3} | {'Alpha (α)':<15} | {'Beta (β)':<15}")
    for i in range(1, intnervalsN):
        A = h[i-1]
        B = 2* (h[i-1] + h[i])
        C = h[i]
        
        rightCoef = 3 * ((y[i+1] - y[i])/ h[i] - (y[i] - y[i-1])/ h[i - 1])
        under = B + A * alpha_arr[i-1]
        alpha_arr[i] = -C / under
        beta_arr[i] = (rightCoef - A * beta_arr[i-1]) / under
        if show_prints:
            print(f"{i:<3d} | {alpha_arr[i]:<15.6f} | {beta_arr[i]:<15.6f}")
        
    c_sec[intnervalsN] = 0
    #-------Backward---------------#
    for i in range(intnervalsN-1, 0 , -1):
        c_sec[i] = alpha_arr[i] * c_sec[i+1] + beta_arr[i]
    c_sec[0] = 0
    a = np.zeros(intnervalsN)
    b = np.zeros(intnervalsN)
    d = np.zeros(intnervalsN)        
    
    for i in range(intnervalsN):
        a[i] = y[i]
        b[i] = ((y[i+1]- y[i])/h[i] - h[i] * (c_sec[i+1]+ 2*c_sec[i]) / 3)
        d[i] = ((c_sec[i+1] - c_sec[i]) / (3 * h[i]))
    
    return{'a':a, 'b':b, 'c':c_sec[:-1], 'd':d, 'h':h, 'x':x}


def evaluate_spline(coefs, x_eval):
    nodes = coefs['x']
    n = len(nodes) - 1
    if x_eval <= nodes[0]: return coefs['a'][0]
    if x_eval >= nodes[-1]: return coefs['a'][-1] + coefs['b'][-1] * (x_eval - nodes[-2])
    for i in range(n):
        if nodes[i] <= x_eval <= nodes[i+1]:
            dx = x_eval - nodes[i]
            val = (coefs['a'][i] + 
                   coefs['b'][i]*dx + 
                   coefs['c'][i]*dx**2 + 
                   coefs['d'][i]*dx**3)
            return val
    return 0








#-----------------------------------------------------Init--------------------------------------------------------------------#
if __name__ == "__main__":
    data = get_elevation_data()
    try:
        

#--------------------------------------------------File Write----------------------------------------------------------------#
       
        latitudes = [k['latitude'] for k in data]
        longitudes = [k['longitude'] for k in data]
        elevations = [k['elevation'] for k in data]
        
        distances = [0.0]
        for i in range (1, len(data)):
            d = dot_distance(latitudes[i-1], longitudes[i-1], latitudes[i], longitudes[i])
            distances.append(distances[-1] + d)     
        x_arr = np.array(distances)
        y_arr = np.array(elevations)
        route_coefs = spline_coef(x_arr,y_arr, show_prints=True)
        
        with open("results.txt", 'w', encoding="utf-8") as f:
            header = header = f"{'Індекс':<6} | {'Широта':<10} | {'Довгота':<10} | {'Висота (м)':<10} | {'Дистанція (м)':<12}\n"
            separ = '-'*60 + "\n"
            
            f.write(header)
            f.write(separ)
            print(header.strip())
            print(separ.strip())
            
            for i in range(len(data)):
                line = f"{i:<6d} | {latitudes[i]:<10.6f} | {longitudes[i]:<10.6f} | {y_arr[i]:<10.2f} | {x_arr[i]:<12.2f}\n"
                f.write(line)
                print(line.strip())
        print("Запис успішно завершено")
        


#-----------------------------------------------------Graph Draw---------------------------------------------------------------#
        
        node_counts=[10,15,20]
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,10))
        ax1.plot(x_arr, y_arr, 'k--', label='Вхідні дані', linewidth=2, alpha=0.7)
        ax1.set_title("Профіль маршруту: Інтерполяція кубічними сплайнами")
        ax1.set_xlabel("Кумулятивна відстань (м)")
        ax1.set_ylabel("Висота (м)")
        ax1.grid(True)
        
        ax2.set_title("Абсолютна похибка ε = |y - S(x)|", fontsize=13)
        ax2.set_xlabel("Кумулятивна відстань (м)")
        ax2.set_ylabel("Похибка ε (м)")
        ax2.grid(True)
        
        for k in node_counts:
            indexes = np.linspace(0, len(data) - 1, k, dtype=int)
            x_sub = x_arr[indexes]
            y_sub = y_arr[indexes]
            
            coefs_sub = spline_coef(x_sub, y_sub)
            
            x_smooth = np.linspace(x_arr[0], x_arr[-1], 500)   
            y_smooth = [evaluate_spline(coefs_sub, x_val) for x_val in x_smooth]
            
            ax1.plot(x_smooth, y_smooth, label=f'Сплайн ({k} вузлів)')       
            y_approx = np.array([evaluate_spline(coefs_sub, x_val) for x_val in x_arr])
            
            err = np.abs(y_arr - y_approx)
            ax2.plot(x_arr, err, marker='o', markersize=4, label=f'Похибка ({k} вузлів)')
        ax1.legend()
        ax2.legend()
        
        plt.subplots_adjust(hspace=0.5)


#-----------------------------------------------------Useful info--------------------------------------------------------------------#

        print("\n" + "="*50)
        print("АНАЛІТИКА МАРШРУТУ")
        
        total_dis = x_arr[-1]
        total_ascent = sum(max(y_arr[i] - y_arr[i-1], 0) for i in range(1, len(data)))
        total_descent = sum(max(y_arr[i-1] - y_arr[i], 0) for i in range(1, len(data)))
        print(f"Загальна відстань:       {total_dis:.2f} м") 
        print(f"Сумарний набір висоти:   {total_ascent:.2f} м")
        print(f"Сумарний спуск:          {total_descent:.2f} м") 
        
        
        grads = np.gradient(y_arr, x_arr) * 100
        print(f"Максимальний підйом:     {np.max(grads):.2f} %")
        print(f"Максимальний спуск:      {np.min(grads):.2f} %")
        print(f"Середній градієнт:       {np.mean(np.abs(grads)):.2f} %")
        
        
        mass = 80   
        g = 9.81
        energy_j = mass * g * total_ascent
        energy_kj = energy_j / 1000
        energy_kcal = energy_j / 4184 
        print("\n--- Енерговитрати (для маси 80 кг) ---")
        print(f"Механічна робота:        {energy_j:.2f} Дж") 
        print(f"Механічна робота:        {energy_kj:.2f} кДж") 
        print(f"Енергія (у калоріях):    {energy_kcal:.2f} ккал")
        print("="*50 + "\n")
        plt.show()
    except:
        print("Unexpected error occured")