import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import tkinter as tk
from tkinter import ttk
import json

class ChemicalHazardSimulation:
    def __init__(self):
        self.X0 = np.zeros(11)  # Начальные данные (X1-X13 и X0)
        self.F = np.zeros(7)     # Влияющие факторы (F1-F7)
        self.k = np.zeros((43, 4))  # Коэффициенты для полиномов (43 полинома)
        self.Xmax = np.zeros(11)  # Нормировочные множители (Xmax1-Xmax11)

    def set_initial_conditions(self, X0):
        self.X0 = X0

    def set_external_factors(self, F):
        self.F = F

    def set_coefficients(self, k):
        self.k = k

    def set_Xmax(self, Xmax):
        self.Xmax = Xmax

    def system_ode(self, X, t):
        dXdt = np.zeros(11)
        
        
        # Уравнение для dX1/dt
        f1_F1 = self.k[0, 0] * self.F[0]**3 + self.k[0, 1] * self.F[0]**2 + self.k[0, 2] * self.F[0] + self.k[0, 3]
        f1_X9 = self.k[1, 0] * X[8]**3 + self.k[1, 1] * X[8]**2 + self.k[1, 2] * X[8] + self.k[1, 3]
        f1_X10 = self.k[2, 0] * X[9]**3 + self.k[2, 1] * X[9]**2 + self.k[2, 2] * X[9] + self.k[2, 3]
        f1_F3 = self.k[3, 0] * self.F[2]**3 + self.k[3, 1] * self.F[2]**2 + self.k[3, 2] * self.F[2] + self.k[3, 3]
        f1_F4 = self.k[4, 0] * self.F[3]**3 + self.k[4, 1] * self.F[3]**2 + self.k[4, 2] * self.F[3] + self.k[4, 3]

        dXdt[0] = (1/self.Xmax[0]) * (f1_F1 - f1_X9 * f1_X10 * (f1_F3 + f1_F4))

        # Уравнение для dX2/dt
        f2_X3 = self.k[5, 0] * X[2]**3 + self.k[5, 1] * X[2]**2 + self.k[5, 2] * X[2] + self.k[5, 3]
        f2_X7 = self.k[6, 0] * X[6]**3 + self.k[6, 1] * X[6]**2 + self.k[6, 2] * X[6] + self.k[6, 3]
        f2_X8 = self.k[7, 0] * X[7]**3 + self.k[7, 1] * X[7]**2 + self.k[7, 2] * X[7] + self.k[7, 3]
        f2_F1 = self.k[8, 0] * self.F[0]**3 + self.k[8, 1] * self.F[0]**2 + self.k[8, 2] * self.F[0] + self.k[8, 3]
        f2_F5 = self.k[9, 0] * self.F[4]**3 + self.k[9, 1] * self.F[4]**2 + self.k[9, 2] * self.F[4] + self.k[9, 3]
        f2_X9 = self.k[10, 0] * X[8]**3 + self.k[10, 1] * X[8]**2 + self.k[10, 2] * X[8] + self.k[10, 3]
        f2_X10 = self.k[11, 0] * X[9]**3 + self.k[11, 1] * X[9]**2 + self.k[11, 2] * X[9] + self.k[11, 3]
        
        dXdt[1] = (1/self.Xmax[1]) * (f2_X3 * f2_X7 * f2_X8 * (f2_F1 + f2_F5) - f2_X9 * f2_X10)

        # Уравнение для dX3/dt
        f3_X1 = self.k[12, 0] * X[0]**3 + self.k[12, 1] * X[0]**2 + self.k[12, 2] * X[0] + self.k[12, 3]
        f3_F1 = self.k[13, 0] * self.F[0]**3 + self.k[13, 1] * self.F[0]**2 + self.k[13, 2] * self.F[0] + self.k[13, 3]
        f3_F4 = self.k[14, 0] * self.F[3]**3 + self.k[14, 1] * self.F[3]**2 + self.k[14, 2] * self.F[3] + self.k[14, 3]
        f3_F3 = self.k[15, 0] * self.F[2]**3 + self.k[15, 1] * self.F[2]**2 + self.k[15, 2] * self.F[2] + self.k[15, 3]
        
        dXdt[2] = (1/self.Xmax[2]) * (f3_X1 * (f3_F1 + f3_F4) - f3_F3)

        # Уравнение для dX4/dt
        f4_X1 = self.k[16, 0] * X[0]**3 + self.k[16, 1] * X[0]**2 + self.k[16, 2] * X[0] + self.k[16, 3]
        f4_F1 = self.k[17, 0] * self.F[0]**3 + self.k[17, 1] * self.F[0]**2 + self.k[17, 2] * self.F[0] + self.k[17, 3]
        f4_F3 = self.k[18, 0] * self.F[2]**3 + self.k[18, 1] * self.F[2]**2 + self.k[18, 2] * self.F[2] + self.k[18, 3]
        f4_F4 = self.k[19, 0] * self.F[3]**3 + self.k[19, 1] * self.F[3]**2 + self.k[19, 2] * self.F[3] + self.k[19, 3]
        
        dXdt[3] = (1/self.Xmax[3]) * (f4_X1 - (f4_F1 + f4_F3 + f4_F4))

        # Уравнение для dX5/dt
        f5_X1 = self.k[20, 0] * X[0]**3 + self.k[20, 1] * X[0]**2 + self.k[20, 2] * X[0] + self.k[20, 3]
        f5_F2 = self.k[21, 0] * self.F[1]**3 + self.k[21, 1] * self.F[1]**2 + self.k[21, 2] * self.F[1] + self.k[21, 3]
        
        dXdt[4] = (1/self.Xmax[4]) * (f5_X1 - f5_F2)

        # Уравнение для dX6/dt
        f6_X2 = self.k[22, 0] * X[1]**3 + self.k[22, 1] * X[1]**2 + self.k[22, 2] * X[1] + self.k[22, 3]
        f6_F5 = self.k[23, 0] * self.F[4]**3 + self.k[23, 1] * self.F[4]**2 + self.k[23, 2] * self.F[4] + self.k[23, 3]
        f6_F6 = self.k[24, 0] * self.F[5]**3 + self.k[24, 1] * self.F[5]**2 + self.k[24, 2] * self.F[5] + self.k[24, 3]
        f6_X4 = self.k[25, 0] * X[3]**3 + self.k[25, 1] * X[3]**2 + self.k[25, 2] * X[3] + self.k[25, 3]
        f6_X10 = self.k[26, 0] * X[9]**3 + self.k[26, 1] * X[9]**2 + self.k[26, 2] * X[9] + self.k[26, 3]
        f6_X11 = self.k[27, 0] * X[10]**3 + self.k[27, 1] * X[10]**2 + self.k[27, 2] * X[10] + self.k[27, 3]
        f6_F7 = self.k[28, 0] * self.F[6]**3 + self.k[28, 1] * self.F[6]**2 + self.k[28, 2] * self.F[6] + self.k[28, 3]
        
        dXdt[5] = (1/self.Xmax[5]) * (f6_X2 * (f6_F5 + f6_F6) - f6_X4 * f6_X10 * f6_X11 * f6_F7)

        # Уравнение для dX7/dt
        f7_X5 = self.k[29, 0] * X[4]**3 + self.k[29, 1] * X[4]**2 + self.k[29, 2] * X[4] + self.k[29, 3]
        f7_X6 = self.k[30, 0] * X[5]**3 + self.k[30, 1] * X[5]**2 + self.k[30, 2] * X[5] + self.k[30, 3]
        f7_X10 = self.k[31, 0] * X[9]**3 + self.k[31, 1] * X[9]**2 + self.k[31, 2] * X[9] + self.k[31, 3]
        
        dXdt[6] = (1/self.Xmax[6]) * (f7_X5 * f7_X6 * f7_X10)

        # Уравнение для dX8/dt
        f8_X3 = self.k[32, 0] * X[2]**3 + self.k[32, 1] * X[2]**2 + self.k[32, 2] * X[2] + self.k[32, 3]
        f8_X9 = self.k[33, 0] * X[8]**3 + self.k[33, 1] * X[8]**2 + self.k[33, 2] * X[8] + self.k[33, 3]
        f8_X10 = self.k[34, 0] * X[9]**3 + self.k[34, 1] * X[9]**2 + self.k[34, 2] * X[9] + self.k[34, 3]
        
        dXdt[7] = (1/self.Xmax[7]) * (f8_X3 - f8_X9 * f8_X10)

        # Уравнение для dX9/dt
        f9_X3 = self.k[35, 0] * X[2]**3 + self.k[35, 1] * X[2]**2 + self.k[35, 2] * X[2] + self.k[35, 3]
        f9_X8 = self.k[36, 0] * X[7]**3 + self.k[36, 1] * X[7]**2 + self.k[36, 2] * X[7] + self.k[36, 3]
        
        dXdt[8] = (1/self.Xmax[8]) * (f9_X3 * f9_X8)

        # Уравнение для dX10/dt
        f10_X3 = self.k[37, 0] * X[2]**3 + self.k[37, 1] * X[2]**2 + self.k[37, 2] * X[2] + self.k[37, 3]
        f10_F6 = self.k[38, 0] * self.F[5]**3 + self.k[38, 1] * self.F[5]**2 + self.k[38, 2] * self.F[5] + self.k[38, 3]
        
        dXdt[9] = (1/self.Xmax[9]) * (f10_X3 * f10_F6)

        # Уравнение для dX11/dt
        f11_X10 = self.k[39, 0] * X[9]**3 + self.k[39, 1] * X[9]**2 + self.k[39, 2] * X[9] + self.k[39, 3]
        f11_F6 = self.k[40, 0] * self.F[5]**3 + self.k[40, 1] * self.F[5]**2 + self.k[40, 2] * self.F[5] + self.k[40, 3]
        f11_F7 = self.k[41, 0] * self.F[6]**3 + self.k[41, 1] * self.F[6]**2 + self.k[41, 2] * self.F[6] + self.k[41, 3]
        f11_F5 = self.k[42, 0] * self.F[4]**3 + self.k[42, 1] * self.F[4]**2 + self.k[42, 2] * self.F[4] + self.k[42, 3]
        
        dXdt[10] = (1/self.Xmax[10]) * (f11_X10 * (f11_F6 + f11_F7) - f11_F5)

        return dXdt

    def solve_system(self, t_span):
        sol = odeint(self.system_ode, self.X0, t_span)
        return sol

    def visualize_results(self, t_span, sol):
        fig, axs = plt.subplots(4, 3, figsize=(15, 15))
        axs = axs.ravel()

        for i in range(11):
            axs[i].plot(t_span, sol[:, i])
            axs[i].set_title(f'X{i+1}')
            axs[i].set_xlabel('t')
            axs[i].set_ylabel('значение')         
            
        plt.subplots_adjust(hspace=0.5)
        plt.tight_layout()
        plt.show()

        
        # Общая диаграмма
        plt.figure(figsize=(15, 10))
        for i in range(11):
            plt.plot(t_span, sol[:, i], label=f'X{i+1}')
            plt.title('Общая диаграмма для X1-X11')
            plt.xlabel('t')
            plt.ylabel('значение')
            
        plt.legend()
        plt.grid()
        plt.show()

         # Круговые диаграммы для каждой переменной
        for i in range(11):
            plt.figure(figsize=(6, 6))
            plt.pie(sol[-1, :], labels=[f'X{j+1}' for j in range(11)], autopct='%1.1f%%')
            plt.title(f'Круговая диаграмма для X{i+1}')
        plt.show()

    def save_parameters(self, filename):
        params = {
            'X0': self.X0.tolist(),
            'F': self.F.tolist(),
            'k': self.k.tolist(),
            'Xmax': self.Xmax.tolist()
        }
        with open(filename, 'w') as f:
            json.dump(params, f)

    def load_parameters(self, filename):
        with open(filename, 'r') as f:
            params = json.load(f)
        self.X0 = np.array(params['X0'])
        self.F = np.array(params['F'])
        self.k = np.array(params['k'])
        self.Xmax = np.array(params['Xmax'])

class SimulationGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Химическое происшествие: симуляция")
        self.sim = ChemicalHazardSimulation()

        self.create_widgets()

    def create_widgets(self):
        # Начальные параметры (X)
        ttk.Label(self.master, text="Начальные параметры (X1-X11):").grid(row=0, column=0, sticky='w')
        self.X0_entries = []
        for i in range(11):
            ttk.Label(self.master, text=f"X{i+1}:").grid(row=i+1, column=0, sticky='e')
            entry = ttk.Entry(self.master)
            entry.grid(row=i+1, column=1)
            self.X0_entries.append(entry)

        # Внешние факторы (F)
        ttk.Label(self.master, text="Внешние факторы (F):").grid(row=0, column=2, sticky='w')
        self.F_entries = []
        for i in range(7):
            ttk.Label(self.master, text=f"F{i+1}:").grid(row=i+1, column=2, sticky='e')
            entry = ttk.Entry(self.master)
            entry.grid(row=i+1, column=3)
            self.F_entries.append(entry)


          # Нормировочные множители (Xmax)
        ttk.Label(self.master, text="Нормировочные множители (Xmax):").grid(row=0, column=14, sticky='w')
        self.Xmax_entries = []
        for i in range(11):  # 11 Xmax
            ttk.Label(self.master, text=f"Xmax{i+1}:").grid(row=i+1, column=14, sticky='e')
            entry = ttk.Entry(self.master)
            entry.grid(row=i+1, column=15)
            self.Xmax_entries.append(entry)

            
        # Коэффициенты для полиномов
        self.k_entries = []
        polynom_label_names = ["f1(F1):","-f1(X9):","-f1(X10)","-f1(F3)","-f1(F4)","f2(X3)","f2(X7)","f2(X8)","f2(F1)","f2(F5)","-f2(X9)","-f2(X10)","f3(X1)","f3(F1)","f3(F4)","-f3(F3)","f4(X1)","-f4(F1)","-f4(F3)","-f4(F4)","f5(X1)","f5(F2)","f6(X2)","f6(F5)","f6(F6)","-f6(X4)","-f6(X10)","-f6(X11)","-f6(F7)","f7(X5)","f7(X6)","f7(10)","f8(X3)","-f8(X9)","-f8(X10)","f9(X3)","f9(X8)","f10(X3)","f10(F6)","f11(X10)","f11(F6)","f11(F7)","-f11(F5)"]
        for i in range(43):  # 43 полинома
            ttk.Label(self.master, text=polynom_label_names[i]).grid(row=i+1, column=4, sticky='e')
            for j in range(4):
                entry = ttk.Entry(self.master)
                entry.grid(row=i+1, column=5+j*2)
                self.k_entries.append(entry)

      

        # Buttons
        ttk.Button(self.master, text="Запустить симуляцию", command=self.run_simulation).grid(row=46, column=0, columnspan=2)
        ttk.Button(self.master, text="Сохранить параметры", command=self.save_params).grid(row=46, column=2, columnspan=2)
        ttk.Button(self.master, text="Загрузить параметры", command=self.load_params).grid(row=46, column=4, columnspan=2)

    def run_simulation(self):
        # Get values from entries
        X0 = [float(entry.get()) for entry in self.X0_entries]
        F = [float(entry.get()) for entry in self.F_entries]
        k = np.array([float(entry.get()) for entry in self.k_entries]).reshape(43, 4)
        Xmax = [float(entry.get()) for entry in self.Xmax_entries]

        # Set values in simulation
        self.sim.set_initial_conditions(X0)
        self.sim.set_external_factors(F)
        self.sim.set_coefficients(k)
        self.sim.set_Xmax(Xmax)

        # Запускаем симуляцию
        t_span = np.linspace(0, 1, 10)
        self.sim.sol = self.sim.solve_system(t_span)

        # Визуализируем результаты
        self.sim.visualize_results(t_span, self.sim.sol)

    def save_params(self):
        self.sim.save_parameters('simulation_params.json')

    def load_params(self):
        self.sim.load_parameters('simulation_params.json')
        # Update GUI with loaded parameters
        for i, entry in enumerate(self.X0_entries):
            entry.delete(0, tk.END)
            entry.insert(0, str(self.sim.X0[i]))
        for i, entry in enumerate(self.F_entries):
            entry.delete(0, tk.END)
            entry.insert(0, str(self.sim.F[i]))
        for i, entry in enumerate(self.k_entries):
            entry.delete(0, tk.END)
            entry.insert(0, str(self.sim.k.flatten()[i]))
        for i, entry in enumerate(self.Xmax_entries):
            entry.delete(0, tk.END)
            entry.insert(0, str(self.sim.Xmax[i]))

if __name__ == "__main__":
    root = tk.Tk()
    app = SimulationGUI(root)
    root.mainloop()
