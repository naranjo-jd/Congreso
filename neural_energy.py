'''
Energy Neural Network Congreso
Authors: 
- Daniel Alejandro Posada Noguera
- Juan Diego Zapata Naranjo

'''

#%% Usual Libraries for numeric analysis and graphs
import networkx as nx
import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import pearsonr


#%% Libraries for Neural Network
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# %% Auxiliar functions for Numeric Analysis

# Función para normalizar los datos usando min-max scaling
def normalize(data):
    data_array = np.array(data)
    return (data_array - np.min(data_array)) / (np.max(data_array) - np.min(data_array))

# Función para graficar la pérdida vs la energía del grafo con gradiente
def plot_loss_vs_energy(loss, energy, log_scale=False):
    loss_array = normalize(loss)
    energy_array = normalize(energy)
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(loss_array)))
    plt.scatter(loss_array, energy_array, color=colors, label='Datos')
    plt.plot(loss_array, energy_array, color='blue', alpha=0.5, label='Curva de tendencia')
    plt.title('Gráfico de pérdida vs Energía del grafo')
    plt.xlabel('Función de pérdida (normalizada)')
    plt.ylabel('Energía del grafo (normalizada)')
    if log_scale:
        plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

# Función para calcular el error cuadrático medio (MSE)
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Función para entrenar una regresión lineal para la pérdida vs energía
def linear_regression(loss, energy):
    loss_array = normalize(loss)
    energy_array = normalize(energy)
    slope, intercept, r_value, _, _ = stats.linregress(loss_array, energy_array)
    predicted_energy = slope * loss_array + intercept
    mse = mean_squared_error(energy_array, predicted_energy)
    plt.scatter(loss_array, energy_array, color='blue', label='Datos')
    plt.plot(loss_array, predicted_energy, color='red', label='Regresión lineal')
    plt.title(f'Regresión Lineal - Pérdida vs Energía (R² = {r_value**2:.4f})')
    plt.xlabel('Función de pérdida (normalizada)')
    plt.ylabel('Energía del grafo (normalizada)')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f"Coeficiente de correlación (r): {r_value:.4f}")
    print(f"Error cuadrático medio (MSE): {mse:.4f}")

# Función para entrenar una regresión polinómica para la pérdida vs energía
def polynomial_regression(loss, energy, degree=2):
    loss_array = normalize(loss)
    energy_array = normalize(energy)
    p = Polynomial.fit(loss_array, energy_array, degree)
    predicted_energy = p(loss_array)
    mse = mean_squared_error(energy_array, predicted_energy)
    loss_fit = np.linspace(loss_array.min(), loss_array.max(), 500)
    energy_fit = p(loss_fit)
    plt.scatter(loss_array, energy_array, color='blue', label='Datos')
    plt.plot(loss_fit, energy_fit, color='green', label=f'Regresión Polinómica (grado {degree})')
    plt.title(f'Regresión Polinómica - Pérdida vs Energía (grado {degree})')
    plt.xlabel('Función de pérdida (normalizada)')
    plt.ylabel('Energía del grafo (normalizada)')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f"Coeficientes del polinomio: {p.convert().coef}")
    print(f"Error cuadrático medio (MSE): {mse:.4f}")

# Función exponencial para el ajuste
def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c

# Función para entrenar una regresión exponencial para la pérdida vs energía
def exponential_regression(loss, energy):
    loss_array = normalize(loss)
    energy_array = normalize(energy)
    popt, _ = curve_fit(exponential_func, loss_array, energy_array, maxfev=5000)
    a, b, c = popt
    predicted_energy = exponential_func(loss_array, a, b, c)
    mse = mean_squared_error(energy_array, predicted_energy)
    plt.scatter(loss_array, energy_array, color='blue', label='Datos')
    loss_fit = np.linspace(loss_array.min(), loss_array.max(), 500)
    energy_fit = exponential_func(loss_fit, a, b, c)
    plt.plot(loss_fit, energy_fit, color='green', label='Ajuste exponencial')
    plt.title('Regresión Exponencial - Pérdida vs Energía')
    plt.xlabel('Función de pérdida (normalizada)')
    plt.ylabel('Energía del grafo (normalizada)')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f"Parámetros del ajuste exponencial: a = {a:.4f}, b = {b:.4f}, c = {c:.4f}")
    print(f"Error cuadrático medio (MSE): {mse:.4f}")

# Función para calcular la correlación entre pérdida y energía
def correlation_analysis(loss, energy):
    loss_array = normalize(loss)
    energy_array = normalize(energy)
    corr_coefficient, p_value = pearsonr(loss_array, energy_array)
    print(f"Coeficiente de correlación de Pearson entre Pérdida y Energía: {corr_coefficient:.4f}")
    print(f"Valor p (significancia estadística): {p_value:.4e}")


# %% Auxiliar functions for Neural Network

# -------------------------------------------------------------------
# Diferentes formas de calcular matrices asociadas a redes neuronales
# -------------------------------------------------------------------

# Función para calcular la matriz de adyacencia de una red neuronal
def red_a_matriz_adyacencia(modelo):
    capas = [layer for layer in modelo.children() if isinstance(layer, nn.Linear)]

    total_neuronas = sum(layer.weight.shape[1] for layer in capas) + capas[-1].weight.shape[0]
    A = np.zeros((total_neuronas, total_neuronas))

    idx_inicial = 0
    idx_final = capas[0].weight.shape[1]  

    for layer in capas:
        n_out, n_in = layer.weight.shape  
        W = np.abs(layer.weight.detach().cpu().numpy())

        A[idx_inicial : idx_final, idx_final : idx_final + n_out] = W.T

        idx_inicial = idx_final
        idx_final += n_out

    return A

# Función para calcular la matriz de bipartita de una red neuronal, tomando cada capa como un conjunto de nodos
def red_a_matriz_bipartita(modelo):
    capas = [layer for layer in modelo.children() if isinstance(layer, torch.nn.Linear)]

    # Determinar el tamaño total de la matriz
    n_total = sum(layer.weight.shape[1] + layer.weight.shape[0] for layer in capas)
    
    # Crear matriz global vacía
    A_global = np.zeros((n_total, n_total))
    
    idx = 0
    for layer in capas:
        W = layer.weight.detach().cpu().numpy()
        n_in, n_out = W.shape

        # Colocar la matriz bipartita en la posición adecuada
        A_global[idx : idx + n_in, idx + n_in : idx + n_in + n_out] = np.abs(W)
        A_global[idx + n_in : idx + n_in + n_out, idx : idx + n_in] = np.abs(W.T)
        
        idx += n_in

    return A_global

# Función para calcular la matriz Laplaciana de una red neuronal 
def red_a_matriz_laplaciana(modelo):
    A = red_a_matriz_adyacencia(modelo)  # Reutiliza la función de adyacencia
    D = np.diag(A.sum(axis=1))          # Matriz diagonal de grados
    L = D - A                           # L = D - A
    return L

# ---------------------------------------------------------------------------------
# Diferentes formas de calcular energías de los grafos asociados a redes neuronales
# ---------------------------------------------------------------------------------

# Función para calcular la energía de una red neuronal a partir de su matriz de adyacencia (Estandar)
def calcular_energia_estandar_ady(modelo):
    A = red_a_matriz_adyacencia(modelo)
    eigenvalues = np.linalg.eigvals(A)
    return np.sum(np.abs(eigenvalues))

# Función para calcular la energía de una red neuronal a partir de su matriz bipartita (Estandar)
def calcular_energia_estandar_bip(modelo):
    A = red_a_matriz_bipartita(modelo)
    eigenvalues = np.linalg.eigvals(A)
    return np.sum(np.abs(eigenvalues))

# Función para calcular la energía de una red neuronal a partir de su matriz de Laplaciana
def calcular_energia_laplaciana(modelo):
    L = red_a_matriz_laplaciana(modelo)
    eigenvalues = np.linalg.eigvals(L)
    return np.sum(np.abs(eigenvalues))


# %% Configuración e Hiperparámetros para Pytorch

# Selecciona "cuda" (GPU) si está disponible, de lo contrario usa "cpu" (CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hiperparámetros del entrenamiento
batch_size = 64       # Número de muestras por lote (batch)
learning_rate = 0.01  # Tasa de aprendizaje para el optimizador
epochs = 10           # Número de veces que el modelo verá todo el conjunto de datos

# Transformaciones para preprocesar los datos
transform = transforms.Compose([
    transforms.ToTensor(),              # Convierte la imagen en un tensor y escala los valores a [0, 1]
    transforms.Normalize((0.5,), (0.5,)) # Normaliza los valores al rango [-1, 1]
])

# Cargar el conjunto de datos MNIST: Un conjunto de datos de dígitos escritos a mano (0-9).
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

# Crear un DataLoader para manejar los datos en lotes
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


#%% Definir la red neuronal de 3 capas para el experimento (SimpleNN)
class SimpleNN(nn.Module): 
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Aplanar la imagen de 28x28 a vector de 784 elementos
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNN().to(device)

#%% Entrenamiento de la red neuronal y cálculo de las energías
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
criterion = nn.CrossEntropyLoss()

loss_history = []
lr_history = []
energy_ady_history = []
energy_bip_history = []
energy_lapl_history = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Calcular las energías usando las funciones proporcionadas
    energia_ady = calcular_energia_estandar_ady(model)
    energia_bip = calcular_energia_estandar_bip(model)
    energia_lapl = calcular_energia_laplaciana(model)
    
    energy_ady_history.append(energia_ady)
    energy_bip_history.append(energia_bip)
    energy_lapl_history.append(energia_lapl)
    
    lr_history.append(scheduler.get_last_lr()[0])
    loss_history.append(total_loss / len(train_loader))
    
    print(f"Época {epoch+1}/{epochs}, Pérdida: {loss_history[-1]:.4f}, "
          f"Energía Ady: {energia_ady:.2f}, Energía Bip: {energia_bip:.2f}, "
          f"Energía Lapl: {energia_lapl:.2f}, LR: {lr_history[-1]:.6f}")
    
    scheduler.step()


#Análisis y Visualización: Pérdida vs Energías

# %% Matriz de adyacencia estándar
print("Análisis para Energía (Matriz de Adyacencia Estándar)")
plot_loss_vs_energy(loss_history, energy_ady_history, log_scale=True)
linear_regression(loss_history, energy_ady_history)
polynomial_regression(loss_history, energy_ady_history, degree=2)
exponential_regression(loss_history, energy_ady_history)  
correlation_analysis(loss_history, energy_ady_history)


# %% Para la energía de la matriz bipartita
print("Análisis para Energía (Matriz Bipartita)")
plot_loss_vs_energy(loss_history, energy_bip_history)
linear_regression(loss_history, energy_bip_history)
polynomial_regression(loss_history, energy_bip_history, degree=2)
exponential_regression(loss_history, energy_bip_history)  
correlation_analysis(loss_history, energy_bip_history)


# %% Para la energía de la matriz laplaciana
print("Análisis para Energía (Matriz Laplaciana)")
plot_loss_vs_energy(loss_history, energy_lapl_history)
linear_regression(loss_history, energy_lapl_history)
polynomial_regression(loss_history, energy_lapl_history, degree=2)
exponential_regression(loss_history, energy_lapl_history) 
correlation_analysis(loss_history, energy_lapl_history)


# %%
