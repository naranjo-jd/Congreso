'''
Energy Neural Network Congreso
Author: 
- Daniel Alejandro Posada Noguera
- Juan Diego Zapata Naranjo

'''

#%% Usual Libraries for numeric analysis and graphs
import networkx as nx
import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
from scipy import stats


#%% Libraries for Neural Network
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# %% Auxiliar functions for Numeric Analysis

# Function to plot the loss vs energy
def plot_loss_vs_energy(loss, energy):
    plt.plot(loss, energy, color='blue', label='Función de pérdida vs Energía del grafo')
    plt.title('Gráfico de pérdida vs Energía del grafo')
    plt.xlabel('Función de pérdida')
    plt.ylabel('Energía del grafo')
    plt.legend()
    plt.grid(True)
    plt.show()

#  Function to train a linear regresion for the loss vs energy
def linear_regression(loss, energy):
    slope, intercept, r_value, _, _ = stats.linregress(loss, energy)
    plt.scatter(loss, energy, color='blue', label='Datos')
    plt.plot(loss, slope * loss + intercept, color='red', label='Regresión lineal')
    plt.title(f'Regresión Lineal - Pérdida vs Energía (R² = {r_value**2:.4f})')
    plt.xlabel('Función de pérdida')
    plt.ylabel('Energía del grafo')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f"Coeficiente de correlación (r): {r_value:.4f}")

# Function to train a polynomial regresion for the loss vs energy
def polynomial_regression(loss, energy, degree=2):
    p = Polynomial.fit(loss, energy, degree)
    loss_fit = np.linspace(min(loss), max(loss), 500)
    energy_fit = p(loss_fit)
    plt.scatter(loss, energy, color='blue', label='Datos')
    plt.plot(loss_fit, energy_fit, color='green', label=f'Regresión Polinómica (grado {degree})')
    plt.title(f'Regresión Polinómica - Pérdida vs Energía (grado {degree})')
    plt.xlabel('Función de pérdida')
    plt.ylabel('Energía del grafo')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f"Coeficientes del polinomio: {p.convert().coef}")

# Function to calculate the correlation between loss and energy
def correlation_analysis(loss, energy):
    corr_coefficient = np.corrcoef(loss, energy)[0, 1]
    print(f"Coeficiente de correlación de Pearson entre Pérdida y Energía: {corr_coefficient:.4f}")


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
def calcular_energia_estandar(modelo):
    A = red_a_matriz_adyacencia(modelo)
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


#%% Funciones para calcular las energías de las redes basada en los pesos de las capas
def calcular_energia(modelo):
    energia_total = 0
    # Recorremos cada capa lineal
    for layer in [modelo.fc1, modelo.fc2, modelo.fc3]:
        # Extraer los pesos y convertir a numpy
        W = layer.weight.detach().cpu().numpy()  # W de forma (n_in, n_out)
        # Construir la matriz bipartita A_bi:
        n_in, n_out = W.shape
        A_top = np.hstack([np.zeros((n_in, n_in)), np.abs(W)])
        A_bottom = np.hstack([np.abs(W).T, np.zeros((n_out, n_out))])
        A_bi = np.vstack([A_top, A_bottom])
        
        # Calcular la matriz de grados y el Laplaciano L = D - A_bi
        D = np.diag(np.sum(A_bi, axis=1))
        L = D - A_bi
        
        # Calcular los autovalores y sumar sus valores absolutos
        eigenvalues = np.linalg.eigvals(L)
        energia_total += np.sum(np.abs(eigenvalues))
    return energia_total