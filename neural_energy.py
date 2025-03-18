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


#%% Auxiliar functions

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


# %% Configuración e Hiperparámetros

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


#%% Definir la red neuronal
class SimpleNN(nn.Module):  # Hereda de nn.Module, la clase base para todas las redes en PyTorch
    def __init__(self):
        super(SimpleNN, self).__init__()  # Llama al constructor de la clase padre (nn.Module)
        
        # -> Entrada: 28*28 = 784 (porque las imágenes son de 28x28 píxeles)                                     
        # -> Salida: 128 neuronas
        self.fc1 = nn.Linear(28*28, 128)

        # -> Entrada: 128 (salida de la capa anterior)
        # -> Salida: 64 neuronas
        self.fc2 = nn.Linear(128, 64)     

        # -> Entrada: 64 (salida de la capa anterior)
        # Salida: 10 neuronas (una por cada clase/dígito del 0 al 9)
        self.fc3 = nn.Linear(64, 10)                  

        self.relu = nn.ReLU()             # Función de activación ReLU (Rectified Linear Unit)

    def forward(self, x):
        # Aplanar la imagen de 28x28 a un vector de 784 elementos
        # x.size(0) es el tamaño del lote (batch size), y -1 indica que se calcule automáticamente el tamaño
        x = x.view(x.size(0), -1)  # Transforma la imagen de (batch_size, 1, 28, 28) a (batch_size, 784)

        # Pasar los datos a través de las capas de la red
        x = self.relu(self.fc1(x))  # Capa 1 + ReLU
        x = self.relu(self.fc2(x))  # Capa 2 + ReLU
        x = self.fc3(x)             # Capa 3 (sin activación, ya que se usará para la salida)

        return x  # Devuelve la salida de la red (logits)

# Crear una instancia del modelo y moverlo al dispositivo (GPU o CPU)
model = SimpleNN().to(device)

