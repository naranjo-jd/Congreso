'''
Energy Neural Network Congreso
Authors: 
- Daniel Alejandro Posada Noguera
- Juan Diego Zapata Naranjo

'''

#%% Usual Libraries for numeric analysis and graphs
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

def plot_loss_vs_energy(loss_history, energy_history, title, ax=None, 
                        marker="o", color="C0", show_regression=False):
    """
    Grafica Pérdida vs Energía en un eje dado (o crea uno nuevo si ax=None).
    - loss_history: lista o arreglo de pérdidas (eje X).
    - energy_history: lista o arreglo de energías (eje Y).
    - title: título de la gráfica.
    - ax: objeto Axes de matplotlib donde dibujar (opcional).
    - marker, color: opciones de estilo para los puntos.
    - show_regression: si es True, ajusta y dibuja una regresión lineal simple.
    """
    # Si no se proporciona un eje, crear una figura nueva
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        created_fig = True

    # Convertir a numpy arrays
    x = np.array(loss_history)
    y = np.array(energy_history)

    # Dibujar puntos
    ax.scatter(x, y, marker=marker, color=color, label="Datos", alpha=0.8)

    # (Opcional) Ajuste lineal simple y línea resultante
    if show_regression:
        # fit polinomial de grado 1
        coef = np.polyfit(x, y, 1)
        line_x = np.linspace(x.min(), x.max(), 100)
        line_y = np.polyval(coef, line_x)
        ax.plot(line_x, line_y, color=color, linestyle="--",
                label=f"Regresión (y={coef[0]:.3f}·x+{coef[1]:.3f})")

    # Etiquetas y título
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("Pérdida", fontsize=12)
    ax.set_ylabel("Energía", fontsize=12)

    # Grilla y leyenda
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=10)

    # Ajuste de márgenes
    if created_fig:
        plt.tight_layout()
        plt.show()

# Función para entrenar una regresión lineal para la energía vs pérdida
def linear_regression(loss, energy, network_name):
    loss_array = normalize(loss)
    energy_array = normalize(energy)
    slope, intercept, r_value, _, _ = stats.linregress(energy_array, loss_array)
    predicted_loss = slope * energy_array + intercept
    plt.scatter(energy_array, loss_array, color='blue', label='Datos')
    plt.plot(energy_array, predicted_loss, color='red', label='Regresión lineal')
    plt.title(f'{network_name} - Regresión Lineal (R² = {r_value**2:.4f})')
    plt.xlabel('Energía del grafo (normalizada)')
    plt.ylabel('Función de pérdida (normalizada)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Función para entrenar una regresión polinómica para la energía vs pérdida
def polynomial_regression(loss, energy, network_name, degree=2):
    loss_array = normalize(loss)
    energy_array = normalize(energy)
    p = Polynomial.fit(energy_array, loss_array, degree)
    predicted_loss = p(energy_array)
    ss_total = np.sum((loss_array - np.mean(loss_array)) ** 2)
    ss_residual = np.sum((loss_array - predicted_loss) ** 2)
    r_squared = 1 - (ss_residual / ss_total)

    energy_fit = np.linspace(energy_array.min(), energy_array.max(), 500)
    loss_fit = p(energy_fit)
    plt.scatter(energy_array, loss_array, color='blue', label='Datos')
    plt.plot(energy_fit, loss_fit, color='green', label=f'Regresión Polinómica (grado {degree})')
    plt.title(f'{network_name} - Regresión Polinómica (grado {degree}, R² = {r_squared:.4f})')
    plt.xlabel('Energía del grafo (normalizada)')
    plt.ylabel('Función de pérdida (normalizada)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Función exponencial para el ajuste
def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c

# Función para entrenar una regresión exponencial para la energía vs pérdida
def exponential_regression(loss, energy, network_name):
    loss_array = normalize(loss)
    energy_array = normalize(energy)
    popt, _ = curve_fit(exponential_func, energy_array, loss_array, maxfev=5000)
    a, b, c = popt
    predicted_loss = exponential_func(energy_array, a, b, c)
    ss_total = np.sum((loss_array - np.mean(loss_array)) ** 2)
    ss_residual = np.sum((loss_array - predicted_loss) ** 2)
    r_squared = 1 - (ss_residual / ss_total)

    plt.scatter(energy_array, loss_array, color='blue', label='Datos')
    energy_fit = np.linspace(energy_array.min(), energy_array.max(), 500)
    loss_fit = exponential_func(energy_fit, a, b, c)
    plt.plot(energy_fit, loss_fit, color='green', label=f'Ajuste exponencial (R² = {r_squared:.4f})')
    plt.title(f'{network_name} - Regresión Exponencial')
    plt.xlabel('Energía del grafo (normalizada)')
    plt.ylabel('Función de pérdida (normalizada)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Función para calcular la correlación entre energía y pérdida
def correlation_analysis(loss, energy):
    loss_array = normalize(loss)
    energy_array = normalize(energy)
    corr_coefficient, p_value = pearsonr(energy_array, loss_array)
    print(f"Coeficiente de correlación de Pearson entre Energía y Pérdida: {corr_coefficient:.4f}")
    print(f"Valor p (significancia estadística): {p_value:.4e}")

def plot_energy_vs_loss_with_stats(
    loss1, energy1,
    loss2, energy2,
    energy_label="Energía",
    network1_name="SimpleNN",
    network2_name="AdvancedNN"
):
    """
    Grafica en una misma figura:
      - Puntos de (loss1, energy1) (Red1) en color C0
      - Puntos de (loss2, energy2) (Red2) en color C1
      - Ajuste polinómico grado 3 separado en cada conjunto (normalizado)
      - Calcula y muestra en la leyenda:
          * coeficiente de correlación de Pearson (r)
          * coeficiente de determinación (R²) del polinomio grado 3
    Parámetros:
      loss1, energy1   : listas o arrays (raw) de pérdida/energía para Red Simple
      loss2, energy2   : listas o arrays (raw) de pérdida/energía para Red Advanced
      energy_label     : texto para poner en el eje Y y en el título
      network1_name    : etiqueta para la Red 1 en la leyenda
      network2_name    : etiqueta para la Red 2 en la leyenda
    """

    # 1) Normalizar pérdidas y energías (Min-Max) para cada red
    x1 = normalize(loss1)     # pérdida normalizada Red Simple
    y1 = normalize(energy1)   # energía normalizada Red Simple

    x2 = normalize(loss2)     # pérdida normalizada Red Advanced
    y2 = normalize(energy2)   # energía normalizada Red Advanced

    # 2) Calcular coeficiente de correlación Pearson (r) para cada red
    r1, _ = pearsonr(x1, y1)
    r2, _ = pearsonr(x2, y2)

    # 3) Ajuste polinómico grado 3 para cada red
    coefs1 = np.polyfit(x1, y1, deg=2)      # coefs [a3, a2, a1, a0]
    poly1  = np.poly1d(coefs1)
    y1_pred = poly1(x1)
    ss_tot1 = np.sum((y1 - np.mean(y1))**2)
    ss_res1 = np.sum((y1 - y1_pred)**2)
    r2_1    = 1 - (ss_res1 / ss_tot1)

    coefs2 = np.polyfit(x2, y2, deg=2)
    poly2  = np.poly1d(coefs2)
    y2_pred = poly2(x2)
    ss_tot2 = np.sum((y2 - np.mean(y2))**2)
    ss_res2 = np.sum((y2 - y2_pred)**2)
    r2_2    = 1 - (ss_res2 / ss_tot2)

    # 4) Crear la figura
    plt.figure(figsize=(7, 5))

    # 5) Dibujar puntos de cada red
    plt.scatter(
        x1, y1,
        marker='o', color='C0', alpha=0.8,
        label=f'{network1_name} (datos)'
    )
    plt.scatter(
        x2, y2,
        marker='s', color='C1', alpha=0.8,
        label=f'{network2_name} (datos)'
    )

    # 6) Dibujar curva polinómica grado 3 para cada red
    x_fit = np.linspace(0, 1, 200)  # rango de normalización [0,1]

    y1_fit = poly1(x_fit)
    plt.plot(
        x_fit, y1_fit,
        linestyle='--', color='C0', linewidth=2,
        label=f'{network1_name} (poly3, R²={r2_1:.3f}, r={r1:.3f})'
    )

    y2_fit = poly2(x_fit)
    plt.plot(
        x_fit, y2_fit,
        linestyle='--', color='C1', linewidth=2,
        label=f'{network2_name} (poly3, R²={r2_2:.3f}, r={r2:.3f})'
    )

    # 7) Etiquetas, título, grilla y leyenda
    plt.title(f"Pérdida vs {energy_label}", fontsize=14, pad=10)
    plt.xlabel("Pérdida (normalizada)", fontsize=12)
    plt.ylabel(f"{energy_label} (normalizada)", fontsize=12)
    plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


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
    n, m = A.shape
    D = np.diag(A.sum(axis=1))          # Matriz diagonal de grados
    L = D - A                           # L = D - A
    L = L - (2 * m / n) * np.eye(n)      # Ajuste para que la matriz sea positiva semidefinida
    return L

# ---------------------------------------------------------------------------------
# Diferentes formas de calcular energías de los grafos asociados a redes neuronales
# ---------------------------------------------------------------------------------

# Función para calcular la energía a partir de una matriz de adyacencia (Estandar)
def calcular_energia_estandar_ady_matriz(A):
    eigenvalues = np.linalg.eigvals(A)
    return np.sum(np.abs(eigenvalues))

# Función para calcular la energía a partir de una matriz bipartita (Estandar)
def calcular_energia_estandar_bip_matriz(A):
    eigenvalues = np.linalg.eigvals(A)
    return np.sum(np.abs(eigenvalues))

# Función para calcular la energía a partir de una matriz Laplaciana
def calcular_energia_laplaciana_matriz(L):
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

model_simple = SimpleNN().to(device)

# %% Definir la red neuronal de 5 capas para el experimento (AdvancedNN)
class AdvancedNN(nn.Module):
    def __init__(self):
        super(AdvancedNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Instanciar el modelo y enviarlo al dispositivo
model_advanc = AdvancedNN().to(device)

#%% Entrenamiento de ambas redes neuronales y cálculo de las energías
def entrenar_red(model, optimizer, scheduler, criterion, train_loader, epochs, device):
    loss_history = []
    lr_history = []
    energy_bip_history = []
    energy_lapl_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            data, target = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Calcular las energías usando las funciones optimizadas
        A_bip = red_a_matriz_bipartita(model)
        L_lapl = red_a_matriz_laplaciana(model)

        energia_bip = calcular_energia_estandar_bip_matriz(A_bip)
        energia_lapl = calcular_energia_laplaciana_matriz(L_lapl)

        energy_bip_history.append(energia_bip)
        energy_lapl_history.append(energia_lapl)

        lr_history.append(scheduler.get_last_lr()[0])
        loss_history.append(total_loss / len(train_loader))

        print(f"Época {epoch+1}/{epochs}, Pérdida: {loss_history[-1]:.4f}, "
              f"Energía Bip: {energia_bip:.2f}, "
              f"Energía Lapl: {energia_lapl:.2f}, LR: {lr_history[-1]:.6f}")

        scheduler.step()

    return loss_history, energy_bip_history, energy_lapl_history

# Inicialización de modelos y optimizadores
model1 = SimpleNN().to(device)
optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate)
scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=30, gamma=0.1)
criterion1 = nn.CrossEntropyLoss()

model2 = AdvancedNN().to(device)
optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate)
scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=30, gamma=0.1)
criterion2 = nn.CrossEntropyLoss()

# Entrenamiento de ambos modelos
print("Entrenando SimpleNN...")
loss_history1, energy_bip1, energy_lapl1 = entrenar_red(model1, optimizer1, scheduler1, criterion1, train_loader, epochs, device)

print("Entrenando AdvancedNN...")
loss_history2, energy_bip2, energy_lapl2 = entrenar_red(model2, optimizer2, scheduler2, criterion2, train_loader, epochs, device)


# %% Análisis y Visualización: Pérdida vs Energías
# --- Gráfica 1: Energía Bipartita con r y R² en la leyenda ---
plot_energy_vs_loss_with_stats(
    loss1         = loss_history1,
    energy1       = energy_bip1,
    loss2         = loss_history2,
    energy2       = energy_bip2,
    energy_label  = "Energía Bipartita",
    network1_name = "SimpleNN",
    network2_name = "AdvancedNN"
)

# --- Gráfica 2: Energía Laplaciana con r y R² en la leyenda ---
plot_energy_vs_loss_with_stats(
    loss1         = loss_history1,
    energy1       = energy_lapl1,
    loss2         = loss_history2,
    energy2       = energy_lapl2,
    energy_label  = "Energía Laplaciana",
    network1_name = "SimpleNN",
    network2_name = "AdvancedNN"
)

# %%

# %% Red Simple
# Gráficos de comparación
plot_loss_vs_energy(loss_history1, energy_bip1, 'Red Simple - Bipartita')
plot_loss_vs_energy(loss_history1, energy_lapl1, 'Red Simple - Laplaciana')

# Regresiones y correlación para la Red Simple
print("=== Red Simple - Bipartita ===")
linear_regression(loss_history1, energy_bip1, 'Red Simple - Bipartita')
polynomial_regression(loss_history1, energy_bip1, 'Red Simple - Bipartita', degree=2)
polynomial_regression(loss_history1, energy_bip1, 'Red Simple - Bipartita', degree=3)
exponential_regression(loss_history1, energy_bip1, 'Red Simple - Bipartita')
correlation_analysis(loss_history1, energy_bip1)

print("\n=== Red Simple - Laplaciana ===")
linear_regression(loss_history1, energy_lapl1, 'Red Simple - Laplaciana')
polynomial_regression(loss_history1, energy_lapl1, 'Red Simple - Laplaciana', degree=2)
polynomial_regression(loss_history1, energy_lapl1, 'Red Simple - Laplaciana', degree=3)
exponential_regression(loss_history1, energy_lapl1, 'Red Simple - Laplaciana')
correlation_analysis(loss_history1, energy_lapl1)


# %% Red Avanzada
# Gráficos de comparación
plot_loss_vs_energy(loss_history2, energy_bip2, 'Red Avanzada - Bipartita')
plot_loss_vs_energy(loss_history2, energy_lapl2, 'Red Avanzada - Laplaciana')

# Regresiones y correlación para la Red Avanzada
print("\n=== Red Avanzada - Bipartita ===")
linear_regression(loss_history2, energy_bip2, 'Red Avanzada - Bipartita')
polynomial_regression(loss_history2, energy_bip2, 'Red Avanzada - Bipart', degree=2)
polynomial_regression(loss_history2, energy_bip2, 'Red Avanzada - Bipart', degree=3)
polynomial_regression(loss_history2, energy_bip2, 'Red Avanzada - Bipart')
exponential_regression(loss_history2, energy_bip2, 'Red Avanzada - Bipart')
correlation_analysis(loss_history2, energy_bip2)

print("\n=== Red Avanzada - Laplaciana ===")
linear_regression(loss_history2, energy_lapl2, 'Red Avanzada - Laplaciana')
polynomial_regression(loss_history2, energy_lapl2, 'Red Avanzada - Laplaciana',degree=2)
polynomial_regression(loss_history2, energy_lapl2, 'Red Avanzada - Laplaciana', degree=3)
exponential_regression(loss_history2, energy_lapl2, 'Red Avanzada - Laplaciana')
correlation_analysis(loss_history2, energy_lapl2)
# %%