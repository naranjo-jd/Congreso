
#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

#%%
# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
learning_rate = 0.01
epochs = 10

# Cargar MNIST con normalización (media y desviación ajustadas)
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#%%
# Definir la red neuronal
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

#%%
# Función para calcular la energía de la red basada en los pesos de las capas
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

# Ejemplo dummy para probar la función calcular_energia
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)
        self.fc3 = nn.Linear(2, 1)

dummy_model = DummyModel()
energia_dummy = calcular_energia(dummy_model)
print(f"Energía del modelo dummy: {energia_dummy:.2f}")

#%%
# Entrenamiento
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
criterion = nn.CrossEntropyLoss()

loss_history = []
lr_history = []
energy_history = []

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
    
    # Calcular energía y registrar la tasa de aprendizaje actual
    energia = calcular_energia(model)
    energy_history.append(energia)
    lr_history.append(scheduler.get_last_lr()[0])
    loss_history.append(total_loss / len(train_loader))
    
    print(f"Época {epoch+1}/{epochs}, Pérdida: {loss_history[-1]:.4f}, Energía: {energia:.2f}, LR: {lr_history[-1]:.6f}")
    
    scheduler.step()

# Visualización de la tasa de aprendizaje y la energía a lo largo de las épocas
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_history, marker='o', color="blue")
plt.xlabel("Época")
plt.ylabel("loss history")
plt.title("loss por Época")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(energy_history, marker='o', color="red")
plt.xlabel("Época")
plt.ylabel("Energía")
plt.title("Energía de la Red")
plt.grid(True)

plt.tight_layout()
plt.show()
# %%
