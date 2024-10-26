import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def campo_electrico(q, r0, r):
    k = 8.9875517873681764e9  # Constante de Coulomb en N·m²/C²
    r_vec = r - r0
    r_mag = np.linalg.norm(r_vec)
    if r_mag == 0:
        return np.zeros(3)  # Evitar división por cero
    E = k * q * r_vec / r_mag**3
    return E

def campo_electrico_total(cargas, posiciones, r):
    E_total = np.zeros(3)
    for q, r0 in zip(cargas, posiciones):
        E_total += campo_electrico(q, r0, r)
    return E_total

lado = 1
n_filas = 3
n_columnas = 4
cargas = [1e-9] * (n_filas * n_columnas)
posiciones = []

for i in range(n_filas):
    for j in range(n_columnas):
        posiciones.append(np.array([0, i * lado / (n_filas - 1), j * lado / (n_columnas - 1)]))

x = np.linspace(-2, 2, 10)
y = np.linspace(-2, 2, 10)
z = np.linspace(-2, 2, 10)
X, Y, Z = np.meshgrid(x, y, z)

Ex = np.zeros_like(X)
Ey = np.zeros_like(Y)
Ez = np.zeros_like(Z)
E_magnitud = np.zeros_like(X)

data = []

# Calcular el campo eléctrico en cada punto de la malla
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        for k in range(X.shape[2]):
            punto = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]])
            E = campo_electrico_total(cargas, posiciones, punto)
            Ex[i, j, k] = E[0]
            Ey[i, j, k] = E[1]
            Ez[i, j, k] = E[2]
            E_magnitud[i, j, k] = np.linalg.norm(E)
            data.append([X[i, j, k], Y[i, j, k], Z[i, j, k], E_magnitud[i, j, k]])

df = pd.DataFrame(data, columns=['X', 'Y', 'Z', 'E_magnitude'])

# Graficar el campo eléctrico y las cargas
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for q, r0 in zip(cargas, posiciones):
    ax.scatter(r0[0], r0[1], r0[2], color='red', s=100 * abs(q), label=f'Carga {q:.1e} C')

# Graficar las líneas del campo eléctrico
ax.quiver(X, Y, Z, Ex, Ey, Ez, length=1, normalize=True)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Campo Eléctrico y Cargas')
plt.show()

# Graficar los componentes del campo eléctrico
componentes = [('X', Ex, np.zeros_like(Ey), np.zeros_like(Ez)),
              ('Y', np.zeros_like(Ex), Ey, np.zeros_like(Ez)),
              ('Z', np.zeros_like(Ex), np.zeros_like(Ey), Ez)]

for title, u, v, w in componentes:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, u, v, w, length=1, normalize=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Componente {title} del Campo Eléctrico')
    plt.show()

Ex_matrix, Ey_matrix, Ez_matrix = Ex, Ey, Ez