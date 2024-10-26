import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def campo_magnetico(I, dl, r0, r):
    mu_0 = 4 * np.pi * 1e-7
    r_vec = r - r0
    r_mag = np.linalg.norm(r_vec)
    if r_mag == 0:
        return np.zeros(3)
    dB = (mu_0 / (4 * np.pi)) * (I * np.cross(dl, r_vec) / r_mag**3)
    return dB

def campo_magnetico_total(corrientes, segmentos, posiciones, r):
    B_total = np.zeros(3)
    for I, dl, r0 in zip(corrientes, segmentos, posiciones):
        B_total += campo_magnetico(I, dl, r0, r)
    return B_total

lado = 1
n_filas = 3
n_columnas = 4
I = 1e-3
corrientes = [I] * (n_filas * n_columnas)
segmentos = [np.array([0, 0, 1]) for _ in range(n_filas * n_columnas)]
posiciones = []

for i in range(n_filas):
    for j in range(n_columnas):
        posiciones.append(np.array([0, i * lado / (n_filas - 1), j * lado / (n_columnas - 1)]))

x = np.linspace(-2, 2, 10)
y = np.linspace(-2, 2, 10)
z = np.linspace(-2, 2, 10)
X, Y, Z = np.meshgrid(x, y, z)

Bx = np.zeros_like(X)
By = np.zeros_like(Y)
Bz = np.zeros_like(Z)
B_magnitud = np.zeros_like(X)

data_magnetico = []

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        for k in range(X.shape[2]):
            punto = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]])
            B = campo_magnetico_total(corrientes, segmentos, posiciones, punto)
            Bx[i, j, k] = B[0]
            By[i, j, k] = B[1]
            Bz[i, j, k] = B[2]
            B_magnitud[i, j, k] = np.linalg.norm(B)
            data_magnetico.append([X[i, j, k], Y[i, j, k], Z[i, j, k], B_magnitud[i, j, k]])

df_magnetico = pd.DataFrame(data_magnetico, columns=['X', 'Y', 'Z', 'B_magnitude'])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for I, r0 in zip(corrientes, posiciones):
    ax.scatter(r0[0], r0[1], r0[2], color='red', s=100 * abs(I), label=f'Corriente ={I:.1e} A')

ax.quiver(X, Y, Z, Bx, By, Bz, length=0.4, normalize=True)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Campo Magnetico y corrientes')
plt.show()

componentes_magneticos = [('X', Bx, np.zeros_like(By), np.zeros_like(Bz)),
                       ('Y', np.zeros_like(Bx), By, np.zeros_like(Bz)),
                       ('Z', np.zeros_like(Bx), np.zeros_like(By), Bz)]

for title, u, v, w in componentes_magneticos:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, u, v, w, length=0.4, normalize=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Componente {title} del campo magnetico')
    plt.show()

Bx_matrix, By_matrix, Bz_matrix = Bx, By, Bz