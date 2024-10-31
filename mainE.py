import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def campo_electrico(q, r0, r):
    epsilon_0 = 8.854e-12
    r_vec = r - r0
    r_mag = np.linalg.norm(r_vec)
    if r_mag == 0:
        return np.zeros(3)
    E = (1 / (4 * np.pi * epsilon_0)) * (q * r_vec / r_mag**3)
    return E

def campo_electrico_total(cargas, posiciones, r):
    E_total = np.zeros(3)
    for q, r0 in zip(cargas, posiciones):
        E_total += campo_electrico(q, r0, r)
    return E_total

lado = 1
total_charge = 12e-9
n_sides = 4
segments_per_side = 10
q_per_differential = total_charge / (n_sides * segments_per_side)

cargas = [q_per_differential] * (n_sides * segments_per_side)

posiciones = []

side_spacing = lado / (segments_per_side - 1)

for i in range(segments_per_side):
    posiciones.append(np.array([0, 0, i * side_spacing]))

for i in range(segments_per_side):
    posiciones.append(np.array([0, i * side_spacing, lado]))

for i in range(segments_per_side):
    posiciones.append(np.array([0, lado, lado - i * side_spacing]))

for i in range(segments_per_side):
    posiciones.append(np.array([0, lado - i * side_spacing, 0]))

x = np.linspace(-2, 2, 10)
y = np.linspace(-2, 2, 10)
z = np.linspace(-2, 2, 10)
X, Y, Z = np.meshgrid(x, y, z)

Ex = np.zeros_like(X)
Ey = np.zeros_like(Y)
Ez = np.zeros_like(Z)
E_magnitud = np.zeros_like(X)

data_electrico = []

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        for k in range(X.shape[2]):
            punto = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]])
            E = campo_electrico_total(cargas, posiciones, punto)
            Ex[i, j, k] = E[0]
            Ey[i, j, k] = E[1]
            Ez[i, j, k] = E[2]
            E_magnitud[i, j, k] = np.linalg.norm(E)
            data_electrico.append([X[i, j, k], Y[i, j, k], Z[i, j, k], E_magnitud[i, j, k]])

df_electrico = pd.DataFrame(data_electrico, columns=['X', 'Y', 'Z', 'E_magnitude'])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for q, r0 in zip(cargas, posiciones):
    ax.scatter(r0[0], r0[1], r0[2], color='red', s=100 * abs(q), label=f'Carga {q:.1e} C')

ax.quiver(X, Y, Z, Ex, Ey, Ez, length=1, normalize=True)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Campo Eléctrico y Cargas')

componentes = [('X', Ex, np.zeros_like(Ey), np.zeros_like(Ez)),
              ('Y', np.zeros_like(Ex), Ey, np.zeros_like(Ez)),
              ('Z', np.zeros_like(Ex), np.zeros_like(Ey), Ez)]

while True:
    print("Menu:")
    print("1. Graficar componentes del campo Eléctrico y componentes")
    print("2. Mostrar magnitud del campo Eléctrico en un punto")
    print("3. Mostrar matriz completa de un componente")
    print("4. Mostrar plano con objetos de carga y líneas")
    print("5. Salir")
    choice = input("Opciones: (1-5) ")

    if choice == '1':
        for title, u, v, w in componentes:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.quiver(X, Y, Z, u, v, w, length=1, normalize=True)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Componente {title} del Campo Eléctrico')
            plt.show()

    elif choice == '2':
        x_point = float(input("Ingrese coordenada x : "))
        y_point = float(input("Ingrese coordenada y: "))
        z_point = float(input("Ingrese coordenada z: "))

        x_values = X[:, 0, 0]
        y_values = Y[0, :, 0]
        z_values = Z[0, 0, :]

        idx_x = (np.abs(x_values - x_point)).argmin()
        idx_y = (np.abs(y_values - y_point)).argmin()
        idx_z = (np.abs(z_values - z_point)).argmin()

        Ex_point = Ex[idx_x, idx_y, idx_z]
        Ey_point = Ey[idx_x, idx_y, idx_z]
        Ez_point = Ez[idx_x, idx_y, idx_z]

        magnitude = np.sqrt(Ex_point**2 + Ey_point**2 + Ez_point**2)
        print(f"Magnitud en el punto: ({x_point}, {y_point}, {z_point}): {magnitude}" + "N/C")

    elif choice == '3':
        print("Seleccione el componente a mostrar:")
        print("1. Componente X")
        print("2. Componente Y")
        print("3. Componente Z")
        component_choice = input("Opciones: (1-3) ")

        if component_choice == '1':
            print("Matriz del componente X:")
            print(Ex)
        elif component_choice == '2':
            print("Matriz del componente Y:")
            print(Ey)
        elif component_choice == '3':
            print("Matriz del componente Z:")
            print(Ez)
        else:
            print("Opcion invalida. Intente de nuevo.")

    elif choice == '4':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for q, r0 in zip(cargas, posiciones):
            ax.scatter(r0[0], r0[1], r0[2], color='red', s=100 * abs(q), label=f'Carga {q:.1e} C')

        for i in range(segments_per_side):
            ax.plot([0, 0], [0, 0], [i * side_spacing, (i + 1) * side_spacing], color='blue')
        for i in range(segments_per_side):
            ax.plot([0, 0], [i * side_spacing, (i + 1) * side_spacing], [lado, lado], color='blue')
        for i in range(segments_per_side):
            ax.plot([0, 0], [lado, lado], [lado - i * side_spacing, lado - (i + 1) * side_spacing], color='blue')
        for i in range(segments_per_side):
            ax.plot([0, 0], [lado - i * side_spacing, lado - (i + 1) * side_spacing], [0, 0], color='blue')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Plano con objetos de carga y líneas')
        plt.show()

    elif choice == '5':
        print("Saliendo.")
        break
    else:
        print("Opcion invalida. Intente de nuevo.")