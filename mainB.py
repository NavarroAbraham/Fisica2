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
total_current = 12e-3  
n_sides = 4            
segments_per_side = 10  
I_per_differential = total_current / (n_sides * segments_per_side)  

corrientes = [I_per_differential] * (n_sides * segments_per_side)
segmentos = [np.array([0, 0, 1]) for _ in range(n_sides * segments_per_side)] 

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

componentes_magneticos = [('X', Bx, np.zeros_like(By), np.zeros_like(Bz)),
                       ('Y', np.zeros_like(Bx), By, np.zeros_like(Bz)),
                       ('Z', np.zeros_like(Bx), np.zeros_like(By), Bz)]


Bx_matrix, By_matrix, Bz_matrix = Bx, By, Bz

while True:
    print("Menu:")
    print("1. Graficar componentes del campo Magnetico y componentes")
    print("2. Mostrar magnitud del campo Magnetico en un punto")
    print("3. Mostrar matriz completa de un componente")
    print("4. Mostrar plano con objetos de corriente y líneas")
    print("5. Salir")
    choice = input("Opciones: (1-5) ")

    if choice == '1':
        for title, u, v, w in componentes_magneticos:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.quiver(X, Y, Z, u, v, w, length=1, normalize=True)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Componente {title} del Campo Magnético')
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

        Bx_point = Bx[idx_x, idx_y, idx_z]
        By_point = By[idx_x, idx_y, idx_z]
        Bz_point = Bz[idx_x, idx_y, idx_z]

        magnitude = np.sqrt(Bx_point**2 + By_point**2 + Bz_point**2)
        print(f"Magnitud en el punto: ({x_point}, {y_point}, {z_point}): {magnitude}" + "T")

    elif choice == '3':
        print("Seleccione el componente a mostrar:")
        print("1. Componente X")
        print("2. Componente Y")
        print("3. Componente Z")
        component_choice = input("Opciones: (1-3) ")

        if component_choice == '1':
            print("Matriz del componente X:")
            print(Bx)
        elif component_choice == '2':
            print("Matriz del componente Y:")
            print(By)
        elif component_choice == '3':
            print("Matriz del componente Z:")
            print(Bz)
        else:
            print("Opcion invalida. Intente de nuevo.")

    elif choice == '4':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for I, r0 in zip(corrientes, posiciones):
            ax.scatter(r0[0], r0[1], r0[2], color='red', s=100 * abs(I), label=f'Corriente ={I:.1e} A')

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
        ax.set_title('Plano con objetos de corriente y líneas')
        plt.show()

    elif choice == '5':
        print("Saliendo.")
        break
    else:
        print("Opcion invalida. Intente de nuevo.")