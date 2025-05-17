import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def decomposicaoLU(A):
    n = len(A)

    L = np.eye(n)
    U = np.zeros((n,n))

    for p in range(n):
        if p == 0:
            for j in range(n):
                U[p][j] = A[p][j]
        else:
            for j in range(p,n):
                somatorio = 0
                for k in range(p):
                    somatorio = somatorio + L[p][k] * U[k][j]
                U[p][j] = A[p][j] - somatorio

        if p==0:
            for i in range(p+1, n):
                L[i][p] = A[i][p]/U[p][p]
        else:
            for i in range(p+1,n):
                somatorio = 0
                for k in range(p):
                    somatorio = somatorio + L[i][k] * U[k][p]
                L[i][p] = (A[i][p] - somatorio)/U[p][p]
    return L, U

def triangluarInferior(A, b):
    n = len(A)

    x = np.zeros(n)
    #primeiro valor de x
    x[0] = b[0]/A[0][0]

    #demais valores de x
    for i in range(1,n):
        somatorio = 0
        for k in range(i):
            somatorio = somatorio + A[i][k] * x[k]
        x[i] = (b[i] - somatorio)/A[i][i]
    return x

def triangularSuperior(A, b):
    n = len(A)
    x = np.zeros(n)
    x[n-1] = b[n-1]/A[n-1][n-1]

    for i in range(1, n):
        p = n-i-1
        somatorio = 0
        for k in range(p+1, n):
            somatorio = somatorio + A[p][k]*x[k]
        x[p] = (b[p] - somatorio)/A[p][p]
    return x

def prod_escalar(a, b):
    return float(np.dot(a, b))

def plot_grau_2(x, y):
    e0 = []
    e1 = []
    e2 = []

    for xi in x:
        e0.append(1)
        e1.append(xi)
        e2.append(xi ** 2)

    e = [e0, e1, e2]
    a = []
    b = []

    for i in range(3):
        row_a = []
        for j in range(3):
            row_a.append(prod_escalar(e[i], e[j]))
        a.append(row_a)
        b.append(prod_escalar(e[i], y))

    L, U = decomposicaoLU(a)
    sis_y = triangluarInferior(L, b)
    sis_x = triangularSuperior(U, sis_y)
    print("Coeficientes grau 2:", sis_x)

    # Definir função quadrática
    function = lambda x: sis_x[2] * x ** 2 + sis_x[1] * x + sis_x[0]

    # Gerar pontos para o eixo x
    res_x = np.linspace(min(x), max(x), 200)
    res_y = function(res_x)

    # Plotar os dados originais
    plt.clf()
    plt.scatter(x, y, color='red', label="Dados")

    # Plotar o polinômio
    plt.plot(res_x, res_y, label="Ajuste grau 2")

    # Adicionar títulos e rótulos
    plt.title("Ajuste Polinomial de Grau 2")
    plt.xlabel("Ano")
    plt.ylabel("População")
    plt.legend()

    # Salvar o gráfico
    plt.savefig("grau_2.png")

def plot_grau_3(x, y):
    e0 = []
    e1 = []
    e2 = []
    e3 = []

    for xi in x:
        e0.append(1)
        e1.append(xi)
        e2.append(xi ** 2)
        e3.append(xi ** 3)

    e = [e0, e1, e2, e3]
    a = []
    b = []

    for i in range(4):
        row_a = []
        for j in range(4):
            row_a.append(prod_escalar(e[i], e[j]))
        a.append(row_a)
        b.append(prod_escalar(e[i], y))

    L, U = decomposicaoLU(a)
    sis_y = triangluarInferior(L, b)
    sis_x = triangularSuperior(U, sis_y)
    print("Coeficientes grau 3:", sis_x)

    # Definir função cúbica
    function = lambda x: sis_x[3] * x ** 3 + sis_x[2] * x ** 2 + sis_x[1] * x + sis_x[0]

    # Gerar pontos para o eixo x
    res_x = np.linspace(min(x), max(x), 200)
    res_y = function(res_x)

    # Plotar os dados originais
    plt.scatter(x, y, color='red', label="Dados")

    # Plotar o polinômio
    plt.plot(res_x, res_y, label="Ajuste grau 3")

    # Adicionar títulos e rótulos
    plt.title("Ajuste Polinomial de Grau 3")
    plt.xlabel("Ano")
    plt.ylabel("População")
    plt.legend()

    # Salvar o gráfico
    plt.savefig("grau_3.png")

def plot_exponencial(x, y):
    e0 = []
    e1 = []

    for xi in x:
        e0.append(1)
        e1.append(xi)

    e = [e0, e1]
    a = []
    b = []

    for i in range(2):
        row_a = []
        for j in range(2):
            row_a.append(prod_escalar(e[i], e[j]))
        a.append(row_a)
        b.append(prod_escalar(e[i], y))

    L, U = decomposicaoLU(a)
    sis_y = triangluarInferior(L, b)
    sis_x = triangularSuperior(U, sis_y)
    print("Coeficientes grau 3:", sis_x)

    # Definir função cúbica
    function = lambda x: sis_x[3] * x ** 3 + sis_x[2] * x ** 2 + sis_x[1] * x + sis_x[0]

    # Gerar pontos para o eixo x
    res_x = np.linspace(min(x), max(x), 200)
    res_y = function(res_x)

    # Plotar os dados originais
    plt.scatter(x, y, color='red', label="Dados")

    # Plotar o polinômio
    plt.plot(res_x, res_y, label="Ajuste grau 3")

    # Adicionar títulos e rótulos
    plt.title("Ajuste Polinomial de Grau 3")
    plt.xlabel("Ano")
    plt.ylabel("População")
    plt.legend()

    # Salvar o gráfico
    plt.savefig("grau_3.png")


#def plot_hiperbolica():

#def plot_geometrica():


file_path = "Populacao_PresidentePrudente.dat"

try:
    df = pd.read_csv(file_path, delimiter='\t')
    print(df)

except FileNotFoundError:
    print(f"File '{file_path}' not found.")
except Exception as e:
    print(f"An error occurred: {e}")

ano = df["Ano"].astype(np.float64).to_numpy()
pop = df["Pop"].astype(np.float64).to_numpy()

print(ano)

#Plot points
#plt.scatter(ano, pop)
#plt.savefig("points.png")

plot_grau_2(ano, pop)
