# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calcular_r2(y_real, y_estimado):
    y_media = np.mean(y_real)
    ss_tot = np.sum((y_real - y_media) ** 2)
    ss_res = np.sum((y_real - y_estimado) ** 2)
    return 1 - (ss_res / ss_tot)


def ajustar_e_plotar(x, y, modelo_fung, nome_modelo):
    y_estimado = modelo_func(x)
    r2 = calcular_r2(y, y_estimado)
    print(f"{nome_modelo} R²: {r2:.5f}")

    plt.clf()
    plt.scatter(x, y, color="red", label="Dados reais")

    x_smooth = np.linspace(min(x), max(x), 200)
    y_smooth = modelo_func(x_smooth)
    plt.plot(x_smooth, y_smooth, label=f"Ajuste {nome_modelo}")

    plt.title(f"Ajuste - {nome_modelo}")
    plt.xlabel("Ano")
    plt.ylabel("População")
    plt.legend()
    plt.show()

    return r2


# Modelos geradores de função:


def gerar_modelo_polinomial_grau2(coef):
    a, b, c = coef
    return lambda x: a * x**2 + b * x + c


def gerar_modelo_polinomial_grau3(coef):
    a, b, c, d = coef
    return lambda x: a * x**3 + b * x**2 + c * x + d


def gerar_modelo_exponencial(coef):
    a, b = coef
    return lambda x: a * np.exp(b * x)


def gerar_modelo_hiperbolico(coef):
    a, b = coef
    return lambda x: a / (x + b)


def gerar_modelo_geometrico(coef):
    a, b = coef
    return lambda x: a * (b**x)


def decomposicaoLU(A):
    n = len(A)

    L = np.eye(n)
    U = np.zeros((n, n))

    for p in range(n):
        if p == 0:
            for j in range(n):
                U[p][j] = A[p][j]
        else:
            for j in range(p, n):
                somatorio = 0
                for k in range(p):
                    somatorio = somatorio + L[p][k] * U[k][j]
                U[p][j] = A[p][j] - somatorio

        if p == 0:
            for i in range(p + 1, n):
                L[i][p] = A[i][p] / U[p][p]
        else:
            for i in range(p + 1, n):
                somatorio = 0
                for k in range(p):
                    somatorio = somatorio + L[i][k] * U[k][p]
                L[i][p] = (A[i][p] - somatorio) / U[p][p]
    return L, U


def triangluarInferior(A, b):
    n = len(A)

    x = np.zeros(n)
    x[0] = b[0] / A[0][0]

    for i in range(1, n):
        somatorio = 0
        for k in range(i):
            somatorio = somatorio + A[i][k] * x[k]
        x[i] = (b[i] - somatorio) / A[i][i]
    return x


def triangularSuperior(A, b):
    n = len(A)
    x = np.zeros(n)
    x[n - 1] = b[n - 1] / A[n - 1][n - 1]

    for i in range(1, n):
        p = n - i - 1
        somatorio = 0
        for k in range(p + 1, n):
            somatorio = somatorio + A[p][k] * x[k]
        x[p] = (b[p] - somatorio) / A[p][p]
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
        e2.append(xi**2)

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

    a, b, c = sis_x[2], sis_x[1], sis_x[0]

    print("f(x) = ax^2 + bx + c")
    print(f"Coeficientes grau 2: a = {a}, b = {b}, c = {c}")

    function = lambda x: a * x**2 + b * x + c

    res_x = np.linspace(min(x), max(x), 200)
    res_y = function(res_x)

    y_estimado = function(np.array(x))
    r2 = calcular_r2(y, y_estimado)
    print(f"R²: {r2:.5f}")

    plt.clf()
    plt.scatter(x, y, color="red", label="Dados")

    plt.plot(res_x, res_y, label="Ajuste grau 2")

    plt.title("Ajuste Polinomial de Grau 2")
    plt.xlabel("Ano")
    plt.ylabel("População")
    plt.legend()

    plt.savefig("grau_2.png")

    return (a, b, c), r2


def plot_grau_3(x, y):
    e0 = []
    e1 = []
    e2 = []
    e3 = []

    for xi in x:
        e0.append(1)
        e1.append(xi)
        e2.append(xi**2)
        e3.append(xi**3)

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

    a, b, c, d = sis_x[3], sis_x[2], sis_x[1], sis_x[0]

    print("f(x) = ax^3 + bx^2 + cx + d")
    print(f"Coeficientes grau 3: a = {a}, b = {b}, c = {c}, d = {d}")

    function = lambda x: a * x**3 + b * x**2 + c * x + d

    res_x = np.linspace(min(x), max(x), 200)
    res_y = function(res_x)

    y_estimado = function(np.array(x))
    r2 = calcular_r2(y, y_estimado)
    print(f"R²: {r2:.5f}")

    plt.clf()
    plt.scatter(x, y, color="red", label="Dados")

    plt.plot(res_x, res_y, label="Ajuste grau 3")

    plt.title("Ajuste Polinomial de Grau 3")
    plt.xlabel("Ano")
    plt.ylabel("População")
    plt.legend()

    plt.savefig("grau_3.png")

    return (a, b, c, d), r2


def plot_exponencial(x, y):
    ln_y = [np.log(yi) for yi in y]

    e0 = [1] * len(x)
    e1 = list(x)
    e = [e0, e1]

    a = []
    b_vec = []

    for i in range(2):
        row_a = []
        for j in range(2):
            row_a.append(prod_escalar(e[i], e[j]))
        a.append(row_a)
        b_vec.append(prod_escalar(e[i], ln_y))

    L, U = decomposicaoLU(a)
    y_lin = triangluarInferior(L, b_vec)
    coef = triangularSuperior(U, y_lin)

    A, B = coef[0], coef[1]
    a_exp = np.exp(A)
    b_exp = B
    print("f(x) = a*e^(bx)")
    print(f"Coeficientes exponencial: a = {a_exp}, b = {b_exp}")

    function = lambda x: a_exp * np.exp(b_exp * x)

    res_x = np.linspace(min(x), max(x), 200)
    res_y = [function(xi) for xi in res_x]

    y_estimado = function(np.array(x))
    r2 = calcular_r2(y, y_estimado)
    print(f"R²: {r2:.5f}")

    plt.clf()
    plt.scatter(x, y, color="red", label="Dados")

    plt.plot(res_x, res_y, label="Ajuste exponencial")

    plt.title("Ajuste Exponencial")
    plt.xlabel("Ano")
    plt.ylabel("População")
    plt.legend()

    plt.savefig("exponencial.png")

    return (a_exp, b_exp), r2


def plot_hiperbolica(x, y):
    inv_y = [1 / yi for yi in y]

    e0 = [1] * len(x)
    e1 = list(x)
    e = [e0, e1]

    a = []
    b_vec = []

    for i in range(2):
        row = [prod_escalar(e[i], e[j]) for j in range(2)]
        a.append(row)
        b_vec.append(prod_escalar(e[i], inv_y))

    L, U = decomposicaoLU(a)
    y_lin = triangluarInferior(L, b_vec)
    coef = triangularSuperior(U, y_lin)

    b, a_coef = coef[0], coef[1]
    print("f(x) = 1/(ax + b)")
    print(f"Coeficientes hiperbola invertida: a = {a_coef}, b = {b}")

    function = lambda x: 1 / (a_coef * x + b)

    res_x = np.linspace(min(x), max(x), 15)
    res_y = [function(xi) for xi in res_x]

    y_estimado = function(np.array(x))
    r2 = calcular_r2(y, y_estimado)
    print(f"R²: {r2:.5f}")

    plt.clf()
    plt.scatter(x, y, color="red", label="Dados")

    plt.plot(res_x, res_y, label="Ajuste Hiperbólico")

    plt.title("Ajuste Hiperbólico")
    plt.xlabel("Ano")
    plt.ylabel("População")
    plt.legend()

    plt.savefig("hiperbolico.png")

    return (a_coef, b), r2


def plot_geometrica(x, y):
    ln_y = [np.log(yi) for yi in y]

    e0 = [1] * len(x)
    e1 = list(x)
    e = [e0, e1]

    a = []
    b_vec = []

    for i in range(2):
        row = [prod_escalar(e[i], e[j]) for j in range(2)]
        a.append(row)
        b_vec.append(prod_escalar(e[i], ln_y))

    L, U = decomposicaoLU(a)
    y_lin = triangluarInferior(L, b_vec)
    coef = triangularSuperior(U, y_lin)

    A, B = coef[0], coef[1]
    a_geo = np.exp(A)
    r_geo = np.exp(B)
    print("f(x) = a*r^x")
    print(f"Coeficientes geométrica: a = {a_geo}, r = {r_geo}")

    function = lambda x: a_geo * (r_geo**x)

    res_x = np.linspace(min(x), max(x), 200)
    res_y = [function(xi) for xi in res_x]

    y_estimado = function(np.array(x))
    r2 = calcular_r2(y, y_estimado)
    print(f"R²: {r2:.5f}")

    plt.clf()
    plt.scatter(x, y, color="red", label="Dados")

    plt.plot(res_x, res_y, label="Ajuste Geométrico")

    plt.title("Ajuste Geométrico")
    plt.xlabel("Ano")
    plt.ylabel("População")
    plt.legend()

    plt.savefig("geometrica.png")

    return (a_geo, r_geo), r2


def plot_all_aprox(ano, pop):
    coef_grau2, r2_grau2 = plot_grau_2(ano, pop)
    print()

    coef_grau3, r2_grau3 = plot_grau_3(ano, pop)
    print()

    coef_exp, r2_exponencial = plot_exponencial(ano, pop)
    print()

    coef_hip, r2_hiperbolica = plot_hiperbolica(ano, pop)
    print()

    coef_geo, r2_geometrica = plot_geometrica(ano, pop)
    print()

    r2_dict = {
        "Polinomial grau 2": r2_grau2,
        "Polinomial grau 3": r2_grau3,
        "Exponencial": r2_exponencial,
        "Hiperbólica": r2_hiperbolica,
        "Geométrica": r2_geometrica,
    }

    melhor_modelo = max(r2_dict, key=r2_dict.get)
    print(f"Melhor ajuste: {melhor_modelo} com R² = {r2_dict[melhor_modelo]:.4f}")

    # Estimativa para 2030
    if melhor_modelo == "Polinomial grau 2":
        modelo = gerar_modelo_polinomial_grau2(coef_grau2)
    elif melhor_modelo == "Polinomial grau 3":
        modelo = gerar_modelo_polinomial_grau3(coef_grau3)
    elif melhor_modelo == "Exponencial":
        modelo = gerar_modelo_exponencial(coef_exp)
    elif melhor_modelo == "Hiperbólica":
        modelo = gerar_modelo_hiperbolico(coef_hip)
    elif melhor_modelo == "Geométrica":
        modelo = gerar_modelo_geometrico(coef_geo)

    pop_2030 = modelo(2030)
    print(f"População estimada para 2030: {pop_2030:.0f} habitantes")


def load_dataset():
    file_path = "populacao_presidenteprudente.dat"

    try:
        df = pd.read_csv(file_path, delimiter="\t")
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    ano = df["Ano"].astype(np.float64).to_numpy()
    pop = df["Pop"].astype(np.float64).to_numpy()

    return ano, pop


if __name__ == "__main__":
    ano, pop = load_dataset()
    plot_all_aprox(ano, pop)
