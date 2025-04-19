import numpy as np
import time

def multiplicacao_matrizes(a, b):
    n = len(a)
    c = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c[i][j] += a[i][k] * b[k][j]
    return c

def adicao(a, b):
    return a + b

def subtracao(a, b):
    return a - b

def strassen(a, b):
    n = len(a)
    if n == 1:
        return a * b
    
    p_medio = n // 2
    a11, a12, a21, a22 = a[:p_medio, :p_medio], a[:p_medio, p_medio:], a[p_medio:, :p_medio], a[p_medio:, p_medio:]
    b11, b12, b21, b22 = b[:p_medio, :p_medio], b[:p_medio, p_medio:], b[p_medio:, :p_medio], b[p_medio:, p_medio:]

    m1 = strassen(adicao(a11, a22), adicao(b11, b22))
    m2 = strassen(adicao(a21, a22), b11)
    m3 = strassen(a11, subtracao(b12, b22))
    m4 = strassen(a22, subtracao(b21, b11))
    m5 = strassen(adicao(a11, a12), b22)
    m6 = strassen(subtracao(a21, a11), adicao(b11, b12))
    m7 = strassen(subtracao(a12, a22), adicao(b21, b22))

    c11 = m1 + m4 - m5 + m7
    c12 = m3 + m5
    c21 = m2 + m4
    c22 = m1 - m2 + m3 + m6

    return combine_submatrizes(c11, c12, c21, c22)

def combine_submatrizes(c11, c12, c21, c22):
    return np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22))))
    
def gera_matriz(n):
    return np.random.randint(0, 10, size=(n, n))

def resultado_final(n):
    print(f"multiplicando as matrizes de {n}x{n}...\n")

    a,b = gera_matriz(n), gera_matriz(n)

    for method, name in [(multiplicacao_matrizes, "padrão"), (strassen, "strassen")]:
        start = time.time()
        c = method(a, b)
        print(f"tempo({name}): {time.time() - start:.4f}s")

    diff = np.max(np.abs(multiplicacao_matrizes(a, b) - strassen(a,b)))
    print(f"erro entre os métodos: {diff}")

if __name__ == "__main__":
    resultado_final(512)    





