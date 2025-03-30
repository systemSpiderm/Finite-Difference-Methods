import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# 定义f(x)
def f(x: np.float64) -> np.float64:
    return (2 * np.pi) / (np.pi**2 - 1) * np.cos(np.pi * x) * np.exp(x)

# 参数设置
a, b = -1, 1
alpha, beta = 0, -np.pi * np.exp(1)
p = -1 / (np.pi**2 - 1)
q = 1

# 定义解析解
def exact_solution(x: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x) * np.exp(x)

# 构建差分方程
def build_system(h: np.float64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = int((b - a) / h)
    x = np.linspace(a, b, N+1)
    A = np.zeros((N+1, N+1))
    b_vec = np.zeros(N+1)
    c = h * (f((x[N-1] + x[N]) / 2) + f(x[N])) / 4
    
    # 内部节点
    for i in range(1, N):
        A[i, i-1] = -p / h**2
        A[i, i] = 2 * p / h**2 + q
        A[i, i+1] = -p / h**2
        b_vec[i] = f(x[i])
    
    # 边界条件
    A[0, 0] = 2 * p / h**2 + q
    A[0, 1] = -p / h**2
    b_vec[0] = f(x[0]) + p / h**2 * alpha

    A[N, N-1] = p / h
    A[N, N] = -p / h - h / 2
    b_vec[N] = -c - p * beta
    
    return A, b_vec, x

# 选主元的高斯消去法
def gauss_elimination(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.float64]:
    A = deepcopy(A)
    b = deepcopy(b)
    det, N = 1, A.shape[0]
    for k in range(N - 1):
        slice = A[:, k][k:N]
        a, ik = np.max(slice), np.argmax(slice) + k
        if a == 0:
            raise ValueError("det = 0, abort")
        if ik != k:
            A[[k, ik], :] = A[[ik, k], :]           #swap ak,j aik,j
            b[[k, ik]] = b[[ik, k]]                 #swap bk bik
            det *= -1
        for i in range(k + 1, N):
            mik = A[i, k] / A[k, k]
            A[i, k] = mik
            A[i, k + 1:] = A[i, k + 1:] - mik * A[k, k + 1:]
            b[i] = b[i] - mik * b[k]
        det *= A[k][k]
    if A[N - 1, N - 1] == 0:
        raise ValueError("det = 0, abort")
    #回代求解
    b[N - 1] = b[N - 1] / A[N - 1, N - 1]
    for i in range(N - 2, -1, -1):
        b[i] = (b[i] - np.sum(A[i, i + 1:] * b[i + 1:])) / A[i, i]
    det *= A[N - 1, N - 1]
    return b, det

# 雅可比迭代法
def jacobi_iteration(A: np.ndarray, b: np.ndarray, iterations: int=30, epsilon: np.float64=1e-6) -> np.ndarray:
    n = A.shape[0]  # 矩阵的大小
    x = np.zeros_like(b)  # 初始化解向量为零向量
    
    for _ in range(iterations):
        x_old = x.copy()  # 保存上一次的解向量
        for i in range(n):
            # 计算当前迭代的 x[i]
            x[i] = (b[i] - np.dot(A[i, :i], x_old[:i]) - np.dot(A[i, i+1:], x_old[i+1:])) / A[i, i]
        
        # 检查收敛条件
        if np.linalg.norm(x - x_old, np.inf) < epsilon:
            print(f"Converged after {_ + 1} iterations.")
            break
    
    return x

def analysis_for_jacobi(h: np.float64=0.20):
    A, _, _ = build_system(h)
    J = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i == j:
                J[i, j] = 1 / A[i, j]
            else:
                J[i, j] = -A[i, j]
    eigenvalues, _ = np.linalg.eigh(J)
    print(eigenvalues)
    a = np.max(np.abs(eigenvalues))
    if a < 1:
        print("Jacobi method is convergent.")
    else:
        print("Jacobi method is divergent.")

def main():
    h_values = [0.20, 0.10, 0.05, 0.02]
    colors = ['b', 'g', 'r', 'c']  # 不同步长的颜色

    # 绘制数值解和精确解
    plt.figure(figsize=(10, 6))

    for h, color in zip(h_values, colors):
        A, b_vec, x = build_system(h)
        u_h_gauss, _ = gauss_elimination(A, b_vec)
        u_h_jacobi = jacobi_iteration(A, b_vec)
    
        # 绘制数值解
        plt.plot(x, u_h_gauss, color=color, label=f'Gauss (h={h})')
        plt.plot(x, u_h_jacobi, ':', color=color, label=f'Jacobi (h={h})')
        
    # 绘制精确解
    x_exact = np.linspace(a, b, 1000)
    u_exact = exact_solution(x_exact)
    plt.plot(x_exact, u_exact, 'k-', label='Exact Solution')
    plt.ylim(min(u_exact) - 0.5, max(u_exact) + 0.5)
    plt.title('Numerical Solutions and Exact Solution')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 绘制误差图像
    plt.figure(figsize=(10, 6))

    for h, color in zip(h_values, colors):
        A, b_vec, x = build_system(h)
        u_h_gauss, _ = gauss_elimination(A, b_vec)
        
        # 插值精确解到数值解的网格点
        u_exact_interp = np.interp(x, x_exact, u_exact)
        # 计算误差
        error = u_h_gauss - u_exact_interp
        
        # 绘制误差图像
        plt.plot(x, error, color=color, label=f'Error (h={h})')

    plt.title('Error between Numerical Solutions and Exact Solution')
    plt.xlabel('x')
    plt.ylabel('Error u_h(x) - u(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
    analysis_for_jacobi()