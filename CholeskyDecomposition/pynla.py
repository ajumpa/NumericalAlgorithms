import numpy as np

'''
Create a symmetric positive definite matrix
'''
def create_spd_matrix(n, v_range=[0.1, 1], dtype=np.float32):
    A = np.random.randn(n, n).astype(dtype, copy=False)
    
    Q, _ = np.linalg.qr(A)
    
    eigenvalues = np.random.uniform(v_range[0], v_range[1], size=n).astype(dtype)
    D = np.diag(eigenvalues)
    
    SPD = Q @ D @ Q.T
    
    SPD = (SPD + SPD.T) / 2.0
    
    return SPD 

'''
Create an positive definite matrix
'''
def create_pd_matrix(n, v_range=[0.1, 1],  dtype=np.float32):
    A = create_spd_matrix(n, v_range, dtype=dtype)
    B = np.random.randn(n,n).astype(dtype, copy=False)
    B = (B - B.T) / 2

    C = A + B

    return C

def create_pascal_matrix(n, p, dtype=np.int64):
    P = np.zeros((n,n)).astype(dtype=dtype)

    P[0:, ::] = p
    P[::, 0] = p
    for i in range(1, n):
        for j in range(1, n):
            P[i,j] = P[i,j-1] + P[i-1, j]

    return P

'''
Computer determinant of LL^T
'''
def determinant(A, n):
    A_cpy = A.copy()
    L_pla = cholesky_decomposition(A_cpy, n)

    p = 1
    for i in range(n):
        p *= L_pla[i,i]
    p *= p

    return p

'''
Solve Ly = b, where L is lower triangular
'''
def forward_substitution(L, b, n):
    x = np.zeros(n, dtype=np.float64)

    for i in range(n):
        x[i] = b[i]
        for j in range(i):
            x[i] -= L[i, j] * x[j]
        x[i] /= L[i, i]
    return x

"""
Solve Ux = y, where U = L^T, L is lower triangular
"""
def backward_substitution(L, y, n):
    x = np.zeros(n,dtype=np.float64)

    for i in range(n-1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= L[j, i] * x[j]
        x[i] /= L[i, i]

    return x

"""
Use cholesky factorization to conver Ax=b to LL^Tx=b, then solve for x
with forward subsitution, then backward substiution
"""
def solve_simple_linear(A, b, n):
    x = np.zeros(n)

    A_cpy = A.copy()
    L = cholesky_decomposition(A_cpy, n)

    y = forward_substitution(L,b, n)

    x = backward_substitution(L,y, n)

    return x


"""
Invert a matrix by repeatedly solving Ax=e_i
"""
def matrix_inverse(A,n):
    X = np.zeros((n,n))
    I = np.eye(n)
    for i in range(n):
        X[i] = solve_simple_linear(A, I[i], n)

    return X

"""
Column-wise cholesky decomposition
This version will overwrite A instead of allocating a new nxn matrix
"""
def cholesky_decomposition(A,n):

    if np.allclose(A, A.T, atol=1e-16) == False:
        print("Error: A is not symmetric")
        return None

    for i in range(0, n):

        # Diagonal element 
        sum = A[i,i]
        for k in range(0,i):
            sum -= A[i, k] * A[i,k]

        # Not positive definite 
        if (sum <= 0):
            print("Error: A is not positive definite")
            return None

        A[i,i] = np.sqrt(sum)

        # Off Diagonal
        for j in range(i+1, n):
            sum = A[j,i]
            for k in range(i):
                sum -= A[j,k] * A[i,k]
            A[j,i] = sum / A[i,i] 
            A[i,j] = 0

    return A