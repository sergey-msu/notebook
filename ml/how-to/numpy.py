# Create array

x = np.array([1, 2, 3, 5])


# Create array with type

x = np.array([1, 2, 3, 5], dtype=float)


# Distinct values

x = np.array([1, 1, 2, 3, 4, 3, 1])
y = np.unique(x)  # [1 2 3 4]


# Indices of k smallest elements

x = np.array([4, 3, 3, 1, 5, 3, 2])
idxs = np.argpartition(x, 3)[:3]   # [3 6 2]


# Type casting

x = np.array([1.2, 2, -1.8])
x.astype(int)  # [1 2 -1]

# 1D Indexing

x = np.array([1, 2, 3, 4, 5])
x[2:]    # [3 4 5]
x[-1:]   # [5]
x[[1,3]] # [2 4]
x[x>3]   # [4 5]


# 2D Indexing

x = np.array([[1, 2, 3], 
              [4, 5, 6], 
              [7, 8, 9]])
x[1][2]   # 6
x[1, 2]   # 6
x[1:, :2] # [[4 5] [7 8]]


# Operations

x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 0, -1, 2, 1])
x + 1  # [2 3 4 5 6]
2*x    # [2 4 6 8 10]
x**2   # [1 4 9 16 25]
x*y    # [1 0 -3 8 5]


# Conditionally set some elements to some value

x = [1, 2, 5, 6, 7, 9, 10]
np.place(x, x < 5, 5)
np.place(x, x > 7, 7)  # [5, 5, 5, 6, 7, 7, 7]


# Norm

from numpy.linalg import norm

norm(x, ord=q)  # lq norm, q=2 - default value


# Reshape

x = np.array([1, 2, 3, 4, 5, 6])
x.reshape((2, 3))  # [[1 2 3] [4 5 6]]


# Add fictive dimensions

x = np.array([1, 2, 3, 4, 5, 6])
x[np.newaxis, :] # [[1 2 3 4 5 6]]
x[:, np.newaxis] # [[1] [2] [3] [4] [5] [6]]


# Dot/matrix product

a = np.array([[1, 0, 1], [0, 1, 0]])
x = np.array([1, 2, 3])
y = np.array([1, 0, 2])
np.dot(x, y)  # 7
np.dot(a, x)  # [4 2]


# Mean and Variance

x = np.array([[1, 2],
              [0, 2],
              [3, 2]])
np.var(x, axis=0)  # [1.55555556, 0.0]
np.mean(x, axis=0) # [1.33333333, 2.0]


# Transpose matrix

b = a.T
c = np.transpose(a)  # the same


# Determinant

det = np.linalg.det(a)


# Rank

rk = np.linalg.matrix_rank(a)


# Linear system solve

x = np.linalg.solve(a, b)


# L2 Pseudo-solution of linear system

x, res, r, s = np.linalg.lstsq(a, b)


# Inverse matrix

b = np.linalg.inv(a)


# Eigenvectors, eigenvalues

w, v = np.linalg.eig(a)


# Random

np.random.rand(4, 5) # 4x5 random matrix from normal


# Quick range

np.arange(0, 10000) # quicker than range()


# Custom discrete random variable

sample = np.random.choice([1,2,3,4,5,6], size=100, p=[0.1, 0.2, 0.1, 0.05, 0.05, 0.5])


# Генерирование бутстреп выборок - n подвыборок с возвращением из исходной выборки

def get_bootstrap_samples(data, n_samples):
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    samples = data[indices]
    return samples
    
 
# Shuffle given list

x = np.arange(10)
np.random.shuffle(x)  # x = [1 7 5 2 9 4 3 6 0 8]
