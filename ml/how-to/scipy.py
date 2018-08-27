import scipy.stats as sts


# Optimization

from scipy import optimize

def f(x):
    return (x[0] - 3.2)**2 + (x[1] - 1)**4 + 3
   
x_min = optimize.minimize(f)
x_min.x  # [3.2 1]


# Solve SLE

from scipy import linalg

A = np.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]])
b = np.array([2, 4, -1])

x = linalg.solve(A, b) # [2. -2. 9.]


# Interpolation

from scipy import interpolate

x = np.arange(0, 10)
y = np.exp(-x/3.0)
f = interpolate.interpld(x, y, kind='quadratic')
x_new = np.arange(0, 10, 0.1)
y_new = f(x_new)


# Generate normal distribution

mu = 2.0
sigma = 0.5
norm_rv = sts.norm(loc=mu, scale=sigma)
x = norm_rv.rvs(size=4)  # [2.42471807,  2.89001427,  1.5406754 ,  2.218372]


# Generate uniform distribution

a = 1
b = 4
uniform_rv = sts.uniform(a, b-a)
x = uniform_rv.rvs(size=4)  # [2.90068986,  1.30900927,  2.61667386,  1.82853085]


# Generate Bernoulli distribution

p = 0.7
bernoulli_rv = sts.bernoulli(p)
x = bernoulli_rv.rvs(size=4)  # [1, 1, 1, 0]


# Generate binomial distribution

n = 20
p = 0.7
binom_rv = sts.binom(n, p)
x = binom_rv.rvs(size=4)  # [13, 15, 13, 14]


# Generate Poisson distribution

lam = 5
poisson_rv = sts.poisson(lam)
x = poisson_rv.rvs(size=4)  # [6, 10,  4,  4]


# Custom discrete random variable

elements = np.array([1, 5, 12])
probabilities = [0.05, 0.7, 0.25]
np.random.choice(elements, 4, p=probabilities)  # [5, 12,  5,  5]


# z - quantiles of normal distribution

sts.norm.ppf(1-0.05/2)  # for 95% interval

