import numpy as np

x = np.arange(500, 5500, 1000)
f = np.array([15, 28, 32, 18, 7])
x_mean = (x * f).sum() / f.sum()

print(np.absolute(x - x_mean))
print(np.absolute(x - x_mean)*f)
print(np.absolute(x - x_mean)**2 * f)
print((np.absolute(x - x_mean)*f).sum())
