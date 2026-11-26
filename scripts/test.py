import numpy as np

n = 9
dx = 0.25
fx = 1 / (n * dx)

a = np.fft.fftshift(np.fft.fftfreq(n, dx))
b = (np.arange(n) - n // 2) * fx
print((a == b).all())

print(1j**0)

c = np.arange(n)
print(c[1::2])
print(c[2:-1:2])