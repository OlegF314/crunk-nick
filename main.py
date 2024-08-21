import numpy as np
import matplotlib.pyplot as plt


def q2(x, t):
    amp = np.sqrt((mu * np.exp((x - 2 * k * t - x0) * np.sqrt(mu)))/((0.5 * np.exp((x - 2 * k * t - x0) * np.sqrt(mu)) + 1) ** 2 - alpha0 * mu / 3 * np.exp(2 * (x - 2 * k * t - x0) * np.sqrt(mu))))
    phase = k * x - w * t + theta
    return amp * np.cos(phase), amp * np.sin(phase)


alpha = 0.3
alpha0 = 0.3
k = 1
w = 0.88
x0 = -30
theta = 0
mu = k * k - w

T = 10
xl = -70
xr = 30
nx = 100
nt = 600
tau = T / (nt - 1)
h = (xr - xl) / (nx - 1)
nslices = 5

x = np.linspace(xl, xr, num=nx)
t = np.linspace(0, T, num=nt)
xs, ts = np.meshgrid(x, t)
us = np.zeros((nt, nx))
vs = np.zeros((nt, nx))
us[0:1, :], vs[0:1, :] = q2(x, 0)
absqs2 = us ** 2 + vs ** 2
uhats, vhats = q2(xs, ts)
absqhats = np.sqrt(uhats ** 2 + vhats ** 2)
for i in range(nt - 1):
    for j in range(1, nx - 1):
        us[i + 1, j] = us[i, j] - tau * vs[i, j] * absqs2[i, j] * (1 - alpha * absqs2[i, j]) - tau / h / h * (vs[i, j + 1] - 2 * vs[i, j] + vs[i, j - 1])
        vs[i + 1, j] = vs[i, j] + tau * us[i, j] * absqs2[i, j] * (1 - alpha * absqs2[i, j]) + tau / h / h * (us[i, j + 1] - 2 * us[i, j] + us[i, j - 1])
        absqs2[i + 1, j] = us[i + 1, j] ** 2 + vs[i + 1, j] ** 2
diffs = np.sqrt((uhats - us) ** 2 + (vhats - vs) ** 2)
fig = plt.figure()
for i in range(nslices):
    plt.plot(x, np.sqrt(absqs2[nt * i // nslices, :]), 'blue')
    plt.plot(x, absqhats[nt * i // nslices, :], 'orange')
#    plt.plot(x, diffs[nt * i // nslices, :])
#    plt.show()
plt.savefig(f'absq_tau={tau}_h={h}.png')
plt.close(fig)
rel = np.array([sum(diffs[i])/sum(absqhats[i]) for i in range(nt)])
fig = plt.figure()
plt.plot(t, rel)
plt.savefig(f'rel_tau={tau}_h={h}.png')
plt.close(fig)
