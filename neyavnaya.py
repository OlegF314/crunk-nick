import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



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
mu = 4 * (k * k - w)

T = 10
xl = -100
xr = 100
nx = 3000
nt = 3000
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
matrix = np.zeros((2 * nx, 2 * nx))
for i in range(2 * nx):
    matrix[i, i] = 1
for i in range(1, nx - 1):
    matrix[i, i + nx] = -2 * tau / h ** 2
    matrix[i + nx, i] = 2 * tau / h ** 2
    matrix[i, i + nx - 1] = tau / h ** 2
    matrix[i + nx - 1, i] = -tau / h ** 2
    matrix[i, i + nx + 1] = tau / h ** 2
    matrix[i + nx + 1, i] = -tau / h ** 2
print(matrix[:,nx:])
for i in range(nt - 1):
    right_part = np.array([0] + [us[i, j] - tau * vs[i, j] * (absqs2[i, j] * (1 - alpha * absqs2[i, j])) for j in range(1, nx - 1)] +
                          [0, 0] + [vs[i, j] + tau * us[i, j] * (absqs2[i, j] * (1 - alpha * absqs2[i, j])) for j in range(1, nx - 1)] + [0])
    uvs_new = np.linalg.solve(matrix, right_part)
    us[i + 1] = uvs_new[:nx]
    vs[i + 1] = uvs_new[nx:]
    absqs2[i + 1] = us[i + 1] ** 2 + vs[i + 1] ** 2
diffs = np.sqrt((uhats - us) ** 2 + (vhats - vs) ** 2)
absqs = np.sqrt(absqs2)
fig = plt.figure()
for i in range(nslices):
    plt.plot(x, absqs[nt * i // nslices, :], 'blue')
    plt.plot(x, absqhats[nt * i // nslices, :], 'orange')
plt.savefig(f'./neyavn/absq_tau={tau}_h={h}.png')
plt.close(fig)
ts_1d = np.reshape(ts, nx * nt)
xs_1d = np.reshape(xs, nx * nt)
us_1d = np.reshape(us, nx * nt)
vs_1d = np.reshape(vs, nx * nt)
absqs_1d = np.reshape(absqs, nx * nt)
uhats_1d = np.reshape(uhats, nx * nt)
vhats_1d = np.reshape(vhats, nx * nt)
absqhats_1d = np.reshape(absqhats, nx * nt)
d = {'x': xs_1d, 't': ts_1d,
     'pred_u': us_1d, 'pred_v': vs_1d, 'pred_h': absqs_1d,
     'true_u': uhats_1d, 'true_v': vhats_1d, 'true_h': absqhats_1d}
df = pd.DataFrame(d)
df.to_csv(f'df_{xl}_{xr}_{nx}_{nt}.csv')
rel = np.array([sum(diffs[i, :])/sum(absqhats[i, :]) for i in range(nt)])
fig = plt.figure()
plt.plot(t, rel)
plt.savefig(f'./neyavn/rel_tau={tau}_h={h}.png')
plt.close(fig)
