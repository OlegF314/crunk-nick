import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time


def q2(x, t):
    amp = np.sqrt((mu * np.exp((x - 2 * k * t - x0) * np.sqrt(mu)))/((0.5 * np.exp((x - 2 * k * t - x0) * np.sqrt(mu)) + 1) ** 2 - alpha0 * mu / 3 * np.exp(2 * (x - 2 * k * t - x0) * np.sqrt(mu))))
    phase = k * x - w * t + theta
    return amp * np.cos(phase) + (amp * np.sin(phase))*1j


alpha = 0.3
alpha0 = 0.3
k = 1
w = 0.88
x0 = -30
theta = 0
mu = 4 * (k * k - w)
nslices = 5

T = 10
xl = -70
xr = 30

nx = 3000                 # Здесь выставляются параметры nx, nt, Δ
nt = 1000
delta = 0.001


tau = T / (nt - 1)
h = (xr - xl) / (nx - 1)
lam = tau / h / h
beg_t = time()

x = np.linspace(xl, xr, num=nx)
t = np.linspace(0, T, num=nt)
xs, ts = np.meshgrid(x, t)
#qs = np.zeros((nt, nx))
qs = [[None] * nx for i in range(nt)]
qs[0] = [q2(x[i], 0) for i in range(nx)]
qs = np.array(qs)
qhats = q2(xs, ts)
#matrix = np.zeros((nx, nx))
matrix = [[0j] * nx for i in range(nx)]
for i in range(nx):
    matrix[i][i] = 1j - lam
    matrix[i - 1][i] = 0j + lam / 2
    matrix[i][i - 1] = 0j + lam / 2
matrix = np.array(matrix)
for i in range(nt - 1):
    qcurr = qs[i]
    right_part = np.array([-lam / 2 * (qs[i, j+1] + qs[i, j-1]) + (1j+lam) * qs[i, j] -
                           tau / 2 * (qs[i, j] * abs(qs[i, j]) ** 2 * (1 - alpha * abs(qs[i, j]) ** 2) +
                                      qcurr[j] * abs(qcurr[j]) ** 2 * (1 - alpha * abs(qcurr[j]) ** 2))
                           for j in range(nx - 1)] +
                          [-lam / 2 * (qs[i, 0] + qs[i, nx-2]) + (1j+lam) * qs[i, nx-1] -
                           tau / 2 * (qs[i, nx-1] * abs(qs[i, nx-1]) ** 2 * (1 - alpha * abs(qs[i, nx-1]) ** 2) +
                                      qcurr[nx-1] * abs(qcurr[nx-1]) ** 2 * (1 - alpha * abs(qcurr[nx-1]) ** 2))])
    qnew = np.linalg.solve(matrix, right_part)
    while max(abs(qcurr - qnew)) > delta:
        qcurr = qnew
        right_part = np.array([-lam / 2 * (qs[i, j + 1] + qs[i, j - 1]) + (1j + lam) * qs[i, j] -
                               tau / 2 * (qs[i, j] * abs(qs[i, j]) ** 2 * (1 - alpha * abs(qs[i, j]) ** 2) +
                                          qcurr[j] * abs(qcurr[j]) ** 2 * (1 - alpha * abs(qcurr[j]) ** 2))
                               for j in range(nx - 1)] +
                              [-lam / 2 * (qs[i, 0] + qs[i, nx - 2]) + (1j + lam) * qs[i, nx - 1] -
                               tau / 2 * (qs[i, nx - 1] * abs(qs[i, nx - 1]) ** 2 * (
                                          1 - alpha * abs(qs[i, nx - 1]) ** 2) +
                                          qcurr[nx - 1] * abs(qcurr[nx - 1]) ** 2 * (
                                                      1 - alpha * abs(qcurr[nx - 1]) ** 2))])
        qnew = np.linalg.solve(matrix, right_part)
    qs[i+1] = qnew

absq = abs(qs)
absqhat = abs(qhats)
fig = plt.figure()
for i in range(nslices):
    plt.plot(x, absq[nt * i // nslices, :], 'blue')
    plt.plot(x, absqhat[nt * i // nslices, :], 'orange')
plt.savefig(f'./neyavn/absq_tau={tau}_h={h}.png')
plt.close(fig)
ts_1d = np.reshape(ts, nx * nt)
xs_1d = np.reshape(xs, nx * nt)
us_1d = np.reshape(np.array([[qs[i][j].real for j in range(nx)] for i in range(nt)]), nx * nt)
vs_1d = np.reshape(np.array([[qs[i][j].imag for j in range(nx)] for i in range(nt)]), nx * nt)
absqs_1d = np.reshape(abs(qs), nx * nt)
uhats_1d = np.reshape(qhats.real, nx * nt)
vhats_1d = np.reshape(qhats.imag, nx * nt)
absqhats_1d = np.reshape(abs(qhats), nx * nt)
d = {'x': xs_1d, 't': ts_1d,
     'pred_u': us_1d, 'pred_v': vs_1d, 'pred_h': absqs_1d,
     'true_u': uhats_1d, 'true_v': vhats_1d, 'true_h': absqhats_1d}
df = pd.DataFrame(d)
df.to_csv(f'df_{xl}_{xr}_{nx}_{nt}.csv')
exec_time = int(time() - beg_t)
print(f'Time: {"0" * (2-len(str(exec_time // 3600)))}{exec_time // 3600}:{"0" * (2-len(str(exec_time % 3600 // 60)))}{exec_time % 3600 // 60}:{"0" * (2-len(str(exec_time % 60)))}{exec_time % 60}')