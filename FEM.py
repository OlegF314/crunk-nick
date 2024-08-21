import json
import numpy as np


def q2(x, t):
    amp = np.sqrt((mu * np.exp((x - 2 * k * t - x0) * np.sqrt(mu)))/((0.5 * np.exp((x - 2 * k * t - x0) * np.sqrt(mu)) + 1) ** 2 - alpha0 * mu / 3 * np.exp(2 * (x - 2 * k * t - x0) * np.sqrt(mu))))
    phase = k * x - w * t + theta
    return amp * np.cos(phase), amp * np.sin(phase)


np.sqrt((mu * np.exp((x - x0) * np.sqrt(mu)))/((0.5 * np.exp((x - x0) * np.sqrt(mu)) + 1) ** 2 - alpha0 * mu / 3 * np.exp(2 * (x - x0) * np.sqrt(mu)))) * np.cos(k * x + theta)
alpha = 0.3
alpha0 = 0.3
k = 1
w = 0.88
x0 = -30
theta = 0
mu = 4 * (k * k - w)

T = 10
xl = -50
xr = 50
nx = 8000
nt = 100
tau = T / (nt - 1)
h = (xr - xl) / (nx - 1)
x = np.linspace(xl, xr, num=nx)
t = np.linspace(0, T, num=nt)
xx = [[x[i]] for i in range(nx)]
tt = [t[i] for i in range(nt)]
dict_x = {"0": xx}
dict_t = {"0": tt}
all_d = {"mesh_coord": dict_x, "dt_coord": dict_t}
with open('eval_data_test.json', 'w') as wfile:
    json.dump(all_d, wfile)
xs, ts = np.meshgrid(x, t)
u, v = q2(xs, ts)
absq = np.sqrt(u ** 2 + v ** 2)
js_2 = [u.tolist(), v.tolist(), absq.tolist()]
with open('eval_data_test_mat.json', 'w') as wfile2:
    json.dump(js_2, wfile2)