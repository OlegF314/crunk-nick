# Схема Кранка-Николсона для BVP2 один солитон
# Автор основного кода Фёдоров Олег
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time

import argparse as ap
try:
    import mlflow
    have_mlflow = True
except ImportError:
    print("Mlflow не установлено и не будет использоваться для логирования")
    have_mlflow = False

from src.for_reports import relative_l2_norm, get_lw_errs
from tempfile import TemporaryDirectory
from pathlib import Path


def q2(x, t, d_task):
    mu = d_task["mu"]
    k = d_task["k"]
    x0 = d_task["x0"]
    alpha0 = d_task["alpha0"]
    w = d_task["w"]
    theta = d_task["theta"]
    amp = np.sqrt((mu * np.exp((x - 2 * k * t - x0) * np.sqrt(mu)))/((0.5 * np.exp((x - 2 * k * t - x0) * np.sqrt(mu)) + 1) ** 2 - alpha0 * mu / 3 * np.exp(2 * (x - 2 * k * t - x0) * np.sqrt(mu))))
    phase = k * x - w * t + theta
    return amp * np.cos(phase) + (amp * np.sin(phase))*1j


def main(d_config):
    # Параметры задачи
    alpha = d_config["alpha"]
    # alpha0 = d_config["alpha0"]
    # k = d_config["k"]
    # w = d_config["w"]
    # x0 = d_config["x0"]
    # theta = d_config["theta"]
    # mu = 4 * (k * k - w)

    # Область определения
    T = d_config["T"]
    xl = d_config["xl"]
    xr = d_config["xr"]

    # Параметры алгоритма
    # Здесь выставляются параметры nx, nt, Δ
    nx = d_config["nx"]
    nt = d_config["nt"]
    delta = d_config["delta"]
    nslices = d_config["nslices"]

    if have_mlflow:
        run = mlflow.start_run()
        mlflow.log_params(d_config)

    tau = T / (nt - 1)
    h = (xr - xl) / (nx - 1)
    lam = tau / h / h
    beg_t = time()

    # Вычисления
    x = np.linspace(xl, xr, num=nx)
    t = np.linspace(0, T, num=nt)
    xs, ts = np.meshgrid(x, t)
    #qs = np.zeros((nt, nx))
    qs = [[None] * nx for i in range(nt)]
    qs[0] = [q2(x[i], 0, d_config) for i in range(nx)]
    qs = np.array(qs)
    qhats = q2(xs, ts, d_config)
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
    # Оценка
    fig = plt.figure()
    for i in range(nslices):
        plt.plot(x, absq[nt * i // nslices, :], 'blue')
        plt.plot(x, absqhat[nt * i // nslices, :], 'orange')
    # plt.savefig(f'absq_tau={tau}_h={h}.png')
    # plt.close(fig)
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
    # df.to_csv(f'df_{xl}_{xr}_{nx}_{nt}.csv')
    exec_time = int(time() - beg_t)
    print(f'Time: {"0" * (2-len(str(exec_time // 3600)))}{exec_time // 3600}:{"0" * (2-len(str(exec_time % 3600 // 60)))}{exec_time % 3600 // 60}:{"0" * (2-len(str(exec_time % 60)))}{exec_time % 60}')
    return df, fig


if __name__ == "__main__":
    args_parser = ap.ArgumentParser()
    args_parser.add_argument("--alpha", default=0.3, type=float)
    args_parser.add_argument("--alpha0", default=0.3, type=float)
    args_parser.add_argument("--k", default=1., type=float)
    args_parser.add_argument("--w", default=0.8, type=float)
    args_parser.add_argument("--x0", default=-30., type=float)
    args_parser.add_argument("--theta", default=0., type=float)
    args_parser.add_argument("--T", default=10., type=float)
    args_parser.add_argument("--xl", default=-50., type=float)
    args_parser.add_argument("--xr", default=50., type=float)
    args_parser.add_argument("--nx", default=500, type=int)
    args_parser.add_argument("--nt", default=100, type=int)
    args_parser.add_argument("--delta", default=0.001, type=float)
    args_parser.add_argument("--nslices", default=5, type=float)
    args_parser.add_argument("--save-df-pred", action="store_true", default=False)
    args_parser.add_argument("--save-fig", action="store_true", default=False)
    args = args_parser.parse_args()

    save_df_pred = args.save_df_pred
    save_fig = args.save_fig

    # Параметры задачи
    alpha = args.alpha
    alpha0 = args.alpha0
    k = args.k
    w = args.w
    x0 = args.x0
    theta = args.theta
    mu = 4 * (k * k - w)

    # Область определения
    T = args.T
    xl = args.xl
    xr = args.xr

    # Параметры алгоритма
    # Здесь выставляются параметры nx, nt, Δ
    nx = args.nx
    nt = args.nt
    delta = args.delta
    nslices = args.nslices

    d_config = dict(alpha=alpha, alpha0=alpha0, k=k, w=w, x0=x0, theta=theta,
                    mu=mu, T=T, xl=xl, xr=xr, nx=nx, nt=nt, delta=delta, nslices=nslices)
    df_pred, fig = main(d_config)

    # Оценка результата
    # Расчёт основных метрик
    df_laws_err = get_lw_errs(df_pred)
    d_scores = {"Lw1_per_max": df_laws_err["ErrLw1_per"].max(),
                "Lw1_per_mean": df_laws_err["ErrLw1_per"].mean(),
                "Lw2_per_max": df_laws_err["ErrLw2_per"].max(),
                "Lw2_per_mean": df_laws_err["ErrLw2_per"].mean()}

    if "true_h" in df_pred.columns:
        d_scores["Rel_h"] = float(relative_l2_norm(df_pred["true_h"].values, df_pred["pred_h"].values))

    if have_mlflow:
        mlflow.log_metrics(d_scores)
        with TemporaryDirectory() as tmp_dir:
            if save_df_pred:
                df_pred_path = Path(tmp_dir, "df_pred.orc")
                df_pred.to_orc(df_pred_path)
                mlflow.log_artifact(df_pred_path)
            if save_fig:
                fig_path = Path(tmp_dir, "fig_pred.png")
                fig.savefig(fig_path)
                mlflow.log_artifact(fig_path)
        mlflow.end_run()
    else:
        print(d_scores)
