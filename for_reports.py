# Функции для расчёта метрик и визуализации результатов предсказания моделей
import numpy as np
import pandas as pd
import pathlib
import os
import json
import argparse as ap

import plotly.express as px
import plotly.graph_objects as go
import scipy
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation


def get_lw_errs(df_pred):
    t0 = df_pred["t"].min()
    df_pred_t0 = df_pred[df_pred["t"] == t0]
    i_inp_x = df_pred_t0["x"].values

    if "true_u" in df_pred.columns and "true_v" in df_pred.columns:
        # Вариант, когда известно точное решение и оно есть в df_pred, колонки true_u, true_v
        i_inp_u = df_pred_t0["true_u"].values
        i_inp_v = df_pred_t0["true_v"].values
    elif "ic_u" in df_pred.columns and "ic_v" in df_pred.columns:
        # Вариант, когда известно только начальное условие
        i_inp_u = df_pred_t0["ic_u"].values
        i_inp_v = df_pred_t0["ic_v"].values
    else:
        raise Exception("В df_pred должны быть колонки true_u, true_v или ic_u, ic_v для расчёта I1, I2")

    I1_true, I2_true = compute_I1_I2(inp_x=i_inp_x, inp_u=i_inp_u, inp_v=i_inp_v)
    df_lw_errs = compute_lw_errs(df_pred, t0=t0, I1_true=I1_true, I2_true=I2_true)

    return df_lw_errs


def check_mesh(df_pred, diff_threshold=0.001):
    """
    Функция проверяет, что x и t представлены в виде равномерной сетки
    Parameters
    ----------
    df_pred: pd.DataFrame
        Данные для анализа

    diff_threshold: float
        Граница, что считать не равномерным.
        В силу погрешности некоторые промежутки между элементами сетки могут быть чуть больше или меньше.
        diff_threshold - задаёт допустимую разницу между промежутками в сетке  для x и t

    Returns
    -------
    is_mesh: bool
        True - если x и t заданы равномерной сеткой
    n_x: int
        Число точек по x в одном срезе
    n_t: int
        Число точек по t в одном срезе
    """
    is_mesh = True

    t_mesh = sorted(df_pred["t"].unique())
    x_mesh = sorted(df_pred["x"].unique())
    n_x = len(x_mesh)
    n_t = len(t_mesh)

    # Проверка, что интервалы между значениями в сетке равномерные с погрешностью diff_threshold
    dt1_t = t_mesh[1] - t_mesh[0]
    dt1_x = x_mesh[1] - x_mesh[0]
    for i in range(len(t_mesh) - 1):
        dt_t = t_mesh[i + 1] - t_mesh[i]
        if np.abs(dt_t - dt1_t) > diff_threshold:
            is_mesh = False
            n_x = None
            n_t = None
    for i in range(len(x_mesh) - 1):
        dt_x = x_mesh[i + 1] - x_mesh[i]
        if np.abs(dt_x - dt1_x) > diff_threshold:
            is_mesh = False
            n_x = None
            n_t = None


    return is_mesh, n_x, n_t


# Оценка по двум законам сохранения
# производные для законов вычисляются численно с помоьщю фукции numpy gradint
def compute_I1_I2(inp_x, inp_u, inp_v):
    """
    Функуия для численного расчёта констант I1, I2 по точкам
    """
    mod_q2 = inp_u ** 2 + inp_v ** 2
    I1 = scipy.integrate.trapezoid(y=mod_q2, x=inp_x)

    dx = inp_x[1] - inp_x[0]
    inp_u_x = np.gradient(inp_u, dx)
    inp_v_x = np.gradient(inp_v, dx)
    C2 = inp_u * inp_v_x - inp_v * inp_u_x
    I2 = scipy.integrate.trapezoid(C2, dx=dx)

    return I1, I2


def compute_lw_errs(df_pred, t0=None, I1_true=None, I2_true=None):
    if (t0 is not None) and (I1_true is None) and (I2_true is None):
        df_vis_0 = df_pred[df_pred["t"] == t0]
        x_int = df_vis_0["x"].values
        u_int_true, v_int_true = df_vis_0["true_u"].values, df_vis_0["true_v"].values
        I1_true, I2_true = compute_I1_I2(inp_x=x_int, inp_u=u_int_true, inp_v=v_int_true)

    l_errors = []
    for current_t, g in df_pred.groupby("t"):
        pred_u, pred_v = g["pred_u"].values, g["pred_v"].values
        inp_x = g["x"].values
        I1_pred, I2_pred = compute_I1_I2(inp_x=inp_x, inp_u=pred_u, inp_v=pred_v)
        I1_err = np.abs(I1_true - I1_pred) / np.abs(I1_true) * 100
        I2_err = np.abs(I2_true - I2_pred) / np.abs(I2_true) * 100
        d_current_err = {"t": current_t, "ErrLw1_per": I1_err, "ErrLw2_per": I2_err,
                         "pred_I1": I1_pred, "pred_I2": I2_pred}
        l_errors.append(d_current_err)
    df_laws_errors = pd.DataFrame(l_errors)
    return df_laws_errors


# Для оценки на известном точном решении
def relative_l2_norm(u_true, u_pred):
    """relative L2-norm из оригинального PINN"""
    res = np.linalg.norm(u_pred - u_true, 2) / np.linalg.norm(u_true, 2)
    return res


def vis_run(df_pred_all, df_laws_err, vis_relh=False, rcount=300, ccount=1000, dpi=300):
    fig = plt.figure(figsize=(15, 7), dpi=dpi)
    if vis_relh:
        specs = fig.add_gridspec(nrows=3, ncols=3)
    else:
        specs = fig.add_gridspec(nrows=2, ncols=3)
    ax = fig.add_subplot(specs[:, 0:2], projection="3d")
    df_vis_pivot_pred = pd.pivot_table(df_pred_all, values="pred_h", index="t", columns="x")
    x_vis = df_vis_pivot_pred.columns.values
    t_vis = df_vis_pivot_pred.index.values
    x_mesh, t_mesh = np.meshgrid(x_vis, t_vis)
    ax.plot_surface(x_mesh, t_mesh, df_vis_pivot_pred.values, cmap="jet",
                    rcount=rcount, ccount=ccount,
                    # rcount=50, ccount=50,
                    # rstride=1, cstride=1,
                    linewidth=0, antialiased=True)
    ax.contourf(x_mesh, t_mesh, df_vis_pivot_pred.values, 100, zdir="z", offset=3.5, cmap="jet")
    ax.set_zlim([0, 3.5])
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title("Predicted |q|")

    if "dde_err" in df_laws_err:
        ax = fig.add_subplot(specs[0, 2])
        df_pred_err_vis = df_pred_all.groupby("t").agg({"dde_err": "max"})
        df_pred_err_vis.plot(ax=ax)
        ax.grid()
        ax.set_title("pde max error")

    ax = fig.add_subplot(specs[1, 2])
    df_laws_err.plot(x="t", y=["ErrLw1_per", "ErrLw2_per"], ax=ax)
    ax.grid()
    ax.set_title("Error on laws, %")

    if vis_relh:
        ax = fig.add_subplot(specs[2, 2])
        l_rel_h = []
        l_rel_h_t = []
        for t, g in df_pred_all.groupby("t"):
            rel_h = relative_l2_norm(g["true_h"], g["pred_h"])
            l_rel_h.append(rel_h)
            l_rel_h_t.append(t)
        ax.plot(l_rel_h_t, l_rel_h)
        ax.set_xlabel("t")
        ax.set_ylabel("Rel h")
        ax.grid()
        ax.set_title("Rel. err with analytic solution")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    args_parser = ap.ArgumentParser()
    args_parser.add_argument("dfpred", help="Путь до файла df_pred, поддерживаются форматы csv и orc")
    args_parser.add_argument("--res-dir", default=None,
                             help="Путь до папки с сохранением результатов, "
                                  "Если не указан, то формируется из имени файла")
    args = args_parser.parse_args()

    df_pred_file_name = args.dfpred
    if ".csv" in df_pred_file_name:
        df_pred = pd.read_csv(df_pred_file_name)
    elif ".orc" in df_pred_file_name:
        df_pred = pd.read_orc(df_pred_file_name)
    else:
        raise Exception("Неизвестный формат файла dfpred, поддерживаются только csv и orc")

    if args.res_dir is None:
        base_file_name = pathlib.Path(df_pred_file_name).stem
        base_file_name = base_file_name.replace(".", "-")
        res_dir = base_file_name
    else:
        res_dir = args.res_dir

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # В случае, если на один по t есть несколько значений,
    # случай на границах сегментов, то выбираются первые по списку результаты для сегмента
    df_pred = df_pred.groupby(["t", "x"]).agg("first")
    df_pred = df_pred.reset_index()

    is_mesh, n_x, n_t = check_mesh(df_pred)
    if not is_mesh:
        raise Exception("Точки x, t в df_pred должны быть представлены в виде равномерной сетки")
    print(f"Размерность сетки по x: {n_x}, по t: {n_t}")

    # Расчёт основных метрик
    df_laws_err = get_lw_errs(df_pred)
    d_scores = {"Lw1_per_max": df_laws_err["ErrLw1_per"].max(),
                "Lw1_per_mean": df_laws_err["ErrLw1_per"].mean(),
                "Lw2_per_max": df_laws_err["ErrLw2_per"].max(),
                "Lw2_per_mean": df_laws_err["ErrLw2_per"].mean()}

    if "true_h" in df_pred.columns:
        d_scores["Rel_h"] = relative_l2_norm(df_pred["true_h"].values, df_pred["pred_h"].values)

    with open(os.path.join(res_dir, "d_scores.json"), "w") as f:
        json.dump(d_scores, f, indent=2, ensure_ascii=False)

    # Визуализации 3d + тепловая карта + графики по законам
    if "true_h" in df_pred.columns:
        vis_relh = True
    else:
        vis_relh = False
    fig = vis_run(df_pred_all=df_pred, df_laws_err=df_laws_err, vis_relh=vis_relh)
    fig.savefig(os.path.join(res_dir, "fig_1.png"))