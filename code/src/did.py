# -*- coding: utf-8 -*-
"""双向固定效应 DID 与事件研究（含区域聚类稳健标准误）。

模型（主设定）：

    Y_{r,t} = α_r + γ_t + β · (Post_t × Intensity_r) + ε_{r,t}

- Post_t    = 1{period ≥ 22}（2020Q2 起），被时间固定效应 γ_t 吸收；
- Intensity_r 为处理前特征（时不变量），被区域固定效应 α_r 吸收；
- 故可识别的系数是二者**交互项** β，即"管控强度的区域差异效应"。

由于社区级协变量为时不变量、与 α_r 共线，不能直接进入主设定；
协变量的用途见 covariates.py。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.special import erf

from . import config


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------
@dataclass
class FitResult:
    params: pd.Series
    se: pd.Series
    tstat: pd.Series
    pvalue: pd.Series
    n_obs: int
    n_clusters: int
    k_params: int
    r2: float
    resid: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    def summary_table(self, focus: Optional[Sequence[str]] = None) -> pd.DataFrame:
        idx = list(self.params.index) if focus is None else [c for c in focus if c in self.params.index]
        return pd.DataFrame({
            "coef": self.params.loc[idx],
            "std_err": self.se.loc[idx],
            "t": self.tstat.loc[idx],
            "p": self.pvalue.loc[idx],
        })


# ---------------------------------------------------------------------------
# 核心估计器
# ---------------------------------------------------------------------------
def ols_cluster(y: np.ndarray, X: np.ndarray, cluster: np.ndarray,
                weights: Optional[np.ndarray] = None) -> FitResult:
    """OLS / WLS + 一维聚类稳健方差（小样本校正）。

    Parameters
    ----------
    weights : 可选，观测权重（如 PSM 的 ATT 权重）；None 等价于等权。
    """
    if not np.isfinite(X).all():
        raise ValueError("设计矩阵含非有限值（NaN/Inf），请检查强度映射与缺失处理")
    if not np.isfinite(y).all():
        raise ValueError("结果变量含非有限值（NaN/Inf）")
    n, k = X.shape
    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if not np.isfinite(w).all() or np.any(w < 0):
            raise ValueError("权重须为非负有限值")
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw

    XtX_inv = np.linalg.pinv(Xw.T @ Xw)
    beta = XtX_inv @ (Xw.T @ yw)
    resid = y - X @ beta
    ew = sw * resid

    # 聚类 meat（WLS 形式：Σ_g (Xw_g' ew_g)(...)）
    clusters = np.unique(cluster)
    G = len(clusters)
    meat = np.zeros((k, k), dtype=np.float64)
    for g in clusters:
        m = cluster == g
        s = Xw[m].T @ ew[m]
        meat += np.outer(s, s)

    V = XtX_inv @ meat @ XtX_inv
    if G > 1 and n > k:
        V *= (G / (G - 1.0)) * ((n - 1.0) / (n - k))
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))

    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    p = 2.0 * (1.0 - _norm_cdf(np.abs(t)))

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return FitResult(params=pd.Series(beta, index=[f"b{i}" for i in range(k)]),
                     se=pd.Series(se, index=[f"b{i}" for i in range(k)]),
                     tstat=pd.Series(t, index=[f"b{i}" for i in range(k)]),
                     pvalue=pd.Series(p, index=[f"b{i}" for i in range(k)]),
                     n_obs=n, n_clusters=G, k_params=k, r2=r2, resid=resid)


# 向后兼容别名
_ols_cluster = ols_cluster


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


# ---------------------------------------------------------------------------
# 设计矩阵工具
# ---------------------------------------------------------------------------
def _dummies(series: pd.Series, drop_first: bool = True) -> pd.DataFrame:
    return pd.get_dummies(series.astype("category"), prefix=str(series.name),
                          drop_first=drop_first, dtype=np.float64)


# ---------------------------------------------------------------------------
# 主设定
# ---------------------------------------------------------------------------
def run_twfe_did(panel: pd.DataFrame,
                 intensity: pd.Series,
                 y_col: str = "Y",
                 extra_time_varying: Optional[pd.DataFrame] = None) -> FitResult:
    """主设定：Y ~ α_r + γ_t + β (Post × Intensity)。

    Parameters
    ----------
    panel : 含 community_area, period_id, y_col
    intensity : 以 community_area 为索引的标准化处理强度
    extra_time_varying : 可选，与 panel 同索引的额外回归元（须为时期可变，否则与 FE 共线）
    """
    df = panel.copy()
    df["Intensity"] = df[config.CLUSTER_COL].map(intensity)
    df = df.dropna(subset=["Intensity", y_col]).reset_index(drop=True)
    df["Post"] = (df["period_id"] >= config.COVID_POST_PERIOD).astype(float)
    df["Post_x_Intensity"] = df["Post"] * df["Intensity"]

    D_area = _dummies(df[config.CLUSTER_COL], drop_first=True)
    D_time = _dummies(df["period_id"], drop_first=True)
    parts = [pd.Series(1.0, index=df.index, name="const"),
             df[["Post_x_Intensity"]].reset_index(drop=True),
             D_area.reset_index(drop=True), D_time.reset_index(drop=True)]
    names = ["const", "Post_x_Intensity"] + list(D_area.columns) + list(D_time.columns)
    if extra_time_varying is not None:
        ev = extra_time_varying.reindex(df.index).reset_index(drop=True)
        parts.append(ev)
        names += list(ev.columns)

    X = pd.concat(parts, axis=1).to_numpy(dtype=np.float64)
    y = df[y_col].to_numpy(dtype=np.float64)
    cluster = df[config.CLUSTER_COL].to_numpy()

    res = _ols_cluster(y, X, cluster)
    res.params.index = names
    res.se.index = names
    res.tstat.index = names
    res.pvalue.index = names
    return res


# ---------------------------------------------------------------------------
# 事件研究
# ---------------------------------------------------------------------------
def run_event_study(panel: pd.DataFrame,
                    intensity: pd.Series,
                    y_col: str = "Y",
                    base_period: int = config.COVID_POST_PERIOD - 1) -> pd.DataFrame:
    """事件研究：Y ~ α_r + γ_t + Σ_k β_k (1{period=k} × Intensity)。

    返回每个时期的 β_k 及其聚类稳健标准误；基准期 β=0（被省略）。
    """
    df = panel.copy()
    df["Intensity"] = df[config.CLUSTER_COL].map(intensity)
    df = df.dropna(subset=["Intensity", y_col]).reset_index(drop=True)

    periods = sorted(df["period_id"].unique())
    D_area = _dummies(df[config.CLUSTER_COL], drop_first=True)
    D_time = _dummies(df["period_id"], drop_first=True)

    inter = {}
    for p in periods:
        if p == base_period:
            continue
        inter[f"Intensity_x_Q{int(p)}"] = ((df["period_id"] == p).astype(float)
                                           * df["Intensity"]).to_numpy()
    inter_df = pd.DataFrame(inter, index=df.index).reset_index(drop=True)

    X = pd.concat([pd.Series(1.0, index=df.index, name="const"),
                   inter_df, D_area.reset_index(drop=True), D_time.reset_index(drop=True)],
                  axis=1).to_numpy(dtype=np.float64)
    names = ["const"] + list(inter.keys()) + list(D_area.columns) + list(D_time.columns)
    y = df[y_col].to_numpy(dtype=np.float64)
    cluster = df[config.CLUSTER_COL].to_numpy()

    res = _ols_cluster(y, X, cluster)
    res.params.index = names
    res.se.index = names
    res.tstat.index = names
    res.pvalue.index = names

    rows = []
    for p in periods:
        key = f"Intensity_x_Q{int(p)}"
        if p == base_period:
            rows.append(dict(period_id=p, coef=0.0, se=0.0, t=np.nan, p=np.nan))
        elif key in res.params.index:
            rows.append(dict(period_id=p, coef=float(res.params[key]),
                             se=float(res.se[key]), t=float(res.tstat[key]),
                             p=float(res.pvalue[key])))
    out = pd.DataFrame(rows)
    out["period_label"] = out["period_id"].map(
        lambda p: f"{2015 + (int(p) - 1) // 4}Q{(int(p) - 1) % 4 + 1}")
    return out


def test_pre_trends(es: pd.DataFrame, base_period: int = config.COVID_POST_PERIOD - 1,
                    alpha: float = 0.05) -> dict:
    """对事件研究中**处理前**系数做联合检验（Wald，近似）。"""
    pre = es[(es["period_id"] < config.COVID_POST_PERIOD) & (es["period_id"] != base_period)]
    pre = pre.dropna(subset=["coef", "se"])
    if len(pre) == 0:
        return {"n_pre": 0, "wald": np.nan, "p": np.nan, "pass": None}
    # 独立近似：Σ (β/se)^2 ~ χ²(q)
    z2 = (pre["coef"] / pre["se"]).to_numpy() ** 2
    wald = float(np.nansum(z2))
    q = len(pre)
    # Wilson–Hilferty 正态近似
    z = ((wald / q) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * q))) / np.sqrt(2.0 / (9.0 * q))
    p = 1.0 - 0.5 * (1.0 + float(erf(z / np.sqrt(2.0))))
    return {"n_pre": int(q), "wald": wald, "p": float(p), "pass": bool(p > alpha)}
