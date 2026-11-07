# -*- coding: utf-8 -*-
r"""尺度谱诊断：**是否存在特征尺度**、可行域边界与谱的稳定性（B3）。

背景
----
AMS-CLQ 定理 1 给出 :math:`\mathrm{CLQ}_{AB}(b)=\mathbb E[g_{AB}(R_b)]/\mathbb E[g_{A\cdot}(R_b)]`
（:math:`R_b\sim\mathrm{Rayleigh}(b)`），并据此提出**谱峰判据** C3：
:math:`b^*=\arg\max_b\mathrm{CLQ}_{AB}(b)` 作为"协同作用于什么尺度"的估计
（原文合成实验中该判据与真尺度的 Spearman $=+1.000$，中位比值 0.875）。

**本模块的关键补充**：C3 的推导前提是格局存在**单一特征分离尺度**。
当格局是**多尺度聚集**（真实地理点格局的常态）时，CLQ 谱受
**命题 3（压缩极限）**支配而**单调衰减**，$\arg\max$ 退化为取最小可用带宽、
不再携带尺度信息。故使用 C3 前必须**先检验谱峰是否存在**。

三重约束
--------
| 约束 | 判据 | 作用 |
|---|---|---|
| **κ** | :math:`\kappa_{AB}(b)\approx\lambda_B\pi b^2\ge30` | 一阶项估计可靠（下界约束） |
| **ρ** | 邻域覆盖研究区比例 :math:`\rho(b)\lesssim0.01` | 压缩偏差可控（**上界**约束） |
| **谱形** | 内部极大值存在？ | 决定能否用 $b^*$ 刻画尺度 |

**无谱峰时**：只能报告整条 :math:`\mathrm{CLQ}(b)` 谱，且**跨数据集比较必须固定 b**
（不能用 $b^*$ 归一化）。
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import ams_clq as ac

#: ρ-可行域默认上界（原文 §2.2.3）
RHO_MAX = 0.01
#: κ 判据阈值
KAPPA_MIN = ac.KAPPA_THRESHOLD


# ---------------------------------------------------------------- 网格与可行域

def log_bandwidths(b_min: float, b_max: float, n: int = 14) -> np.ndarray:
    """对数等距带宽网格（米）。"""
    return np.logspace(np.log10(b_min), np.log10(b_max), int(n))


def neighbourhood_fraction(xy: np.ndarray, b: float,
                           sample: Optional[int] = 2000) -> float:
    r"""ρ(b)：点到点邻域覆盖研究区比例的中位估计。

    以"半径 :math:`b` 内邻点数 / (N−1)"为代理（与原文 `bandwidth.neighbourhood_fraction`
    同口径），抽样计算以省时。
    """
    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    if n < 2:
        return 0.0
    tree = cKDTree(xy)
    idx = np.arange(n)
    if sample is not None and n > sample:
        idx = idx[np.unique(np.linspace(0, n - 1, int(sample)).astype(int))]
    cnt = tree.query_ball_point(xy[idx], r=float(b), return_length=True) - 1
    return float(np.median(cnt)) / (n - 1.0)


def feasible_range(xy: np.ndarray, bs: Sequence[float],
                   rho_max: float = RHO_MAX) -> Tuple[float, float]:
    """返回同时满足 **κ 下界** 与 **ρ 上界** 的带宽区间（米）。

    κ 下界由数据决定（需给定类型对），故此处只给 **ρ 上界**；κ 需在具体类型对上判断。
    """
    bs = np.asarray(bs, dtype=np.float64)
    rho = np.array([neighbourhood_fraction(xy, b) for b in bs])
    ok = rho <= rho_max
    if not ok.any():
        return (np.nan, np.nan)
    return (float(bs[ok].min()), float(bs[ok].max()))


# ---------------------------------------------------------------- 谱峰诊断

def peak_diagnostics(clq: np.ndarray, bs: np.ndarray) -> Dict[str, float]:
    r"""判断 CLQ 谱是否含**内部极大值**（C3 判据的适用前提）。

    Returns
    -------
    dict
        * ``has_interior_max``：存在内部极大值（严格大于两端）
        * ``argmax_b``：$\arg\max_b\mathrm{CLQ}(b)$
        * ``argmax_is_boundary``：argmax 落在网格端点（⇒ C3 退化）
        * ``monotone_dec`` / ``monotone_inc``：单调性
        * ``n_interior_extrema``：内部极值个数
        * ``curvature_at_end``：两端斜率符号（判断"单调"是否只是网格未覆盖峰）
    """
    clq = np.asarray(clq, dtype=np.float64)
    bs = np.asarray(bs, dtype=np.float64)
    good = np.isfinite(clq)
    if good.sum() < 3:
        return {'has_interior_max': False, 'argmax_b': np.nan,
                'argmax_is_boundary': True, 'monotone_dec': np.nan,
                'monotone_inc': np.nan, 'n_interior_extrema': 0}
    c, x = clq[good], bs[good]
    d = np.diff(c)
    k = int(np.argmax(c))
    interior = 0 < k < len(c) - 1 and c[k] > c[0] and c[k] > c[-1]
    sgn = np.sign(d)
    n_ext = int(np.sum(np.diff(sgn) != 0))
    return {'has_interior_max': bool(interior),
            'argmax_b': float(x[k]),
            'argmax_is_boundary': bool(k in (0, len(c) - 1)),
            'monotone_dec': bool((d < 0).all()),
            'monotone_inc': bool((d > 0).all()),
            'n_interior_extrema': n_ext}


# ---------------------------------------------------------------- 谱表

def spectrum(xy: np.ndarray, types: np.ndarray, pairs: Sequence[Tuple[int, int, str]],
             bs: Sequence[float], region: Optional[float] = None,
             origin: Optional[Tuple[float, float]] = None, grid_n: int = 120,
             radius_mult: float = 3.0, first_order: str = 'grid',
             smooth_mult: float = 1.0, W_cache: bool = True) -> pd.DataFrame:
    r"""在带宽网格上计算 CLQ / CLQ⁽¹⁾ / CLQ⁽²⁾ / κ / ρ（grid 一阶项）。

    Parameters
    ----------
    pairs : ``[(a_t, b_t, label), ...]``
    """
    xy = np.asarray(xy, dtype=np.float64)
    types = np.asarray(types)
    rows = []
    rho_cache: Dict[float, float] = {}
    for b in np.asarray(bs, dtype=np.float64):
        W = ac.build_weight_matrix(xy, float(b), radius_mult)
        kcache: Dict[Tuple[int, int], float] = {}
        for a, c, lab in pairs:
            if (a, c) not in kcache:
                kcache[(a, c)] = ac.kappa_ab(xy, types, a, c, float(b))
            d = ac.clq_decompose(W, xy, types, a, c, float(b),
                                 first_order=first_order, grid_n=grid_n,
                                 region=region, origin=origin,
                                 smooth_mult=smooth_mult, kappa=kcache[(a, c)])
            bkey = float(b)
            if bkey not in rho_cache:
                rho_cache[bkey] = neighbourhood_fraction(xy, bkey)
            d.update({'pair': lab, 'b_km': float(b) / 1000.0,
                      'rho': rho_cache[bkey]})
            rows.append(d)
        del W
    return pd.DataFrame(rows)


def spectrum_diagnostics(spec: pd.DataFrame) -> pd.DataFrame:
    """对谱表逐类型对做峰形诊断。"""
    rows = []
    for lab, g in spec.groupby('pair', sort=False):
        g = g.sort_values('b_km')
        d = peak_diagnostics(g['CLQ'].to_numpy(), g['b_km'].to_numpy())
        d.update({'pair': lab,
                  'clq_min': float(g.CLQ.min()), 'clq_max': float(g.CLQ.max()),
                  'clq_at_bmin': float(g.CLQ.iloc[0]),
                  'kappa_min': float(g.kappa.min()), 'kappa_max': float(g.kappa.max()),
                  'rho_min': float(g.rho.min()), 'rho_max': float(g.rho.max())})
        rows.append(d)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 自相似性

def self_similarity_check(spec_by_region: Dict[str, pd.DataFrame],
                          region_span_km: Dict[str, float]) -> pd.DataFrame:
    r"""检验不同研究区上的 CLQ 谱是否**自相似**（命题 1 的可比性前提）。

    做法：把每个区域的 :math:`b` 按该区跨度归一化（:math:`b/L`），
    比较归一化谱形；若格局自相似，各曲线应在归一化轴上重合。

    Returns
    -------
    DataFrame(``region``, ``span_km``, ``n_interior_extrema``, ``monotone_dec``,
    ``argmax_is_boundary``, ``clq_at_1pct``, ``clq_at_5pct``)
    """
    rows = []
    for reg, g in spec_by_region.items():
        L = float(region_span_km[reg])
        g = g.sort_values('b_km')
        diag = peak_diagnostics(g['CLQ'].to_numpy(), g['b_km'].to_numpy())
        b_norm = g['b_km'].to_numpy() / L
        clq = g['CLQ'].to_numpy()
        def _at(p):
            j = int(np.argmin(np.abs(b_norm - p)))
            return float(clq[j])
        rows.append({'region': reg, 'span_km': L, **diag,
                     'clq_at_1pct_of_span': _at(0.01),
                     'clq_at_5pct_of_span': _at(0.05)})
    return pd.DataFrame(rows)
