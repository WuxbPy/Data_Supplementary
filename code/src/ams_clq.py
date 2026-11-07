# -*- coding: utf-8 -*-
r"""AMS-CLQ：分类点格局的**一阶/二阶分离**、**尺度谱**与**一阶保留检验**
（College Scorecard 应用）。

理论对应
--------
本模块实现 AMS-CLQ 的三个层次（`IJGIS/amsclq/AMS-CLQ_中文投稿稿.md`）：

* **尺度层（定理 1）**：核加权 CLQ 在平稳各向同性下精确归约为配对相关函数
  :math:`g_{AB}` 在 :math:`R_b\sim\mathrm{Rayleigh}(b)` 上的加权平均，
  故**带宽平滑等价于高斯尺度空间**，尺度谱的峰位是真实特征尺度的固定倍数。
* **结构层（定理 2）**：:math:`\mathrm{CLQ}_{AB}=\mathrm{CLQ}^{(1)}_{AB}\cdot(1+\eta_{AB})/(1+\eta_{A\cdot})`，
  把"**空间分布重叠**"（一阶）与"**点间交互**"（二阶）严格分离。
* **κ 判据**：以带宽内目标类型的期望邻域计数 :math:`\kappa_{AB}(b)\approx\lambda_B\pi b^2`
  判定闭式一阶项估计是否可用（:math:`\kappa\ge30\Rightarrow|\mathrm{CLQ}^{(2)}-1|\le0.015`）。

三个统计量
----------
令 :math:`w_{ij}=K_b(\|x_i-x_j\|)`、:math:`p_B=N_B/(N-1)`、:math:`Y_a=\sum_{i:t_i=a}\sum_{j\neq i}w_{ij}`：

.. math::
    \mathrm{CLQ}_{AB}       &= \frac{\sum_{i:t_i=a}\sum_{j}w_{ij}\mathbb 1\{t_j=b\}}{Y_a\,p_B}
    \qquad\text{（观测）}\\
    \mathrm{CLQ}^{(1)}_{AB} &= \frac{\sum_{i:t_i=a}\sum_{j}w_{ij}\,q_B(x_j)}{Y_a\,p_B}
    \qquad\text{（一阶：零交互，保留类型强度场）}\\
    \mathrm{CLQ}^{(2)}_{AB} &= \mathrm{CLQ}_{AB}\big/\mathrm{CLQ}^{(1)}_{AB}

其中 :math:`q_B(x)=\lambda_B(x)/\lambda(x)` 为**局部类型份额**，由 Nadaraya–Watson 核回归
直接估计（归一化常数在比值中相消）：

.. math::
    q_B(x)=\frac{\sum_{k:t_k=B}K_h(x-x_k)}{\sum_{k}K_h(x-x_k)}

**含义**：在"两个类型各自按自身强度场独立落点"的零假设下，A 的典型邻域中 B 的期望占比。
若 :math:`\mathrm{CLQ}^{(2)}\approx1`，则表面的"协同/排斥"完全由**分布重叠**解释。

尺度一致性
----------
估计 :math:`\hat q_B` 的平滑尺度 ``smooth_scale`` 不应细于待检交互尺度 ``b``，
否则会把二阶结构**吸收进一阶项**、人为把 :math:`\mathrm{CLQ}^{(2)}` 拉向 1。
默认 ``smooth_scale = b``，并提供敏感性接口。

实现要点
--------
全部统计量都归约为**稀疏权重矩阵** :math:`W`（CSR，去自环）上的矩阵向量积，
故置换检验每次只需两次稀疏乘法（:math:`O(\mathrm{nnz})`），可上千次重复。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

# ---------------------------------------------------------------- 常量

CONTROL_LABELS = {1: 'Public', 2: 'Private nonprofit', 3: 'Private for-profit'}
ICLEVEL_LABELS = {1: '4-year+', 2: '2-year', 3: '<2-year'}
CCBASIC_NA = -2.0
KAPPA_THRESHOLD = 30.0          # AMS-CLQ 判据阈值
KAPPA_BOUND = 0.015             # κ≥30 时 |CLQ⁽²⁾−1| 的上界

CONUS_STATES = frozenset(
    'AL AZ AR CA CO CT DE DC FL GA ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT '
    'NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY'.split())

_ALBERS = dict(lat1=29.5, lat2=45.5, lat0=23.0, lon0=-96.0, R=6371007.181)


# ---------------------------------------------------------------- 投影

def project_albers_conus(lon, lat, lat1=_ALBERS['lat1'], lat2=_ALBERS['lat2'],
                         lat0=_ALBERS['lat0'], lon0=_ALBERS['lon0'],
                         R=_ALBERS['R']):
    """球面 Albers 等积圆锥投影（CONUS）→ 米制 ``(x, y)``。

    等积性保证密度估计不受面积变形影响；距离变形在 CONUS 内约 **0.8%**（已用 NYC–LA 大圆校验）。
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    p1, p2 = np.radians(lat1), np.radians(lat2)
    p0, l0 = np.radians(lat0), np.radians(lon0)
    phi, lam = np.radians(lat), np.radians(lon)
    n = (np.sin(p1) + np.sin(p2)) / 2.0
    C = np.cos(p1) ** 2 + 2.0 * n * np.sin(p1)
    rho = np.sqrt(np.maximum(C - 2.0 * n * np.sin(phi), 0.0)) / n
    rho0 = np.sqrt(C - 2.0 * n * np.sin(p0)) / n
    theta = n * (lam - l0)
    return R * rho * np.sin(theta), R * (rho0 - rho * np.cos(theta))


# ---------------------------------------------------------------- 数据加载

_SCORECARD_COLS = [
    'UNITID', 'INSTNM', 'CITY', 'STABBR', 'LATITUDE', 'LONGITUDE',
    'MAIN', 'NUMBRANCH', 'CONTROL', 'ICLEVEL', 'PREDDEG', 'HIGHDEG',
    'CCBASIC', 'CCUGPROF', 'CCSIZSET', 'LOCALE', 'REGION', 'ST_FIPS',
    'CURROPER', 'DISTANCEONLY', 'UGDS', 'PCTPELL', 'PCTFLOAN',
    'ADM_RATE', 'SAT_AVG', 'TUITIONFEE_IN', 'COSTT4_A', 'AVGFACSAL',
    'PFTFAC', 'C150_4', 'MD_EARN_WNE_P10', 'DEBT_MDN', 'HBCU', 'TRIBAL', 'HSI',
]


def load_scorecard(path=None, conus: bool = True, main_only: bool = True,
                   active_only: bool = True, drop_distance_only: bool = False,
                   verbose: bool = True) -> pd.DataFrame:
    """载入 College Scorecard 院校级截面 → 分析点表（含 Albers 米制坐标）。

    默认筛选：**本土 48 州 + DC**、**主校区**（``MAIN==1``）、**在运营**（``CURROPER==1``）、
    **坐标有效**。返回列含 ``x``/``y``（米）、``control``/``iclevel``/``type6``。
    """
    if path is None:
        path = (Path(__file__).resolve().parents[1] / 'data' / 'Scorecard' / 'raw'
                / 'Most-Recent-Cohorts-Institution.csv')
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'未找到 Scorecard 文件：{path}')
    df = pd.read_csv(path, usecols=lambda c: c in _SCORECARD_COLS, low_memory=False)
    n0 = len(df)
    for c in ('LATITUDE', 'LONGITUDE', 'UGDS', 'PCTPELL'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if conus:
        df = df[df['STABBR'].isin(CONUS_STATES)]
    if main_only:
        df = df[df['MAIN'] == 1]
    if active_only:
        df = df[df['CURROPER'] == 1]
    if drop_distance_only:
        df = df[~(df['DISTANCEONLY'] == 1)]
    df = df[df['LATITUDE'].between(-90, 90) & df['LONGITUDE'].between(-180, 180)]
    df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
    df = df.drop_duplicates(subset=['LATITUDE', 'LONGITUDE', 'INSTNM']).reset_index(drop=True)

    df['x'], df['y'] = project_albers_conus(df['LONGITUDE'].to_numpy(),
                                            df['LATITUDE'].to_numpy())
    df['control'] = df['CONTROL'].map(CONTROL_LABELS)
    df['iclevel'] = df['ICLEVEL'].map(ICLEVEL_LABELS)
    df['type6'] = df['iclevel'].fillna('?') + ' | ' + df['control'].fillna('?')
    df['area_km2'] = np.nan     # 研究区面积在需要时由 hull 估计
    if verbose:
        print(f'  [Scorecard] {n0:,} → {len(df):,} 所'
              f'（conus={conus}, main={main_only}, active={active_only}）')
        print(df['control'].value_counts().to_string())
    return df


def load_cross_section(path=None, verbose: bool = True) -> pd.DataFrame:
    """从打包的交叉截面点文件 ``scorecard.npz`` 构造分析点表。

    与 :func:`load_scorecard` 的输出同构（``x``/``y`` 米制、``control``/``iclevel``
    标签），但只依赖存档自带的 150 KB 点文件，使主结果可在无原始发布包的机器上重跑。
    npz 中的 ``types``/``levels`` 为 0-based，分别对应 CONTROL_LABELS/ICLEVEL_LABELS 的 1-based 键。
    """
    if path is None:
        path = Path(__file__).resolve().parents[1] / 'data' / 'scorecard.npz'
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'未找到交叉截面点文件：{path}')
    z = np.load(path, allow_pickle=True)
    lon, lat = z['coords'][:, 0], z['coords'][:, 1]
    x, y = project_albers_conus(lon, lat)
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'control': [CONTROL_LABELS[int(t) + 1] for t in z['types']],
        'iclevel': [ICLEVEL_LABELS[int(v) + 1] for v in z['levels']],
    })
    if verbose:
        print(f'  [Archive] {len(df):,} 所（交叉截面点文件 {path.name}）')
        print(df['control'].value_counts().to_string())
    return df


def encode_types(labels: Sequence[str], values: pd.Series):
    """类别标签 → 1-based 整型编码（与 CausalCLQ 其他模块口径一致）。"""
    order = list(labels)
    m = {n: i + 1 for i, n in enumerate(order)}
    return values.map(m).to_numpy().astype(np.int64), {v: k for k, v in m.items()}


# ---------------------------------------------------------------- 核 / 稀疏权

def _kernel(d: np.ndarray, b: float, kind: str = 'gauss') -> np.ndarray:
    """核权：``gauss`` → :math:`\\exp(-d^2/2b^2)`；``uniform`` → :math:`\\mathbb 1\\{d\\le b\\}`。"""
    if kind == 'gauss':
        return np.exp(-np.square(d) / (2.0 * b * b))
    if kind == 'uniform':
        return (d <= b).astype(np.float64)
    raise ValueError(f"kernel 必须是 'gauss' 或 'uniform'，收到 {kind!r}")


def build_weight_matrix(xy: np.ndarray, b: float, radius_mult: float = 3.0,
                        kernel: str = 'gauss') -> sparse.csr_matrix:
    """构造**去自环**的核权稀疏矩阵 :math:`W`（``N×N``，CSR）。"""
    xy = np.asarray(xy, dtype=np.float64)
    tree = cKDTree(xy)
    r = b * radius_mult
    pairs = tree.query_pairs(r, output_type='ndarray')
    if len(pairs) == 0:
        return sparse.csr_matrix((len(xy), len(xy)))
    i, j = pairs[:, 0], pairs[:, 1]
    d = np.linalg.norm(xy[i] - xy[j], axis=1)
    w = _kernel(d, b, kernel)
    W = sparse.coo_matrix((np.concatenate([w, w]),
                           (np.concatenate([i, j]), np.concatenate([j, i]))),
                          shape=(len(xy), len(xy))).tocsr()
    W.setdiag(0.0)
    W.eliminate_zeros()
    return W


def local_type_share(xy: np.ndarray, is_target: np.ndarray, h: float,
                     radius_mult: float = 3.0) -> np.ndarray:
    r"""Nadaraya–Watson 估计**局部类型份额** :math:`q_B(x)`（在**每个点**处求值）。"""
    xy = np.asarray(xy, dtype=np.float64)
    is_target = np.asarray(is_target, dtype=np.float64)
    tree = cKDTree(xy)
    pairs = tree.query_pairs(max(h * radius_mult, 1e-9), output_type='ndarray')
    num = np.zeros(len(xy))
    den = np.zeros(len(xy))
    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        d = np.linalg.norm(xy[i] - xy[j], axis=1)
        w = np.exp(-np.square(d) / (2.0 * h * h))
        for a, b_ in ((i, j), (j, i)):
            np.add.at(den, a, w)
            np.add.at(num, a, w * is_target[b_])
    # 自身也计入（权重 1，保证分母非零）
    num += is_target
    den += 1.0
    return num / den


def local_type_shares_all(xy: np.ndarray, types: np.ndarray, n_types: int,
                          h: float, radius_mult: float = 3.0) -> np.ndarray:
    """返回 ``(N, K)`` 的**局部类型份额矩阵**（每行和为 1），供一阶保留检验重抽类型。"""
    Q = np.empty((len(xy), n_types + 1), dtype=np.float64)
    Q[:, 0] = 0.0
    for k in range(1, n_types + 1):
        Q[:, k] = local_type_share(xy, (types == k).astype(np.float64), h,
                                  radius_mult=radius_mult)
    Q = np.clip(Q, 1e-12, None)
    return Q / Q.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- 网格式一阶项（原文规格）

def grid_densities(xy: np.ndarray, types: np.ndarray, a_t: int, b_t: int,
                   region: Optional[float] = None, grid_n: int = 120,
                   origin: Optional[Tuple[float, float]] = None):
    """把点格局栅格化为**密度场**（原文 `heterogeneous.intensity_grid` 的等价实现）。

    Returns
    -------
    ``(lam, lam_a, lam_b, cell, h, region)``；``lam*`` 单位为"每平方米"。

    Raises
    ------
    ValueError
        若显式给定的 ``region`` 不足以覆盖全部点（**原文实现的隐含前提**：
        网格外部的点会被静默丢弃，必须显式检查）。
    """
    xy = np.asarray(xy, dtype=np.float64)
    types = np.asarray(types)
    if origin is not None:
        x0, y0 = float(origin[0]), float(origin[1])
    else:
        x0, y0 = float(xy[:, 0].min()), float(xy[:, 1].min())
    if region is None:
        region = float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))
    region = float(region)
    oob = ((xy[:, 0] < x0 - 1e-9) | (xy[:, 0] > x0 + region + 1e-9)
           | (xy[:, 1] < y0 - 1e-9) | (xy[:, 1] > y0 + region + 1e-9))
    if oob.any():
        raise ValueError(
            f'{int(oob.sum())} 个点落在网格外（region={region:g} / origin={origin}）；'
            f'x 跨度 {np.ptp(xy[:, 0]):.1f}、y 跨度 {np.ptp(xy[:, 1]):.1f}')
    h = region / grid_n
    cell = h * h
    edges = [np.linspace(x0, x0 + region, grid_n + 1),
             np.linspace(y0, y0 + region, grid_n + 1)]

    def _dens(mask):
        H, _, _ = np.histogram2d(xy[mask, 0], xy[mask, 1], bins=edges)
        return H / cell

    lam = _dens(np.ones(len(xy), dtype=bool))
    return lam, _dens(types == a_t), _dens(types == b_t), cell, h, region


def first_order_clq_grid(xy: np.ndarray, types: np.ndarray, b: float,
                         a_t: int, b_t: int, region: Optional[float] = None,
                         grid_n: int = 120, smooth_mult: float = 1.0,
                         origin: Optional[Tuple[float, float]] = None) -> float:
    r"""式 (8) 的**网格式实现（原文规格）**。

    .. math::
        \mathrm{CLQ}^{(1)}(b;h)=\frac{\sum_c \lambda_A(c)\,(G_{\sigma}*\lambda_B)(c)}
        {\big[\sum_c \lambda_A(c)\,(G_{\sigma}*\lambda)(c)\big]\,p_B},
        \qquad \sigma=h_{\rm mult}\cdot b

    ``smooth_mult=1`` 即原文（代码里 ``sig = b / h``，$h$ 为网格边长）；
    $\sigma\ne b$ 得到的是另一个估计量，仅用于敏感性分析。

    .. important::
        本函数的路线（**分子、分母分别做同带宽核卷积后取比值**）才是式 (8) 的一致估计。
        常见的替代实现"逐点估计局部份额 $\hat q_B$、再乘核权求和"
        （见 :func:`local_type_share` + :func:`clq_decompose` 的 legacy 路线）
        与之相差一个 $1/\lambda$ 权重，**在强度场尖锐时有系统偏差**
        （实测 $\kappa=689$ 时两者 CLQ⁽²⁾ 相差 0.05）。详见
        `amsclq/examples_bias_study/README.md`。
    """
    from scipy.ndimage import gaussian_filter

    lam, lam_a, lam_b, cell, h, _ = grid_densities(xy, types, a_t, b_t, region,
                                                   grid_n, origin)
    p_B = float((np.asarray(types) == b_t).mean())
    sig = float(b) * float(smooth_mult) / h
    sm_lam = gaussian_filter(lam, sig, mode='constant')
    sm_b = gaussian_filter(lam_b, sig, mode='constant')
    num = float((lam_a * sm_b).sum()) * cell
    den = float((lam_a * sm_lam).sum()) * cell
    return num / (den * p_B) if den > 0 else np.nan


def local_share_grid_all(xy: np.ndarray, types: np.ndarray, n_types: int,
                         smooth: float, region: Optional[float] = None,
                         grid_n: int = 120,
                         origin: Optional[Tuple[float, float]] = None) -> np.ndarray:
    r"""**网格式局部份额场** ``(N, n_types+1)``（每行和 1；第 0 列恒 0）。

    用于**一阶保留零假设**的重抽，与原文 `heterogeneous.first_order_resample`
    同路线：网格化 → 高斯滤波 → 在每个点所在格取值。
    """
    from scipy.ndimage import gaussian_filter

    xy = np.asarray(xy, dtype=np.float64)
    types = np.asarray(types)
    if origin is not None:
        x0, y0 = float(origin[0]), float(origin[1])
    else:
        x0, y0 = float(xy[:, 0].min()), float(xy[:, 1].min())
    if region is None:
        region = float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))
    region = float(region)
    h = region / grid_n
    if (xy[:, 0] < x0 - 1e-9).any() or (xy[:, 0] > x0 + region + 1e-9).any() \
            or (xy[:, 1] < y0 - 1e-9).any() or (xy[:, 1] > y0 + region + 1e-9).any():
        raise ValueError(f'有点落在网格外（region={region:g} / origin={origin}）')
    edges = [np.linspace(x0, x0 + region, grid_n + 1)] * 2

    def _dens(mask):
        H, _, _ = np.histogram2d(xy[mask, 0], xy[mask, 1], bins=edges)
        return H / (h * h)

    sig = float(smooth) / h
    tot = np.maximum(gaussian_filter(_dens(np.ones(len(xy), dtype=bool)), sig,
                                     mode='constant'), 1e-12)
    ix = np.clip(((xy[:, 0] - x0) / h).astype(int), 0, grid_n - 1)
    iy = np.clip(((xy[:, 1] - y0) / h).astype(int), 0, grid_n - 1)
    tot_pt = tot[ix, iy]
    Q = np.zeros((len(xy), n_types + 1))
    for t in range(1, n_types + 1):
        sm = gaussian_filter(_dens(types == t), sig, mode='constant')
        Q[:, t] = sm[ix, iy] / tot_pt
    Q = np.clip(Q, 1e-12, None)
    return Q / Q.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- 核心分解

def weighted_sums(W: sparse.csr_matrix, types: np.ndarray, src: int, tgt: int,
                  q_pt: Optional[np.ndarray] = None):
    """返回 ``(X, Y, X1, n_A, n_B)``：观测 B 计数、总权、一阶期望计数。"""
    one_t = (types == tgt).astype(np.float64)
    in_a = (types == src)
    s_c = W @ one_t
    X = float(s_c[in_a].sum())
    Y = float(np.asarray(W.sum(axis=1)).ravel()[in_a].sum())
    X1 = np.nan
    if q_pt is not None:
        s_q = W @ q_pt
        X1 = float(s_q[in_a].sum())
    return X, Y, X1, int(in_a.sum()), int(one_t.sum())


def clq_decompose(W: sparse.csr_matrix, xy: np.ndarray, types: np.ndarray,
                  src: int, tgt: int, b: float,
                  smooth_scale: Optional[float] = None,
                  radius_mult: float = 3.0, q_pt: Optional[np.ndarray] = None,
                  kappa: Optional[float] = None,
                  first_order: str = 'grid', grid_n: int = 120,
                  region: Optional[float] = None, smooth_mult: float = 1.0,
                  origin: Optional[Tuple[float, float]] = None,
                  nw_reference: bool = False) -> Dict[str, float]:
    r"""**单带宽**的 CLQ 三因子分解（``W`` 可预先构造以复用）。

    Parameters
    ----------
    first_order : {'grid', 'nw'}
        * ``'grid'``（**默认，原文规格**）：式 (8) 的网格 + 同带宽卷积实现
          :func:`first_order_clq_grid`；
        * ``'nw'``（**legacy，已知有偏**）：逐点条件份额 × 核权。
          在强度场尖锐时系统性偏离（实测 $\kappa=689$、CLQ⁽²⁾ 相差 0.05），
          保留仅供对照与复现历史结果。
    smooth_mult : float
        grid 路线的一阶平滑倍数 $\sigma=h_{\rm mult}b$（``1`` = 原文设定）。
    nw_reference : bool
        额外返回 NW 路线的 ``CLQ1_nw``/``CLQ2_nw`` 作为对照。
    """
    types = np.asarray(types)
    if smooth_scale is None:
        smooth_scale = b
    N = len(types)
    n_B = int((types == tgt).sum())
    p_B = n_B / (N - 1.0)

    # 观测分子（与一阶项估计器无关）
    X, Y, _, n_A, n_B = weighted_sums(W, types, src, tgt, None)
    clq = X / (Y * p_B) if Y > 0 else np.nan

    if first_order == 'grid':
        clq1 = first_order_clq_grid(xy, types, b, src, tgt, region, grid_n,
                                    smooth_mult, origin)
    elif first_order == 'nw':
        if q_pt is None:
            q_pt = local_type_share(xy, (types == tgt).astype(np.float64),
                                    smooth_scale, radius_mult=radius_mult)
        _, _, clq1, _, _ = weighted_sums(W, types, src, tgt, q_pt)
        clq1 = clq1 / (Y * p_B) if Y > 0 else np.nan
    else:
        raise ValueError(f"first_order 必须是 'grid' 或 'nw'，收到 {first_order!r}")

    clq2 = clq / clq1 if (np.isfinite(clq1) and clq1 > 0) else np.nan
    out = {'X': X, 'Y': Y, 'p_B': float(p_B), 'CLQ': float(clq),
           'CLQ1': float(clq1), 'CLQ2': float(clq2),
           'kappa': float(kappa_ab(xy, types, src, tgt, b)) if kappa is None
           else float(kappa),
           'n_A': n_A, 'n_B': n_B, 'N': N, 'b_m': float(b),
           'first_order': first_order, 'smooth_mult': float(smooth_mult),
           'grid_n': int(grid_n), 'smooth_scale_m': float(smooth_scale)}
    if nw_reference:
        q = q_pt if q_pt is not None else local_type_share(
            xy, (types == tgt).astype(np.float64), smooth_scale,
            radius_mult=radius_mult)
        _, _, x1n, _, _ = weighted_sums(W, types, src, tgt, q)
        c1n = x1n / (Y * p_B) if Y > 0 else np.nan
        out['CLQ1_nw'] = float(c1n)
        out['CLQ2_nw'] = float(clq / c1n) if (np.isfinite(c1n) and c1n > 0) else np.nan
    return out


def kappa_ab(xy: np.ndarray, types: np.ndarray, src: int, tgt: int, b: float) -> float:
    r"""κ 判据：A 点带宽内目标 B 的**中位计数**（:math:`\approx\lambda_B\pi b^2`）。"""
    tree = cKDTree(xy)
    src_idx = np.where(types == src)[0]
    is_t = (types == tgt)
    if len(src_idx) == 0:
        return np.nan
    pairs = tree.query_pairs(b, output_type='ndarray')
    cnt = np.zeros(len(types))
    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        hit = is_t[j]
        np.add.at(cnt, i[hit], 1.0)
        hit2 = is_t[i]
        np.add.at(cnt, j[hit2], 1.0)
    return float(np.median(cnt[src_idx]))


# ---------------------------------------------------------------- 尺度谱 / 全对

def scale_spectrum(xy: np.ndarray, types: np.ndarray, src: int, tgt: int,
                   bs: Sequence[float], smooth_mult: float = 1.0,
                   radius_mult: float = 3.0, kernel: str = 'gauss',
                   first_order: str = 'grid', grid_n: int = 120,
                   region: Optional[float] = None,
                   origin: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
    r"""**尺度谱**：扫描带宽，返回 CLQ / CLQ⁽¹⁾ / CLQ⁽²⁾ / κ。

    定理 1 保证带宽平滑等价于高斯尺度空间，故谱峰位置是真实特征尺度的固定倍数。
    一阶项默认用 grid 路线（原文规格），``smooth_mult`` 即 :math:`\sigma=b\cdot h_{\rm mult}`。
    """
    rows = []
    for b in bs:
        W = build_weight_matrix(xy, float(b), radius_mult, kernel)
        d = clq_decompose(W, xy, types, src, tgt, float(b),
                          smooth_scale=float(b) * smooth_mult,
                          radius_mult=radius_mult, first_order=first_order,
                          grid_n=grid_n, region=region, smooth_mult=smooth_mult,
                          origin=origin)
        d['b_km'] = float(b) / 1000.0
        rows.append(d)
    return pd.DataFrame(rows)


def all_pairs(xy: np.ndarray, types: np.ndarray, labels: Dict[int, str],
              b: float, smooth_scale: Optional[float] = None,
              radius_mult: float = 3.0, kernel: str = 'gauss',
              W: Optional[sparse.csr_matrix] = None,
              first_order: str = 'grid', grid_n: int = 120,
              region: Optional[float] = None, smooth_mult: float = 1.0,
              origin: Optional[Tuple[float, float]] = None,
              nw_reference: bool = False) -> pd.DataFrame:
    """所有类型对的 CLQ 分解表。

    ``first_order='grid'``（默认）时无需逐类构造 NW 份额场；``'nw'`` 时每类只估一次并复用。
    """
    if W is None:
        W = build_weight_matrix(xy, b, radius_mult, kernel)
    ids = sorted(labels)
    qcache = {}
    if first_order == 'nw' or nw_reference:
        qcache = {c: local_type_share(xy, (types == c).astype(np.float64),
                                      smooth_scale or b, radius_mult=radius_mult)
                  for c in ids}
    kcache = {}
    rows = []
    for a in ids:
        for c in ids:
            key = (a, c)
            if key not in kcache:
                kcache[key] = kappa_ab(xy, types, a, c, b)
            d = clq_decompose(W, xy, types, a, c, b, smooth_scale=smooth_scale,
                              q_pt=qcache.get(c), kappa=kcache[key],
                              first_order=first_order, grid_n=grid_n,
                              region=region, smooth_mult=smooth_mult,
                              origin=origin, nw_reference=nw_reference)
            d['source'], d['target'] = labels[a], labels[c]
            d['b_km'] = b / 1000.0
            rows.append(d)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 两类零假设

def simulate_independent_placement(xy: np.ndarray, types: np.ndarray, n_types: int,
                                   smooth: float = 0.0,
                                   region: Optional[float] = None,
                                   grid_n: int = 120, seed: int = 0):
    r"""**独立落点零格局**：各类型**从自身的平滑强度场独立重抽位置**。

    与"在固定位置重抽类型"（:func:`local_share_grid_all` + 重抽）相比，本方案
    **精确保留每个类型的一阶强度场** $\lambda_t$（因为直接从它抽样），
    只切断类型之间的点间关联 ⇒ 二阶项 $\eta\equiv0$。

    .. important::
        **``smooth`` 必须取 0**（默认）。若先对密度做高斯平滑再抽样，等同于把强度场
        再卷积一次（$G_b\circ G_b=G_{b\sqrt2}$），零分布的一阶水平被削弱、
        检验过度激进。本项目实测（非营利→非营利，$\mathrm{CLQ}^{(1)}=1.301$）：

        ============  ==========  ==========
        零假设平滑尺度   零分布均值        p
        ============  ==========  ==========
        0（不平滑）       1.277      0.250
        $b/5$            1.260      0.087
        $b/3$            1.244      0.037
        $b$              1.187      0.013
        ============  ==========  ==========

        故本函数以**原始网格直方图**抽样（网格须足够细，$h\ll b$）。
        该零假设比"$b$-平滑场"更保守（保留更多一阶结构，更难拒绝）。

    .. note::
        "在固定位置重抽类型"同样会把类型场再平滑一次，本项目实测非营利→非营利的
        $\mathrm{CLQ}^{(1)}=1.301$ 而该零分布均值仅 1.032，故不用该路线。

    Returns
    -------
    ``(xy_null, types_null)``（点数与各自类型计数均保持不变）
    """
    from scipy.ndimage import gaussian_filter

    xy = np.asarray(xy, dtype=np.float64)
    types = np.asarray(types)
    x0, y0 = float(xy[:, 0].min()), float(xy[:, 1].min())
    if region is None:
        region = float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))
    h = float(region) / grid_n
    edges = [np.linspace(x0, x0 + region, grid_n + 1)] * 2
    sig = float(smooth) / h
    rng = np.random.default_rng(seed)
    xs, ts = [], []
    for t in range(1, n_types + 1):
        m = (types == t)
        n = int(m.sum())
        if n == 0:
            continue
        H, _, _ = np.histogram2d(xy[m, 0], xy[m, 1], bins=edges)
        p = np.maximum(gaussian_filter(H, sig, mode='constant'), 0.0).ravel()
        if p.sum() <= 0:
            xs.append(xy[m])
            ts.append(np.full(n, t))
            continue
        k = rng.choice(len(p), size=n, p=p / p.sum())
        cx = x0 + (k % grid_n + rng.random(n)) * h
        cy = y0 + (k // grid_n + rng.random(n)) * h
        xs.append(np.column_stack([cx, cy]))
        ts.append(np.full(n, t))
    return np.vstack(xs), np.concatenate(ts)


def _clq_from_types(W: sparse.csr_matrix, types: np.ndarray, wsum: np.ndarray,
                    src: int, tgt: int) -> float:
    """给定类型向量快速算 CLQ（供置换复用）。"""
    in_a = (types == src)
    n_B = int((types == tgt).sum())
    N = len(types)
    if n_B == 0 or N < 2 or not in_a.any():
        return np.nan
    X = float((W @ (types == tgt).astype(np.float64))[in_a].sum())
    Y = float(wsum[in_a].sum())
    p_B = n_B / (N - 1.0)
    return X / (Y * p_B) if Y > 0 else np.nan


def null_clq_test(W: sparse.csr_matrix, xy: np.ndarray, types: np.ndarray,
                  src: int, tgt: int, q_all: Optional[np.ndarray] = None,
                  mode: str = 'first_order', n_perm: int = 499,
                  seed: int = 42) -> Dict[str, float]:
    r"""两类零假设下的 CLQ 置换检验。

    * ``mode='global'``：**全局类型标签重排**（H0：CLQ=1，**不保留**一阶结构）。
      一阶异质存在时会被大量拒绝 → 这正是定理 4.1 的机制。
    * ``mode='first_order'``：**一阶保留检验**（H0：CLQ⁽²⁾=1）。在每个点处按
      :math:`\hat q_t(x_i)` 独立重抽类型（给定位置下类型独立）→ 保留一阶强度场、
      切断点间交互。

    Returns
    -------
    dict(``clq_obs``, ``p``, ``n_perm``, ``mean_null``, ``mode``)
    """
    rng = np.random.default_rng(seed)
    types = np.asarray(types)
    wsum = np.asarray(W.sum(axis=1)).ravel()
    obs = _clq_from_types(W, types, wsum, src, tgt)
    K = int(types.max())
    if mode == 'first_order' and q_all is None:
        raise ValueError("mode='first_order' 需要 q_all（local_type_shares_all 的输出）")

    cnt = 0
    tot = 0
    null_sum = 0.0
    if mode == 'global':
        for _ in range(n_perm):
            t2 = rng.permutation(types)
            v = _clq_from_types(W, t2, wsum, src, tgt)
            if np.isfinite(v):
                tot += 1
                null_sum += v
                cnt += int(abs(v - 1.0) >= abs(obs - 1.0))
    elif mode == 'first_order':
        # 累积分布抽样：一次生成 (n_perm, N) 的均匀数与类别
        cum = np.cumsum(q_all, axis=1)
        cum[:, -1] = 1.0
        labels = np.arange(q_all.shape[1])
        for _ in range(n_perm):
            u = rng.random(len(types))
            t2 = labels[np.argmax(cum >= u[:, None], axis=1)]
            v = _clq_from_types(W, t2, wsum, src, tgt)
            if np.isfinite(v):
                tot += 1
                null_sum += v
                cnt += int(abs(v - 1.0) >= abs(obs - 1.0))
    else:
        raise ValueError(f"mode 必须是 'global' 或 'first_order'，收到 {mode!r}")
    return {'clq_obs': float(obs), 'p': (cnt + 1) / (tot + 1) if tot else np.nan,
            'n_perm': tot, 'mean_null': (null_sum / tot) if tot else np.nan,
            'mode': mode}


def _clq_pairs(xy: np.ndarray, types: np.ndarray, src: int, tgt: int, b: float,
               radius_mult: float = 3.0, kernel: str = 'gauss') -> float:
    """由配对直接计算 CLQ（**不建稀疏矩阵**，供大量置换复用）。"""
    xy = np.asarray(xy, dtype=np.float64)
    types = np.asarray(types)
    pairs = cKDTree(xy).query_pairs(b * radius_mult, output_type='ndarray')
    if len(pairs) == 0:
        return np.nan
    i, j = pairs[:, 0], pairs[:, 1]
    d = np.linalg.norm(xy[i] - xy[j], axis=1)
    w = _kernel(d, b, kernel)
    ia, ib = (types[i] == src), (types[j] == tgt)
    ia2, ib2 = (types[j] == src), (types[i] == tgt)
    X = float(w[ia & ib].sum() + w[ia2 & ib2].sum())
    Y = float(w[ia].sum() + w[ia2].sum())
    n_B = int((types == tgt).sum())
    p_B = n_B / (len(types) - 1.0)
    return X / (Y * p_B) if (Y > 0 and p_B > 0) else np.nan


def null_clq_placement(xy: np.ndarray, types: np.ndarray, src: int, tgt: int,
                       b: float, n_types: int, smooth: float = 0.0,
                       region: Optional[float] = None, grid_n: int = 120,
                       n_perm: int = 199, seed: int = 42,
                       radius_mult: float = 3.0, kernel: str = 'gauss') -> Dict[str, float]:
    r"""**独立落点零假设**检验（$H_0:\ \mathrm{CLQ}_{AB}=\mathrm{CLQ}^{(1)}_{AB}$）。

    每次置换按 :func:`simulate_independent_placement` 重抽位置并重建权重矩阵，
    故零分布应围绕 $\mathrm{CLQ}^{(1)}$（而非 1）——与"全局重排"零假设形成对照：
    前者检验"是否只是分布重叠"，后者检验"是否偏离 1"。

    Returns
    -------
    dict(``clq_obs``, ``p``, ``n_perm``, ``mean_null``, ``sd_null``, ``mode``)
    """
    xy = np.asarray(xy, dtype=np.float64)
    types = np.asarray(types)
    if smooth is None:
        smooth = b
    obs = _clq_pairs(xy, types, src, tgt, b, radius_mult, kernel)
    vals = []
    for r in range(n_perm):
        xn, tn = simulate_independent_placement(xy, types, n_types, smooth,
                                                region, grid_n, seed + r)
        v = _clq_pairs(xn, tn, src, tgt, b, radius_mult, kernel)
        if np.isfinite(v):
            vals.append(v)
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size == 0:
        return {'clq_obs': float(obs), 'p': np.nan, 'n_perm': 0,
                'mean_null': np.nan, 'sd_null': np.nan,
                'mode': 'independent_placement'}
    p = (np.sum(np.abs(vals - 1.0) >= abs(obs - 1.0)) + 1.0) / (len(vals) + 1.0)
    return {'clq_obs': float(obs), 'p': float(p), 'n_perm': int(len(vals)),
            'mean_null': float(vals.mean()),
            'sd_null': float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
            'mode': 'independent_placement'}
