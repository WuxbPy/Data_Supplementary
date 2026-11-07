# -*- coding: utf-8 -*-
"""AMS-CLQ 模块测试（``src/ams_clq.py``）。

**核心是估计量验证**：用已知真值的合成点格局检验
（i）H0（无二阶交互）下 :math:`\\mathrm{CLQ}^{(2)}=1`；
（ii）H1（真实吸引）下 :math:`\\mathrm{CLQ}^{(2)}>1` 且随强度单调；
（iii）一阶/二阶分离对平滑尺度的依赖方向。
其余为投影、κ、两类零假设与真实数据冒烟。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import ams_clq as ac      # noqa: E402

SCORECARD = (Path(__file__).resolve().parents[1] / 'data' / 'Scorecard' / 'raw'
             / 'Most-Recent-Cohorts-Institution.csv')


# ---------------------------------------------------------------- 合成 DGP

def synthetic(seed=0, nA=2000, nB=2000, b=25e3, h=100e3, frac=0.0, r=2e3,
              L=500e3, blob=False):
    """合成两类点格局。

    ``frac`` 比例的 B 点被搬到 A 点的 ``r`` 邻域内（制造**真实二阶吸引**）；
    ``blob=True`` 时两类型各为部分重叠的高斯斑块（制造**一阶重叠**）。
    """
    rng = np.random.default_rng(seed)
    if blob:
        A = np.column_stack([rng.normal(300e3, 150e3, nA), rng.normal(500e3, 150e3, nA)])
        B = np.column_stack([rng.normal(650e3, 150e3, nB), rng.normal(500e3, 150e3, nB)])
    else:
        A = rng.uniform(0, L, (nA, 2))
        B = rng.uniform(0, L, (nB, 2))
    if frac > 0:
        k = int(frac * nB)
        c = A[rng.integers(0, len(A), k)]
        B[:k] = c if r <= 0 else c + rng.normal(0, r, size=(k, 2))
    xy = np.vstack([A, B])
    types = np.r_[np.ones(nA, int), np.full(nB, 2, int)]
    return xy, types, b, h


# ---------------------------------------------------------------- 投影

def test_projection_accuracy_vs_great_circle():
    """Albers 投影距离 vs 大圆距离，误差应 < 2%。"""
    R = 6371.0088

    def gc(lon1, lat1, lon2, lat2):
        la1, lo1, la2, lo2 = map(np.radians, [lat1, lon1, lat2, lon2])
        return 2 * R * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 +
                                         np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))
    pts = [((-74.006, 40.7128), (-118.2437, 34.0522)),      # NYC–LA
           ((-87.6298, 41.8781), (-95.3698, 29.7604)),      # Chicago–Houston
           ((-71.0589, 42.3601), (-80.1918, 25.7617))]      # Boston–Miami
    for (lo1, la1), (lo2, la2) in pts:
        x1, y1 = ac.project_albers_conus(np.array([lo1]), np.array([la1]))
        x2, y2 = ac.project_albers_conus(np.array([lo2]), np.array([la2]))
        proj = np.hypot(x1[0] - x2[0], y1[0] - y2[0]) / 1000
        assert abs(proj / gc(lo1, la1, lo2, la2) - 1) < 0.02


# ---------------------------------------------------------------- 估计量验证

def test_clq2_equals_one_under_no_interaction():
    """**H0**：无二阶交互 → CLQ⁽²⁾ = 1（且 CLQ = CLQ⁽¹⁾）。"""
    vals = []
    for s in range(4):
        xy, types, b, h = synthetic(seed=s, frac=0.0)
        W = ac.build_weight_matrix(xy, b)
        d = ac.clq_decompose(W, xy, types, 1, 2, b, smooth_scale=h,
                             first_order='nw')
        vals.append((d['CLQ'], d['CLQ1'], d['CLQ2']))
    clq, clq1, clq2 = np.mean(vals, axis=0)
    assert abs(clq - 1.0) < 0.02
    assert abs(clq1 - 1.0) < 0.02
    assert abs(clq2 - 1.0) < 0.02


def test_clq2_detects_attraction_monotonically():
    """**H1**：真实吸引 → CLQ⁽²⁾ 显著 > 1，且随吸引比例单调。

    .. note:: 显式使用 ``first_order='nw'``：该合成点数相对 120×120 网格过少，
       grid 路线受网格噪声影响（约 1.7% 偏移）。grid 路线的正确性由
       :func:`test_grid_first_order_matches_upstream` 与
       `amsclq/examples_bias_study/` 的实验保证。
    """
    out = []
    for frac in (0.0, 0.5, 1.0):
        v = []
        for s in range(4):
            xy, types, b, h = synthetic(seed=s, frac=frac)
            W = ac.build_weight_matrix(xy, b)
            v.append(ac.clq_decompose(W, xy, types, 1, 2, b, smooth_scale=h,
                                      first_order='nw')['CLQ2'])
        out.append(float(np.mean(v)))
    assert out[0] == pytest.approx(1.0, abs=0.02)
    assert out[1] > out[0] + 0.003           # 50% 吸引可检出
    assert out[2] > out[1] + 0.003           # 100% 更强
    assert out[2] > 1.01


def test_first_order_explains_blob_structure():
    """一阶重叠（两个部分重叠斑块）：CLQ 明显≠1，但一阶分离吸收**大部分**偏离。

    注：两斑块近乎不重叠时，比值型 CLQ 估计量本身不稳定（邻域内 B 计数接近 0），
    故这里检验**定性性质**——一阶分离后偏离至少缩小一半。
    """
    xy, types, b, h = synthetic(seed=1, blob=True, frac=0.0)
    W = ac.build_weight_matrix(xy, b)
    d = ac.clq_decompose(W, xy, types, 1, 2, b, smooth_scale=h, first_order='nw')
    assert d['CLQ'] < 0.9                              # 明显的"排斥"信号
    assert abs(d['CLQ2'] - 1) < 0.5 * abs(d['CLQ'] - 1)  # 一阶吸收 > 一半


def test_clq2_sensitive_to_smoothing_scale():
    """**尺度一致性（NW 路线回归）**：一阶平滑尺度改变 CLQ⁽²⁾。

    grid 路线下同样的敏感性由 ``smooth_mult`` 控制，见
    `amsclq/examples_bias_study/exp3_smoothing_sensitivity.py`。
    """
    fine, coarse = [], []
    for s in range(4):
        xy, types, b, _ = synthetic(seed=10 + s, frac=1.0)
        W = ac.build_weight_matrix(xy, b)
        fine.append(ac.clq_decompose(W, xy, types, 1, 2, b, smooth_scale=b,
                                     first_order='nw')['CLQ2'])
        coarse.append(ac.clq_decompose(W, xy, types, 1, 2, b, smooth_scale=b * 4.0,
                                       first_order='nw')['CLQ2'])
    f, c = float(np.mean(fine)), float(np.mean(coarse))
    assert f == pytest.approx(1.0, abs=0.03)   # h=b：二阶被吸收，CLQ⁽²⁾≈1
    assert c > f                               # h=4b：二阶显现，偏离更大


# ---------------------------------------------------------------- κ 与核

def test_kappa_scales_as_area():
    """κ ≈ λ_B π b²：b 加倍 → κ 约 4 倍（小尺度下）。"""
    xy, types, b, h = synthetic(seed=3, frac=0.0)
    k1 = ac.kappa_ab(xy, types, 1, 2, b)
    k2 = ac.kappa_ab(xy, types, 1, 2, 2 * b)
    assert k2 > 3.0 * k1


def test_kernel_shapes():
    d = np.array([0.0, 1.0, 10.0])
    assert ac._kernel(d, 1.0, 'gauss')[0] == pytest.approx(1.0)
    assert np.all(ac._kernel(d, 1.0, 'gauss') <= 1.0)
    assert ac._kernel(d, 1.0, 'uniform').tolist() == [1.0, 1.0, 0.0]
    with pytest.raises(ValueError):
        ac._kernel(d, 1.0, 'bogus')


def test_local_type_share_bounds():
    xy, types, b, h = synthetic(seed=4, frac=0.0)
    q = ac.local_type_share(xy, (types == 2).astype(float), h)
    assert np.all((q >= 0) & (q <= 1))
    assert abs(q.mean() - 0.5) < 0.05


# ---------------------------------------------------------------- 零假设检验

def test_two_null_tests_ordering():
    """一阶重叠下：标准检验显著，而**一阶保留零分布围绕"一阶水平"而非 1**。

    这正是 AMS-CLQ 的机制：CLQ≠1 由一阶分布重叠造成；
    以一阶场为参照后，观测偏离被大幅削弱。
    """
    xy, types, b, h = synthetic(seed=5, blob=True, frac=0.0)
    W = ac.build_weight_matrix(xy, b)
    q_all = ac.local_type_shares_all(xy, types, 2, h)
    g = ac.null_clq_test(W, xy, types, 1, 2, mode='global', n_perm=299)
    f = ac.null_clq_test(W, xy, types, 1, 2, q_all=q_all, mode='first_order',
                         n_perm=299)
    assert g['p'] < 0.05                              # 标准检验：显著
    # 全局零分布居中于 1（不保留一阶）；一阶保留零分布居中于观测的一阶水平
    assert abs(g['mean_null'] - 1.0) < 0.03
    assert abs(f['mean_null'] - 1.0) > abs(g['mean_null'] - 1.0)
    # 以"偏离 1 的程度"衡量，一阶保留检验应弱于标准检验
    assert f['p'] >= g['p']
    with pytest.raises(ValueError):
        ac.null_clq_test(W, xy, types, 1, 2, mode='first_order', n_perm=5)


def test_all_pairs_shape():
    xy, types, b, h = synthetic(seed=6, frac=0.0)
    labels = {1: 'A', 2: 'B'}
    tab = ac.all_pairs(xy, types, labels, b, smooth_scale=h, first_order='nw')
    assert len(tab) == 4                  # 2×2 有向
    assert set(tab.columns) >= {'source', 'target', 'CLQ', 'CLQ1', 'CLQ2', 'kappa'}


# ---------------------------------------------------------------- 真实数据冒烟

def test_load_scorecard_smoke():
    if not SCORECARD.exists():
        pytest.skip('未找到 Scorecard 原始文件')
    df = ac.load_scorecard(verbose=False)
    assert len(df) > 3000
    assert {'x', 'y', 'control', 'iclevel', 'type6'} <= set(df.columns)
    assert df['x'].abs().mean() > 1e5     # 米制坐标
    assert df['control'].nunique() == 3
    # 仅保留本土
    assert not (df['STABBR'] == 'AK').any()
    assert not (df['STABBR'] == 'HI').any()


# ---------------------------------------------------------------- grid 路线（原文规格）

def test_grid_first_order_matches_upstream():
    """**对齐测试**：本文 grid 一阶项与上游 ``amsclq.heterogeneous`` 逐位一致。

    这是本项目最重要的回归测试——2026-09-30 的教训：任何"与原文的差异"
    必须在完成此对拍后才允许解释。
    """
    pytest.importorskip('amsclq.heterogeneous', reason='未安装上游 amsclq 包')
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from amsclq.heterogeneous import first_order_clq as up
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 6000, (3000, 2))
    types = rng.integers(1, 4, 3000).astype(np.int64)
    for b in (150.0, 500.0, 1200.0):
        mine = ac.first_order_clq_grid(pts, types, b, 1, 2, region=6000.0,
                                       grid_n=120)
        theirs = float(np.asarray(up(pts, types, [b], 1, 2, [1, 2, 3],
                                     6000.0, 120)).ravel()[0])
        assert mine == pytest.approx(theirs, rel=1e-12, abs=1e-12)


def test_grid_is_default_first_order():
    """``clq_decompose`` 默认走 grid 路线，``smooth_mult=1`` 即原文设定。"""
    xy, types, b, _ = synthetic(seed=1, frac=0.0)
    W = ac.build_weight_matrix(xy, b)
    d = ac.clq_decompose(W, xy, types, 1, 2, b)
    assert d['first_order'] == 'grid' and d['smooth_mult'] == 1.0
    assert d['CLQ1'] == pytest.approx(ac.first_order_clq_grid(xy, types, b, 1, 2),
                                      rel=1e-12)


def test_grid_guard_rejects_region_too_small():
    """显式 ``region`` 过小时必须**报错**，而非静默丢点。"""
    xy, types, b, _ = synthetic(seed=3, frac=0.0)
    with pytest.raises(ValueError, match='落在网格外'):
        ac.first_order_clq_grid(xy, types, b, 1, 2, region=1.0)


def test_independent_placement_preserves_counts():
    """独立落点零格局应保持各类型点数不变。"""
    xy, types, b, _ = synthetic(seed=5, frac=0.0)
    xn, tn = ac.simulate_independent_placement(xy, types, 2, smooth=0.0,
                                               grid_n=60, seed=1)
    assert len(xn) == len(xy)
    assert (np.bincount(tn, minlength=3) == np.bincount(types, minlength=3)).all()
    assert np.isfinite(xn).all()


def test_placement_null_mean_near_first_order():
    """独立落点零分布均值应接近 **CLQ⁽¹⁾**（而非 1）——零假设构造正确性。"""
    xy, types, b, _ = synthetic(seed=1, blob=True, frac=0.0)
    W = ac.build_weight_matrix(xy, b)
    d = ac.clq_decompose(W, xy, types, 1, 2, b)
    r = ac.null_clq_placement(xy, types, 1, 2, b, n_types=2, smooth=0.0,
                              grid_n=80, n_perm=40, seed=3)
    assert abs(r['mean_null'] - d['CLQ1']) < 6 * max(r['sd_null'], 1e-6), \
        f"零分布均值 {r['mean_null']:.4f} 偏离 CLQ1={d['CLQ1']:.4f} 过多"
