# -*- coding: utf-8 -*-
"""尺度谱诊断模块测试（``src/clq_scale.py``）。

覆盖：对数带宽网格、ρ（邻域覆盖比例）、可行域、**谱峰诊断的边界情形**
（单调 / 内部极大 / 内部极小 / 序列过短）、谱表与自相似性检查。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import clq_scale as cs      # noqa: E402


# ---------------------------------------------------------------- 带宽网格

def test_log_bandwidths():
    bs = cs.log_bandwidths(1e3, 1e5, 3)
    assert bs[0] == pytest.approx(1e3) and bs[-1] == pytest.approx(1e5)
    assert bs[1] == pytest.approx(1e4)          # 对数等距
    assert np.all(np.diff(bs) > 0)


# ---------------------------------------------------------------- ρ

def test_neighbourhood_fraction_direction():
    """ρ 随 b 单调不减；聚集格局的 ρ 大于均匀格局。"""
    rng = np.random.default_rng(0)
    uni = rng.uniform(0, 10_000, (1500, 2))
    blob = np.vstack([rng.normal(c, 300, (500, 2))
                      for c in [(3000, 3000), (7000, 7000), (5000, 2000)]])
    r1 = cs.neighbourhood_fraction(uni, 500.0)
    r2 = cs.neighbourhood_fraction(uni, 2000.0)
    assert r2 > r1
    assert cs.neighbourhood_fraction(blob, 500.0) > r1   # 聚集 ⇒ 邻域占比更高


def test_feasible_range_respects_rho():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 10_000, (1500, 2))
    bs = np.array([100.0, 300.0, 600.0, 1000.0, 2000.0])
    lo, hi = cs.feasible_range(xy, bs, rho_max=0.01)
    assert lo <= hi
    assert cs.neighbourhood_fraction(xy, hi) <= 0.01 * 1.05


# ---------------------------------------------------------------- 谱峰诊断

def test_peak_diagnostics_monotone_decreasing():
    d = cs.peak_diagnostics(np.array([3.0, 2.0, 1.5, 1.2]),
                            np.array([1.0, 2.0, 3.0, 4.0]))
    assert d['monotone_dec'] and not d['monotone_inc']
    assert not d['has_interior_max']
    assert d['argmax_is_boundary']            # argmax 落在左端
    assert d['n_interior_extrema'] == 0
    assert d['argmax_b'] == 1.0


def test_peak_diagnostics_interior_max():
    d = cs.peak_diagnostics(np.array([1.2, 1.8, 2.4, 1.9, 1.3]),
                            np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert d['has_interior_max']
    assert not d['argmax_is_boundary']
    assert d['argmax_b'] == 3.0
    assert d['n_interior_extrema'] == 1


def test_peak_diagnostics_interior_min_only():
    """只有内部**极小**（浅谷）时不应判为有内部极大值。"""
    d = cs.peak_diagnostics(np.array([1.18, 1.11, 1.08, 1.09, 1.11]),
                            np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert not d['has_interior_max']          # 谷不算峰
    assert not d['monotone_dec'] and not d['monotone_inc']
    assert d['n_interior_extrema'] >= 1


def test_peak_diagnostics_short_and_nan():
    d = cs.peak_diagnostics(np.array([1.0, 1.1]), np.array([1.0, 2.0]))
    assert not d['has_interior_max'] and d['argmax_is_boundary']
    d2 = cs.peak_diagnostics(np.array([1.0, np.nan, 1.2, 1.1]),
                             np.array([1.0, 2.0, 3.0, 4.0]))
    assert np.isfinite(d2['argmax_b'])


# ---------------------------------------------------------------- 谱表

def _synthetic(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 20_000, (n, 2))
    types = rng.integers(1, 3, n).astype(np.int64)
    return xy, types


def test_spectrum_columns_and_shape():
    xy, types = _synthetic()
    pairs = [(1, 2, 'P12'), (1, 1, 'P11')]
    bs = np.array([200.0, 500.0, 1000.0])
    spec = cs.spectrum(xy, types, pairs, bs, region=20_000.0,
                       origin=(0.0, 0.0), grid_n=60)
    assert len(spec) == len(pairs) * len(bs)
    assert {'pair', 'b_km', 'CLQ', 'CLQ1', 'CLQ2', 'kappa', 'rho'} <= set(spec.columns)
    assert np.isfinite(spec.CLQ).all()
    assert (spec.rho >= 0).all()


def test_spectrum_diagnostics_shape():
    xy, types = _synthetic()
    bs = np.array([200.0, 500.0, 1000.0, 2000.0])
    spec = cs.spectrum(xy, types, [(1, 2, 'P12')], bs, region=20_000.0,
                       origin=(0.0, 0.0), grid_n=60)
    diag = cs.spectrum_diagnostics(spec)
    assert len(diag) == 1
    assert {'pair', 'has_interior_max', 'argmax_b', 'kappa_min', 'rho_max'} <= set(diag.columns)


def test_spectrum_uniform_data_is_neutral():
    """完全均匀且独立标记时，CLQ 应在全部带宽上 ≈1（无任何结构）。"""
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 40_000, (4000, 2))
    types = np.r_[np.ones(2000, int), np.full(2000, 2, int)]
    rng.shuffle(types)
    bs = np.array([500.0, 1500.0, 3000.0])
    spec = cs.spectrum(xy, types, [(1, 2, 'P12')], bs, region=40_000.0,
                       origin=(0.0, 0.0), grid_n=60)
    assert np.allclose(spec.CLQ, 1.0, atol=0.06)


# ---------------------------------------------------------------- 自相似性

def test_self_similarity_shape():
    rng = np.random.default_rng(7)
    xy = rng.uniform(0, 10_000, (800, 2))          # 落在 [0,1e4]² 内
    types = rng.integers(1, 3, 800).astype(np.int64)
    bs = np.array([200.0, 500.0, 1000.0])
    sbd, span = {}, {}
    for i, (org, reg) in enumerate([((0.0, 0.0), 10_000.0), ((0.0, 0.0), 20_000.0)]):
        g = cs.spectrum(xy, types, [(1, 2, 'P12')], bs, region=reg, origin=org,
                        grid_n=40)
        sbd[f'R{i}'] = g
        span[f'R{i}'] = reg / 1000.0
    ss = cs.self_similarity_check(sbd, span)
    assert len(ss) == 2
    assert {'region', 'span_km', 'has_interior_max', 'clq_at_1pct_of_span'} <= set(ss.columns)
