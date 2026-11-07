# -*- coding: utf-8 -*-
"""College Scorecard 30 期面板模块测试（``src/scorecard_panel.py``）。

覆盖：ZIP5 解析、ZIP5→坐标查询表、单期载入、网格单元、平衡计数面板。
真实数据缺失时整组跳过。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import scorecard_panel as sp      # noqa: E402


def _zip_or_skip():
    try:
        return sp.default_zip()
    except FileNotFoundError:
        pytest.skip('未找到 College Scorecard zip')


# ---------------------------------------------------------------- ZIP5 解析

def test_zip5_of_handles_zip4_and_invalid():
    s = pd.Series(['35294-0110', '10001', ' 02138 ', 'ABCDE', None, '1234'])
    out = sp.zip5_of(s)
    assert out.iloc[0] == '35294'
    assert out.iloc[1] == '10001'
    assert out.iloc[2] == '02138'
    assert out.iloc[3:].isna().all()      # 非 5 位数字 → NaN


# ---------------------------------------------------------------- 期次清单

def test_list_periods_is_30_and_sorted():
    zp = _zip_or_skip()
    p = sp.list_periods(zp)
    assert len(p) == 30
    yrs = [y for y, _ in p]
    assert yrs == sorted(yrs)
    assert yrs[0] == 1996 and yrs[-1] == 2025


# ---------------------------------------------------------------- 查询表

def test_zip_lookup_valid():
    zp = _zip_or_skip()
    lut = sp.build_zip_lookup(zp, 2025)
    assert len(lut) > 3000
    assert {'lat', 'lon', 'n_inst'} <= set(lut.columns)
    assert lut['lat'].between(-90, 90).all()
    assert lut['lon'].between(-180, 180).all()
    assert (lut['n_inst'] >= 1).all()


# ---------------------------------------------------------------- 单期载入

@pytest.mark.parametrize('year', [1996, 2012, 2025])
def test_load_merged_period(year):
    zp = _zip_or_skip()
    lut = sp.build_zip_lookup(zp, 2025)
    d = sp.load_merged_period(zp, year, zip_lut=lut)
    assert len(d) > 3000
    assert {'x', 'y', 'control', 'iclevel', 'year', 'zip5'} <= set(d.columns)
    assert (d['year'] == year).all()
    assert np.isfinite(d['x']).all() and np.isfinite(d['y']).all()
    assert d['control'].nunique() == 3
    assert d['UNITID'].is_unique
    # 仅本土：不应出现 AK / HI
    assert not d['STABBR'].isin(['AK', 'HI']).any()
    # 坐标范围应落在 CONUS 尺度内（Albers，km）
    assert d['x'].abs().max() < 6e6
    assert d['y'].between(0, 4e6).all()


def test_coordinate_imputation_coverage():
    """ZIP5 回链覆盖率应 ≥75%（实测均值 87.5%，1996 最低 79.6%）。"""
    zp = _zip_or_skip()
    lut = sp.build_zip_lookup(zp, 2025)
    for y in (1996, 2012, 2025):
        d = sp.load_merged_period(zp, y, zip_lut=lut)
        n_mapped = int(d['zip5'].isin(lut.index).sum())
        assert n_mapped / len(d) > 0.75


# ---------------------------------------------------------------- 网格单元

def test_add_grid_unit_consistency():
    df = pd.DataFrame({'x': [0.0, 1.0, 120000.0, 239999.0, 240000.0],
                       'y': [0.0, 1.0, 0.0, 0.0, 0.0]})
    out, origin = sp.add_grid_unit(df, cell_m=120000.0, origin=(0.0, 0.0))
    # 同一网格内前两点应同 id；跨 120 km 边界换 id
    assert out['cell'].iloc[0] == out['cell'].iloc[1]      # [0,120k) → 格 0
    assert out['cell'].iloc[2] != out['cell'].iloc[0]      # 恰好 120k → 格 1
    assert out['cell'].iloc[3] == out['cell'].iloc[2]      # 239,999 仍在格 1
    assert out['cell'].iloc[4] != out['cell'].iloc[3]      # 240,000 → 格 2
    assert out['cell'].iloc[2] == '1_0'
    assert origin == (0.0, 0.0)


def test_cell_type_counts_balanced():
    """计数面板应为平衡面板：每个 cell × 每个 year 一行。"""
    zp = _zip_or_skip()
    lut = sp.build_zip_lookup(zp, 2025)
    yrs = [2010, 2011, 2012]
    panel = pd.concat([sp.load_merged_period(zp, y, zip_lut=lut) for y in yrs],
                      ignore_index=True)
    ctp = sp.cell_type_counts(panel, cell_m=200_000.0, origin=(0.0, 0.0))
    assert set(ctp.columns) >= {'cell', 'year', 'N', 'n_public'}
    assert len(ctp) == ctp['cell'].nunique() * len(yrs)
    # 计数非负且 N 等于三类之和
    ncols = [c for c in ctp.columns if c.startswith('n_')]
    assert (ctp[ncols].to_numpy() >= 0).all()
    assert np.allclose(ctp['N'], ctp[ncols].sum(axis=1))
    # 总计数应等于机构总数（无重复计入）
    assert ctp['N'].sum() == len(panel)


def test_slug_stability():
    assert sp._slug('Private for-profit') == 'private_for_profit'
    assert sp._slug('Public') == 'public'
    assert sp._slug('4-year+') == '4_yearp'
