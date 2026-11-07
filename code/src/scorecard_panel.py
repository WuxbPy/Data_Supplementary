# -*- coding: utf-8 -*-
r"""College Scorecard **30 期 MERGED 面板**载入与网格面板构造（B2 用）。

数据事实（已实测）
------------------
* ``College_Scorecard_Raw_Data_*.zip`` 内含 ``MERGED1996_97_PP.csv`` … ``MERGED2025_26_PP.csv``
  共 **30 期**，每期 **3,308 列**，关键列（``UNITID/CONTROL/ICLEVEL/LATITUDE/LONGITUDE/STABBR``）
  **跨期齐备**；
* **``MAIN`` 在 MERGED 各期不可用**（筛 ``MAIN==1`` 得 0 行）——该列仅 ``Most-Recent-Cohorts`` 有。
  故面板口径为 **CONUS × 在运营 × 坐标有效**（含分校区），与 B1 的"主校区"口径不同，须在文中声明；
* 为省磁盘，直接**从 zip 流式读取**（``usecols`` 只取所需列，30 期约 20 s）。

单位与分析单元
--------------
* **机构**：``UNITID``；**类型**：``CONTROL``（办学性质）、``ICLEVEL``（办学层次）
* **空间单元**：Albers 米制平面上的**正方形网格**（默认 120 km）。
  Albers 为绝对投影（不依赖数据），故各期坐标**天然可比**，无需共同原点。
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import ams_clq as ac

_ROOT = Path(__file__).resolve().parents[2]          # 项目根（含 CausalCLQ/ 与 书稿/）
#: zip 候选位置（按序探测）
ZIP_CANDIDATES = (
    _ROOT / 'CausalCLQ' / 'data' / 'Scorecard' / 'raw'
    / 'College_Scorecard_Raw_Data_06102026.zip',
    _ROOT / '书稿' / '校园案例' / 'data' / 'College_Scorecard_Raw_Data_06102026.zip',
)


def default_zip() -> Path:
    """返回存在的 zip 路径（按 :data:`ZIP_CANDIDATES` 顺序探测）。"""
    for p in ZIP_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        '未找到 College Scorecard zip。请放置于以下任一位置：\n  '
        + '\n  '.join(str(p) for p in ZIP_CANDIDATES))


DEFAULT_ZIP = None    #: 兼容占位；请用 :func:`default_zip`

#: 面板所需列（实测 30 期齐备）
PANEL_COLS = ('UNITID', 'INSTNM', 'STABBR', 'ZIP', 'CITY',
              'LATITUDE', 'LONGITUDE', 'CONTROL', 'ICLEVEL', 'CURROPER', 'UGDS')

_MEMBER_RE = re.compile(r'MERGED(\d{4})_\d{2}_PP\.csv$')


def list_periods(zip_path=None) -> List[Tuple[int, str]]:
    """返回 ``[(起始年, zip 内成员名), ...]``，按年份升序（30 期）。"""
    zp = Path(zip_path) if zip_path else default_zip()
    if not zp.exists():
        raise FileNotFoundError(f'未找到 Scorecard zip：{zp}')
    out = []
    with zipfile.ZipFile(zp) as z:
        for nm in z.namelist():
            m = _MEMBER_RE.search(nm)
            if m:
                out.append((int(m.group(1)), nm))
    return sorted(out)


def _read_member(zp: Path, nm: str, cols: Sequence[str]) -> pd.DataFrame:
    with zipfile.ZipFile(zp) as z:
        with z.open(nm) as f:
            return pd.read_csv(io.TextIOWrapper(f, encoding='latin-1'),
                               usecols=lambda c: c in set(cols), low_memory=False)


def zip5_of(series: pd.Series) -> pd.Series:
    """Scorecard 的 ``ZIP`` 为 ZIP+4（如 ``35294-0110``）→ 取**前 5 位** ZIP5。"""
    return (series.astype(str).str.strip().str.slice(0, 5)
            .where(lambda s: s.str.fullmatch(r'\d{5}')))


def build_zip_lookup(zip_path=None, ref_year: int = 2025,
                     verbose: bool = False) -> pd.DataFrame:
    r"""构造 **ZIP5 → (lat, lon)** 查询表，来源为**唯一有坐标的期**（默认 2025）。

    **必要性（关键数据约束）**：实测 MERGED 1996–2024 共 29 期
    ``LATITUDE``/``LONGITUDE``/``CURROPER`` **全为空**，仅 2025_26 期有坐标
    （5,731 个）。而 ``ZIP`` 在 **30 期全部非缺**，故以 ZIP5 为桥梁把坐标
    回链到历史期。

    Returns
    -------
    DataFrame(index=ZIP5, columns=``lat``, ``lon``, ``n_inst``)
    """
    zp = Path(zip_path) if zip_path else default_zip()
    nm = dict(list_periods(zp))[ref_year]
    ref = _read_member(zp, nm, ['UNITID', 'LATITUDE', 'LONGITUDE', 'ZIP'])
    ref['LATITUDE'] = pd.to_numeric(ref['LATITUDE'], errors='coerce')
    ref['LONGITUDE'] = pd.to_numeric(ref['LONGITUDE'], errors='coerce')
    ref = ref.dropna(subset=['LATITUDE', 'LONGITUDE'])
    ref['zip5'] = zip5_of(ref['ZIP'])
    ref = ref.dropna(subset=['zip5'])
    lut = (ref.groupby('zip5')
           .agg(lat=('LATITUDE', 'mean'), lon=('LONGITUDE', 'mean'),
                n_inst=('UNITID', 'size')))
    if verbose:
        print(f'  [ZIP5 查询表] {len(lut):,} 个 ZIP5（来自 {ref_year} 期 '
              f'{len(ref):,} 所带坐标机构）')
    return lut


def load_merged_period(zip_path=None, year: int = 2025,
                       cols: Sequence[str] = PANEL_COLS,
                       zip_lut: Optional[pd.DataFrame] = None,
                       zip_ref_year: int = 2025,
                       conus: bool = True, jitter_km: float = 3.0,
                       seed: int = 0, verbose: bool = False) -> pd.DataFrame:
    """载入单期 MERGED → 机构点表（Albers ``x``/``y`` 由 **ZIP5 回链**得到）。

    Parameters
    ----------
    year : 期起始年（1996–2025）
    zip_lut : :func:`build_zip_lookup` 的输出；None 时内部构造（多期循环请外部传入以复用）
    jitter_km : ZIP5 内均匀抖动半径（km）——ZIP5 是面而非点，抖动避免同位点堆叠
    """
    zp = Path(zip_path) if zip_path else default_zip()
    periods = dict(list_periods(zp))
    if year not in periods:
        raise ValueError(f'无 {year} 期；可用：{sorted(periods)}')
    df = _read_member(zp, periods[year], cols)
    if 'UGDS' in df.columns:
        df['UGDS'] = pd.to_numeric(df['UGDS'], errors='coerce')
    if conus:
        df = df[df['STABBR'].isin(ac.CONUS_STATES)]
    df = df.drop_duplicates(subset=['UNITID']).reset_index(drop=True)

    if zip_lut is None:
        zip_lut = build_zip_lookup(zp, zip_ref_year)
    df['zip5'] = zip5_of(df['ZIP'])
    df = df.merge(zip_lut, left_on='zip5', right_index=True, how='left')

    # 直接在经纬度上做 ZIP5 内抖动（约 ±jitter_km）
    rng = np.random.default_rng(seed + int(year))
    ok = df['lat'].notna().to_numpy()
    if ok.any():
        dlat = jitter_km / 111.32
        dlon = jitter_km / (111.32 * np.cos(np.radians(df.loc[ok, 'lat'].to_numpy())))
        df.loc[ok, 'lat'] += rng.uniform(-dlat, dlat, ok.sum())
        df.loc[ok, 'lon'] += rng.uniform(-1, 1, ok.sum()) * dlon

    df = df.dropna(subset=['lat', 'lon']).reset_index(drop=True)
    df['x'], df['y'] = ac.project_albers_conus(df['lon'].to_numpy(),
                                                df['lat'].to_numpy())
    df['control'] = df['CONTROL'].map(ac.CONTROL_LABELS)
    df['iclevel'] = df['ICLEVEL'].map(ac.ICLEVEL_LABELS)
    df['year'] = int(year)
    if verbose:
        print(f'  [{year}] {len(df):,} 所（坐标经 ZIP5 回链）  '
              f'{df["control"].value_counts().to_dict()}')
    return df


def build_panel(zip_path=None, years: Optional[Sequence[int]] = None,
                zip_ref_year: int = 2025, verbose: bool = True) -> pd.DataFrame:
    """堆叠各期为**机构-期面板**（ZIP5 查询表只建一次）。"""
    zp = Path(zip_path) if zip_path else default_zip()
    periods = [y for y, _ in list_periods(zp)]
    years = list(years) if years is not None else periods
    lut = build_zip_lookup(zp, zip_ref_year, verbose=verbose)
    frames = [load_merged_period(zp, y, zip_lut=lut, zip_ref_year=zip_ref_year)
              for y in years]
    out = pd.concat(frames, ignore_index=True)
    if verbose:
        print(f'  合并面板：{len(out):,} 行（{len(years)} 期 × 机构）')
    return out


# ---------------------------------------------------------------- 网格面板

def add_grid_unit(df: pd.DataFrame, cell_m: float = 120_000.0,
                  origin: Optional[Tuple[float, float]] = None):
    """加网格单元 id（返回 ``(df, origin)``；``origin`` 可固定以跨期对齐）。"""
    if origin is None:
        origin = (float(np.floor(df['x'].min() / cell_m) * cell_m),
                  float(np.floor(df['y'].min() / cell_m) * cell_m))
    ix = np.floor((df['x'] - origin[0]) / cell_m).astype(np.int64)
    iy = np.floor((df['y'] - origin[1]) / cell_m).astype(np.int64)
    df = df.copy()
    df['cell'] = [f'{a}_{b}' for a, b in zip(ix, iy)]
    df['cx'] = origin[0] + (ix + 0.5) * cell_m
    df['cy'] = origin[1] + (iy + 0.5) * cell_m
    return df, origin


def cell_type_counts(panel: pd.DataFrame, cell_m: float = 120_000.0,
                     origin: Optional[Tuple[float, float]] = None,
                     type_col: str = 'control') -> pd.DataFrame:
    """**平衡**单元×期面板：每格的各类型机构计数（无机构的格计 0）。

    Returns
    -------
    DataFrame(``cell``, ``year``, ``n_<type>``…, ``N``, ``cx``, ``cy``)
    """
    df, origin = add_grid_unit(panel, cell_m, origin)
    cats = sorted(df[type_col].dropna().unique())
    cells = sorted(df['cell'].unique())
    years = sorted(df['year'].unique())
    full = pd.MultiIndex.from_product([cells, years], names=['cell', 'year'])
    base = (df.groupby(['cell', 'year', type_col]).size().unstack(fill_value=0)
            .reindex(full, fill_value=0))
    base.columns = [f'n_{_slug(c)}' for c in base.columns]
    base['N'] = base.sum(axis=1)
    cent = (df.groupby('cell')[['cx', 'cy']].first())
    out = base.reset_index()
    out = out.merge(cent, left_on='cell', right_index=True, how='left')
    out.attrs['origin'] = origin
    return out


def _slug(s: str) -> str:
    """类型名 → 安全列名后缀。"""
    return (str(s).lower().replace(' ', '_').replace('-', '_')
            .replace('+', 'p').replace('|', '_'))


TARGET_TYPES = {c: _slug(c) for c in ac.CONTROL_LABELS.values()}
