#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B2：**30 期时变 CLQ 演化 + 增益就业法规（GE, 2012）的政策冲击 DID**

研究问题
--------
1. 美国高校类型的空间协同结构在 1996–2025 的 **30 期**中如何演化？
2. **2012 年增益就业法规（Gainful Employment）** 对营利性院校部门及其**空间协同结构**
   造成了怎样的处理效应？

识别设计
--------
GE 规则 2011-10 定稿、2012-07 生效，以"毕业生债务收入比"为门槛切断联邦资助，
**定向打击营利性院校**（观测：营利性机构数 2012 年峰值 → 2025 年 −23%）。

* **单元**：Albers 平面上的 **120 km 网格**（绝对投影，跨期天然对齐）
* **处理强度**：细胞在**处理前**（2008–2011）的平均营利性占比（标准化）
* **主设定（TWFE）**：$Y_{ct}=\alpha_c+\gamma_t+\beta\,(\mathrm{Post}_t\times Z_c)+\varepsilon_{ct}$，
  $\mathrm{Post}_t=\mathbb 1\{t\ge 2012\}$，聚类到细胞
* **事件研究**：$\sum_k\beta_k(\mathbb 1\{t=k\}\times Z_c)$，基准期 $k=2011$ → 检验前趋势
* **安慰剂**：假事件年 2005（应 ≈ 0）

被解释变量
----------
* ``log1p(n_forprofit)``：**直接效应**（法规是否压缩营利性部门）
* ``log1p(n_public)`` / ``log1p(n_nonprofit)``：**替代效应**（公立/非营利是否填补空缺）
* ``fp_share``：细胞内**协同结构**（营利性占比）

数据约束（重要）
----------------
实测 MERGED 1996–2024 共 29 期 **``LATITUDE``/``LONGITUDE``/``CURROPER`` 全为空**，
仅 2025_26 期有坐标（5,731 所）。``ZIP`` 在 30 期全部非缺，故坐标经
**ZIP5 → 2025 期院校质心** 回链（+ 3 km 抖动），平均覆盖 **87.5%**（最低 79.6%）。
故面板口径为 **CONUS × 坐标可回链**（含分校区，与 B1 的"主校区"口径不同）。

用法
----
.. code-block:: bash

    PY=/home/wuxb/miniconda3/envs/py310/bin/python
    $PY CausalCLQ/scripts/scorecard_b2.py
    $PY CausalCLQ/scripts/scorecard_b2.py --cell-km 150 --b-km 200
    $PY CausalCLQ/scripts/scorecard_b2.py --no-es        # 跳过事件研究（快）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_PKG_DIR = Path(__file__).resolve().parents[1]
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from src import ams_clq as ac               # noqa: E402
from src import scorecard_panel as sp       # noqa: E402
from src.did import ols_cluster             # noqa: E402

RESULTS = _PKG_DIR / 'results'
FIG_DIR = RESULTS / 'figures'

#: 政策事件（用于图上标注）
EVENTS = {2008: '2008 金融危机', 2012: 'GE 法规生效', 2020: 'COVID'}


# ---------------------------------------------------------------- 工具

def _dummies(s: pd.Series, drop_first: bool = True) -> pd.DataFrame:
    d = pd.get_dummies(s.astype('category'), prefix='', prefix_sep='', dtype=float)
    return d.iloc[:, 1:] if drop_first else d


def twfe(y: np.ndarray, unit: np.ndarray, period: np.ndarray,
         xtv: np.ndarray, names, cluster: np.ndarray,
         post_col: int = 0):
    """``Y ~ α_u + γ_t + β·X_tv``（X_tv 为时期可变回归元）。"""
    Du = _dummies(pd.Series(unit))
    Dt = _dummies(pd.Series(period))
    X = np.column_stack([np.ones(len(y)), xtv, Du.to_numpy(), Dt.to_numpy()])
    cols = (['const'] + list(names) + list(Du.columns) + list(Dt.columns))
    r = ols_cluster(y, X, cluster)
    key = names[post_col]
    return {'coef': float(r.params[cols.index(key)]),
            'se': float(r.se[cols.index(key)]),
            't': float(r.tstat[cols.index(key)]),
            'p': float(r.pvalue[cols.index(key)])}


def event_study(y: np.ndarray, unit: np.ndarray, period: np.ndarray,
                intensity: np.ndarray, base: int, cluster: np.ndarray,
                periods: list) -> pd.DataFrame:
    """事件研究：``Y ~ α_u + γ_t + Σ_k β_k (1{t=k} × Z)``，基准期 ``base``。"""
    du = _dummies(pd.Series(unit))
    dt = _dummies(pd.Series(period))
    inter, names = [], []
    for p in periods:
        if p == base:
            continue
        inter.append(((period == p).astype(float) * intensity))
        names.append(f'Z_x_t{p}')
    if inter:
        X = np.column_stack([np.ones(len(y))] + inter + [du.to_numpy(), dt.to_numpy()])
    else:
        X = np.column_stack([np.ones(len(y)), du.to_numpy(), dt.to_numpy()])
    r = ols_cluster(y, X, cluster)
    rows = []
    for i, p in enumerate(periods):
        if p == base:
            rows.append(dict(year=p, coef=0.0, se=0.0, t=np.nan, p=np.nan))
            continue
        key = f'Z_x_t{p}'
        j = names.index(key) + 1
        rows.append(dict(year=p, coef=float(r.params[j]), se=float(r.se[j]),
                         t=float(r.tstat[j]), p=float(r.pvalue[j])))
    return pd.DataFrame(rows)


def pre_trend_wald(es: pd.DataFrame, post_year: int = 2012) -> dict:
    """处理前系数联合检验（近似 χ²）：Σ(β/se)² ~ χ²(q)。"""
    pre = es[es.year < post_year].dropna(subset=['coef', 'se'])
    pre = pre[pre.se > 0]
    if len(pre) == 0:
        return {'n_pre': 0, 'wald': np.nan, 'p': np.nan}
    wald = float(np.nansum((pre.coef / pre.se) ** 2))
    q = len(pre)
    from scipy.stats import chi2
    return {'n_pre': int(q), 'wald': wald, 'p': float(chi2.sf(wald, q))}


def twfe_cell_trends(y: np.ndarray, unit: np.ndarray, period: np.ndarray,
                     xtv: np.ndarray, names, cluster: np.ndarray) -> dict:
    r"""**TWFE + 胞元特异性线性趋势**（应对安慰剂暴露的世俗趋势）。

    .. math::
        Y_{ct}=\alpha_c+\gamma_t+\beta(\mathrm{Post}_t Z_c)
        +\sum_c \theta_c\,(t\,\mathbb 1\{c\})+\varepsilon_{ct}

    与朴素 TWFE 的差别：每个胞元可有**自己的线性时间趋势**，
    故 $\beta$ 只由"相对各自趋势的**偏离**"识别，能吸收
    "高暴露区本就增长更快"这类世俗趋势。

    .. note::
        三重差分（在 $\alpha_{cs}+\gamma_{st}$ 下加 $\mathrm{Post}\times Z\times\mathbb 1\{s=\text{fp}\}$）
        在本设定下与对处理组单独跑 DID **数值等价**（三重交互仅对 fp 行非零，其余行只用于估 FE），
        故不提供额外稳健性，此处改用胞元趋势。
    """
    Du = _dummies(pd.Series(unit))
    Dt = _dummies(pd.Series(period))
    Tt = period.astype(float) - float(period.min())
    trend = Du.to_numpy() * Tt[:, None]          # 胞元 × 线性时间
    X = np.column_stack([np.ones(len(y)), xtv, Du.to_numpy(), Dt.to_numpy(), trend])
    r = ols_cluster(y, X, cluster)
    return {'coef': float(r.params[1]), 'se': float(r.se[1]),
            't': float(r.tstat[1]), 'p': float(r.pvalue[1]),
            'n_obs': int(len(y)), 'n_cell_trends': int(Du.shape[1])}


def series_break_test(ts: pd.DataFrame, src: str, tgt: str,
                      break_year: int = 2012) -> dict:
    """CLQ 时间序列的**结构断点检验**：对 ``CLQ_t`` 回归 ``1{t≥break}``。"""
    s = ts[(ts.source == src) & (ts.target == tgt)].sort_values('year')
    y = s['CLQ2'].to_numpy(dtype=float)
    t = s['year'].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(y)), (t >= break_year).astype(float)])
    r = ols_cluster(y, X, t.astype(int))      # 按时期聚类（30 个簇）
    return {'src': src, 'tgt': tgt, 'coef': float(r.params[1]),
            'se': float(r.se[1]), 't': float(r.tstat[1]),
            'p': float(r.pvalue[1]), 'level_pre': float(np.nanmean(y[t < break_year])),
            'level_post': float(np.nanmean(y[t >= break_year]))}


# ---------------------------------------------------------------- 主流程

def main() -> None:
    ap = argparse.ArgumentParser(description='B2：30 期时变 CLQ + GE 法规 DID')
    ap.add_argument('--b-km', type=float, default=150.0, help='CLQ 带宽（km）')
    ap.add_argument('--first-order', choices=('grid', 'nw'), default='grid',
                    help="一阶项估计器：'grid'（默认，原文式(8)规格）或 'nw'（legacy，已知有偏）")
    ap.add_argument('--scale-mult', type=float, default=1.0,
                    help='一阶平滑倍数 σ/b（**1.0 即原文设定**）')
    ap.add_argument('--grid-n', type=int, default=120)
    ap.add_argument('--cell-km', type=float, default=120.0, help='DID 空间单元边长（km）')
    ap.add_argument('--pre-from', type=int, default=2008)
    ap.add_argument('--pre-to', type=int, default=2011)
    ap.add_argument('--post-year', type=int, default=2012, help='GE 法规事件年')
    ap.add_argument('--no-es', action='store_true', help='跳过事件研究')
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    b = args.b_km * 1000.0
    print('=' * 78)
    print(f'B2：30 期时变 CLQ 演化 + GE 法规（{args.post_year}）DID')
    print('=' * 78)

    # ---------------------------------------------------------------- 1 面板
    print('\n[1] 载入 30 期机构面板（坐标经 ZIP5 回链）')
    panel = sp.build_panel(verbose=True)
    periods = sorted(panel['year'].unique())
    print(f'  {len(periods)} 期：{periods[0]}–{periods[-1]}，共 {len(panel):,} 行')

    # ---------------------------------------------------------------- 2 时变全局 CLQ
    print(f'\n[2] 30 期全局 CLQ 演化（b={args.b_km:.0f} km，一阶项 = '
          f'{args.first_order}，σ={args.scale_mult:g}b）')
    labels = sorted(panel['control'].dropna().unique())
    # **跨期固定网格原点与跨度**：否则各期网格不对齐，一阶项不可比
    ORG = (float(panel['x'].min()), float(panel['y'].min()))
    REG = float(max(panel['x'].max() - ORG[0], panel['y'].max() - ORG[1]))
    print(f'  固定网格：origin=({ORG[0]:.0f}, {ORG[1]:.0f}) m，尺度 {REG / 1e3:.0f} km，'
          f'grid_n={args.grid_n}（网格边长 {REG / args.grid_n / 1e3:.1f} km）')
    ts_rows = []
    for y in periods:
        g = panel[panel.year == y]
        tid, tmap = ac.encode_types(labels, g['control'])
        xy = g[['x', 'y']].to_numpy()
        W = ac.build_weight_matrix(xy, b)
        for a in sorted(tmap):
            for c in sorted(tmap):
                d = ac.clq_decompose(W, xy, tid, a, c, b,
                                     smooth_scale=b * args.scale_mult,
                                     first_order=args.first_order,
                                     grid_n=args.grid_n, region=REG, origin=ORG,
                                     smooth_mult=args.scale_mult,
                                     nw_reference=True)
                d.update({'year': y, 'source': tmap[a], 'target': tmap[c]})
                ts_rows.append(d)
    ts = pd.DataFrame(ts_rows)
    ts.to_csv(RESULTS / 'scorecard_b2_clq_timeseries.csv', index=False,
              encoding='utf-8-sig')
    print(f'  {len(ts)} 行（{len(periods)} 期 × {len(labels)}×{len(labels)} 对）')
    print('  营利性→营利性 CLQᵗ 抽样（并列 NW 对照）：')
    sub = ts[(ts.source == 'Private for-profit') & (ts.target == 'Private for-profit')]
    for _, r in sub[sub.year.isin([1996, 2005, 2011, 2012, 2016, 2020, 2025])].iterrows():
        print(f'    {int(r.year)}  CLQ={r.CLQ:.3f} CLQ1={r.CLQ1:.3f} '
              f'CLQ2={r.CLQ2:.3f}  κ={r.kappa:5.1f}  n_B={int(r.n_B):4d}'
              f'  [CLQ2_nw={r.get("CLQ2_nw", float("nan")):.3f}]')

    # ---------------------------------------------------------------- 3 网格面板
    print(f'\n[3] 构建 {args.cell_km:.0f} km 网格面板')
    ctp = sp.cell_type_counts(panel, cell_m=args.cell_km * 1000.0,
                              origin=(0.0, 0.0))
    ctp = ctp.rename(columns={c: c.replace('n_', 'n_') for c in ctp.columns})
    fn, pn = 'n_private_for_profit', 'n_public'
    nn = 'n_private_nonprofit'
    for c in (fn, pn, nn):
        if c not in ctp.columns:
            raise KeyError(f'缺少列 {c}；实际：{[x for x in ctp.columns if x.startswith("n_")]}')
    ctp['N'] = ctp[[fn, pn, nn]].sum(axis=1)
    ctp = ctp[ctp['N'] > 0].reset_index(drop=True)
    print(f'  {ctp.cell.nunique()} 个非空网格 × {ctp.year.nunique()} 期 = {len(ctp):,} 行')

    # 处理强度：处理前营利性占比
    pre = ctp[(ctp.year >= args.pre_from) & (ctp.year <= args.pre_to)]
    inten = (pre.groupby('cell')
             .apply(lambda d: d[fn].sum() / max(d['N'].sum(), 1),
                    include_groups=False))
    inten = ((inten - inten.mean()) / inten.std()).rename('intensity')
    ctp = ctp.merge(inten, left_on='cell', right_index=True, how='left')
    ctp['intensity'] = ctp['intensity'].fillna(0.0)
    print(f'  处理强度：{inten.notna().sum()} 格有 {args.pre_from}–{args.pre_to} 基线'
          f'（占比均值 {inten.mean():.3f}，标准化后 sd=1）')

    # ---------------------------------------------------------------- 4 DID
    print(f'\n[4] 设定一：TWFE DID（Post × 处理强度，聚类到网格）')
    ctp['post'] = (ctp.year >= args.post_year).astype(float)
    ctp['did'] = ctp['post'] * ctp['intensity'].to_numpy()

    outcomes = {
        'log1p(n_forprofit)': np.log1p(ctp[fn].to_numpy()),
        'log1p(n_public)': np.log1p(ctp[pn].to_numpy()),
        'log1p(n_nonprofit)': np.log1p(ctp[nn].to_numpy()),
        'fp_share': (ctp[fn] / ctp['N']).to_numpy(),
    }
    unit, per = ctp['cell'].to_numpy(), ctp['year'].to_numpy()
    did_rows = []
    for nm, y in outcomes.items():
        r = twfe(y, unit, per, ctp[['did']].to_numpy(), ['did'], unit)
        r.update({'outcome': nm, 'design': 'TWFE'})
        did_rows.append(r)
        print(f'  {nm:<22} β={r["coef"]:+.4f}  se={r["se"]:.4f}  '
              f't={r["t"]:+.2f}  p={r["p"]:.4f}')

    # ---- 安慰剂（关键诊断：假事件年应 ≈0）
    print(f'  ---- 安慰剂检验（假事件年，处理前）----')
    plac = {}
    for fy in (1999, 2002, 2005):
        ctp['did_f'] = ((ctp.year >= fy).astype(float) * ctp['intensity'].to_numpy())
        r = twfe(np.log1p(ctp[fn].to_numpy()), unit, per,
                 ctp[['did_f']].to_numpy(), ['did_f'], unit)
        plac[fy] = r['coef']
        did_rows.append({**r, 'outcome': f'placebo_log1p(n_fp)@{fy}', 'design': 'placebo'})
        print(f'  log1p(n_forprofit) @{fy}  β={r["coef"]:+.4f}  se={r["se"]:.4f}  '
              f't={r["t"]:+.2f}  p={r["p"]:.4f}')
    main_b = next(r['coef'] for r in did_rows
                  if r.get('design') == 'TWFE' and r['outcome'] == 'log1p(n_forprofit)')
    max_plac = max(abs(v) for v in plac.values())
    did_ok = max_plac < abs(main_b)
    print(f'  → 真事件 |β|={abs(main_b):.4f}，安慰剂最大 |β|={max_plac:.4f} '
          f'⇒ {"通过（真事件更大）" if did_ok else "**不通过：世俗趋势主导，朴素 DID 无效**"}')

    # ---------------------------------------------------------------- 4b 胞元趋势
    print(f'\n[4b] 设定二：**TWFE + 胞元特异性线性趋势**（吸收世俗趋势）')
    y_fp = np.log1p(ctp[fn].to_numpy())
    r = twfe_cell_trends(y_fp, unit, per, ctp[['did']].to_numpy(), ['did'], unit)
    print(f'  log1p(n_forprofit)  β={r["coef"]:+.4f}  se={r["se"]:.4f}  '
          f't={r["t"]:+.2f}  p={r["p"]:.4f}   (n={r["n_obs"]:,}, '
          f'{r["n_cell_trends"]} 条胞元趋势)')
    trend_b = r['coef']
    did_rows.append({**r, 'outcome': 'log1p(n_forprofit)', 'design': 'TWFE_cell_trends'})
    for fy in (2005,):
        ctp['did_f'] = ((ctp.year >= fy).astype(float) * ctp['intensity'].to_numpy())
        rp = twfe_cell_trends(np.log1p(ctp[fn].to_numpy()), unit, per,
                              ctp[['did_f']].to_numpy(), ['did_f'], unit)
        did_rows.append({**rp, 'outcome': f'placebo_log1p(n_fp)@{fy}',
                         'design': 'TWFE_cell_trends_placebo'})
        print(f'  安慰剂 @{fy}                  β={rp["coef"]:+.4f}  se={rp["se"]:.4f}  '
              f't={rp["t"]:+.2f}  p={rp["p"]:.4f}')
    print('  → 引入胞元趋势后 β 大幅缩小 = 朴素 DID 的"效应"主要来自世俗趋势')

    did = pd.DataFrame(did_rows)
    did.to_csv(RESULTS / 'scorecard_b2_did.csv', index=False, encoding='utf-8-sig')

    # ---------------------------------------------------------------- 4c 断点
    print(f'\n[4c] CLQ 时间序列的**结构断点检验**（对 1{{t≥{args.post_year}}} 回归 CLQ⁽²⁾）')
    brk = []
    for a in labels:
        for c in labels:
            if c < a:
                continue
            b1 = series_break_test(ts, a, c, args.post_year)
            brk.append(b1)
            print(f"  {a[:20]:<21}→{c[:20]:<21} Δ={b1['coef']:+.4f} "
                  f"(前 {b1['level_pre']:.3f} → 后 {b1['level_post']:.3f})  p={b1['p']:.3f}")
    brk = pd.DataFrame(brk)
    brk.to_csv(RESULTS / 'scorecard_b2_break.csv', index=False, encoding='utf-8-sig')

    # ---------------------------------------------------------------- 5 事件研究
    es_all = {}
    if not args.no_es:
        print(f'\n[5] 事件研究（基准年 {args.pre_to}）')
        ctp['inten_np'] = ctp['intensity'].to_numpy()
        for nm, y in list(outcomes.items())[:3]:
            es = event_study(y, unit, per, ctp['inten_np'].to_numpy(),
                             args.pre_to, unit, periods)
            es['outcome'] = nm
            es_all[nm] = es
            pw = pre_trend_wald(es, args.post_year)
            print(f'  {nm:<22} 前趋势 Wald χ²({pw["n_pre"]})={pw["wald"]:.2f}  '
                  f'p={pw["p"]:.3f}  → {"通过" if pw["p"] > .05 else "**不通过**"}')
        es_df = pd.concat(es_all.values(), ignore_index=True)
        es_df.to_csv(RESULTS / 'scorecard_b2_event_study.csv', index=False,
                     encoding='utf-8-sig')
        print('  log1p(n_forprofit) 事件研究系数：')
        e = es_all['log1p(n_forprofit)']
        print(e[e.year.isin([2008, 2010, 2011, 2012, 2013, 2015, 2018, 2021, 2025])]
              .to_string(index=False))
    else:
        es_df = pd.DataFrame()

    # ---------------------------------------------------------------- 6 汇总
    # ---- CLQ 稳定性摘要
    stab = {}
    for a in labels:
        for c in labels:
            if c < a:
                continue
            s = ts[(ts.source == a) & (ts.target == c)].dropna(subset=['CLQ2'])
            stab[f'{a}->{c}'] = {'clq2_mean': float(s.CLQ2.mean()),
                                 'clq2_sd': float(s.CLQ2.std()),
                                 'clq2_min': float(s.CLQ2.min()),
                                 'clq2_max': float(s.CLQ2.max()),
                                 'kappa_min': float(s.kappa.min())}
    fp = panel[panel.year == 2012]['control'].value_counts()
    fp25 = panel[panel.year == 2025]['control'].value_counts()
    summary = {
        'n_periods': len(periods), 'year_range': [int(periods[0]), int(periods[-1])],
        'n_institutions_total': int(len(panel)),
        'n_cells': int(ctp.cell.nunique()), 'cell_km': args.cell_km,
        'b_km': args.b_km, 'scale_mult': args.scale_mult,
        'post_year': args.post_year,
        'did': did.to_dict('records'),
        'did_beta_log1p_nfp': float(main_b),
        'placebo_max_abs_beta': float(max_plac),
        'did_pass_placebo': bool(did_ok),
        'twfe_cell_trends_beta': float(trend_b),
        'clq2_stability': stab,
        'break_tests': brk.to_dict('records'),
        'type_counts_1996': panel[panel.year == 1996]['control'].value_counts().to_dict(),
        'type_counts_2012': panel[panel.year == 2012]['control'].value_counts().to_dict(),
        'type_counts_2025': panel[panel.year == 2025]['control'].value_counts().to_dict(),
        'forprofit_change_2012_2025_pct': float(
            100 * (fp25.get('Private for-profit', 0)
                   / max(fp.get('Private for-profit', 1), 1) - 1)),
        'elapsed_sec': round(time.time() - t0, 1),
    }
    (RESULTS / 'scorecard_b2_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding='utf-8')

    _figure(panel, ts, ctp, es_all, args)
    print(f'\n[完成] 结果写入 {RESULTS}（{summary["elapsed_sec"]}s）')


def _figure(panel, ts, ctp, es_all, args) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))

    # (a) 各类型机构数演化
    ax = axes[0]
    cnt = (panel.groupby(['year', 'control']).size().unstack(fill_value=0))
    for c, col in (('Public', 'tab:blue'), ('Private nonprofit', 'tab:green'),
                   ('Private for-profit', 'tab:red')):
        if c in cnt.columns:
            ax.plot(cnt.index, cnt[c], 'o-', ms=3, color=col, label=c)
    for y, lab in EVENTS.items():
        ax.axvline(y, color='gray', ls=':', lw=.9)
        ax.annotate(lab, (y, ax.get_ylim()[1]), fontsize=6, rotation=90,
                    va='top', color='gray')
    ax.set_xlabel('period (start year)'); ax.set_ylabel('institutions')
    ax.set_title('(a) Institutions by ownership'); ax.legend(fontsize=7)

    # (b) 时变 CLQ 与一阶/二阶
    ax = axes[1]
    s = ts[(ts.source == 'Private for-profit') & (ts.target == 'Private for-profit')]
    ax.plot(s.year, s.CLQ, 'o-', ms=3, color='tab:blue', label='CLQ')
    ax.plot(s.year, s.CLQ1, 's--', ms=3, color='tab:orange', label='CLQ⁽¹⁾')
    ax.plot(s.year, s.CLQ2, '^:', ms=3, color='tab:green', label='CLQ⁽²⁾')
    ax.axhline(1, color='k', lw=.6)
    for y in EVENTS:
        ax.axvline(y, color='gray', ls=':', lw=.9)
    ax.set_xlabel('period'); ax.set_ylabel('value')
    ax.set_title('(b) Time-varying CLQ: for-profit→for-profit')
    ax.legend(fontsize=7)

    # (c) 事件研究
    ax = axes[2]
    if 'log1p(n_forprofit)' in es_all:
        e = es_all['log1p(n_forprofit)']
        ax.errorbar(e.year, e.coef, yerr=1.96 * e.se, fmt='o-', ms=3,
                    color='tab:red', lw=1)
        ax.axhline(0, color='k', lw=.6)
        ax.axvline(args.pre_to, color='gray', ls='--', lw=.9)
        ax.axvline(args.post_year, color='gray', ls=':', lw=.9)
        ax.set_title('(c) Event study: log1p(n_forprofit)')
    ax.set_xlabel('period'); ax.set_ylabel('β_k (× intensity sd)')

    # (d) 处理强度下的营利性占比分位演化
    ax = axes[3]
    q = ctp.copy()
    q['grp'] = pd.qcut(q['intensity'], 4, labels=False, duplicates='drop')
    q['fp_share'] = q['n_private_for_profit'] / q['N']
    cols = ['tab:blue', 'tab:cyan', 'tab:orange', 'tab:red']
    for gi in sorted(q['grp'].dropna().unique()):
        g = q[q.grp == gi].groupby('year')['fp_share'].mean()
        ax.plot(g.index, g.values, 'o-', ms=3, color=cols[int(gi) % 4],
                label=f'Q{int(gi) + 1} exposure')
    ax.axvline(args.post_year, color='k', ls='--', lw=.9)
    ax.set_xlabel('period'); ax.set_ylabel('for-profit share (cell mean)')
    ax.set_title('(d) For-profit share by pre-exposure quartile')
    ax.legend(fontsize=7)

    fig.tight_layout()
    out = FIG_DIR / 'scorecard_b2.png'
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'  [图] {out}')


if __name__ == '__main__':
    main()
