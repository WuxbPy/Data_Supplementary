#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B1：**美国高校类型的空间协同——一阶混淆有多严重？**（College Scorecard × AMS-CLQ）

研究问题
--------
不同类型高校在地图上的"聚集/分离"（CLQ ≠ 1），究竟是**类型间的点间互作**（二阶），
还是**各自空间分布本身就不重叠**（一阶）？AMS-CLQ 定理 2 给出精确分解：

.. math::
    \\mathrm{CLQ}_{AB}=\\mathrm{CLQ}^{(1)}_{AB}\\cdot\\frac{1+\\eta_{AB}}{1+\\eta_{A\\cdot}},
    \\qquad \\mathrm{CLQ}^{(2)}_{AB}=\\mathrm{CLQ}_{AB}/\\mathrm{CLQ}^{(1)}_{AB}

若 :math:`\\mathrm{CLQ}^{(2)}\\approx1`，则地图上的"协同"完全由**分布重叠**解释。

数据与设计
----------
* **点**：一所院校；**坐标**：真实经纬度 → Albers 等积圆锥（米）
* **样本**：本土 48 州+DC × 主校区 × 在运营 × 坐标有效 = 4,785 所
* **类别**：``CONTROL``（公立/私立非营利/私立营利）、``ICLEVEL``（4年/2年/<2年）、及 9 类交叉
* **尺度**：带宽扫描 25–300 km；**κ 判据** :math:`\\kappa\\approx\\lambda_B\\pi b^2` 决定闭式一阶项是否可用
* **两类零假设**：
  - ``global``：全局类型重排（H0：CLQ=1，**不保留**一阶结构）
  - ``first_order``：一阶保留检验（H0：CLQ⁽²⁾=1），按 :math:`\\hat q_t(x_i)` 重抽类型

用法
----
.. code-block:: bash

    PY=/home/wuxb/miniconda3/envs/py310/bin/python
    $PY CausalCLQ/scripts/scorecard_clq.py                 # 默认 b=150km
    $PY CausalCLQ/scripts/scorecard_clq.py --b-km 200 --n-perm 299
    $PY CausalCLQ/scripts/scorecard_clq.py --no-null       # 跳过置换检验（快）
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

from src import ams_clq as ac          # noqa: E402

RESULTS = _PKG_DIR / 'results'
FIG_DIR = RESULTS / 'figures'
SCALES_KM = [25, 50, 100, 150, 200, 300]


def main() -> None:
    ap = argparse.ArgumentParser(description='B1：高校类型空间协同的 AMS-CLQ 一阶/二阶分离')
    ap.add_argument('--b-km', type=float, default=150.0, help='主带宽（km）')
    ap.add_argument('--n-perm', type=int, default=99)
    ap.add_argument('--no-null', action='store_true', help='跳过两类零假设检验')
    ap.add_argument('--first-order', choices=('grid', 'nw'), default='grid',
                    help="一阶项估计器：'grid'（默认，原文式(8)规格）或 'nw'（legacy，已知有偏）")
    ap.add_argument('--scale-mult', type=float, default=1.0,
                    help='一阶平滑倍数 σ/b（**1.0 即原文设定**；仅作敏感性分析）')
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    b = args.b_km * 1000.0
    print('=' * 76)
    print(f'B1 高校类型空间协同 · AMS-CLQ 一阶/二阶分离（主带宽 {args.b_km:.0f} km）')
    print('=' * 76)

    # ---------------------------------------------------------------- 数据
    # 优先使用存档自带的交叉截面点文件；缺失时才回到原始发布包
    _npz = Path(__file__).resolve().parent / 'data' / 'scorecard.npz'
    df = ac.load_cross_section(_npz) if _npz.exists() else ac.load_scorecard()
    xy = df[['x', 'y']].to_numpy()
    print(f'  分析样本 {len(df):,} 所；经纬度→Albers 等积圆锥（CONUS）')

    schemes = {}
    for col in ('control', 'iclevel'):
        labels = sorted(df[col].dropna().unique())
        tid, tmap = ac.encode_types(labels, df[col])
        schemes[col] = (labels, tid, tmap)

    # ---------------------------------------------------------------- 主表
    print(f'\n[1] 主分解表（b = {args.b_km:.0f} km，一阶项 = {args.first_order}）')
    print(f"  {'A':<22}{'B':<22}{'CLQ':>7}{'CLQ1':>8}{'CLQ2':>8}{'kappa':>7}"
          f"{'CLQ2_nw':>9}{'n_A':>7}{'n_B':>7}")
    main_rows = []
    for col, (labels, tid, tmap) in schemes.items():
        W = ac.build_weight_matrix(xy, b)
        tab = ac.all_pairs(xy, tid, tmap, b, smooth_scale=b * args.scale_mult,
                           W=W, first_order=args.first_order,
                           smooth_mult=args.scale_mult, nw_reference=True)
        tab.insert(0, 'scheme', col)
        main_rows.append(tab)
        for _, r in tab.iterrows():
            if r['source'] > r['target']:
                continue
            print(f"  {r['source'][:21]:<22}{r['target'][:21]:<22}"
                  f"{r['CLQ']:>7.3f}{r['CLQ1']:>8.3f}{r['CLQ2']:>8.3f}"
                  f"{r['kappa']:>7.1f}{r.get('CLQ2_nw', float('nan')):>9.3f}"
                  f"{r['n_A']:>7.0f}{r['n_B']:>7.0f}")
    main = pd.concat(main_rows, ignore_index=True)
    main.to_csv(RESULTS / 'scorecard_clq_main.csv', index=False, encoding='utf-8-sig')

    # ---------------------------------------------------------------- 尺度谱
    print(f'\n[2] 尺度谱（带宽 {"–".join(str(s) for s in (SCALES_KM[0], SCALES_KM[-1]))} km）')
    spec_rows = []
    for col, (labels, tid, tmap) in schemes.items():
        for a in sorted(tmap):
            for c in sorted(tmap):
                if c < a:
                    continue
                s = ac.scale_spectrum(xy, tid, a, c, [k * 1000.0 for k in SCALES_KM],
                                      smooth_mult=args.scale_mult,
                                      first_order=args.first_order)
                s['scheme'], s['source'], s['target'] = col, tmap[a], tmap[c]
                spec_rows.append(s)
    spec = pd.concat(spec_rows, ignore_index=True)
    spec.to_csv(RESULTS / 'scorecard_clq_spectrum.csv', index=False,
                encoding='utf-8-sig')
    print('  CLQ⁽²⁾ 随尺度（control 方案，对角 A=B）：')
    piv = spec[(spec.scheme == 'control') & (spec.source == spec.target)] \
        .pivot_table(index='source', columns='b_km', values='CLQ2').round(3)
    print(piv.to_string())

    # ------------------------------------------------- [2b] 平滑尺度敏感性（核心）
    print(f'\n[2b] **平滑尺度敏感性**（一阶份额的 h = mult×b；mult 过小会把二阶吸收进一阶）')
    sens = []
    for col, (labels, tid, tmap) in schemes.items():
        for mult in (1.0, 2.0, 3.0, 5.0):
            W = ac.build_weight_matrix(xy, b)
            for a in sorted(tmap):
                for c in sorted(tmap):
                    if c < a:
                        continue
                    d = ac.clq_decompose(W, xy, tid, a, c, b,
                                         smooth_mult=mult,
                                         first_order=args.first_order)
                    sens.append({'scheme': col, 'source': tmap[a], 'target': tmap[c],
                                 'mult': mult, 'kappa': d['kappa'],
                                 'CLQ': d['CLQ'], 'CLQ1': d['CLQ1'], 'CLQ2': d['CLQ2']})
    sens = pd.DataFrame(sens)
    sens.to_csv(RESULTS / 'scorecard_clq_scale_sensitivity.csv', index=False,
                encoding='utf-8-sig')
    piv = sens[sens.scheme == 'control'].pivot_table(index=['source', 'target'],
                                                     columns='mult', values='CLQ2')
    print(piv.round(3).to_string())
    dev = sens.groupby('mult')['CLQ2'].apply(lambda v: (v - 1).abs().max())
    print('  各 mult 下 |CLQ⁽²⁾−1| 的最大值：' +
          ', '.join(f'h={m:g}b→{v:.3f}' for m, v in dev.items()))

    # ---------------------------------------------------------------- κ 判据校验
    print(f'\n[3] κ 判据校验（|CLQ⁽²⁾−1| ≤ {ac.KAPPA_BOUND} 当 κ ≥ {ac.KAPPA_THRESHOLD:.0f}）')
    good = spec[(spec.kappa >= ac.KAPPA_THRESHOLD) & spec.CLQ2.notna()]
    viol = good[good.CLQ2.sub(1).abs() > ac.KAPPA_BOUND]
    print(f'  κ≥30 单元格 {len(good)} 个，其中越界（|CLQ⁽²⁾−1|>{ac.KAPPA_BOUND}）'
          f' **{len(viol)}** 个')
    if len(good):
        print('  κ≥30 时 CLQ⁽²⁾ 范围：[%.4f, %.4f]'
              % (good.CLQ2.min(), good.CLQ2.max()))
    allc = spec[spec.CLQ2.notna()]
    print('  全部单元格 CLQ⁽²⁾ 范围：[%.4f, %.4f]（%d 个）'
          % (allc.CLQ2.min(), allc.CLQ2.max(), len(allc)))

    # ---------------------------------------------------------------- 两类零假设
    summary = {'b_km': args.b_km, 'n_points': int(len(df)),
               'kappa_threshold': ac.KAPPA_THRESHOLD,
               'kappa_bound': ac.KAPPA_BOUND}
    if not args.no_null:
        print(f'\n[4] 两类零假设检验（{args.n_perm} 次置换，b = {args.b_km:.0f} km）')
        null_rows = []
        for col, (labels, tid, tmap) in schemes.items():
            W = ac.build_weight_matrix(xy, b)
            for a in sorted(tmap):
                for c in sorted(tmap):
                    g = ac.null_clq_test(W, xy, tid, a, c, mode='global',
                                         n_perm=args.n_perm)
                    # 一阶保留零假设：各类型从**自身经验密度**独立重抽位置
                    # （smooth=0；先平滑再抽样会二次卷积、检验过度激进）
                    f = ac.null_clq_placement(xy, tid, a, c, b,
                                              n_types=max(tmap),
                                              n_perm=args.n_perm)
                    k = ac.kappa_ab(xy, tid, a, c, b)
                    null_rows.append({
                        'scheme': col, 'source': tmap[a],
                        'target': tmap[c], 'kappa': k,
                        'clq_obs': g['clq_obs'], 'p_global': g['p'],
                        'null_global': g['mean_null'],
                        'p_first_order': f['p'], 'null_first_order': f['mean_null']})
        nt = pd.DataFrame(null_rows)
        nt.to_csv(RESULTS / 'scorecard_clq_null.csv', index=False,
                  encoding='utf-8-sig')
        sig_g = int((nt.p_global <= 0.05).sum())
        sig_f = int((nt.p_first_order <= 0.05).sum())
        hi = nt[nt.kappa >= ac.KAPPA_THRESHOLD]
        sig_g_hi = int((hi.p_global <= 0.05).sum())
        sig_f_hi = int((hi.p_first_order <= 0.05).sum())
        print(f'  全部 {len(nt)} 个有序类型对：')
        print(f'    标准检验（全局重排）显著 = **{sig_g}** 个')
        print(f'    一阶保留检验显著       = **{sig_f}** 个')
        print(f'  其中 κ≥30 的 {len(hi)} 个：标准 {sig_g_hi} 个 → 一阶保留 {sig_f_hi} 个')
        summary.update({'n_pairs': len(nt), 'sig_global': sig_g,
                        'sig_first_order': sig_f,
                        'n_pairs_kappa30': len(hi),
                        'sig_global_kappa30': sig_g_hi,
                        'sig_first_order_kappa30': sig_f_hi})
        print('  κ≥30 子集的明细（按 CLQ 降序）：')
        cols = ['scheme', 'source', 'target', 'kappa', 'clq_obs',
                'p_global', 'p_first_order']
        print(hi.sort_values('clq_obs', ascending=False)[cols]
              .head(12).to_string(index=False))

    summary['clq2_range_all'] = [float(allc.CLQ2.min()), float(allc.CLQ2.max())]
    if len(good):
        summary['clq2_range_kappa30'] = [float(good.CLQ2.min()), float(good.CLQ2.max())]
    summary['first_order_estimator'] = args.first_order
    if 'CLQ2_nw' in spec.columns:
        vn = spec.dropna(subset=['CLQ2_nw'])
        summary['clq2_nw_range'] = [float(vn.CLQ2_nw.min()), float(vn.CLQ2_nw.max())]
        summary['kappa30_violations'] = int(len(viol))
    # 一阶份额：观测 CLQ 的变异有多少由 CLQ⁽¹⁾ 解释
    v = spec.dropna(subset=['CLQ', 'CLQ1'])
    if len(v):
        r2 = np.corrcoef(v.CLQ, v.CLQ1)[0, 1] ** 2
        slope = np.polyfit(v.CLQ1, v.CLQ, 1)[0]
        summary['corr2_clq_clq1'] = float(r2)
        summary['slope_clq_on_clq1'] = float(slope)
        print(f'\n[5] 一阶解释力：corr²(CLQ, CLQ⁽¹⁾) = **{r2:.4f}**，'
              f'回归斜率 = {slope:.4f}（1 表示 CLQ 完全由一阶决定）')
        dev = (v.CLQ - v.CLQ1).abs()
        summary['max_abs_dev_clq_clq1'] = float(dev.max())
        print(f'    |CLQ − CLQ⁽¹⁾| 最大 {dev.max():.4f}，中位 {dev.median():.4f}')

    summary['elapsed_sec'] = round(time.time() - t0, 1)
    (RESULTS / 'scorecard_clq_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    _figure(df, spec, main, b, tag='')
    print(f'\n[完成] 结果写入 {RESULTS}（{summary["elapsed_sec"]}s）')


def _figure(df, spec, main, b, tag='') -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))

    # (a) 院校分布（按办学性质）
    ax = axes[0]
    for lab, c in (('Public', 'tab:blue'), ('Private nonprofit', 'tab:green'),
                   ('Private for-profit', 'tab:red')):
        s = df[df['control'] == lab]
        ax.scatter(s['x'] / 1000, s['y'] / 1000, s=2.5, alpha=0.5, c=c, label=lab)
    ax.set_aspect('equal'); ax.legend(fontsize=7, markerscale=4)
    ax.set_xlabel('Albers x (km)'); ax.set_ylabel('Albers y (km)')
    ax.set_title(f'(a) US institutions (CONUS, n={len(df):,})')

    # (b) 尺度谱：CLQ vs CLQ⁽¹⁾ vs CLQ⁽²⁾（取对角自对，最能体现一阶主导）
    ax = axes[1]
    sub = spec[(spec.scheme == 'control') & (spec.source == spec.target)]
    if len(sub):
        p0 = sub.groupby('source')['CLQ'].mean().idxmin()
        sub = sub[sub.source == p0].sort_values('b_km')
        ax.plot(sub.b_km, sub.CLQ, 'o-', color='tab:blue',
                label=f'CLQ ({p0[:14]}→self)')
        ax.plot(sub.b_km, sub.CLQ1, 's--', color='tab:orange', label='CLQ⁽¹⁾')
        ax.plot(sub.b_km, sub.CLQ2, '^:', color='tab:green', label='CLQ⁽²⁾')
        ax.axhline(1.0, color='k', lw=0.7)
        first = True
        for _, r in sub.iterrows():
            if r['kappa'] >= 30 and first:
                ax.axvspan(r['b_km'] * 0.92, 400, color='tab:green', alpha=0.08)
                ax.annotate(f'κ≥30 (from {r["b_km"]:.0f} km)', (r['b_km'], 1.35),
                            fontsize=7, color='tab:green')
                first = False
    ax.set_xlabel('bandwidth b (km)'); ax.set_ylabel('value')
    ax.set_title('(b) Scale spectrum: 1st vs 2nd order')
    ax.legend(fontsize=7)

    # (c) CLQ vs CLQ⁽¹⁾ 散点
    ax = axes[2]
    v = spec.dropna(subset=['CLQ', 'CLQ1'])
    ax.scatter(v.CLQ1, v.CLQ, s=22, alpha=0.75, c='tab:purple')
    lim = [min(v.CLQ1.min(), v.CLQ.min()) - .05, max(v.CLQ1.max(), v.CLQ.max()) + .05]
    ax.plot(lim, lim, 'k--', lw=1, label='CLQ = CLQ⁽¹⁾')
    ax.axhline(1, color='gray', lw=.4); ax.axvline(1, color='gray', lw=.4)
    ax.set_xlabel('CLQ⁽¹⁾ (first order)'); ax.set_ylabel('CLQ (observed)')
    r2 = np.corrcoef(v.CLQ, v.CLQ1)[0, 1] ** 2
    ax.set_title(f'(c) Observed ≈ first order ($R^2$={r2:.4f})')
    ax.legend(fontsize=7)

    # (d) CLQ⁽²⁾ 分布（按 κ 分层）
    ax = axes[3]
    v = spec.dropna(subset=['CLQ2'])
    lo, hi = v[v.kappa < 30], v[v.kappa >= 30]
    bins = np.linspace(0.9, 1.10, 25)
    ax.hist(lo.CLQ2, bins=bins, alpha=.7, color='tab:gray', label=f'κ<30 (n={len(lo)})')
    ax.hist(hi.CLQ2, bins=bins, alpha=.8, color='tab:green', label=f'κ≥30 (n={len(hi)})')
    ax.axvline(1.0, color='k', lw=1)
    ax.axvspan(1 - 0.015, 1 + 0.015, color='tab:green', alpha=.12)
    ax.set_xlabel('CLQ⁽²⁾ (second order)'); ax.set_ylabel('count')
    ax.set_title('(d) 2nd-order: ≈1 where κ≥30')
    ax.legend(fontsize=7)

    fig.tight_layout()
    out = FIG_DIR / f'scorecard_clq{tag}.png'
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'  [图] {out}')


if __name__ == '__main__':
    main()
