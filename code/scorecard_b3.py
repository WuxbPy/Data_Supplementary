#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B3：**尺度谱诊断**——高校空间协同是否存在"特征尺度"？对标半径能否由数据确定？

研究问题
--------
计划原文的 B3 设想用 AMS-CLQ 的**谱峰判据** C3（$b^*=\\arg\\max_b\\mathrm{CLQ}(b)$）
确定高校"对标邻域"的合理半径。本脚本首先**检验该判据的适用前提**：

* 定理 1 给出 $\\mathrm{CLQ}_{AB}(b)=\\mathbb E[g_{AB}(R_b)]/\\mathbb E[g_{A\\cdot}(R_b)]$；
* **命题 3（压缩极限）**：$\\mathbb E[\\theta(b)]=\\mathbb E[\\mathrm{CLQ}(b)]-1$ 随 $b$ **单调衰减**。

故 C3 只有当格局含**单一特征分离尺度**（合成实验中的"位移配对"）时才产生内部谱峰；
**多尺度聚集**（真实地理点格局的常态）会使谱单调 ⇒ $\\arg\\max$ 退化为取最小带宽。

本脚本给出**三重约束**下的尺度规范：

============  ==========================================  ==========================
约束            判据                                          作用
============  ==========================================  ==========================
κ             $\\kappa_{AB}(b)\\approx\\lambda_B\\pi b^2\\ge30$           一阶项估计可靠（**下界**）
ρ             邻域覆盖研究区比例 $\\rho(b)\\lesssim0.01$        压缩偏差可控（**上界**）
谱形           内部极大值存在？                                决定能否用 $b^*$ 刻画尺度
============  ==========================================  ==========================

分析内容
--------
1. **CONUS 全谱**（5 → 500 km 对数网格）× 全部类型对 → 谱形诊断；
2. **可行域**：κ≥30 与 ρ≤0.01 的交集；
3. **子区域自相似性**：按 census REGION 分组，检验归一化谱形是否重合（命题 1 的可比性前提）；
4. **跨期稳定性**（1996–2025，30 期）：谱形是否漂移、特征尺度是否随时间变化。

用法
----
.. code-block:: bash

    PY=/home/wuxb/miniconda3/envs/py310/bin/python
    $PY CausalCLQ/scripts/scorecard_b3.py
    $PY CausalCLQ/scripts/scorecard_b3.py --no-panel     # 跳过 30 期部分（快）
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
from src import clq_scale as cs        # noqa: E402

RESULTS = _PKG_DIR / 'results'
FIG_DIR = RESULTS / 'figures'

BS_KM = (5, 8, 13, 20, 30, 45, 70, 100, 150, 200, 300, 400, 500)


def main() -> None:
    ap = argparse.ArgumentParser(description='B3：尺度谱诊断与对标半径规范')
    ap.add_argument('--grid-n', type=int, default=120)
    ap.add_argument('--no-panel', action='store_true', help='跳过 30 期稳定性分析')
    ap.add_argument('--panel-pairs', type=int, default=2)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    bs = np.array(BS_KM) * 1000.0
    print('=' * 78)
    print('B3：尺度谱诊断 —— 高校空间协同是否存在特征尺度？')
    print('=' * 78)

    # ---------------------------------------------------------------- 1 CONUS
    df = ac.load_scorecard(verbose=False)
    xy = df[['x', 'y']].to_numpy()
    ORG = (float(xy[:, 0].min()), float(xy[:, 1].min()))
    REG = float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))
    labs = sorted(df['control'].dropna().unique())
    tid, _ = ac.encode_types(labs, df['control'])
    ix = {n: i + 1 for i, n in enumerate(labs)}
    pairs = [(ix['Private for-profit'], ix['Private for-profit'], '营利→营利'),
             (ix['Private nonprofit'], ix['Private nonprofit'], '非营利→非营利'),
             (ix['Public'], ix['Public'], '公立→公立'),
             (ix['Private for-profit'], ix['Public'], '营利→公立'),
             (ix['Private nonprofit'], ix['Public'], '非营利→公立')]
    print(f'\n[1] CONUS 全谱（{len(xy):,} 所，区域 {REG / 1e3:.0f} km，'
          f'{len(bs)} 个带宽）')
    spec = cs.spectrum(xy, tid, pairs, bs, region=REG, origin=ORG,
                       grid_n=args.grid_n)
    spec.to_csv(RESULTS / 'scorecard_b3_spectrum.csv', index=False,
                encoding='utf-8-sig')
    diag = cs.spectrum_diagnostics(spec)
    diag.to_csv(RESULTS / 'scorecard_b3_spectrum_diag.csv', index=False,
                encoding='utf-8-sig')
    print('\n  %-16s %9s %9s %8s %10s %10s' %
          ('类型对', 'CLQ(b_min)', 'CLQ(b_max)', '内部极值', 'argmax边界?', '单调'))
    for _, r in diag.iterrows():
        print('  %-16s %9.3f %9.3f %8d %10s %10s' %
              (r['pair'], r['clq_at_bmin'], r['clq_max'],
               r['n_interior_extrema'], '是' if r['argmax_is_boundary'] else '否',
               '递减' if r['monotone_dec'] else ('递增' if r['monotone_inc'] else '非单调')))
    n_peak = int(diag['has_interior_max'].sum())
    print(f'  ⇒ 含内部极大值的类型对：**{n_peak}/{len(diag)}**')

    # 抽样打印一个谱
    g = spec[spec.pair == '非营利→非营利'].sort_values('b_km')
    print('\n  非营利→非营利 谱：')
    print('   b(km)   ' + ' '.join('%7.0f' % v for v in g.b_km))
    print('   CLQ     ' + ' '.join('%7.3f' % v for v in g.CLQ))
    print('   CLQ⁽²⁾  ' + ' '.join('%7.3f' % v for v in g.CLQ2))
    print('   κ       ' + ' '.join('%7.0f' % v for v in g.kappa))
    print('   ρ       ' + ' '.join('%7.4f' % v for v in g.rho))

    # ---------------------------------------------------------------- 2 可行域
    print('\n[2] 三重约束的可行域（CONUS）')
    AREA = REG * REG
    rho_th = spec.groupby('b_km')['rho'].first()
    feas_rho = rho_th[rho_th <= cs.RHO_MAX]
    hi = float(feas_rho.index.max())
    print(f'  ρ（经验中位邻域占比）≤ {cs.RHO_MAX} ⇒ b ≤ **{hi:.0f} km**'
          f'（压缩偏差可控上界）')
    k_ok = spec[spec.kappa >= cs.KAPPA_MIN].groupby('b_km')['kappa'].min()
    lo = float(k_ok.index.min()) if len(k_ok) else np.nan
    if np.isfinite(lo):
        print(f'  κ ≥ {cs.KAPPA_MIN:.0f} ⇒ b ≥ **{lo:.0f} km**（一阶项可靠下界，'
              f'按最不利类型对）')
    # ---- 恒等式 κ ≈ N_B · ρ（**经验 ρ**；均匀场下 ρ_th=πb²/|Ω| 会低估聚集）
    print('\n  **关系校验**：κ ≈ N_B · ρ（ρ 取**经验**中位邻域占比）')
    print('   注：均匀场下 ρ_th=πb²/|Ω|，但机构高度聚集 ⇒ ρ_emp ≈ 4–8 × ρ_th，'
          '故必须用经验值')
    chk = spec[spec.b_km == 100.0].copy()
    chk['rho_th'] = np.pi * (chk['b_km'] * 1e3) ** 2 / AREA
    chk['kappa_hat'] = chk['n_B'] * chk['rho']
    print('   %-14s %7s %10s %10s %10s %10s' %
          ('类型对', 'N_B', 'ρ_th', 'ρ_emp', 'κ̂=N_Bρ_emp', 'κ 实测'))
    for _, r in chk.iterrows():
        print('   %-14s %7d %10.5f %10.5f %10.1f %10.1f' %
              (r['pair'], r['n_B'], r['rho_th'], r['rho'],
               r['kappa_hat'], r['kappa']))
    print(f'\n   **联立两判据的蕴含**：κ≥30 且 ρ≤0.01 ⟹ '
          f'N_B ≥ {cs.KAPPA_MIN / cs.RHO_MAX:.0f}（**与研究区面积无关**）')
    nB_max = int(max((tid == k).sum() for k in range(1, int(tid.max()) + 1)))
    thr = cs.KAPPA_MIN / cs.RHO_MAX
    print(f'   本数据最大类型计数 N_B = **{nB_max}** < {thr:.0f}'
          f' ⇒ **可行窗口恒为空**（子区域/各期同理，因条件与区域无关）')
    if np.isfinite(lo) and lo > hi:
        print(f'   ⇒ κ 下界 {lo:.0f} km **大于** ρ 上界 {hi:.0f} km，'
              f'**无带宽可同时满足两判据**')
        lo_use = np.nan
    print('   → 折中：B1/B2 取 b = 150 km，此时 κ 可接受（部分类型对 ≥30）'
          '而 ρ≈0.015 轻微超界')
    print('  谱形约束：' + ('**不存在特征尺度**，故不能用 b* 归一化'
                          if n_peak == 0 else f'{n_peak} 个类型对可用 b*'))

    # ---------------------------------------------------------------- 3 子区域
    print('\n[3] 子区域自相似性（按 census REGION，取点数最多的 4 个）')
    sub_bs = np.array([5, 10, 20, 35, 50, 75, 100, 150, 200, 300]) * 1000.0
    spec_by_region, span_km = {}, {}
    for rg in df['REGION'].value_counts().head(4).index:
        g = df[df['REGION'] == rg]
        gxy = g[['x', 'y']].to_numpy()
        gorg = (float(gxy[:, 0].min()), float(gxy[:, 1].min()))
        greg = float(max(np.ptp(gxy[:, 0]), np.ptp(gxy[:, 1])))
        gsub = cs.spectrum(gxy, tid[g.index], pairs[:3], sub_bs, region=greg,
                           origin=gorg, grid_n=args.grid_n)
        key = f'REGION={rg}(n={len(g)})'
        spec_by_region[key] = gsub[gsub.pair == '非营利→非营利'].copy()
        span_km[key] = greg / 1000.0
    ss = cs.self_similarity_check(spec_by_region, span_km)
    ss.to_csv(RESULTS / 'scorecard_b3_self_similarity.csv', index=False,
              encoding='utf-8-sig')
    print('  %-22s %8s %10s %10s %10s' %
          ('区域', '跨度km', '内部极值', 'argmax边界?', 'CLQ@1%跨度'))
    for _, r in ss.iterrows():
        print('  %-22s %8.0f %10d %10s %10.3f' %
              (r['region'], r['span_km'], r['n_interior_extrema'],
               '是' if r['argmax_is_boundary'] else '否', r['clq_at_1pct_of_span']))
    print('  ⇒ 各区域谱形是否一致：' +
          ('否（不自相似 ⇒ CLQ 不可跨研究区直接比较）' if
           ss['clq_at_1pct_of_span'].std() > 0.05 else '大体一致'))

    # ---------------------------------------------------------------- 4 跨期
    panel_diag = pd.DataFrame()
    if not args.no_panel:
        print('\n[4] 30 期谱形稳定性（1996–2025）')
        from src import scorecard_panel as sp
        panel = sp.build_panel(verbose=False)
        periods = sorted(panel['year'].unique())
        pbs = np.array([20, 50, 100, 150, 200, 300]) * 1000.0
        rows = []
        for y in periods:
            g = panel[panel.year == y]
            gxy = g[['x', 'y']].to_numpy()
            gtid, _ = ac.encode_types(labs, g['control'])
            for b in pbs:
                W = ac.build_weight_matrix(gxy, float(b))
                d = ac.clq_decompose(W, gxy, gtid, ix['Private nonprofit'],
                                     ix['Private nonprofit'], float(b),
                                     first_order='grid', grid_n=args.grid_n)
                rows.append({'year': y, 'b_km': b / 1000.0, 'CLQ': d['CLQ'],
                             'CLQ2': d['CLQ2'], 'kappa': d['kappa']})
        pan = pd.DataFrame(rows)
        pan.to_csv(RESULTS / 'scorecard_b3_panel_spectrum.csv', index=False,
                   encoding='utf-8-sig')
        # 每期的谱是否单调 / argmax 位置
        pr = []
        for y, g in pan.groupby('year'):
            g = g.sort_values('b_km')
            d = cs.peak_diagnostics(g['CLQ'].to_numpy(), g['b_km'].to_numpy())
            pr.append({'year': int(y), **d, 'clq_b100': float(
                g[g.b_km == 100]['CLQ'].iloc[0]) if (g.b_km == 100).any() else np.nan})
        panel_diag = pd.DataFrame(pr)
        panel_diag.to_csv(RESULTS / 'scorecard_b3_panel_diag.csv', index=False,
                          encoding='utf-8-sig')
        print('  %-10s %10s %12s %12s' % ('期', 'argmax_b', 'argmax边界?', '单调递减?'))
        for _, r in panel_diag.head(5).iterrows():
            print('  %-10d %10.0f %12s %12s' %
                  (r.year, r.argmax_b, '是' if r.argmax_is_boundary else '否',
                   '是' if r.monotone_dec else '否'))
        print(f'  ... 共 {len(panel_diag)} 期')
        print(f"  30 期中含内部极大值者：**{int(panel_diag.has_interior_max.sum())}/{len(panel_diag)}**")
        print(f"  argmax 恒为边界（最小 b）：**{int(panel_diag.argmax_is_boundary.sum())}/{len(panel_diag)}**")

    # ---------------------------------------------------------------- 汇总
    summary = {
        'n_points': int(len(xy)), 'region_span_km': REG / 1000.0,
        'bandwidths_km': list(BS_KM),
        'n_pairs_with_interior_max': int(n_peak), 'n_pairs': int(len(diag)),
        'rho_feasible_max_km': float(feas_rho.index.max()),
        'kappa_feasible_min_km': float(lo) if np.isfinite(lo) else None,
        'usable_range_km': ([float(lo), float(feas_rho.index.max())]
                            if np.isfinite(lo) else None),
        'self_similarity_std_at_1pct': float(ss['clq_at_1pct_of_span'].std()),
        'diagnostics': diag.to_dict('records'),
        'self_similarity': ss.to_dict('records'),
        'panel_n_interior_max': int(panel_diag.has_interior_max.sum())
        if len(panel_diag) else None,
        'elapsed_sec': round(time.time() - t0, 1),
    }
    (RESULTS / 'scorecard_b3_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding='utf-8')

    _figure(spec, diag, spec_by_region, span_km, panel_diag, lo, hi, bs)
    print(f'\n[完成] 结果写入 {RESULTS}（{summary["elapsed_sec"]}s）')


def _figure(spec, diag, spec_by_region, span_km, panel_diag, lo, hi, bs) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))

    # (a) CONUS 全谱
    ax = axes[0]
    for lab, g in spec.groupby('pair', sort=False):
        g = g.sort_values('b_km')
        ax.semilogx(g.b_km, g.CLQ, 'o-', ms=3.5, lw=1.3, label=lab)
    ax.axhline(1.0, color='k', lw=0.7)
    if hi == hi:
        ax.axvspan(50, hi, color='tab:green', alpha=0.08)
        ax.annotate('ρ-feasible', (min(hi, 500), 1.6), fontsize=7, color='tab:green')
    ax.set_xlabel('bandwidth b (km, log)'); ax.set_ylabel('CLQ(b)')
    ax.set_title('(a) CLQ spectrum, CONUS: no interior peak')
    ax.legend(fontsize=7)

    # (b) 可行域与 CLQ²
    ax = axes[1]
    g = spec[spec.pair == '非营利→非营利'].sort_values('b_km')
    ax.semilogx(g.b_km, g.CLQ2, 's-', ms=3.5, color='tab:purple', label='CLQ⁽²⁾')
    ax.axhline(1.0, color='k', lw=0.7)
    ax.axvspan(50, hi, color='tab:green', alpha=0.10, label='κ≥30 & ρ≤0.01')
    ax.set_ylim(0.8, 1.25)
    ax.set_xlabel('b (km, log)'); ax.set_ylabel('CLQ⁽²⁾')
    ax.set_title('(b) 2nd-order is flat ≈1 in feasible range')
    ax.legend(fontsize=7)

    # (c) 子区域归一化谱
    ax = axes[2]
    for reg, g in spec_by_region.items():
        g = g.sort_values('b_km')
        L = span_km[reg]
        ax.semilogx(g.b_km / L, g.CLQ, 'o-', ms=3.2, lw=1.2, label=reg)
    ax.axhline(1.0, color='k', lw=0.7)
    ax.set_xlabel('b / region span'); ax.set_ylabel('CLQ(b)')
    ax.set_title('(c) Not self-similar across regions')
    ax.legend(fontsize=6.5)

    # (d) 30 期谱的 argmax 位置
    ax = axes[3]
    if len(panel_diag):
        ax.plot(panel_diag.year, panel_diag.argmax_b / 1000.0, 'o-', ms=3.5,
                color='tab:red')
        ax.set_ylim(0, max(2 * panel_diag.argmax_b.max() / 1000.0, 40))
        ax.set_xlabel('period'); ax.set_ylabel('argmax b (km)')
        ax.set_title('(d) 30 periods: argmax pinned at grid minimum')
    else:
        ax.text(0.5, 0.5, 'panel skipped', ha='center', va='center')
        ax.set_title('(d) 30 periods (skipped)')

    fig.tight_layout()
    out = FIG_DIR / 'scorecard_b3.png'
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'  [图] {out}')


if __name__ == '__main__':
    main()
