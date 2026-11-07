#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kernel-truncation robustness of the second-order term (Table A6).

Motivation
----------
The Gaussian kernel is truncated at ``radius_mult * b`` for sparsity.  A
truncation at 3b discards 0.27 per cent of each kernel's mass, and the loss is
not spatially uniform: it is small where a category is dense and larger where
the category is nearly uniform, because in the latter case the discarded ring
still contributes a non-negligible share of the weight.  The second-order term
is a ratio of two quantities computed with the same truncation, so most of the
loss cancels; what survives is an offset of the order of the discarded mass.

The article attributes the 0.013 gap between C and C(1) in the
public-to-public row to this offset.  That attribution is testable: widen the
truncation and the offset should shrink towards zero if truncation is the
cause.  This script performs that test on the cross-section at b = 150 km for
radius multiples of 3, 4, 5 and 8.  At 8b the discarded mass is 1.2e-15, i.e.
numerically exact.

Usage
-----
    python truncation_test.py                 # writes results/scorecard_truncation.csv

Outputs
-------
``results/scorecard_truncation.csv`` with one row per (pair, radius multiple).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from src import ams_clq as ac          # noqa: E402

RESULTS = _HERE.parent / 'results'
NPZ = _HERE / 'data' / 'scorecard.npz'

LABELS = {1: 'Public', 2: 'Private nonprofit', 3: 'Private for-profit'}


def load_points(npz: Path = NPZ):
    """Cross-section from the archived npz: lon/lat → Albers metres, 1-based types."""
    z = np.load(npz, allow_pickle=True)
    lon, lat = z['coords'][:, 0], z['coords'][:, 1]
    x, y = ac.project_albers_conus(lon, lat)
    types = np.asarray(z['types'], dtype=np.int64) + 1     # npz is 0-based
    return np.column_stack([x, y]), types


def main() -> None:
    ap = argparse.ArgumentParser(description='Kernel-truncation robustness')
    ap.add_argument('--b-km', type=float, default=150.0)
    ap.add_argument('--mults', type=float, nargs='+', default=[3.0, 4.0, 5.0, 8.0])
    ap.add_argument('--npz', type=Path, default=NPZ)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    xy, types = load_points(args.npz)
    b = args.b_km * 1000.0
    print(f'  cross-section {len(xy):,} points; b = {args.b_km:g} km')

    rows = []
    for mult in args.mults:
        tab = ac.all_pairs(xy, types, LABELS, b, first_order='grid',
                           radius_mult=mult)
        tab['radius_mult'] = mult
        rows.append(tab)
        print(f'  radius_mult = {mult:g} done')

    out = pd.concat(rows, ignore_index=True)
    keep = ['radius_mult', 'source', 'target', 'CLQ', 'CLQ1', 'CLQ2', 'kappa']
    out = out[[c for c in keep if c in out.columns]]
    csv = RESULTS / 'scorecard_truncation.csv'
    out.to_csv(csv, index=False, encoding='utf-8-sig')

    for src in ('Public',):
        sub = out[(out.source == src) & (out.target == src)]
        print(f'\n  {src} → {src}:')
        for _, r in sub.iterrows():
            print(f'    {r.radius_mult:>4g}b  C={r.CLQ:.4f}  C1={r.CLQ1:.4f}  '
                  f'C2={r.CLQ2:.4f}  kappa={r.kappa:.0f}')
    print(f'\n  wrote {csv}')


if __name__ == '__main__':
    main()
