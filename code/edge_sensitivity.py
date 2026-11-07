# -*- coding: utf-8 -*-
"""Edge-effect sensitivity for the cross-section (Table A4 of the manuscript).

The estimator is computed on an observed window. Points near the window boundary
have part of their kernel support outside it, so their neighbourhood counts are
truncated. This script compares the full sample with an interior subsample that
drops every point within 3b of the bounding rectangle, at b = 150 km.

Run from the material package root:
    python code/edge_sensitivity.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src import ams_clq as ac  # noqa: E402

NPZ = os.path.join(HERE, "data", "scorecard.npz")
RESULTS = os.path.join(HERE, "..", "results")
B = 150_000.0
PAIRS = [("Public", "Public"), ("Private nonprofit", "Private nonprofit"),
         ("Private for-profit", "Private for-profit")]


def run(xy, types, tag):
    W = ac.build_weight_matrix(xy, B)
    rows = []
    for src, tgt in PAIRS:
        s = ac.CONTROL_LABELS and {v: k for k, v in ac.CONTROL_LABELS.items()}[src]
        t = {v: k for k, v in ac.CONTROL_LABELS.items()}[tgt]
        r = ac.clq_decompose(W, xy, types, s, t, B)
        r.update(sample=tag, source=src, target=tgt)
        rows.append(r)
    return rows


def main():
    z = np.load(NPZ, allow_pickle=True)
    lon, lat = z["coords"][:, 0], z["coords"][:, 1]
    x, y = ac.project_albers_conus(lon, lat)
    xy = np.column_stack([x, y])
    types = z["types"].astype(int)

    # the pipeline encodes control as 1=Public, 2=Private nonprofit, 3=Private for-profit
    types = types + 1

    rows = []
    rows += run(xy, types, "full window (n = %s)" % format(len(xy), ","))

    x0, x1 = xy[:, 0].min(), xy[:, 0].max()
    y0, y1 = xy[:, 1].min(), xy[:, 1].max()
    m = ((xy[:, 0] > x0 + 3 * B) & (xy[:, 0] < x1 - 3 * B) &
         (xy[:, 1] > y0 + 3 * B) & (xy[:, 1] < y1 - 3 * B))
    rows += run(xy[m], types[m], "interior, 3b buffer (n = %s)" % format(int(m.sum()), ","))

    df = pd.DataFrame(rows)[["sample", "source", "target", "N", "CLQ", "CLQ1", "CLQ2", "kappa"]]
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "scorecard_edge_sensitivity.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False, float_format=lambda v: "%.4f" % v))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
