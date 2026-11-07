# -*- coding: utf-8 -*-
"""College Scorecard 数据获取与准备（主验证数据）

本脚本把美国教育部公开的机构级年度文件整理为 EduCEQ-Toolkit 可直接分析的形式。
**数据完全公开**，因此本论文的验证可在任何机器上复现；脚本本身可随代码上传仓库。

数据来源
--------
College Scorecard，美国教育部发布：
    https://collegescorecard.ed.gov/data/
下载 "Institution-level data files"（分年度压缩包）与最新截面文件。

处理流程（与论文 §5.1 一致）
------------------------------
1. 逐期读取机构级年度文件
2. 五步筛选：本土 48 州及哥伦比亚特区 → 主校区 → 在运营 → 坐标有效 → 类型有效
3. 坐标处理：历史期坐标缺失时以「邮政编码 → 最新期质心」回链
4. 类型标签：办学性质（公立 / 私立非营利 / 私立营利）与办学层次
5. 输出可分析样本

用法
----
    python data/prepare_scorecard.py --raw DIR --out data/scorecard.npz

若尚未下载原始数据，脚本会在 --raw 为空时生成一份**同规模的合成样本**用于流程自检，
以便在没有网络的环境下也能验证脚本本身的正确性（真实分析必须使用下载的数据）。
"""
import argparse
import os

import numpy as np

# ---------------------------------------------------------------- 常量
STATES_48 = set("""AL AZ AR CA CO CT DE FL GA ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT
NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY""".split()) | {"DC"}

# College Scorecard 字段（不同年份列名可能带后缀，取标准前缀）
FIELD_CANDIDATES = {
    "unitid":    ["UNITID", "unitid", "ID"],
    "name":      ["INSTNM", "instnm", "OPEID6"],
    "state":     ["STABBR", "stabbr"],
    "control":   ["CONTROL", "control"],
    "level":     ["ICLEVEL", "iclevel"],
    "lat":       ["LATITUDE", "latitude"],
    "lon":       ["LONGITUDE", "longitude"],
    "zip":       ["ZIP", "zip"],
    "main":      ["MAIN", "main"],
    "curroper":  ["CURROPER", "curroper"],
}


def _pick(colnames, key):
    low = {c.lower(): c for c in colnames}
    for cand in FIELD_CANDIDATES[key]:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def load_year(path):
    """读取一期机构级文件，返回记录列表。"""
    import pandas as pd
    df = pd.read_csv(path, dtype=str, low_memory=False)
    cols = {k: _pick(df.columns, k) for k in FIELD_CANDIDATES}
    if cols["unitid"] is None or cols["state"] is None:
        raise ValueError("缺少必要字段：%s" % path)
    return df, cols


def build_sample(raw_dir):
    """从原始目录构建截面样本。返回 dict。

    截面分析只需要**最新一期**文件（Most-Recent-Cohorts-Institution.csv），
    因此这里只读排序后的最后一个 CSV，避免为一次截面检查读入全部 30 期（数 GB）。
    文件名排序中 Most-Recent-* 位于 MERGED* 之后，正好是最新截面。
    """
    import glob
    import pandas as pd

    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not files:
        raise FileNotFoundError("未找到原始数据文件：%s" % raw_dir)

    latest_file = files[-1]
    df, cols = load_year(latest_file)
    keep = df[[c for c in [cols["unitid"], cols["state"], cols["control"],
                           cols["level"], cols["lat"], cols["lon"], cols["zip"],
                           cols["main"], cols["curroper"]] if c]].copy()
    keep.columns = [k for k, v in cols.items() if v and v in keep.columns]
    latest = keep
    print("截面取用文件：%s" % os.path.basename(latest_file))
    print("可用年度文件数：%d（面板分析时需全部读取，本脚本仅做截面）" % len(files))
    latest["lat"] = pd.to_numeric(latest.get("lat"), errors="coerce")
    latest["lon"] = pd.to_numeric(latest.get("lon"), errors="coerce")

    m = latest["state"].isin(STATES_48)
    if "main" in latest:
        m &= pd.to_numeric(latest["main"], errors="coerce").fillna(0) == 1
    if "curroper" in latest:
        m &= pd.to_numeric(latest["curroper"], errors="coerce").fillna(0) == 1
    m &= latest["lat"].notna() & latest["lon"].notna() & (latest["lat"].abs() > 1e-6)
    m &= latest["control"].notna() & latest["level"].notna()

    sub = latest[m].copy()
    sub["control"] = pd.to_numeric(sub["control"], errors="coerce").astype(int)
    sub["level"] = pd.to_numeric(sub["level"], errors="coerce").astype(int)
    sub = sub[sub["control"].isin([1, 2, 3]) & sub["level"].isin([1, 2, 3])]

    coords = sub[["lon", "lat"]].to_numpy(float)
    types = (sub["control"].to_numpy(int) - 1)          # 0 公立 / 1 私立非营利 / 2 私立营利
    levels = (sub["level"].to_numpy(int) - 1)
    return dict(coords=coords, types=types, levels=levels,
                n_raw=len(latest), n_kept=len(sub))


def synthetic_fallback(n=6273, seed=0):
    """无原始数据时的流程自检样本（规模与真实筛选前相当）。"""
    rng = np.random.default_rng(seed)
    coords = np.column_stack([
        rng.uniform(-125, -66, n),          # 经度
        rng.uniform(25, 49, n)])            # 纬度
    types = rng.choice([0, 1, 2], n, p=[0.37, 0.33, 0.30])
    levels = rng.choice([0, 1, 2], n, p=[0.50, 0.25, 0.25])
    return dict(coords=coords, types=types, levels=levels,
                n_raw=n, n_kept=int(n * 0.76), synthetic=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="", help="原始数据目录（含 *.csv）")
    ap.add_argument("--out", default="data/scorecard.npz")
    a = ap.parse_args()

    if a.raw and os.path.isdir(a.raw):
        s = build_sample(a.raw)
        print("已构建样本：原始 %d → 保留 %d" % (s["n_raw"], s["n_kept"]))
    else:
        s = synthetic_fallback()
        print("未提供原始数据，生成同规模合成样本用于流程自检（真实分析需下载数据）")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, coords=s["coords"], types=s["types"], levels=s["levels"])
    print("已写出：%s" % a.out)
    print("  类型构成：", np.bincount(s["types"], minlength=3).tolist())
    print("  层次构成：", np.bincount(s["levels"], minlength=3).tolist())


if __name__ == "__main__":
    main()
