# -*- coding: utf-8 -*-
"""CausalCLQ 全局配置：路径、类型映射、默认参数。

本模块独立于上游 amsclq / netclq / MCCLQ，避免命名与路径冲突。
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
# config.py 位于 <ROOT>/CausalCLQ/src/config.py
SRC_DIR = Path(__file__).resolve().parent
PKG_DIR = SRC_DIR.parent                 # CausalCLQ/
ROOT = PKG_DIR.parent                    # 项目根

CHICAGO_CSV = ROOT / "chicago_mcclq_data" / "chicago_mcclq.csv"
CHICAGO_NPZ = ROOT / "chicago_mcclq_data" / "chicago_mcclq.npz"
CHICAGO_TYPE_MAPPING = ROOT / "chicago_mcclq_data" / "chicago_mcclq_type_mapping.csv"

RESULTS_DIR = PKG_DIR / "results"
FIG_DIR = RESULTS_DIR / "figures"

# ---------------------------------------------------------------------------
# 芝加哥类型编号（与 convert_chicago_to_mcclq.py 的 TYPE_ORDER 一致）
# ---------------------------------------------------------------------------
TYPE_NAME = {
    1: "Homicide",
    2: "Criminal Sexual Assault",
    3: "Robbery",
    4: "Aggravated Assault",
    5: "Aggravated Battery",
    6: "Burglary",
    7: "Theft",
    8: "Motor Vehicle Theft",
    9: "Arson",
}

# P5 的目标类型对：盗窃 -> 入室盗窃（上游 ASW-STCLQ 发现 2020Q2 后显著跃升）
SOURCE_TYPE = 7        # Theft
TARGET_TYPE = 6        # Burglary

# location_type 编码（来自 convert_chicago_to_mcclq.py 的 LOC_CLASSES）
LOC_CLASSES = {
    0: "residence",
    1: "street",
    2: "parking",
    3: "school",
    4: "commercial",
    5: "transit",
    6: "other",
}

# ---------------------------------------------------------------------------
# 时间：2015Q1 = period 1 ... 2022Q4 = period 32
# ---------------------------------------------------------------------------
N_PERIODS = 32
# 疫情管控（准自然实验）：2020Q2 = period 22 起为处理后期
COVID_POST_PERIOD = 22
# 处理前窗口（用于构造前定协变量）：2015Q1–2020Q1 = period 1–21
PRE_WINDOW_END = 21

# ---------------------------------------------------------------------------
# 估计参数
# ---------------------------------------------------------------------------
# 邻域：k 近邻（自适应带宽）。k/N = 200/826073 ≈ 0.024% < 1%，满足 ρ 约束。
DEFAULT_K = 200
# k 近邻内的距离核带宽（米）；半径与 k 取较小者，避免稀疏区过度外扩
DEFAULT_KERNEL_BW = 300.0
DEFAULT_MAX_RADIUS = 1200.0
# 一阶强度估计网格（米）与高斯平滑尺度（格）
DEFAULT_GRID_CELL = 250.0
DEFAULT_GRID_SIGMA = 2.0

# 城市中心（用于投影与"到市中心距离"协变量）
LON0, LAT0 = -87.68, 41.85

# 聚类稳健标准误的聚类单元
CLUSTER_COL = "community_area"
