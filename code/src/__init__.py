# -*- coding: utf-8 -*-
"""CausalCLQ：因果协同区位熵（Causal Co-Location Quotient）系列。

本包为 P5 MVP 的可复现实现，独立于上游 amsclq / netclq / MCCLQ。

模块
----
config               路径、类型映射与默认参数
data_io              数据加载与米制投影
clq_outcome          点级二阶协同 LCLQ2 与面板被解释变量
covariates           前定协变量与处理强度
did                  双向固定效应 DID / 事件研究 / 聚类稳健（含 WLS）
standard_did         标准 DID：二值处理、平衡性、PSM、随机化推断、安慰剂
external_covariates  外部协变量加载/下载/连接（ACS、LSOA Census+IMD）
"""
from . import config  # noqa: F401

__all__ = ["config"]
__version__ = "0.1.0-mvp"
