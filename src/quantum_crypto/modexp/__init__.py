"""Modular exponentiation implementations registry for Shor's algorithm.

各 N 専用の mod-exp 実装を登録・管理します。
"""
from typing import Callable, TypedDict
from qiskit import QuantumCircuit


class ModExpConfig(TypedDict):
    """mod-exp 実装の設定"""
    n: int
    func: Callable[[int, int], QuantumCircuit]
    valid_bases: list[int]
    n_work_qubits: int
    max_denominator: int
    factors: tuple[int, int]  # デバッグ用


# 各 N の実装をインポート
REGISTRY: dict[int, ModExpConfig] = {}

try:
    from .n15 import config as config_15
    REGISTRY[15] = config_15
except ImportError:
    pass

try:
    from .n21 import config as config_21
    REGISTRY[21] = config_21
except ImportError:
    pass

try:
    from .n33 import config as config_33
    REGISTRY[33] = config_33
except ImportError:
    pass


def get_config(n: int) -> ModExpConfig:
    """指定された N の設定を取得

    Parameters
    ----------
    n : int
        因数分解する合成数

    Returns
    -------
    ModExpConfig
        N に対する mod-exp 実装の設定

    Raises
    ------
    NotImplementedError
        N が未実装の場合
    """
    if n not in REGISTRY:
        available = sorted(REGISTRY.keys())
        raise NotImplementedError(
            f"N={n} は未実装です。実装済み: {available}"
        )
    return REGISTRY[n]


def list_supported() -> list[int]:
    """実装済みの N のリストを返す"""
    return sorted(REGISTRY.keys())
