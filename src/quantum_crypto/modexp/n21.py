"""N=21 用の制御付き modular exponentiation 実装

N=21 (= 3 × 7) の因数分解用の量子回路実装。
5量子ビットで 0-31 の状態を表現し、|x⟩ → |ax mod 21⟩ を実現。

注意: この実装はユニタリ行列を直接使用するため、ゲート数が多く、
実機での実行には適していません。教育・シミュレーション目的です。
"""
from math import gcd

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
import numpy as np


# 定数定義
N = 21
FACTORS = (3, 7)
# gcd(a, 21) = 1 となる a を計算
VALID_BASES = [a for a in range(2, N) if gcd(a, N) == 1]
N_WORK_QUBITS = 5  # 21 < 2^5
MAX_DENOMINATOR = 21


def c_amod21(a: int, power: int) -> QuantumCircuit:
    """制御付き a^(2^power) mod 21 の実装

    ユニタリ行列を使用して (a*x) mod 21 の置換を実装します。
    """
    if a not in VALID_BASES:
        raise ValueError(f"a={a} は N={N} に対して無効です（gcd(a,{N})≠1）。有効なベース: {VALID_BASES}")

    U = QuantumCircuit(N_WORK_QUBITS)

    # a^(2^power) mod N を計算
    # NOTE: 入力パラメータを上書きしないようローカル変数を使用
    a_power = a
    for _ in range(power):
        a_power = (a_power * a_power) % N

    # (a * x) mod 21 の置換行列を作成
    # 32x32 のユニタリ行列（置換行列）
    dim = 2 ** N_WORK_QUBITS
    matrix = np.zeros((dim, dim))

    for x in range(dim):
        if x < N:
            # x → (a_power*x) mod N
            y = (a_power * x) % N
            matrix[y, x] = 1
        else:
            # N以上はそのまま
            matrix[x, x] = 1

    # Operator から unitary gate を作成
    unitary_op = Operator(matrix)
    U.unitary(unitary_op, range(N_WORK_QUBITS), label=f"{a}^{2**power} mod {N}")

    U = U.to_gate()
    U.name = f"{a}^{2**power} mod {N}"
    c_U = U.control()
    return c_U


# レジストリ用の設定
config = {
    "n": N,
    "func": c_amod21,
    "valid_bases": VALID_BASES,
    "n_work_qubits": N_WORK_QUBITS,
    "max_denominator": MAX_DENOMINATOR,
    "factors": FACTORS,
}
