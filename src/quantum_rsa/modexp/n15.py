"""N=15 用の制御付き modular exponentiation 実装

N=15 (= 3 × 5) の因数分解用の量子回路実装。
4量子ビットで 0-15 の状態を表現し、|x⟩ → |ax mod 15⟩ を実現。
"""
from qiskit import QuantumCircuit


# 定数定義
N = 15
FACTORS = (3, 5)
VALID_BASES = [2, 4, 7, 8, 11, 13]  # gcd(a, 15) = 1 となる a
N_WORK_QUBITS = 4  # 15 < 2^4
MAX_DENOMINATOR = 15


def c_amod15(a: int, power: int) -> QuantumCircuit:
    """制御付き a^(2^power) mod 15 の実装

    Parameters
    ----------
    a : int
        ベース（2, 4, 7, 8, 11, 13 のいずれか）
    power : int
        指数（2^power として使用）

    Returns
    -------
    QuantumCircuit
        制御付きユニタリゲート

    Raises
    ------
    ValueError
        a が有効なベースでない場合
    """
    if a not in VALID_BASES:
        raise ValueError(
            f"a={a} は N={N} に対して無効です（gcd(a,{N})≠1）。"
            f"有効なベース: {VALID_BASES}"
        )

    U = QuantumCircuit(N_WORK_QUBITS)

    # a^(2^power) mod N を計算
    for _ in range(power):
        a = (a * a) % N

    # 各 a の値に対する変換を SWAP/CNOT で実装
    if a == 1:
        pass  # 恒等変換
    elif a == 2:
        U.swap(0, 3)
        U.swap(1, 3)
        U.swap(2, 3)
    elif a == 4:
        U.swap(0, 2)
        U.swap(1, 3)
    elif a == 7:
        U.swap(2, 3)
        U.swap(1, 3)
        U.swap(2, 3)
        U.cx(3, 2)
        U.cx(3, 1)
        U.cx(3, 0)
    elif a == 8:
        U.swap(0, 3)
        U.swap(1, 3)
        U.swap(0, 3)
    elif a == 11:
        U.swap(0, 3)
        U.swap(1, 2)
    elif a == 13:
        U.swap(2, 3)
        U.swap(1, 3)
        U.swap(0, 3)
        U.cx(3, 2)
        U.cx(3, 0)

    U = U.to_gate()
    U.name = f"{a}^{2**power} mod {N}"
    c_U = U.control()
    return c_U


# レジストリ用の設定
config = {
    "n": N,
    "func": c_amod15,
    "valid_bases": VALID_BASES,
    "n_work_qubits": N_WORK_QUBITS,
    "max_denominator": MAX_DENOMINATOR,
    "factors": FACTORS,
}
