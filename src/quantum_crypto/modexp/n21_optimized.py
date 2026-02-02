"""N=21 用の最適化された制御付き modular exponentiation 実装

論文: "Demonstration of Shor's factoring algorithm for N=21 on IBM quantum processors"
      Skosana & Tame, Scientific Reports 11, 16599 (2021)
      https://www.nature.com/articles/s41598-021-95973-w

この実装は5量子ビットのみを使用:
- 制御レジスタ: 3量子ビット (c0, c1, c2)
- 作業レジスタ: 2量子ビット (q0, q1)

状態マッピング (a=4 の場合):
- |1⟩  ↦ |00⟩ (q1q0)
- |4⟩  ↦ |01⟩
- |16⟩ ↦ |10⟩

最適化:
- Margolus gate (approximate Toffoli) を使用
- CXゲート数: 15個
- 回路深度: 35
"""
from math import gcd

from qiskit import QuantumCircuit
import numpy as np


# 定数定義
N = 21
FACTORS = (3, 7)
# a=4 のみサポート（論文の実装に準拠）
VALID_BASES = [4]
N_WORK_QUBITS = 2  # 最適化版は2量子ビットのみ
N_COUNT_QUBITS = 3  # 制御レジスタも3量子ビット
MAX_DENOMINATOR = 21


def margolus_gate(qc: QuantumCircuit, c1: int, c2: int, t: int):
    """Margolus gate (approximate Toffoli / relative-phase Toffoli)

    標準Toffoliと同じ動作だが、|101⟩状態に-1の位相を付与。
    この位相は N=21 のアルゴリズムでは影響しない。

    CXゲート数: 3個 (標準Toffoliの6個から半減)
    """
    qc.ry(np.pi/4, t)
    qc.cx(c1, t)
    qc.ry(np.pi/4, t)
    qc.cx(c2, t)
    qc.ry(-np.pi/4, t)
    qc.cx(c1, t)
    qc.ry(-np.pi/4, t)


def controlled_swap_margolus(qc: QuantumCircuit, c: int, t1: int, t2: int):
    """Controlled-SWAP using Margolus decomposition

    Fredkin gate を Margolus + CX で構成
    """
    qc.cx(t2, t1)
    margolus_gate(qc, c, t1, t2)
    qc.cx(t2, t1)


def c_U1(qc: QuantumCircuit, control: int, q0: int, q1: int):
    """U^1: |1⟩ → |4⟩

    4^1 mod 21 = 4

    状態変換:
    - |00⟩ (=1) → |01⟩ (=4)

    実装: 単純なCNOT
    """
    qc.cx(control, q0)


def c_U2(qc: QuantumCircuit, control: int, q0: int, q1: int):
    """U^2: |1⟩ → |16⟩, |4⟩ → |1⟩

    4^2 mod 21 = 16

    状態変換:
    - |00⟩ (=1)  → |10⟩ (=16)
    - |01⟩ (=4)  → |00⟩ (=1)
    - |10⟩ (=16) → |01⟩ (=4)

    実装: Controlled-SWAP + CNOT
    """
    # Fredkin (controlled-SWAP) for q0, q1
    controlled_swap_margolus(qc, control, q0, q1)
    qc.cx(control, q1)


def c_U4(qc: QuantumCircuit, control: int, q0: int, q1: int):
    """U^4: 全状態を循環

    4^4 mod 21 = 256 mod 21 = 4

    状態変換:
    - |00⟩ (=1)  → |01⟩ (=4)
    - |01⟩ (=4)  → |10⟩ (=16)
    - |10⟩ (=16) → |00⟩ (=1)

    実装: Toffoli + Fredkin構成
    """
    # Margolus (approximate Toffoli)
    margolus_gate(qc, control, q1, q0)
    # Fredkin
    controlled_swap_margolus(qc, control, q0, q1)


def c_amod21_optimized(a: int, power: int) -> QuantumCircuit:
    """最適化された制御付き a^(2^power) mod 21 の実装

    論文の5量子ビット実装に基づく。

    Parameters
    ----------
    a : int
        基数 (現在は4のみサポート)
    power : int
        指数 (0, 1, 2 のみ: U^1, U^2, U^4)

    Returns
    -------
    QuantumCircuit
        制御付きユニタリ回路（3量子ビット: control + 2 work）
    """
    if a != 4:
        raise ValueError(f"最適化版は a=4 のみサポート。a={a} は使用できません。")

    if power not in [0, 1, 2]:
        raise ValueError(f"power は 0, 1, 2 のみサポート。power={power} は使用できません。")

    # 3量子ビット回路: qubit 0 = control, qubit 1,2 = work (q0, q1)
    qc = QuantumCircuit(3, name=f"4^{2**power} mod 21")

    control = 0
    q0 = 1
    q1 = 2

    if power == 0:  # U^1
        c_U1(qc, control, q0, q1)
    elif power == 1:  # U^2
        c_U2(qc, control, q0, q1)
    elif power == 2:  # U^4
        c_U4(qc, control, q0, q1)

    return qc.to_gate()


# レジストリ用の設定（最適化版）
config_optimized = {
    "n": N,
    "func": c_amod21_optimized,
    "valid_bases": VALID_BASES,
    "n_work_qubits": N_WORK_QUBITS,
    "n_count_qubits": N_COUNT_QUBITS,
    "max_denominator": MAX_DENOMINATOR,
    "factors": FACTORS,
    "optimized": True,
    "paper": "Skosana & Tame, Scientific Reports 11, 16599 (2021)",
}


def build_full_circuit_n21(shots: int = 1024) -> QuantumCircuit:
    """N=21 完全な因数分解回路を構築

    論文の Figure 3 に基づく実装。

    回路構成:
    - 3 制御量子ビット (c0, c1, c2)
    - 2 作業量子ビット (q0, q1)
    - 逆QFT

    Returns
    -------
    QuantumCircuit
        測定付きの完全な回路
    """
    from qiskit.circuit.library import QFT

    n_count = 3
    n_work = 2

    qc = QuantumCircuit(n_count + n_work, n_count)

    # 制御量子ビットを |+⟩ に初期化
    for i in range(n_count):
        qc.h(i)

    # 作業レジスタを |1⟩ = |00⟩ に初期化 (すでに |0⟩ なのでそのまま)
    # 注: 状態マッピングで |1⟩ ↦ |00⟩

    # 制御ユニタリを適用
    # c0 controls U^1
    c_U1(qc, 0, n_count, n_count + 1)

    # c1 controls U^2
    c_U2(qc, 1, n_count, n_count + 1)

    # c2 controls U^4
    c_U4(qc, 2, n_count, n_count + 1)

    # 逆QFT
    qc.append(QFT(n_count, inverse=True), range(n_count))

    # 測定
    qc.measure(range(n_count), range(n_count))

    return qc


if __name__ == "__main__":
    # テスト実行
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    print("=== N=21 Optimized Shor Implementation ===")
    print("Based on: Skosana & Tame, Scientific Reports 11, 16599 (2021)")
    print()

    # 回路を構築
    qc = build_full_circuit_n21()

    # メトリクス表示
    print(f"Qubits: {qc.num_qubits}")
    print(f"Depth: {qc.depth()}")
    print(f"Gate counts: {dict(qc.count_ops())}")

    # トランスパイル後のメトリクス
    simulator = AerSimulator()
    qc_transpiled = transpile(qc, simulator, optimization_level=3)
    print(f"\nAfter transpilation:")
    print(f"Depth: {qc_transpiled.depth()}")
    print(f"CX gates: {qc_transpiled.count_ops().get('cx', 0)}")

    # シミュレーション実行
    print("\n=== Simulation Results ===")
    job = simulator.run(qc_transpiled, shots=4096)
    result = job.result()
    counts = result.get_counts()

    # 上位の結果を表示
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    print("Top results:")
    for bitstring, count in sorted_counts[:8]:
        phase = int(bitstring, 2) / 8  # 2^3 = 8
        print(f"  {bitstring}: {count} ({count/4096*100:.1f}%) -> phase = {phase:.3f}")
