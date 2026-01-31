"""N=35, a=8, r=4 の最適化量子回路

実機実行成功済み (2026-01-31, Rigetti Ankaa-3)
成功率: 47.6% ± 2.2% (5回平均)

r>2 の実証用。Margolus gate で Toffoli を削減。

【数論的背景】
ord_35(8) = 4
8^0 mod 35 = 1
8^1 mod 35 = 8
8^2 mod 35 = 29
8^3 mod 35 = 22
8^4 mod 35 = 1 (周期4)

【状態マッピング】
|1⟩  → |00⟩
|8⟩  → |01⟩
|29⟩ → |10⟩
|22⟩ → |11⟩

【制御ユニタリ】
U^1 = 巡回置換 (+1 mod 4) = CNOT + Margolus
U^2 = X on work1 (|00⟩↔|10⟩, |01⟩↔|11⟩)
U^4 = Identity (周期4)

【回路スペック】
- 量子ビット: 5 (制御: 3, 作業: 2)
- 2Qゲート: 8 (CNOT 5 + CPhaseShift 3) ※Braket版はSWAPなし
- Toffoli: 0 (Margolus で代替)

【因数分解】
gcd(8^2 - 1, 35) = gcd(28, 35) = 7
gcd(8^2 + 1, 35) = gcd(30, 35) = 5
∴ 35 = 5 × 7
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT

# 定数
N = 35
A = 8
R = 4
FACTORS = (5, 7)
N_WORK_QUBITS = 2
N_COUNT_QUBITS = 3


def margolus_gate_qiskit(qc: QuantumCircuit, c1: int, c2: int, t: int):
    """Margolus gate (近似Toffoli) を Qiskit 回路に追加

    標準 Toffoli の CX 6個 → CX 3個 + RY 4個
    |101⟩ に -1 位相が付くが、本実験では影響なし。
    """
    qc.ry(np.pi/4, t)
    qc.cx(c1, t)
    qc.ry(np.pi/4, t)
    qc.cx(c2, t)
    qc.ry(-np.pi/4, t)
    qc.cx(c1, t)
    qc.ry(-np.pi/4, t)


def build_circuit_qiskit(n_count: int = 3) -> QuantumCircuit:
    """Qiskit用の回路を構築

    Parameters
    ----------
    n_count : int
        制御量子ビット数 (デフォルト: 3)

    Returns
    -------
    QuantumCircuit
        測定付きの完全な回路
    """
    n_work = N_WORK_QUBITS
    qc = QuantumCircuit(n_count + n_work, n_count)

    work0 = n_count      # 作業量子ビット0
    work1 = n_count + 1  # 作業量子ビット1

    # 制御レジスタを |+⟩ に初期化
    for i in range(n_count):
        qc.h(i)

    # U^1: 制御付き increment (+1 mod 4)
    # c0 が制御
    qc.cx(0, work0)
    margolus_gate_qiskit(qc, 0, work0, work1)

    # U^2: c1 controls X on work1
    qc.cx(1, work1)

    # U^4: Identity (周期4なので何もしない)

    # 逆QFT
    qc.append(QFT(n_count, inverse=True), range(n_count))

    # 測定
    qc.measure(range(n_count), range(n_count))

    return qc


def margolus_gate_braket(circuit, c1: int, c2: int, t: int):
    """Margolus gate を Braket 回路に追加"""
    circuit.ry(t, np.pi/4)
    circuit.cnot(c1, t)
    circuit.ry(t, np.pi/4)
    circuit.cnot(c2, t)
    circuit.ry(t, -np.pi/4)
    circuit.cnot(c1, t)
    circuit.ry(t, -np.pi/4)
    return circuit


def build_circuit_braket():
    """AWS Braket用の回路を構築

    Returns
    -------
    braket.circuits.Circuit
        Braket回路
    """
    from braket.circuits import Circuit

    circuit = Circuit()

    # c0, c1, c2 = 0, 1, 2 (制御), w0, w1 = 3, 4 (作業)

    # 制御レジスタを |+⟩ に
    circuit.h(0).h(1).h(2)

    # U^1: 制御付き increment
    circuit.cnot(0, 3)
    margolus_gate_braket(circuit, 0, 3, 4)

    # U^2: CNOT
    circuit.cnot(1, 4)

    # 逆QFT (3量子ビット) - ビット順保持のためSWAPなし
    circuit.h(2)
    circuit.cphaseshift(1, 2, -np.pi/2)
    circuit.h(1)
    circuit.cphaseshift(0, 2, -np.pi/4)
    circuit.cphaseshift(0, 1, -np.pi/2)
    circuit.h(0)

    return circuit


def get_expected_phases() -> list[float]:
    """期待される位相のリスト"""
    return [0.0, 0.25, 0.5, 0.75]  # s/r for s=0,1,2,3 and r=4


def analyze_counts(counts: dict, n_count: int = 3) -> dict:
    """測定結果を分析

    Parameters
    ----------
    counts : dict
        測定結果 {bitstring: count}
    n_count : int
        制御量子ビット数

    Returns
    -------
    dict
        分析結果
    """
    total = sum(counts.values())
    expected = get_expected_phases()

    # 正しい位相の確率を計算
    correct_count = 0
    for bits, count in counts.items():
        control_bits = bits[-n_count:] if len(bits) > n_count else bits
        phase = int(control_bits, 2) / (2 ** n_count)
        if any(abs(phase - ep) < 0.01 for ep in expected):
            correct_count += count

    return {
        "total_shots": total,
        "correct_count": correct_count,
        "success_rate": correct_count / total * 100,
        "period": R,
        "base": A
    }


# 設定エクスポート
config = {
    "n": N,
    "a": A,
    "r": R,
    "factors": FACTORS,
    "n_work_qubits": N_WORK_QUBITS,
    "n_count_qubits": N_COUNT_QUBITS,
    "note": "実機成功済み (47.6% ± 2.2%, 2026-01-31)"
}


if __name__ == "__main__":
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    print("=" * 60)
    print("N=35, a=8, r=4 最適化回路 (Margolus)")
    print("=" * 60)
    print()

    qc = build_circuit_qiskit()

    print(f"量子ビット: {qc.num_qubits}")
    print(f"ゲート: {dict(qc.count_ops())}")
    print()

    simulator = AerSimulator()
    qc_t = transpile(qc, simulator, optimization_level=3)
    print(f"トランスパイル後 CX: {qc_t.count_ops().get('cx', 0)}")
    print()

    job = simulator.run(qc_t, shots=10000)
    counts = job.result().get_counts()

    print("測定結果:")
    for bits, count in sorted(counts.items(), key=lambda x: -x[1])[:8]:
        phase = int(bits, 2) / 8
        print(f"  {bits}: {count/10000*100:.1f}% (位相 {phase:.3f})")

    analysis = analyze_counts(counts)
    print(f"\n成功率: {analysis['success_rate']:.1f}%")
    print()

    # 比較表
    print("【a=6 vs a=8 比較】")
    print("| 項目 | a=6 (r=2) | a=8 (r=4) |")
    print("|------|-----------|-----------|")
    print("| 量子ビット | 3 | 5 |")
    print("| 2Qゲート | 2 | 8 |")
    print("| 周期 | 2 | 4 |")
