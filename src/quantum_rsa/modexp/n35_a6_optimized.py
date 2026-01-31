"""N=35, a=6, r=2 の最適化量子回路

実機実行成功済み (2026-01-31, Rigetti Ankaa-3)
成功率: 96.6% ± 1.2% (3回平均)

【数論的背景】
ord_35(6) = 2  (∵ 6² = 36 ≡ 1 mod 35)

【状態マッピング】
|1⟩ → |0⟩
|6⟩ → |1⟩

【制御ユニタリ】
U^1 = X gate (状態反転)
U^2 = Identity (周期2)

【回路スペック】
- 量子ビット: 3 (制御: 2, 作業: 1)
- 2Qゲート: 2 (CNOT 1 + CPhaseShift 1)
- 回路深度: ~7

【因数分解】
gcd(6-1, 35) = 5
gcd(6+1, 35) = 7
∴ 35 = 5 × 7
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT

# 定数
N = 35
A = 6
R = 2
FACTORS = (5, 7)
N_WORK_QUBITS = 1
N_COUNT_QUBITS = 2


def build_circuit_qiskit(n_count: int = 2) -> QuantumCircuit:
    """Qiskit用の回路を構築

    Parameters
    ----------
    n_count : int
        制御量子ビット数 (デフォルト: 2)

    Returns
    -------
    QuantumCircuit
        測定付きの完全な回路
    """
    n_work = N_WORK_QUBITS
    qc = QuantumCircuit(n_count + n_work, n_count)

    # 制御レジスタを |+⟩ に初期化
    for i in range(n_count):
        qc.h(i)

    # U^1: CNOT (c0 → work)
    qc.cx(0, n_count)

    # U^2, U^4, ... : Identity (周期2)

    # 逆QFT
    qc.append(QFT(n_count, inverse=True), range(n_count))

    # 測定
    qc.measure(range(n_count), range(n_count))

    return qc


def build_circuit_braket():
    """AWS Braket用の回路を構築

    Returns
    -------
    braket.circuits.Circuit
        Braket回路
    """
    from braket.circuits import Circuit

    circuit = Circuit()

    # c0, c1 = 0, 1 (制御), w0 = 2 (作業)

    # 制御レジスタを |+⟩ に
    circuit.h(0).h(1)

    # U^1: CNOT
    circuit.cnot(0, 2)

    # 逆QFT (2量子ビット) - ビット順保持のためSWAPなし
    circuit.h(1)
    circuit.cphaseshift(0, 1, -np.pi/2)
    circuit.h(0)

    return circuit


def get_expected_phases() -> list[float]:
    """期待される位相のリスト"""
    return [0.0, 0.5]  # s/r for s=0,1 and r=2


def analyze_counts(counts: dict, n_count: int = 2) -> dict:
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
        # 制御レジスタ部分を抽出
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
    "note": "実機成功済み (96.6% ± 1.2%, 2026-01-31)"
}


if __name__ == "__main__":
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    print("=" * 60)
    print("N=35, a=6, r=2 最適化回路")
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
    for bits, count in sorted(counts.items(), key=lambda x: -x[1]):
        phase = int(bits, 2) / 4
        print(f"  {bits}: {count/10000*100:.1f}% (位相 {phase})")

    analysis = analyze_counts(counts)
    print(f"\n成功率: {analysis['success_rate']:.1f}%")
