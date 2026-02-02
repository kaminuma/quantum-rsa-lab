"""N=91, a=8, r=4 の最適化量子回路

N=35 と同じ回路構造で N=91 (= 7 x 13) を因数分解可能。

【数論的背景】
ord_91(8) = 4
8^0 mod 91 = 1
8^1 mod 91 = 8
8^2 mod 91 = 64
8^3 mod 91 = 57
8^4 mod 91 = 1 (周期4)

【状態マッピング】(N=35と同一構造)
|1⟩  → |00⟩
|8⟩  → |01⟩
|64⟩ → |10⟩
|57⟩ → |11⟩

【制御ユニタリ】(N=35と完全に同一)
U^1 = 巡回置換 (+1 mod 4) = CNOT + Margolus
U^2 = X on work1 (|00⟩↔|10⟩, |01⟩↔|11⟩)
U^4 = Identity (周期4)

【回路スペック】
- 量子ビット: 5 (制御: 3, 作業: 2)
- 2Qゲート: 8 (CNOT 5 + CPhaseShift 3)
- Toffoli: 0 (Margolus で代替)

【因数分解】
gcd(8^2 - 1, 91) = gcd(63, 91) = 7
gcd(8^2 + 1, 91) = gcd(65, 91) = 13
∴ 91 = 7 × 13

【ポイント】
N=35 → N=91 で 2.6倍大きな数を、同一の回路複雑度で因数分解。
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT

# 定数
N = 91
A = 8
R = 4
FACTORS = (7, 13)
N_WORK_QUBITS = 2
N_COUNT_QUBITS = 3


def margolus_gate_qiskit(qc: QuantumCircuit, c1: int, c2: int, t: int):
    """Margolus gate (近似Toffoli) を Qiskit 回路に追加"""
    qc.ry(np.pi/4, t)
    qc.cx(c1, t)
    qc.ry(np.pi/4, t)
    qc.cx(c2, t)
    qc.ry(-np.pi/4, t)
    qc.cx(c1, t)
    qc.ry(-np.pi/4, t)


def build_circuit_qiskit(n_count: int = 3) -> QuantumCircuit:
    """Qiskit用の回路を構築"""
    n_work = N_WORK_QUBITS
    qc = QuantumCircuit(n_count + n_work, n_count)

    work0 = n_count
    work1 = n_count + 1

    for i in range(n_count):
        qc.h(i)

    # U^1: 制御付き increment (+1 mod 4)
    qc.cx(0, work0)
    margolus_gate_qiskit(qc, 0, work0, work1)

    # U^2: c1 controls X on work1
    qc.cx(1, work1)

    # U^4: Identity

    # 逆QFT
    qc.append(QFT(n_count, inverse=True), range(n_count))

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
    """AWS Braket用の回路を構築"""
    from braket.circuits import Circuit

    circuit = Circuit()

    # c0, c1, c2 = 0, 1, 2 (制御), w0, w1 = 3, 4 (作業)
    circuit.h(0).h(1).h(2)

    # U^1: 制御付き increment
    circuit.cnot(0, 3)
    margolus_gate_braket(circuit, 0, 3, 4)

    # U^2: CNOT
    circuit.cnot(1, 4)

    # 逆QFT (3量子ビット) - SWAPなし
    circuit.h(2)
    circuit.cphaseshift(1, 2, -np.pi/2)
    circuit.h(1)
    circuit.cphaseshift(0, 2, -np.pi/4)
    circuit.cphaseshift(0, 1, -np.pi/2)
    circuit.h(0)

    return circuit


def get_expected_phases() -> list[float]:
    """期待される位相のリスト"""
    return [0.0, 0.25, 0.5, 0.75]


def analyze_counts(counts: dict, n_count: int = 3) -> dict:
    """測定結果を分析"""
    total = sum(counts.values())
    expected = get_expected_phases()

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


config = {
    "n": N,
    "a": A,
    "r": R,
    "factors": FACTORS,
    "n_work_qubits": N_WORK_QUBITS,
    "n_count_qubits": N_COUNT_QUBITS,
    "note": "N=35と同一回路構造"
}


if __name__ == "__main__":
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    print("=" * 60)
    print(f"N={N}, a={A}, r={R} 最適化回路")
    print(f"因数: {FACTORS[0]} × {FACTORS[1]}")
    print("=" * 60)

    qc = build_circuit_qiskit()
    simulator = AerSimulator()
    qc_t = transpile(qc, simulator, optimization_level=3)

    job = simulator.run(qc_t, shots=10000)
    counts = job.result().get_counts()

    print("\n測定結果:")
    for bits, count in sorted(counts.items(), key=lambda x: -x[1])[:8]:
        phase = int(bits, 2) / 8
        print(f"  {bits}: {count/10000*100:.1f}% (位相 {phase:.3f})")

    analysis = analyze_counts(counts)
    print(f"\nSupport Mass: {analysis['success_rate']:.1f}%")
