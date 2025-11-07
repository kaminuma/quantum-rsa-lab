"""Unified Shor's algorithm implementation with both classical and quantum methods."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, Optional, TYPE_CHECKING

import numpy as np

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import QFT
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

if TYPE_CHECKING:
    try:
        from qiskit_aer.noise import NoiseModel
    except ImportError:  # pragma: no cover - 型チェック時のみ
        NoiseModel = object  # type: ignore[misc,assignment]

# modexp レジストリのインポート
try:
    from .modexp import get_config, list_supported
    MODEXP_AVAILABLE = True
except ImportError:
    # スタンドアロン実行時
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from modexp import get_config, list_supported
        MODEXP_AVAILABLE = True
    except ImportError:
        MODEXP_AVAILABLE = False


# ============================================================================
# 定数定義
# ============================================================================
DEFAULT_N = 15
DEFAULT_N_COUNT = 8  # 位相推定の精度（カウントレジスタのビット数）
DEFAULT_SHOTS = 2048


@dataclass
class ShorResult:
    number: int
    base: int
    factors: tuple[int, int] | None
    shots: int
    success: bool
    method: str
    measured_phase: float | None = None
    period: int | None = None
    counts: dict | None = None
    mitigated_counts: dict | None = None
    circuit_depth: int | None = None
    total_gates: int | None = None
    two_qubit_gates: int | None = None
    gate_counts: dict | None = None
    noise_model_name: str | None = None


def gcd(a: int, b: int) -> int:
    """最大公約数を計算"""
    while b:
        a, b = b, a % b
    return a


def find_period_classical(a: int, N: int) -> Optional[int]:
    """古典的な周期発見（小さいNのみ）"""
    if gcd(a, N) != 1:
        return None

    r = 1
    value = a
    while value != 1:
        value = (value * a) % N
        r += 1
        if r > N:  # 無限ループ防止
            return None
    return r


def factor_from_period(a: int, N: int, r: int) -> Optional[tuple[int, int]]:
    """周期rを使って因数分解を試みる"""
    if r % 2 != 0:
        return None

    x = pow(a, r // 2, N)
    if x == N - 1:
        return None

    factor1 = gcd(x - 1, N)
    factor2 = gcd(x + 1, N)

    if factor1 > 1 and factor1 < N:
        return (factor1, N // factor1)
    if factor2 > 1 and factor2 < N:
        return (factor2, N // factor2)

    return None


# ============================================================================
# 量子版の実装
# ============================================================================

def qpe_amod_n(a: int, N: int, n_count: int = DEFAULT_N_COUNT) -> QuantumCircuit:
    """汎用的な量子位相推定を使った a^x mod N の周期発見"""
    if not QISKIT_AVAILABLE:
        raise RuntimeError("Qiskit が必要です。pip install qiskit-aer を実行してください。")

    if not MODEXP_AVAILABLE:
        raise RuntimeError("modexp モジュールが必要です。")

    # N に対応する設定を取得
    config = get_config(N)
    c_amod_func = config["func"]
    n_work = config["n_work_qubits"]

    qc = QuantumCircuit(n_count + n_work, n_count)

    # カウントビットを|+⟩状態に初期化
    for q in range(n_count):
        qc.h(q)

    # 作業レジスタを|1⟩に初期化
    qc.x(n_count)

    # 制御付きユニタリ演算を適用
    for q in range(n_count):
        c_U = c_amod_func(a, q)
        qc.append(c_U, [q] + list(range(n_count, n_count + n_work)))

    # 逆量子フーリエ変換
    qc.append(QFT(n_count, inverse=True), range(n_count))

    # 測定
    qc.measure(range(n_count), range(n_count))

    return qc


def find_period_from_phase(phase: float, N: int, max_denominator: int | None = None) -> int | None:
    """測定された位相から周期を推定（連分数展開）"""
    if phase == 0:
        return None

    # max_denominator が指定されていない場合は N を使用
    if max_denominator is None:
        max_denominator = N

    frac = Fraction(phase).limit_denominator(max_denominator)
    r = frac.denominator

    if r > 0 and r <= N:
        return r
    return None


# ============================================================================
# 実験用ユーティリティ
# ============================================================================
TWO_QUBIT_GATES = {
    "cx",
    "cy",
    "cz",
    "cp",
    "cu",
    "cu1",
    "cu3",
    "ecr",
    "iswap",
    "swap",
    "cswap",
    "rxx",
    "ryy",
    "rzz",
    "xx_plus_yy",
    "xx_minus_yy",
}


def _collect_circuit_metrics(transpiled_qc: "QuantumCircuit") -> tuple[int, int, int, dict]:
    """トランスパイル済み回路から深さやゲート数を算出"""
    gate_counts = dict(transpiled_qc.count_ops())
    total_gates = int(sum(gate_counts.values()))
    two_qubit_gates = int(sum(count for gate, count in gate_counts.items() if gate in TWO_QUBIT_GATES))
    depth = transpiled_qc.depth()
    return depth, total_gates, two_qubit_gates, gate_counts


def _build_meas_calibration_matrix(
    simulator: "AerSimulator",
    n_meas_qubits: int,
    shots: int,
) -> "np.ndarray":
    """測定誤り補正用のキャリブレーション行列を取得"""
    if shots <= 0:
        raise ValueError("meas_calibration_shots は正である必要があります。")

    basis_states = 2 ** n_meas_qubits
    cal_circuits = []
    for state in range(basis_states):
        qc = QuantumCircuit(n_meas_qubits, n_meas_qubits)
        for qubit in range(n_meas_qubits):
            if (state >> qubit) & 1:
                qc.x(qubit)
        qc.measure(range(n_meas_qubits), range(n_meas_qubits))
        cal_circuits.append(qc)

    transpiled = transpile(cal_circuits, simulator)
    result = simulator.run(transpiled, shots=shots).result()

    matrix = np.zeros((basis_states, basis_states))
    for state in range(basis_states):
        counts = result.get_counts(state)
        total = sum(counts.values()) or 1
        for bitstring, count in counts.items():
            measured = int(bitstring, 2)
            matrix[measured, state] = count / total
    return matrix


def _apply_readout_mitigation(
    counts: dict,
    shots: int,
    cal_matrix: "np.ndarray",
    n_meas_qubits: int,
) -> dict[str, float]:
    """キャリブレーション行列を使って簡単な readout mitigation を適用"""
    if shots <= 0:
        return counts

    dim = cal_matrix.shape[0]
    vector = np.zeros(dim)
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        vector[idx] = count

    probs = vector / shots
    mitigated_probs = np.linalg.pinv(cal_matrix) @ probs
    mitigated_probs = np.clip(mitigated_probs, 0, None)

    total = mitigated_probs.sum()
    if total > 0:
        mitigated_probs /= total

    mitigated_counts: dict[str, float] = {}
    for idx, prob in enumerate(mitigated_probs):
        if prob <= 0:
            continue
        bitstring = format(idx, f"0{n_meas_qubits}b")
        mitigated_counts[bitstring] = float(prob * shots)
    return mitigated_counts


# ============================================================================
# 統合インターフェース
# ============================================================================

def run_shor(
    number: int = DEFAULT_N,
    base: Optional[int] = None,
    shots: int = DEFAULT_SHOTS,
    method: Literal["quantum", "classical"] = "quantum",
    n_count: int = DEFAULT_N_COUNT,
    noise_model: "NoiseModel" | None = None,
    apply_readout_mitigation: bool = False,
    meas_calibration_shots: int = 2048,
    simulator_options: Optional[dict] = None,
) -> ShorResult:
    """Shorアルゴリズムの統合実行関数

    Parameters
    ----------
    number : int
        因数分解する合成数
        量子モードでは modexp レジストリに登録された N のみ対応
    base : int, optional
        ベース a。Noneの場合は自動選択
    shots : int
        測定回数（quantumモード時のみ使用）
    method : "quantum" | "classical"
        実行方法（"quantum": 量子回路, "classical": 古典シミュレーション）
    n_count : int
        位相推定の精度（quantumモード時のみ）
    noise_model : NoiseModel, optional
        AerSimulator に適用するノイズモデル
    apply_readout_mitigation : bool
        True の場合、測定キャリブレーションを行い単純な readout mitigation を適用
    meas_calibration_shots : int
        測定キャリブレーション回路を実行するショット数
    simulator_options : dict, optional
        AerSimulator に渡す追加オプション（例: seed_simulator 等）

    Returns
    -------
    ShorResult
        実行結果
    """
    if method == "quantum":
        if not QISKIT_AVAILABLE:
            raise RuntimeError(
                "量子モードには Qiskit が必要です。\n"
                "pip install qiskit-aer を実行するか、method='classical' を使用してください。"
            )
        return _run_quantum_shor(
            number=number,
            base=base,
            shots=shots,
            n_count=n_count,
            noise_model=noise_model,
            apply_readout_mitigation=apply_readout_mitigation,
            meas_calibration_shots=meas_calibration_shots,
            simulator_options=simulator_options,
        )
    elif method == "classical":
        return _run_classical_shor(number, base, shots)
    else:
        raise ValueError(f"無効なmethod: {method}。'quantum' または 'classical' を指定してください。")


def _run_classical_shor(number: int, base: Optional[int], shots: int) -> ShorResult:
    """古典的な方法でShorアルゴリズムを実行"""
    if number % 2 == 0:
        return ShorResult(
            number=number,
            base=2,
            factors=(2, number // 2),
            shots=shots,
            success=True,
            method="classical_trivial_even",
        )

    if base is None:
        import random
        base = random.randint(2, number - 1)

    g = gcd(base, number)
    if g > 1:
        return ShorResult(
            number=number,
            base=base,
            factors=(g, number // g),
            shots=shots,
            success=True,
            method="classical_gcd",
        )

    r = find_period_classical(base, number)

    if r is None:
        return ShorResult(
            number=number,
            base=base,
            factors=None,
            shots=shots,
            success=False,
            method="classical_period_not_found",
        )

    result = factor_from_period(base, number, r)

    if result:
        return ShorResult(
            number=number,
            base=base,
            factors=result,
            shots=shots,
            success=True,
            method=f"classical_period_r={r}",
            period=r,
        )
    else:
        return ShorResult(
            number=number,
            base=base,
            factors=None,
            shots=shots,
            success=False,
            method=f"classical_period_r={r}_but_failed",
            period=r,
        )


def _run_quantum_shor(
    number: int,
    base: Optional[int],
    shots: int,
    n_count: int,
    noise_model: "NoiseModel" | None,
    apply_readout_mitigation: bool,
    meas_calibration_shots: int,
    simulator_options: Optional[dict],
) -> ShorResult:
    """量子回路を使ってShorアルゴリズムを実行"""
    if not MODEXP_AVAILABLE:
        raise RuntimeError("modexp モジュールが必要です。")

    # N に対応する設定を取得
    try:
        config = get_config(number)
    except NotImplementedError as e:
        raise NotImplementedError(str(e)) from e

    valid_bases = config["valid_bases"]
    max_denominator = config["max_denominator"]

    if base is None:
        # デフォルトベースを選択（中央付近の値）
        base = valid_bases[len(valid_bases) // 2]

    if base not in valid_bases:
        raise ValueError(f"a={base} は N={number} に対して無効です。有効なベース: {valid_bases}")

    g = gcd(base, number)
    if g > 1:
        return ShorResult(
            number=number,
            base=base,
            factors=(g, number // g),
            shots=shots,
            success=True,
            method="quantum_gcd",
        )

    # 量子回路を構築して実行
    qc = qpe_amod_n(base, number, n_count)

    simulator_kwargs = dict(simulator_options or {})
    noise_model_name = None
    if noise_model is not None:
        simulator_kwargs["noise_model"] = noise_model
        noise_model_name = getattr(noise_model, "name", None) or noise_model.__class__.__name__
        basis_gates = getattr(noise_model, "basis_gates", None)
        if basis_gates and "basis_gates" not in simulator_kwargs:
            simulator_kwargs["basis_gates"] = basis_gates

    simulator = AerSimulator(**simulator_kwargs)
    qc_transpiled = transpile(qc, simulator)
    circuit_depth, total_gates, two_qubit_gates, gate_counts = _collect_circuit_metrics(qc_transpiled)

    meas_calibration_matrix = None
    if apply_readout_mitigation:
        try:
            meas_calibration_matrix = _build_meas_calibration_matrix(
                simulator,
                n_meas_qubits=n_count,
                shots=meas_calibration_shots,
            )
        except Exception:
            meas_calibration_matrix = None

    result = simulator.run(qc_transpiled, shots=shots).result()
    counts = result.get_counts()

    mitigated_counts = None
    if apply_readout_mitigation and meas_calibration_matrix is not None:
        try:
            mitigated_counts = _apply_readout_mitigation(
                counts=counts,
                shots=shots,
                cal_matrix=meas_calibration_matrix,
                n_meas_qubits=n_count,
            )
        except Exception:
            mitigated_counts = None

    search_counts = mitigated_counts if mitigated_counts else counts

    # 上位の測定結果を複数試す
    sorted_counts = sorted(search_counts.items(), key=lambda x: x[1], reverse=True)

    for measured_string, _ in sorted_counts[:10]:
        measured_value = int(measured_string, 2)

        if measured_value == 0:
            continue

        phase = measured_value / (2 ** n_count)
        r = find_period_from_phase(phase, number, max_denominator)

        if r is None or r == 1:
            continue

        factors = factor_from_period(base, number, r)

        if factors:
            return ShorResult(
                number=number,
                base=base,
                factors=factors,
                shots=shots,
                success=True,
                method=f"quantum_qpe_r={r}_phase={phase:.4f}",
                measured_phase=phase,
                period=r,
                counts=counts,
                mitigated_counts=mitigated_counts,
                circuit_depth=circuit_depth,
                total_gates=total_gates,
                two_qubit_gates=two_qubit_gates,
                gate_counts=gate_counts,
                noise_model_name=noise_model_name,
            )

    # すべて失敗
    best_measured = int(sorted_counts[0][0], 2)
    best_phase = best_measured / (2 ** n_count)

    return ShorResult(
        number=number,
        base=base,
        factors=None,
        shots=shots,
        success=False,
        method=f"quantum_qpe_failed_best_phase={best_phase:.4f}",
        measured_phase=best_phase,
        period=None,
        counts=counts,
        mitigated_counts=mitigated_counts,
        circuit_depth=circuit_depth,
        total_gates=total_gates,
        two_qubit_gates=two_qubit_gates,
        gate_counts=gate_counts,
        noise_model_name=noise_model_name,
    )


if __name__ == "__main__":
    print("=" * 70)
    print("Shor アルゴリズム - 古典 vs 量子")
    print("=" * 70)

    # 古典版の実行
    print("\n【古典版（ループで周期発見）】")
    for a in [7, 11, 2]:
        result = run_shor(number=DEFAULT_N, base=a, method="classical")
        status = "✓" if result.success else "✗"
        print(f"  {status} a={a:2d}: N={DEFAULT_N} = {result.factors} | 方法: {result.method}")

    # 量子版の実行
    if QISKIT_AVAILABLE:
        print("\n【量子版（QPE + QFT）】")
        for a in [7, 11, 2]:
            result = run_shor(number=DEFAULT_N, base=a, method="quantum", shots=DEFAULT_SHOTS)
            status = "✓" if result.success else "✗"
            phase_str = f"位相={result.measured_phase:.4f}" if result.measured_phase else ""
            print(f"  {status} a={a:2d}: N={DEFAULT_N} = {result.factors} | {phase_str} | {result.method}")
    else:
        print("\n【量子版】")
        print("  Qiskit がインストールされていません。")
        print("  pip install qiskit-aer を実行してください。")
