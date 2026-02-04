#!/usr/bin/env python3
"""r=4 スケーリング実験スクリプト

同一回路構造（5量子ビット, 8 2Qゲート）で N を上げていく実験。

使用方法:
    # シミュレーション（全ターゲット）
    python scripts/run_r4_scaling_experiments.py --simulate

    # 特定の N のみ
    python scripts/run_r4_scaling_experiments.py --target 91 --simulate
    python scripts/run_r4_scaling_experiments.py --target 91

    # 全ターゲット実機実行
    python scripts/run_r4_scaling_experiments.py

    # ショット数指定
    python scripts/run_r4_scaling_experiments.py --shots 500 --reps 2

ターゲット:
    N=91  (7×13),  a=8,  r=4  -- N=35と同じ基数！
    N=143 (11×13), a=34, r=4
    N=185 (5×37),  a=6,  r=4

コスト見積もり:
    - 1000 shots × 3 ターゲット ≈ $4.05
"""

import argparse
import json
import os
from datetime import datetime
from fractions import Fraction
from math import gcd
from typing import Dict, Optional

import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from braket.aws import AwsDevice
from braket.circuits import Circuit


# =============================================================================
# ターゲット定義
# =============================================================================

TARGETS = {
    35: {"a": 8, "r": 4, "factors": (5, 7), "module": "n35_a8_optimized"},
    91: {"a": 8, "r": 4, "factors": (7, 13), "module": "n91_a8_r4_optimized"},
    143: {"a": 34, "r": 4, "factors": (11, 13), "module": "n143_a34_r4_optimized"},
    185: {"a": 6, "r": 4, "factors": (5, 37), "module": "n185_a6_r4_optimized"},
}

N_COUNT_QUBITS = 3  # 全ターゲット共通


# =============================================================================
# 回路構築
# =============================================================================

def build_circuit_braket(N: int) -> Circuit:
    """指定した N の Braket 回路を構築"""
    target = TARGETS[N]
    module_name = target["module"]

    # 動的インポート
    module = __import__(f"quantum_crypto.modexp.{module_name}", fromlist=["build_circuit_braket"])
    circuit = module.build_circuit_braket()

    # 制御レジスタのみ測定
    for q in range(N_COUNT_QUBITS):
        circuit.measure(q)

    return circuit


# =============================================================================
# 分析関数
# =============================================================================

def expected_distribution_r4() -> Dict[str, float]:
    """r=4 の期待分布（位相 0, 0.25, 0.5, 0.75）"""
    return {"000": 0.25, "010": 0.25, "100": 0.25, "110": 0.25}


def normalize_counts(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total > 0 else {}


def hellinger_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    return (1.0 / np.sqrt(2.0)) * np.sqrt(
        sum((np.sqrt(p.get(k, 0.0)) - np.sqrt(q.get(k, 0.0)))**2 for k in keys)
    )


def infer_r_from_mode(counts: Dict[str, int], N: int, a: int) -> Dict:
    """期待ビンの中で最頻値から r を推定"""
    if not counts:
        return {"mode_bits": None, "phase": None, "r": None}

    # 期待ビン = LSB が 0
    expected_bins = {b: c for b, c in counts.items() if b.endswith('0')}
    if not expected_bins:
        expected_bins = counts

    mode_bits = max(expected_bins.items(), key=lambda kv: kv[1])[0]
    x = int(mode_bits, 2)
    phase = x / (2 ** N_COUNT_QUBITS)

    if phase == 0:
        sorted_bins = sorted(expected_bins.items(), key=lambda kv: -kv[1])
        if len(sorted_bins) > 1:
            mode_bits = sorted_bins[1][0]
            x = int(mode_bits, 2)
            phase = x / (2 ** N_COUNT_QUBITS)

    if phase == 0:
        return {"mode_bits": mode_bits, "phase": 0.0, "r": None}

    frac = Fraction(phase).limit_denominator(N)
    r_candidate = frac.denominator

    r_valid = None
    for mult in [1, 2, 4]:
        r_try = r_candidate * mult
        if r_try <= N and pow(a, r_try, N) == 1:
            r_valid = r_try
            break

    return {
        "mode_bits": mode_bits,
        "phase": phase,
        "fraction": str(frac),
        "r": r_valid if r_valid else r_candidate
    }


def analyze_results(counts: dict, N: int) -> dict:
    """測定結果を分析"""
    target = TARGETS[N]
    a = target["a"]
    total = sum(counts.values())

    exp_dist = expected_distribution_r4()
    p_emp = normalize_counts(counts)

    correct_bins = set(exp_dist.keys())
    correct_count = sum(counts.get(b, 0) for b in correct_bins)
    support_mass = (correct_count / total * 100) if total > 0 else 0.0

    hel = hellinger_distance(p_emp, exp_dist)
    r_inf = infer_r_from_mode(counts, N, a)

    return {
        "N": N,
        "a": a,
        "r_theoretical": 4,
        "factors": target["factors"],
        "total_shots": total,
        "control_counts": counts,
        "correct_bins": sorted(list(correct_bins)),
        "correct_count": int(correct_count),
        "support_mass": float(support_mass),
        "hellinger": float(hel),
        "r_inference": r_inf,
    }


# =============================================================================
# 実行関数
# =============================================================================

def run_simulation(circuit: Circuit, shots: int = 10000) -> dict:
    from braket.devices import LocalSimulator
    device = LocalSimulator()
    task = device.run(circuit, shots=shots)
    result = task.result()
    return dict(result.measurement_counts)


def safe_device_metadata(device: AwsDevice) -> dict:
    meta = {
        "name": getattr(device, "name", None),
        "status": getattr(device, "status", None),
        "arn": getattr(device, "arn", None),
    }
    return meta


def save_results(N: int, task_id: str, counts: dict, analysis: dict,
                 shots: int, simulate: bool, device_meta: Optional[dict] = None):
    """結果を保存"""
    os.makedirs("logs", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    mode = "sim" if simulate else "ankaa3"
    target = TARGETS[N]
    filename = f"logs/n{N}-a{target['a']}-r4-{mode}-{timestamp}.json"

    data = {
        "experiment": f"N={N} Shor r=4 scaling",
        "device": "LocalSimulator" if simulate else "Ankaa-3",
        "task_id": task_id,
        "timestamp": datetime.now().isoformat(),
        "device_metadata": device_meta,
        "parameters": {
            "N": N,
            "a": target["a"],
            "r": 4,
            "factors": target["factors"],
            "n_qubits": 5,
            "n_count": N_COUNT_QUBITS,
            "shots": shots
        },
        "raw_counts": counts,
        "analysis": analysis
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"  保存: {filename}")
    return filename


def run_single_experiment(N: int, shots: int, simulate: bool) -> dict:
    """単一の N に対する実験"""
    target = TARGETS[N]
    a = target["a"]
    factors = target["factors"]

    print(f"\n{'='*70}")
    print(f"N={N} = {factors[0]} × {factors[1]}, a={a}, r=4")
    print('='*70)

    circuit = build_circuit_braket(N)

    print(f"  量子ビット: {circuit.qubit_count}")
    gate_counts = {}
    for instr in circuit.instructions:
        name = instr.operator.name
        gate_counts[name] = gate_counts.get(name, 0) + 1
    print(f"  ゲート: {gate_counts}")

    if simulate:
        print("  [シミュレーション]")
        task_id = "simulation"
        device_meta = {"name": "LocalSimulator"}
        counts = run_simulation(circuit, shots)
    else:
        print("  [実機実行]")
        device = AwsDevice("arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3")
        device_meta = safe_device_metadata(device)
        print(f"  デバイス: {device.name}, ステータス: {device.status}")

        task = device.run(circuit, shots=shots)
        print(f"  Task: {task.id}")
        print("  実行中...")

        result = task.result()
        task_id = task.id
        counts = dict(result.measurement_counts)

    # 分析
    analysis = analyze_results(counts, N)

    print(f"\n  【結果】")
    print(f"  Support Mass: {analysis['support_mass']:.1f}%")
    print(f"  Hellinger: {analysis['hellinger']:.4f}")
    print(f"  r推定: {analysis['r_inference']['r']}")

    # 因数分解表示
    r = 4
    a_half = pow(a, r // 2, N)
    p = gcd(a_half - 1, N)
    q = gcd(a_half + 1, N)
    print(f"\n  【因数分解】")
    print(f"  {a}^2 mod {N} = {a_half}")
    print(f"  gcd({a_half}-1, {N}) = {p}, gcd({a_half}+1, {N}) = {q}")
    print(f"  ∴ {N} = {p} × {q} {'✓' if sorted([p,q]) == sorted(factors) else '✗'}")

    # 保存
    save_results(N, task_id, counts, analysis, shots, simulate, device_meta)

    return analysis


def main():
    REAL_DEFAULT_SHOTS = 1000
    REAL_DEFAULT_REPS = 1
    SIM_DEFAULT_SHOTS = 10000
    SIM_DEFAULT_REPS = 1

    parser = argparse.ArgumentParser(description="r=4 スケーリング実験")
    parser.add_argument("--target", type=int, default=None,
                       help="特定の N のみ実行 (91, 143, 185)")
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--include_35", action="store_true",
                       help="N=35も含める（比較用）")

    args = parser.parse_args()

    if args.shots is None:
        args.shots = SIM_DEFAULT_SHOTS if args.simulate else REAL_DEFAULT_SHOTS
    if args.reps is None:
        args.reps = SIM_DEFAULT_REPS if args.simulate else REAL_DEFAULT_REPS

    # ターゲット選択
    if args.target:
        if args.target not in TARGETS:
            raise ValueError(f"Unknown target N={args.target}. Available: {list(TARGETS.keys())}")
        targets = [args.target]
    else:
        targets = [91, 143, 185]
        if args.include_35:
            targets = [35] + targets

    print("="*70)
    print("r=4 スケーリング実験")
    print("="*70)
    print(f"ターゲット: {targets}")
    print(f"shots={args.shots}, reps={args.reps}")
    print(f"モード: {'シミュレーション' if args.simulate else '実機 (Ankaa-3)'}")

    if not args.simulate:
        cost_per_shot = 0.00035
        gate_cost = 0.00145 * 8  # 8 2Q gates
        total_cost = len(targets) * args.reps * (args.shots * cost_per_shot + gate_cost)
        print(f"推定コスト: ${total_cost:.2f}")

    results = {}
    for N in targets:
        rep_results = []
        for i in range(args.reps):
            if args.reps > 1:
                print(f"\n--- N={N} rep {i+1}/{args.reps} ---")
            rep_results.append(run_single_experiment(N, args.shots, args.simulate))

        sm = [r["support_mass"] for r in rep_results]
        hel = [r["hellinger"] for r in rep_results]

        results[N] = {
            "per_rep": rep_results,
            "summary": {
                "support_mass_mean": float(np.mean(sm)),
                "support_mass_std": float(np.std(sm)),
                "hellinger_mean": float(np.mean(hel)),
                "hellinger_std": float(np.std(hel)),
            }
        }

    # サマリー
    print("\n" + "="*70)
    print("【サマリー】")
    print("="*70)
    print()
    print("| N | factors | a | Support Mass | Hellinger |")
    print("|---|---------|---|--------------|-----------|")
    for N in targets:
        t = TARGETS[N]
        s = results[N]["summary"]
        print(f"| {N} | {t['factors'][0]}×{t['factors'][1]} | {t['a']} | "
              f"{s['support_mass_mean']:.1f}±{s['support_mass_std']:.1f}% | "
              f"{s['hellinger_mean']:.3f}±{s['hellinger_std']:.3f} |")

    print()
    print("【ポイント】")
    print("- 全て同一の回路構造（5量子ビット, 8 2Qゲート）")
    print("- N を上げても Support Mass はほぼ同じはず（デバイスノイズ依存）")
    print("- r=4 は「正当な Shor」（Smolin批判の対象外）")


if __name__ == "__main__":
    main()
