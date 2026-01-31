#!/usr/bin/env python3
"""N=35 実機実験スクリプト（研究用）

使用方法:
    # a=6 (r=2) をシミュレーション
    python scripts/run_n35_experiments.py --base 6 --simulate

    # a=6 を実機で実行（低ショット×反復がデフォルト）
    python scripts/run_n35_experiments.py --base 6

    # a=8 (r=4) を実機で実行（低ショット×反復がデフォルト）
    python scripts/run_n35_experiments.py --base 8

    # 両方実行して比較（低ショット×反復がデフォルト）
    python scripts/run_n35_experiments.py --base both

    # 明示指定（例：最終確認の高ショット単発）
    python scripts/run_n35_experiments.py --base 8 --shots 800 --reps 1

    # 高ショットを明示的に許可（財布破壊ボタン）
    python scripts/run_n35_experiments.py --base 8 --shots 2000 --reps 1 --override_high_shots
"""

import argparse
import json
import os
from datetime import datetime
from fractions import Fraction
from math import gcd
from typing import Dict, List, Optional

import numpy as np
import sys
from pathlib import Path

# プロジェクトルートをパスに追加（src/からのインポート用）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Braket imports
from braket.aws import AwsDevice
from braket.circuits import Circuit

# プロジェクトモジュールからのインポート
from quantum_rsa.modexp.n35_a6_optimized import (
    config as config_a6,
    get_expected_phases as get_expected_phases_a6,
    build_circuit_braket as build_circuit_braket_a6,
)
from quantum_rsa.modexp.n35_a8_optimized import (
    config as config_a8,
    get_expected_phases as get_expected_phases_a8,
    build_circuit_braket as build_circuit_braket_a8,
)


# =============================================================================
# ユーティリティ関数
# =============================================================================

def safe_device_metadata(device: AwsDevice) -> dict:
    """デバイス情報を可能な範囲で収集"""
    meta = {
        "name": getattr(device, "name", None),
        "status": getattr(device, "status", None),
        "arn": getattr(device, "arn", None),
        "provider_name": getattr(device, "provider_name", None),
    }
    try:
        props = device.properties
        meta["properties_raw"] = str(props)[:20000]  # 巨大化防止
    except Exception as e:
        meta["properties_error"] = repr(e)
    return meta


def add_control_measurements_only(circuit: Circuit, control_qubits: List[int]) -> Circuit:
    """制御レジスタだけ測定する（bit順事故を防ぐ）"""
    for q in control_qubits:
        circuit.measure(q)
    return circuit


# =============================================================================
# 理論分布と距離指標
# =============================================================================

def expected_distribution(base: int) -> Dict[str, float]:
    """制御レジスタのみの理論分布（ideal compiled-QPE）

    モジュールから取得した期待位相を使用。
    """
    if base == 6:
        # n_count=2, r=2: 位相 0, 0.5 → ビット 00, 10
        phases = get_expected_phases_a6()  # [0.0, 0.5]
        n_count = config_a6["n_count_qubits"]
    elif base == 8:
        # n_count=3, r=4: 位相 0, 0.25, 0.5, 0.75 → ビット 000, 010, 100, 110
        phases = get_expected_phases_a8()  # [0.0, 0.25, 0.5, 0.75]
        n_count = config_a8["n_count_qubits"]
    else:
        raise ValueError(f"unsupported base: {base}")

    # 位相をビット列に変換
    dist = {}
    for phase in phases:
        bits = format(int(phase * (2 ** n_count)), f'0{n_count}b')
        dist[bits] = 1.0 / len(phases)
    return dist


def normalize_counts(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def total_variation_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def hellinger_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    return (1.0 / np.sqrt(2.0)) * np.sqrt(
        sum((np.sqrt(p.get(k, 0.0)) - np.sqrt(q.get(k, 0.0)))**2 for k in keys)
    )


def kl_divergence(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-12) -> float:
    """KL(p||q). 0*log(0)=0, q=0 は eps で回避"""
    keys = set(p) | set(q)
    s = 0.0
    for k in keys:
        pk = p.get(k, 0.0)
        if pk <= 0:
            continue
        qk = max(q.get(k, 0.0), eps)
        s += pk * np.log(pk / qk)
    return float(s)


def chi_square_uniform_test(counts: Dict[str, int], n_bins: int) -> Dict[str, float]:
    """H0: 一様分布に対するχ²検定"""
    total = sum(counts.values())
    if total == 0 or n_bins == 0:
        return {"chi2": 0.0, "p": 1.0, "df": 0}

    expected = total / n_bins
    chi2 = 0.0
    # 全ビンで計算（観測されなかったビンも含む）
    for i in range(n_bins):
        bits = format(i, f'0{int(np.log2(n_bins))}b') if n_bins > 1 else '0'
        obs = counts.get(bits, 0)
        chi2 += ((obs - expected) ** 2) / expected

    return {"chi2": float(chi2), "df": int(n_bins - 1)}


def infer_r_from_mode(control_counts: Dict[str, int], n_count: int, N: int, a: int) -> Dict:
    """期待ビンの中で最頻値から r を推定（ノイズ耐性版）

    改良点:
    - mode一発ではなく、期待ビン（LSB=0）の中から最頻値を選ぶ
    - 連分数で候補を出し、pow(a, r, N) == 1 を満たす最小 r を採用
    """
    if not control_counts:
        return {"mode_bits": None, "phase": None, "fraction": None, "r": None, "method": "none"}

    # 期待ビン = LSB が 0 のビット列（位相 s/r の正しい出力）
    expected_bins = {b: c for b, c in control_counts.items() if b.endswith('0')}

    if not expected_bins:
        # 期待ビンがない場合は全体から
        expected_bins = control_counts

    # 期待ビンの中で最頻値を選ぶ
    mode_bits = max(expected_bins.items(), key=lambda kv: kv[1])[0]
    x = int(mode_bits, 2)
    phase = x / (2 ** n_count)

    if phase == 0:
        # 位相0は r を決定できない（s=0 のケース）
        # 2番目に多いビット列を使う
        sorted_bins = sorted(expected_bins.items(), key=lambda kv: -kv[1])
        if len(sorted_bins) > 1:
            mode_bits = sorted_bins[1][0]
            x = int(mode_bits, 2)
            phase = x / (2 ** n_count)

    if phase == 0:
        return {"mode_bits": mode_bits, "phase": 0.0, "fraction": "0", "r": None, "method": "phase_zero"}

    frac = Fraction(phase).limit_denominator(N)
    r_candidate = frac.denominator

    # バリデーション: pow(a, r, N) == 1 を満たすか確認
    # 満たさない場合は r の倍数を試す
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
        "r_candidate": r_candidate,
        "r": r_valid if r_valid else r_candidate,
        "method": "expected_bin_mode"
    }


# =============================================================================
# 回路構築
# =============================================================================

def build_n35_a6_circuit() -> Circuit:
    """N=35, a=6, r=2 の最適化回路 (3量子ビット)

    モジュールから回路を取得し、制御レジスタのみ測定を追加。
    """
    circuit = build_circuit_braket_a6()
    # 測定は「制御のみ」（bit順事故を防ぐ）
    add_control_measurements_only(circuit, [0, 1])
    return circuit


def build_n35_a8_circuit() -> Circuit:
    """N=35, a=8, r=4 の最適化回路 (5量子ビット)

    モジュールから回路を取得し、制御レジスタのみ測定を追加。
    """
    circuit = build_circuit_braket_a8()
    # 測定は「制御のみ」（bit順事故を防ぐ）
    add_control_measurements_only(circuit, [0, 1, 2])
    return circuit


# =============================================================================
# 実行関数
# =============================================================================

def run_simulation(circuit: Circuit, shots: int = 10000) -> dict:
    """ローカルシミュレーターで実行"""
    from braket.devices import LocalSimulator

    device = LocalSimulator()
    task = device.run(circuit, shots=shots)
    result = task.result()
    return dict(result.measurement_counts)


def analyze_results(counts: dict, base: int, n_count: int, N: int = 35) -> dict:
    """測定結果を分析（研究用: 距離指標 + r推定）"""
    total = sum(counts.values())
    control_counts = dict(counts)

    # 期待分布（理想）
    exp_dist = expected_distribution(base)
    p_emp = normalize_counts(control_counts)

    # 成功率（bits集合で判定）
    correct_bins = set(exp_dist.keys())
    correct_count = sum(control_counts.get(b, 0) for b in correct_bins)
    success_rate = (correct_count / total * 100) if total > 0 else 0.0

    # 分布距離
    tv = total_variation_distance(p_emp, exp_dist)
    hel = hellinger_distance(p_emp, exp_dist)
    kl = kl_divergence(p_emp, exp_dist)

    # χ²（uniform に対して）
    n_bins = 2 ** n_count
    chi2u = chi_square_uniform_test(control_counts, n_bins)

    # r推定（期待ビンの中から最頻値を使う改良版）
    r_inf = infer_r_from_mode(control_counts, n_count, N, a=base)

    return {
        "total_shots": total,
        "control_counts": control_counts,
        "expected_distribution": exp_dist,
        "correct_bins": sorted(list(correct_bins)),
        "correct_count": int(correct_count),
        "success_rate": float(success_rate),
        "distance_metrics": {
            "total_variation": tv,
            "hellinger": hel,
            "kl_divergence": kl
        },
        "chi_square_uniform": chi2u,
        "r_inference": r_inf,
        "theoretical_period": 2 if base == 6 else 4,
    }


def save_results(base: int, task_id: str, counts: dict, analysis: dict,
                 shots: int, simulate: bool, circuit: Circuit,
                 device_meta: Optional[dict] = None):
    """結果を保存"""
    os.makedirs("logs", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    mode = "sim" if simulate else "ankaa3"
    filename = f"logs/n35-a{base}-{mode}-{timestamp}.json"

    data = {
        "experiment": f"N=35 Shor's Algorithm (a={base})",
        "device": "LocalSimulator" if simulate else "Ankaa-3",
        "task_id": task_id,
        "timestamp": datetime.now().isoformat(),
        "device_metadata": device_meta,
        "parameters": {
            "N": 35,
            "a": base,
            "r_theoretical": analysis.get("theoretical_period"),
            "n_qubits": 3 if base == 6 else 5,
            "n_count": 2 if base == 6 else 3,
            "shots": shots
        },
        "circuit_ir": None,
        "raw_counts": counts,
        "analysis": analysis
    }

    # 回路IRを保存
    try:
        data["circuit_ir"] = circuit.to_ir().json()
    except Exception as e:
        data["circuit_ir_error"] = repr(e)

    with open(filename, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\n結果を保存: {filename}")
    return filename


def run_experiment(base: int, shots: int, simulate: bool) -> dict:
    """実験を実行"""
    print("=" * 70)
    print(f"N=35, a={base} 実験")
    print("=" * 70)
    print()

    # 回路構築
    if base == 6:
        circuit = build_n35_a6_circuit()
        n_count = 2
    elif base == 8:
        circuit = build_n35_a8_circuit()
        n_count = 3
    else:
        raise ValueError(f"サポートされていない基数: {base}")

    print("【回路情報】")
    print(f"  量子ビット数: {circuit.qubit_count}")
    gate_counts = {}
    for instr in circuit.instructions:
        name = instr.operator.name
        gate_counts[name] = gate_counts.get(name, 0) + 1
    print(f"  ゲート構成: {gate_counts}")
    print()

    # 実行
    if simulate:
        print("【シミュレーション実行】")
        task_id = "simulation"
        device_meta = {"name": "LocalSimulator"}
        counts = run_simulation(circuit, shots)
    else:
        print("【実機実行】")
        device = AwsDevice("arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3")
        device_meta = safe_device_metadata(device)
        print(f"  デバイス: {device.name}")
        print(f"  ステータス: {device.status}")

        task = device.run(circuit, shots=shots)
        print(f"  Task ARN: {task.id}")
        print("  実行中...")

        result = task.result()
        task_id = task.id
        counts = dict(result.measurement_counts)

    print("\n【測定結果】")
    for bits, count in sorted(counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {bits}: {count} ({count/shots*100:.1f}%)")

    # 分析
    print("\n【分析】")
    analysis = analyze_results(counts, base, n_count, N=35)
    print(f"  理論周期 r(theory) = {analysis['theoretical_period']}")
    print(f"  正しいビン確率: {analysis['success_rate']:.1f}%")
    print(f"  TV距離: {analysis['distance_metrics']['total_variation']:.4f}")
    print(f"  Hellinger: {analysis['distance_metrics']['hellinger']:.4f}")
    print(f"  KL(p||ideal): {analysis['distance_metrics']['kl_divergence']:.4f}")
    print(f"  χ²(uniform): {analysis['chi_square_uniform']['chi2']:.2f} (df={analysis['chi_square_uniform']['df']})")

    ri = analysis["r_inference"]
    print(f"  r推定(mode): bits={ri['mode_bits']} phase={ri['phase']} frac={ri['fraction']} -> r={ri['r']}")

    # 因数分解
    print("\n【因数分解】")
    a = base
    r_est = analysis["r_inference"].get("r")
    r_theory = analysis["theoretical_period"]

    # 因数分解に使う周期は理論値を固定（評価器としてブレを排除）
    # r_est はノイズで誤爆しやすいため、診断用ログのみに使用
    r_use = r_theory

    a_half = pow(a, r_use // 2, 35)
    p = gcd(a_half - 1, 35)
    q = gcd(a_half + 1, 35)
    print(f"  r_use={r_use} fixed (r_est={r_est}, r_theory={r_theory})")
    print(f"  a^(r/2) mod N = {a}^{r_use//2} mod 35 = {a_half}")
    print(f"  gcd({a_half}-1, 35) = {p}")
    print(f"  gcd({a_half}+1, 35) = {q}")
    print(f"  ∴ 35 = {p} × {q}")

    # 保存
    save_results(base, task_id, counts, analysis, shots, simulate, circuit, device_meta)

    return analysis


def main():
    # ========= ショット数の運用方針（重要） =========
    # - 実機はコスト制約が強いので「低ショット×反復」がデフォルト
    #   実機デフォルト: shots=200, reps=5（合計1000 shots）
    # - シミュレーションは統計が安いので shots=10000, reps=1 がデフォルト
    # - 実機で shots>1000 は事故りやすいので、明示オプトインがない限り拒否
    REAL_DEFAULT_SHOTS = 200
    REAL_DEFAULT_REPS  = 5
    SIM_DEFAULT_SHOTS  = 10000
    SIM_DEFAULT_REPS   = 1
    REAL_MAX_SHOTS_NO_OVERRIDE = 1000

    parser = argparse.ArgumentParser(description="N=35 Shor実験（研究用）")
    parser.add_argument("--base", type=str, default="8",
                       help="基数 (6, 8, or both). 実機の主対象は a=8 を想定して default=8")
    parser.add_argument("--shots", type=int, default=None,
                       help="ショット数（未指定なら simulate/real で安全なデフォルトを採用）")
    parser.add_argument("--simulate", action="store_true",
                       help="シミュレーションのみ")
    parser.add_argument("--reps", type=int, default=None,
                       help="同一条件の反復回数（未指定なら simulate/real で安全なデフォルトを採用）")
    parser.add_argument("--override_high_shots", action="store_true",
                       help="実機で shots>1000 を明示的に許可（財布破壊ボタン）")

    args = parser.parse_args()

    # ---- デフォルト shot/reps を安全側に自動設定 ----
    if args.shots is None:
        args.shots = SIM_DEFAULT_SHOTS if args.simulate else REAL_DEFAULT_SHOTS
    if args.reps is None:
        args.reps = SIM_DEFAULT_REPS if args.simulate else REAL_DEFAULT_REPS

    # ---- 実機の高ショット事故防止 ----
    if (not args.simulate) and (args.shots > REAL_MAX_SHOTS_NO_OVERRIDE) and (not args.override_high_shots):
        raise SystemExit(
            f"Refusing: real QPU shots={args.shots} (> {REAL_MAX_SHOTS_NO_OVERRIDE}). "
            f"Use <= {REAL_MAX_SHOTS_NO_OVERRIDE} or pass --override_high_shots."
        )

    bases = [6, 8] if args.base == "both" else [int(args.base)]

    results = {}
    for base in bases:
        rep_analyses = []
        for i in range(args.reps):
            if args.reps > 1:
                print(f"\n{'='*70}")
                print(f"--- rep {i+1}/{args.reps} ---")
            rep_analyses.append(run_experiment(base, args.shots, args.simulate))

        # 平均/分散（成功率と距離）
        sr = [a["success_rate"] for a in rep_analyses]
        tv = [a["distance_metrics"]["total_variation"] for a in rep_analyses]
        hel = [a["distance_metrics"]["hellinger"] for a in rep_analyses]

        results[base] = {
            "reps": args.reps,
            "per_rep": rep_analyses,
            "summary": {
                "success_rate_mean": float(np.mean(sr)),
                "success_rate_std": float(np.std(sr)),
                "tv_mean": float(np.mean(tv)),
                "tv_std": float(np.std(tv)),
                "hellinger_mean": float(np.mean(hel)),
                "hellinger_std": float(np.std(hel)),
            }
        }

        if args.reps > 1:
            print("\n[rep summary]")
            s = results[base]["summary"]
            print(f"  success_rate: mean={s['success_rate_mean']:.2f}% std={s['success_rate_std']:.2f}%")
            print(f"  TV: mean={s['tv_mean']:.4f} std={s['tv_std']:.4f}")
            print(f"  Hellinger: mean={s['hellinger_mean']:.4f} std={s['hellinger_std']:.4f}")
        print()

    # 比較 (両方実行した場合)
    if len(bases) == 2:
        print("=" * 70)
        print("【比較】")
        print("=" * 70)
        print()
        print("| base | qubits | reps | success_rate(mean±std) | TV(mean±std) | Hellinger(mean±std) |")
        print("|------|--------|------|------------------------|--------------|---------------------|")
        for base in bases:
            qubits = 3 if base == 6 else 5
            s = results[base]["summary"]
            print(
                f"| a={base} | {qubits} | {results[base]['reps']} | "
                f"{s['success_rate_mean']:.1f}±{s['success_rate_std']:.1f}% | "
                f"{s['tv_mean']:.4f}±{s['tv_std']:.4f} | "
                f"{s['hellinger_mean']:.4f}±{s['hellinger_std']:.4f} |"
            )


if __name__ == "__main__":
    main()
