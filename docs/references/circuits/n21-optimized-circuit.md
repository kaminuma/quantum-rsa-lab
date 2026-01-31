---
title: N=21 Optimized Circuit Complete Reference
source: Skosana & Tame (2021), このリポジトリの実装
last_verified: 2026-01-30
---

# N=21 最適化回路の完全詳細

## 概要

N=21を5量子ビットのみで因数分解する最適化回路。
標準QPE実装（カウント8 + 作業5 = 13量子ビット、深度10,000+）と比較して劇的に効率化。
※ 理論的な最小値は 2n+n = 15量子ビット（n=5）だが、カウントビット数は精度とトレードオフ

| 項目 | 標準実装 | 最適化版 |
|------|---------|---------|
| 量子ビット | 13 | **5** |
| 回路深度 | 10,600 | **35** |
| CXゲート | 大量 | **25** |
| 実機実行 | 不可能 | **可能** |

※ CXゲート数25はSkosana & Tame (2021)論文の値

## 量子ビット構成

```
制御レジスタ: c0, c1, c2 (3量子ビット)
  - QPEの位相情報を記録
  - 各ビットが U^1, U^2, U^4 を制御

作業レジスタ: q0, q1 (2量子ビット)
  - |a^x mod 21⟩ の状態を保持
  - 3状態のみ使用: {|1⟩, |4⟩, |16⟩}
```

## 状態マッピング

通常、21までの状態を表現するには5量子ビット（0-31）が必要。
しかし、a=4 の場合に実際に現れる状態は3つだけ:

```
4^0 mod 21 = 1   → |00⟩ (q1q0)
4^1 mod 21 = 4   → |01⟩
4^2 mod 21 = 16  → |10⟩
4^3 mod 21 = 1   (周期3で循環)
```

**エンコーディング**:
```
実際の値  2量子ビット表現
   1    →    |00⟩
   4    →    |01⟩
  16    →    |10⟩
```

## 制御ユニタリ演算

### U^1: 4^1 mod 21 = 4

```
変換: |1⟩ → |4⟩
      |00⟩ → |01⟩

回路: 単純なCNOT（状態圧縮により簡略化）
c ──●──
    │
q0 ─X──
q1 ────

※ このCNOTは、{ |00⟩, |01⟩, |10⟩ } に限定した状態圧縮符号化のもとで、
   4倍写像が単一ビット反転として表現できることを利用している。
   汎用的なmod-expはより多くのゲートが必要。
```

```python
def c_U1(qc, control, q0, q1):
    # 状態圧縮版：{|1⟩,|4⟩,|16⟩}のみを扱う
    qc.cx(control, q0)
```

### U^2: 4^2 mod 21 = 16

```
変換: |1⟩ → |16⟩, |4⟩ → |1⟩, |16⟩ → |4⟩
      |00⟩ → |10⟩
      |01⟩ → |00⟩
      |10⟩ → |01⟩

回路: Controlled-SWAP + CNOT
```

```python
def c_U2(qc, control, q0, q1):
    # Controlled-SWAP (Fredkin)
    qc.cx(q1, q0)
    margolus_gate(qc, control, q0, q1)
    qc.cx(q1, q0)
    # CNOT
    qc.cx(control, q1)
```

### U^4: 4^4 mod 21 = 256 mod 21 = 4

```
注意:
r = 3 のため、4^4 ≡ 4^(4 mod 3) ≡ 4^1 (mod 21)。
したがって U⁴ は U¹ と同型のユニタリであり、
状態空間では同じ巡回置換を実装する。

変換: |1⟩ → |4⟩, |4⟩ → |16⟩, |16⟩ → |1⟩
      |00⟩ → |01⟩, |01⟩ → |10⟩, |10⟩ → |00⟩

回路: Margolus + Fredkin
```

```python
def c_U4(qc, control, q0, q1):
    # Margolus gate
    margolus_gate(qc, control, q1, q0)
    # Fredkin
    qc.cx(q1, q0)
    margolus_gate(qc, control, q0, q1)
    qc.cx(q1, q0)
```

## 完全な回路図

```
        ┌───┐                                              ┌───────────┐
c0: |0⟩─┤ H ├──●───────────────────────────────────────────┤           ├─M─→ b0
        └───┘  │                                           │           │
        ┌───┐  │  ┌──────────────────────────────────┐     │   QFT†    │
c1: |0⟩─┤ H ├──┼──┤ controlled-SWAP + CNOT (U²)      ├─────┤           ├─M─→ b1
        └───┘  │  └──────────────────────────────────┘     │           │
        ┌───┐  │                                     ┌───┐ │           │
c2: |0⟩─┤ H ├──┼─────────────────────────────────────┤U⁴ ├─┤           ├─M─→ b2
        └───┘  │                                     └───┘ └───────────┘
               │
q0: |0⟩────────X────────(U² gates)───────────────────(U⁴ gates)────────────

q1: |0⟩─────────────────(U² gates)───────────────────(U⁴ gates)────────────
```

## 期待される測定結果

周期 r = 3 に対応する位相:

```
s = 0: φ = 0/3 = 0.000  → 測定値 |000⟩
s = 1: φ = 1/3 = 0.333  → 測定値 |010⟩ or |011⟩
s = 2: φ = 2/3 = 0.667  → 測定値 |101⟩ or |110⟩
```

3量子ビットなので 2^3 = 8 レベルの精度:

| s/r | 理想位相 | 最近接測定値 | 2進数 |
|-----|---------|-------------|-------|
| 0/3 | 0.000 | 0/8 = 0.000 | 000 |
| 1/3 | 0.333 | 3/8 = 0.375 | 011 |
| 2/3 | 0.667 | 5/8 = 0.625 | 101 |

## シミュレーション結果

```python
from quantum_rsa.modexp.n21_optimized import build_full_circuit_n21
from qiskit_aer import AerSimulator

qc = build_full_circuit_n21()
simulator = AerSimulator()
job = simulator.run(qc, shots=4096)
counts = job.result().get_counts()

# 典型的な結果:
# 000: 27.8% (s=0)
# 010: 19.1% (s≈1)
# 110: 19.1% (s≈2)
# 100:  9.3%
# ...
```

## 連分数展開による周期復元

```python
from fractions import Fraction

def get_period(measured_bits, n_count=3, N=21):
    # 測定値を位相に変換
    measured_int = int(measured_bits, 2)
    phase = measured_int / (2 ** n_count)

    # 連分数展開
    frac = Fraction(phase).limit_denominator(N)
    r = frac.denominator

    return r

# 例:
# "010" → phase = 2/8 = 0.25 → 1/4 → r=4 (不正解)
# "011" → phase = 3/8 = 0.375 → 3/8 → limit_denom(21) → 1/3 → r=3 ✓
# "101" → phase = 5/8 = 0.625 → 5/8 → limit_denom(21) → 2/3 → r=3 ✓
```

## 因数の計算

```python
from math import gcd

r = 3  # 周期
a = 4  # 基数
N = 21

# r が奇数の場合、通常は失敗するが...
# a^(r/2) を整数として扱うため、特殊処理が必要

# 論文では別のアプローチを使用:
# a^r - 1 = 4^3 - 1 = 63 = 3 × 21
# gcd(63, 21) = 21 (自明)

# 実際には、測定結果から直接因数を推定:
# 複数回の測定で統計的に因数を特定
```

## このリポジトリでの使用方法

```python
# 最適化回路を直接使用
from quantum_rsa.modexp.n21_optimized import (
    build_full_circuit_n21,
    config_optimized
)

# 完全な回路を構築
qc = build_full_circuit_n21()
print(f"Qubits: {qc.num_qubits}")  # 5
print(f"Depth: {qc.depth()}")      # ~27

# 実行
from qiskit_aer import AerSimulator
simulator = AerSimulator()
job = simulator.run(qc, shots=4096)
counts = job.result().get_counts()
```

## 実機実行時の注意

1. **トランスパイル**: IBM Quantumの基底ゲートに変換
   ```python
   from qiskit import transpile
   qc_hw = transpile(qc, backend, optimization_level=3)
   ```

2. **エラー緩和**: Readout error mitigation を使用
   ```python
   from qiskit_ibm_runtime import Sampler, Options
   options = Options()
   options.resilience_level = 1  # 軽量なエラー緩和
   ```

3. **ショット数**: 実機では1000-4000 shots推奨

## 参考ファイル

- 実装コード: [src/quantum_rsa/modexp/n21_optimized.py](../../../src/quantum_rsa/modexp/n21_optimized.py)
- 論文要約: [papers/shor-n21-skosana-2021.md](../papers/shor-n21-skosana-2021.md)
- Margolus gate: [margolus-gate.md](margolus-gate.md)
