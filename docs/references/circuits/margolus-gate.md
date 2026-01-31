---
title: Margolus Gate (Approximate Toffoli) Reference
source: Maslov (2016), Skosana & Tame (2021)
last_verified: 2026-01-30
---

# Margolus Gate（近似Toffoliゲート）

## 概要

Margolus gateは、標準Toffoliゲートの**CXゲート数を半減**させる近似実装。
特定の条件下で、標準Toffoliと同等の機能を提供する。

| ゲート | CXゲート数 | 1量子ビットゲート |
|--------|-----------|------------------|
| 標準Toffoli | 6 | 多数 |
| **Margolus** | **3** | 4 (RY) |

## 動作の違い

### 標準Toffoli (CCX)

```
|a⟩|b⟩|c⟩ → |a⟩|b⟩|c ⊕ (a·b)⟩

真理値表:
|000⟩ → |000⟩
|001⟩ → |001⟩
|010⟩ → |010⟩
|011⟩ → |011⟩
|100⟩ → |100⟩
|101⟩ → |101⟩
|110⟩ → |111⟩  ← 反転
|111⟩ → |110⟩  ← 反転
```

### Margolus Gate（相対位相Toffoli）

Margolus gate（相対位相Toffoli）は、CCX（Toffoli）と同じ古典的な写像（ビット反転）を実現するが、いくつかの計算基底状態に相対位相が付与される点が異なる。位相が付与される基底状態は分解（実装）に依存し得るため、回路全体として位相が観測結果に影響しないことを確認した上で利用する。

```
CCXと同じビット反転 + 一部の計算基底状態に相対位相

真理値表（本実装の場合）:
|000⟩ → |000⟩
|001⟩ → |001⟩
|010⟩ → |010⟩
|011⟩ → |011⟩
|100⟩ → |100⟩
|101⟩ → -|101⟩  ← 位相 -1（実装依存）
|110⟩ → |111⟩
|111⟩ → |110⟩
```

※ 本質は「CCX と対角位相（diagonal phase）の積」である。位相が乗る基底は実装バリアントにより異なる場合がある。

## 使用条件

Margolus gateは以下の場合に標準Toffoliの代わりに使える:

1. **位相が乗る部分空間が到達不能な場合**
   - アルゴリズム中で位相が付与される計算基底状態に到達しない
   - N=21最適化回路では作業レジスタが{|00⟩, |01⟩, |10⟩}の
     計算基底状態のみを取るため、|101⟩の-1位相は問題にならない
   - **注意**: 制御ビットが重ね合わせ状態の場合は慎重な解析が必要

2. **compute–uncompute でペアになっている場合**
   - 同じ相対位相Toffoliが前後に現れて位相が相殺される
   - 典型例：可逆計算の補助レジスタ計算 → 後で巻き戻す

3. **ゲートが測定直前に適用される場合**
   - 測定は計算基底で行われるため、相対位相は結果に影響しない

4. **位相が相殺される回路構成**
   - 複数のMargolus gateで位相が打ち消し合う設計

## 回路実装

### 標準Toffoli（6 CX版）

```
c1 ──●────────────────●────────●───────●──
     │                │        │       │
c2 ──┼────●───────────┼────●───┼───────┼──
     │    │           │    │   │       │
 t ──┼────┼──H──T†─┬──X──T─┬───X──T†─┬──X──T──H──
          │        │       │         │
          └────────X───────X─────────X
```

### Margolus Gate（3 CX版）

```python
def margolus_gate(qc: QuantumCircuit, c1: int, c2: int, t: int):
    """
    Margolus gate: Toffoliの近似実装
    CXゲート数: 3（標準の半分）

    Parameters:
        qc: 量子回路
        c1: 制御ビット1
        c2: 制御ビット2
        t: ターゲットビット
    """
    qc.ry(np.pi/4, t)      # RY(π/4)
    qc.cx(c1, t)           # CX
    qc.ry(np.pi/4, t)      # RY(π/4)
    qc.cx(c2, t)           # CX
    qc.ry(-np.pi/4, t)     # RY(-π/4)
    qc.cx(c1, t)           # CX
    qc.ry(-np.pi/4, t)     # RY(-π/4)
```

### 回路図

```
c1 ──────●─────────●──────
         │         │
c2 ──────┼────●────┼──────
         │    │    │
 t ─RY─┬─X─RY─X─RY─X─RY───
   π/4     π/4  -π/4 -π/4
```

## Qiskitでの実装

```python
import numpy as np
from qiskit import QuantumCircuit

def margolus_gate(qc: QuantumCircuit, c1: int, c2: int, t: int):
    """Margolus gate (relative-phase Toffoli)
    Note: CCX up to a diagonal phase, not exact CCX on all superposition states.
    """
    qc.ry(np.pi/4, t)
    qc.cx(c1, t)
    qc.ry(np.pi/4, t)
    qc.cx(c2, t)
    qc.ry(-np.pi/4, t)
    qc.cx(c1, t)
    qc.ry(-np.pi/4, t)

# 使用例
qc = QuantumCircuit(3)
margolus_gate(qc, 0, 1, 2)  # q0, q1 が制御、q2 がターゲット
```

## 検証コード

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
import numpy as np

# 標準Toffoliの行列
toffoli = QuantumCircuit(3)
toffoli.ccx(0, 1, 2)
toffoli_matrix = Operator(toffoli).data

# Margolus gateの行列
margolus = QuantumCircuit(3)
margolus_gate(margolus, 0, 1, 2)
margolus_matrix = Operator(margolus).data

# CCX† * Margolus が対角行列になることを確認（相対位相Toffoliの本質）
D = toffoli_matrix.conj().T @ margolus_matrix
print("CCX† * Margolus の対角成分:")
print(np.diag(D))
# → 対角行列であれば「CCX と diagonal phase の積」であることを確認できる
# → 位相が乗る基底を特定可能
```

## N=21での使用例

```python
def c_U4(qc: QuantumCircuit, control: int, q0: int, q1: int):
    """U^4: 4^4 mod 21 の実装"""
    # Margolus gate を使用（標準Toffoliの代わり）
    margolus_gate(qc, control, q1, q0)
    # Fredkin (controlled-SWAP)
    controlled_swap_margolus(qc, control, q0, q1)

def controlled_swap_margolus(qc, c, t1, t2):
    """Margolus版 Controlled-SWAP"""
    qc.cx(t2, t1)
    margolus_gate(qc, c, t1, t2)
    qc.cx(t2, t1)
```

## パフォーマンス比較

N=21のShor回路での比較:

| 実装 | CXゲート数 | 回路深度 | 実機実行 |
|------|-----------|---------|---------|
| 標準Toffoli | ~45 | ~60 | 困難 |
| **Margolus** | **25** | **35** | **可能** |

※ CX数・深さは一例（特定の分解と最適化設定に依存）。実機のカップリング制約により変動する。

## 注意事項

1. **位相の影響を確認**: 位相が付与される計算基底状態にアルゴリズムが到達しないことを確認
2. **連続使用時の位相蓄積**: 複数回使用時は位相が累積する可能性（compute–uncompute で相殺するか確認）
3. **実装依存性**: 位相が乗る基底は分解バリアントにより異なる場合がある
4. **エラー削減**: CXゲートが減るため、ノイズ耐性が向上

## 参考文献

- Maslov, D. "Advantages of using relative-phase Toffoli gates" (2016)
- Skosana & Tame, Scientific Reports 11, 16599 (2021)
- [このリポジトリの実装](../../../src/quantum_rsa/modexp/n21_optimized.py)
