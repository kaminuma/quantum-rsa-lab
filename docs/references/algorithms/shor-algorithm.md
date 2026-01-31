---
title: Shor's Algorithm Complete Reference
source: Multiple academic sources
last_verified: 2026-01-30
---

# Shorのアルゴリズム完全リファレンス

## 概要

Shorのアルゴリズムは、整数Nを多項式時間で因数分解する量子アルゴリズム。
古典計算機では指数時間かかる問題を、量子コンピュータでは効率的に解ける。

**計算量**:
- 古典（最良）: O(exp(n^(1/3)))  ※ 一般数体篩法
- 量子（Shor）: O(n³)  ※ n = log₂(N)、modular exponentiation の実装に依存

## アルゴリズムの流れ

```
入力: 合成数 N（因数分解したい数）

1. N が偶数なら → 2 が因数

2. N = a^b の形か確認 → a が因数

3. ランダムに a を選ぶ (1 < a < N)

4. g = gcd(a, N) を計算
   → g > 1 なら g が因数（終了）

5. 【量子パート】f(x) = a^x mod N の周期 r を見つける
   → 量子位相推定（QPE）を使用

6. r が奇数、または a^(r/2) ≡ -1 (mod N) なら
   → ステップ3に戻る

7. 因数を計算:
   p = gcd(a^(r/2) - 1, N)
   q = gcd(a^(r/2) + 1, N)

8. p, q が非自明な因数なら成功
```

## 成功確率

ランダムなaを選んだとき、有用な周期rが得られる確率は **少なくとも 1/2**。

理由:
- r が偶数である確率: 高い
- a^(r/2) ≢ -1 (mod N) である確率: 高い
- 数回試行すれば、ほぼ確実に因数が見つかる

※ この確率はNが2つの異なる奇素数の積である場合の理論値。Nの素因数構成によって変動する。

## 量子パート：周期発見

### なぜ量子が必要か

関数 f(x) = a^x mod N の周期 r を見つけるには:

| 手法 | 計算量 |
|------|--------|
| 古典（全探索） | O(r) ≈ O(N) |
| 古典（Baby-step giant-step） | O(√r) |
| **量子（Shor）** | **O(n³)** ※ n = log₂N |

※ QPE自体は O(n) 回の制御ユニタリ演算。ボトルネックは制御modular exponentiation。

### 量子回路の構成

```
カウントレジスタ (n qubits)     作業レジスタ (m qubits)
     |0⟩ ─H─────●─────────────────────── QFT† ─ M
     |0⟩ ─H─────┼────●────────────────── QFT† ─ M
     |0⟩ ─H─────┼────┼────●───────────── QFT† ─ M
     ...        │    │    │
     |0⟩ ─H─────┼────┼────┼────●──────── QFT† ─ M
                │    │    │    │
     |1⟩ ───────U────U²───U⁴───U⁸─...────────────
     |0⟩ ───────┴────┴────┴────┴─────────────────
     ...
```

**各部分の役割**:
1. **H ゲート**: 全ての x の重ね合わせ |0⟩ + |1⟩ + ... + |2^n-1⟩
2. **制御U**: |x⟩|y⟩ → |x⟩|a^x · y mod N⟩
3. **逆QFT**: 位相を測定可能な形に変換

### 量子ビット数

| レジスタ | 量子ビット数 | 説明 |
|---------|-------------|------|
| カウント | 2n ～ 2n+1 | 位相の精度を決める |
| 作業 | n | N < 2^n を満たす |
| **合計** | **3n ～ 3n+1** | |

**例**: N=21 (n=5ビット) → 標準QPE: 2n+n = 15量子ビット
**最適化後**: 5量子ビット（状態圧縮 + カウントビット削減）

※ 5量子ビット版はN=21に特化したcompiled回路であり、汎用的なShor実装ではない。

## 数学的背景

### 周期と位相の関係

量子状態は周期 r に関連した位相を持つ:

```
|ψ⟩ = (1/√r) Σ_{s=0}^{r-1} e^{2πis/r} |u_s⟩
```

測定すると、s/r に近い値が高確率で得られる。

### 連分数展開

測定値 φ ≈ s/r から r を復元:

```python
from fractions import Fraction

# 測定値 φ = 0.333... ≈ 1/3
phase = measured_value / 2**n_qubits
frac = Fraction(phase).limit_denominator(N)
r = frac.denominator  # → 3
```

## 実装パターン比較

### パターン1: フルQPE（非反復型）

```
量子ビット: 2n + n = 3n
回路深度: 深い
mid-circuit測定: 不要
実機実行: 可能（小さいN）
```

### パターン2: 反復型（Semi-classical QFT）

```
量子ビット: 1 + n = n+1
回路深度: 浅い × n回
mid-circuit測定: 必要
実機実行: 難しい（フィードバック必要）
```

### パターン3: 最適化（状態圧縮）

```
量子ビット: 3-5（Nに依存）
回路深度: 最小
条件: 特定のNに特化
実機実行: 最も現実的
```

## 具体例: N=15

```
N = 15 = 3 × 5
a = 7 を選ぶ

周期を計算:
  7^1 mod 15 = 7
  7^2 mod 15 = 49 mod 15 = 4
  7^3 mod 15 = 28 mod 15 = 13
  7^4 mod 15 = 91 mod 15 = 1  ← 周期 r = 4

因数を計算:
  gcd(7^2 - 1, 15) = gcd(48, 15) = 3 ✓
  gcd(7^2 + 1, 15) = gcd(50, 15) = 5 ✓
```

## 具体例: N=21

```
N = 21 = 3 × 7
a = 4 を選ぶ

周期を計算:
  4^1 mod 21 = 4
  4^2 mod 21 = 16
  4^3 mod 21 = 64 mod 21 = 1  ← 周期 r = 3

r=3 は奇数なので、通常は失敗...
しかし、論文では特殊な手法で成功
（詳細は papers/shor-n21-skosana-2021.md 参照）
```

## 世界記録

| カテゴリ | 記録 | 年 | 手法 |
|----------|------|-----|------|
| 実機・純粋Shor | N=21 | 2021 | 5量子ビット最適化 |
| 実機・反復Shor | N=35 | 2021 | 反復QPE |
| 実機・ハイブリッド | N=1591 | 2025 | Schnorr-QAOA |
| シミュレータ | N=549B | 2023 | GPUスパコン |

## このリポジトリの実装

```python
from quantum_rsa.runner import run_shor

# 古典版（周期をループで計算）
result = run_shor(15, method='classical')

# 量子版（QPE使用）
result = run_shor(15, method='quantum', shots=1024)

print(result.factors)  # (3, 5)
print(result.period)   # 4
```

## 参考資料

- [Shor's original paper (1994)](https://arxiv.org/abs/quant-ph/9508027)
- [IBM Quantum Tutorial](https://quantum.cloud.ibm.com/docs/en/tutorials/shors-algorithm)
- [Wikipedia](https://en.wikipedia.org/wiki/Shor's_algorithm)
