# Shorアルゴリズム世界記録

本ドキュメントは、Shorアルゴリズムおよび関連因数分解研究の「実装上の到達点と研究動向」を俯瞰するための整理資料である。

## 現在の世界記録（2025-2026年時点）

| カテゴリ | 記録 | 年 | 手法 | 出典 |
|---------|------|-----|------|------|
| **純粋Shor実機** | **21** (3×7) | 2021 | QPE + QFT, 5量子ビット | [Skosana & Tame](https://www.nature.com/articles/s41598-021-95973-w) |
| **反復型Shor** | **35** (5×7) | - | Iterative QPE (要出典確認) | |
| **ハイブリッド (Schnorr-QAOA)** | **1591** (37×43) | 2025年3月 | 6量子ビット, イオントラップ | [arXiv:2503.10588](https://arxiv.org/abs/2503.10588) |
| **断熱/NMR** | 143 (11×13) | 2012 | NMR量子プロセッサ | [Phys. Rev. Lett.](https://link.aps.org/doi/10.1103/PhysRevLett.108.130501) |
| **シミュレーション (GPU)** | 549,755,813,701 | - | 実機ではない | |

## 主要な研究進展（2024-2025年）

### 1. Regevのアルゴリズム（2023-2024）
- **内容**: 数のサイズと量子操作の関係を改善する新しい変種
- **影響**: 30年ぶりのShorの計算量改善
- **出典**: [Quanta Magazine](https://www.quantamagazine.org/thirty-years-later-a-speed-boost-for-quantum-factoring-20231017/)

### 2. 量子ビット削減（Chevignard et al., 2024）
- **内容**: nビットRSAを〜n/2論理量子ビットで因数分解
- **影響**: RSA-2048に必要な量子ビットが〜1,730個に（従来の4,099個から）
- **トレードオフ**: 実行時間が長くなる

### 3. Gidneyの最適化（2025）
- **内容**: RSA-2048に〜1,000-1,400論理量子ビット
- **実行時間**: 現代のエラー訂正で約1週間

### 4. 楕円曲線最適化（2025）
- Tカウント97%削減
- T深度60%削減
- 量子ビット使用量16%削減

### 5. ハイブリッドSchnorr-QAOA（2025）
- ロシア量子センターによるデモンストレーション
- 6量子ビットのみで1591を因数分解
- 古典的格子縮小と量子最適化を組み合わせ

## ハードウェアの進歩

| 企業 | プロセッサ | 量子ビット | 年 |
|------|-----------|-----------|-----|
| IBM | Condor | 1,121 物理 | 2023 |
| IBM | Flamingo | 462（量子リンク付き） | 2024 |
| Google | Willow | 105（エラー閾値以下） | 2024 |
| Quantinuum | - | 50 エンタングル論理 | 2024-2025 |
| Microsoft | - | 24 エンタングル論理 | 2024-2025 |

## 今後の研究方針

### 方針A: 純粋Shor (QPE) の拡張
- **目標**: N=35, 51, 55, 77
- **アプローチ**: モジュラー指数回路の最適化
- **課題**: 回路深度が指数的に増加

### 方針B: ハイブリッドSchnorr-QAOAの実装
- **目標**: 1591記録を超える
- **アプローチ**: Schnorrの格子縮小とQAOAを組み合わせ
- **利点**: 必要な量子ビット数が大幅に削減

### 方針C: Regevのアルゴリズムの実装
- **目標**: 量子操作数の削減
- **アプローチ**: フィボナッチベースの指数計算
- **利点**: 大きなNに対するスケーリングが改善

### 方針D: 回路最適化
- **目標**: 実機での実行（IBM Quantum, AWS Braket）
- **アプローチ**:
  - TカウントとT深度の削減
  - エラー緩和の実装
  - トランスパイラ最適化の使用

## 追跡すべき指標

1. **回路指標**
   - 総ゲート数
   - 2量子ビットゲート数
   - 回路深度
   - Tカウント / T深度

2. **成功指標**
   - 因数分解成功率
   - 正解に必要なショット数
   - ノイズ耐性

3. **リソース指標**
   - 量子ビット数
   - 古典前処理時間
   - メモリ使用量

## 参考文献

- [Shor's Algorithm - Wikipedia](https://en.wikipedia.org/wiki/Shor's_algorithm)
- [The State of Factoring on Quantum Computers (arXiv:2410.14397)](https://arxiv.org/html/2410.14397v1)
- [Quanta Magazine - Speed Boost for Quantum Factoring](https://www.quantamagazine.org/thirty-years-later-a-speed-boost-for-quantum-factoring-20231017/)
- [Demonstration of Shor's for N=21 on IBM](https://www.nature.com/articles/s41598-021-95973-w)
- [Schnorr-QAOA Factorization of 1591 (arXiv:2503.10588)](https://arxiv.org/abs/2503.10588)
- [Quantum Factorization of 143 (Phys. Rev. Lett.)](https://link.aps.org/doi/10.1103/PhysRevLett.108.130501)
