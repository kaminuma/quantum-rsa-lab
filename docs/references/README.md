# 参考資料ライブラリ

このディレクトリには、Shorのアルゴリズム実装に必要な参考資料をまとめています。
ネットで調べなくても、ここを参照すれば実装できるようになっています。

> **注記**: 本リポジトリのShor実装は教育・実証目的のcompiled circuitを含み、大規模Nに対する汎用RSA破りを目的とするものではありません。

## 資料索引

### 論文要約 (`papers/`)

| ファイル | 内容 | 重要度 |
|---------|------|--------|
| [shor-n21-skosana-2021.md](papers/shor-n21-skosana-2021.md) | N=21実機実装論文（5量子ビット、Margolus gate） | ★★★ |
| [shor-world-record.md](papers/shor-world-record.md) | Shor因数分解の世界記録と研究動向 | ★★ |

### アルゴリズム解説 (`algorithms/`)

| ファイル | 内容 | 重要度 |
|---------|------|--------|
| [shor-algorithm.md](algorithms/shor-algorithm.md) | Shorアルゴリズム全体フロー（計算量・成功確率の注記付き） | ★★★ |
| [shor-implementation-comparison.md](algorithms/shor-implementation-comparison.md) | 実装手法の比較（純粋Shorの定義明記） | ★★ |

### 回路実装パターン (`circuits/`)

| ファイル | 内容 | 重要度 |
|---------|------|--------|
| [margolus-gate.md](circuits/margolus-gate.md) | 相対位相Toffoli（CCX + diagonal phase、使用条件付き） | ★★★ |
| [n21-optimized-circuit.md](circuits/n21-optimized-circuit.md) | N=21最適化回路の完全詳細（状態圧縮、U⁴≡U¹の説明付き） | ★★★ |

### ハードウェア情報 (`hardware/`)

| ファイル | 内容 | 重要度 |
|---------|------|--------|
| [real-hardware-guide.md](hardware/real-hardware-guide.md) | 実機実行ガイド（AWS Braket / IBM Quantum、価格注記付き） | ★★★ |

---

## クイックリファレンス

### Shorアルゴリズムの流れ

```
1. N を因数分解したい
2. ランダムに a を選ぶ (1 < a < N)
3. gcd(a, N) > 1 なら終了（因数発見）
4. 【量子】 f(x) = a^x mod N の周期 r を見つける
5. r が偶数なら: p = gcd(a^(r/2) - 1, N)
6. p と N/p が因数
```

※ 成功確率1/2はNが2つの異なる奇素数の積である場合の理論値

### 実機実装の要点

| N | 量子ビット | 回路深度 | CXゲート | 実装方式 |
|---|-----------|---------|----------|---------|
| 15 | 4-5 | ~20 | ~10 | 状態圧縮 |
| 21 | 5 | 35 | 25 | Margolus gate（compiled） |
| 35 (a=6) | 3 | ~10 | 2 | r=2最小構成（compiled） |
| 35 (a=8) | 5 | ~30 | 8 | Margolus gate（compiled） |

※ 上記はすべてNに特化したcompiled回路であり、汎用的なShor実装ではない

### 重要な最適化テクニック

1. **Margolus gate**: CCXを3 CXで実装（通常6 CX）
   - 本質は「CCX + diagonal phase」
   - 位相が乗る部分空間が到達不能な場合に使用可能

2. **状態圧縮**: 作業レジスタを圧縮（log₂(N) → 2-3量子ビット）
   - a^x mod N で実際に現れる状態のみをエンコード

3. **Semi-classical QFT**: mid-circuit測定で量子ビット再利用
   - 実機ではリアルタイムフィードバックが必要

---

## 使い方

```python
# 例: N=21の最適化実装を参照したい場合
# 1. circuits/n21-optimized-circuit.md を読む
# 2. papers/shor-n21-skosana-2021.md で理論背景を確認
# 3. src/quantum_rsa/modexp/n21_optimized.py のコードを参照
```

## 更新履歴

- 2026-01-31: 各ドキュメントの注記・免責・数学的説明を強化
- 2026-01-30: 初期作成
