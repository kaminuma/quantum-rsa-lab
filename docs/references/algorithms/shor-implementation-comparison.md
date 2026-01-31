# Shor's Algorithm Implementation Comparison

## 私たちの実装 vs 他の手法

### 私たちの実装（quantum-rsa-lab）

```
手法: 非反復型 QPE + Full QFT
カウント量子ビット: 8個（固定、2^8=256レベルの位相精度を確保するため）
作業量子ビット: ⌈log₂N⌉ 個（N < 2^n を満たす最小の n）
合計: 8 + n_work 個

例: N=21 → n_work=5, 合計 8 + 5 = 13 量子ビット
    N=77 → n_work=7, 合計 8 + 7 = 15 量子ビット
```

**特徴:**
- Full QFT（逆量子フーリエ変換）を使用
- 全てのカウント量子ビットを同時に使用
- mid-circuit測定不要
- シンプルで教育目的に最適

**回路構造:**
```
|0⟩ ─H─────●─────────────────────────────── QFT† ─ M
|0⟩ ─H─────┼────●────────────────────────── QFT† ─ M
|0⟩ ─H─────┼────┼────●───────────────────── QFT† ─ M
...        │    │    │
|0⟩ ─H─────┼────┼────┼────●──────────────── QFT† ─ M
           │    │    │    │
|1⟩ ───────U────U²───U⁴───U⁸─...── (work register)
|0⟩ ───────┴────┴────┴────┴────── (work register)
...
```

---

## 他の手法との比較

### 1. 反復型 Shor（Semi-classical QFT）

**世界記録**: 549,755,813,701（GPUスパコンシミュレーション、実機ではない）

```
手法: 反復型 QPE + Semi-classical QFT
カウント量子ビット: 1個（再利用）
作業量子ビット: 2n 個
合計: 2n + 1 個
```

**特徴:**
- 1つのカウント量子ビットを繰り返し測定・リセット
- mid-circuit測定が**必須**
- 量子ビット数が約1/3に削減
- 実機では難しい（リアルタイムフィードバック必要）

**回路構造:**
```
反復1: |0⟩ ─H─●─H─ M → classical bit b₁
              │
反復2: |0⟩ ─H─●─P(θ₁)─H─ M → classical bit b₂  (θ₁はb₁に依存)
              │
反復3: |0⟩ ─H─●─P(θ₂)─H─ M → classical bit b₃  (θ₂はb₁,b₂に依存)
...
```

### 2. 実機での純粋Shor（N=21達成）

**記録**: N=21（IBM quantum processors, 2021）

```
手法: コンパイル済みQPE（非反復）
カウント量子ビット: 3個
作業量子ビット: 2個
合計: 5 量子ビット
```

**特徴:**
- 回路を手動で最適化・コンパイル
- 近似Toffoliゲートを使用
- 特定のNに特化した実装

### 3. Compiled/Adiabatic手法（N=143達成）

**記録**: N=143（2012年、4量子ビット）

```
手法: 断熱量子計算 + 事前知識の活用
量子ビット: 4個のみ
```

**特徴:**
- 143の特殊な構造を利用
- 「純粋なShor」とは言えない
- 事前知識（解の構造）を活用

### 4. ハイブリッド Schnorr-QAOA（N=1591達成）

**記録**: N=1591（2025年3月、6量子ビット）

```
手法: 古典的格子縮小 + QAOA
量子ビット: 6個
```

**特徴:**
- Schnorrの格子アルゴリズムと量子最適化を組み合わせ
- 大幅に量子ビット数を削減
- 純粋なShorではない（QPEを使用せず、古典アルゴリズムの一部を量子で加速）

---

## 比較表

※「純粋なShor」の定義：QPEによる周期発見を量子回路で実行し、問題固有の事前知識を使わない実装

| 手法 | 最大N | 量子ビット数 | mid-circuit測定 | 純粋なShor? |
|------|-------|-------------|----------------|-------------|
| 非反復Full QPE | シミュレータ依存 | 8 + n_work | 不要 | ✅ Yes |
| 反復Shor（Semi-classical） | 549B* | 2n+1 | 必要 | ✅ Yes |
| 実機純粋Shor | 21 | 5 | 不要 | ✅ Yes |
| Compiled/Adiabatic | 143 | 4 | - | ❌ No |
| Schnorr-QAOA | 1591 | 6 | - | ❌ No |

*GPUスパコンでのシミュレーション

---

## 私たちの実装の位置づけ

### 強み
1. **純粋なShorアルゴリズム** - QPE + QFTの教科書的実装
2. **mid-circuit測定不要** - 現在の実機で実行可能
3. **コードが読みやすい** - 教育・学習目的に最適
4. **拡張しやすい** - 新しいNを追加するのが簡単

### 弱み
1. **量子ビット数が多い** - 8 + n_work 個必要
2. **スケーラビリティ** - ユニタリ行列方式は2^n_workのメモリ

### 実機実行のためのアプローチ

**アプローチ A: 回路最適化**
- 状態圧縮（N=21方式）で量子ビット削減
- Margolus gate等で2Qゲート数削減

**アプローチ B: 反復型への変更**
- Semi-classical QFTを実装
- mid-circuit測定が必要だが量子ビット削減可能

**アプローチ C: ハイブリッド手法**
- Schnorr-QAOAなどの検討

---

## 結論

非反復型Full QPEは教育目的として明快な実装。実機実行には回路最適化が鍵となる。

---

## References

- [Shor's algorithm - Wikipedia](https://en.wikipedia.org/wiki/Shor's_algorithm)
- [IBM Quantum - Shor's Algorithm](https://quantum.cloud.ibm.com/docs/en/tutorials/shors-algorithm)
- [Demonstration of Shor's for N=21](https://www.nature.com/articles/s41598-021-95973-w)
- [Large-Scale Simulation of Shor's Algorithm](https://arxiv.org/abs/2308.05047)
- [Reducing the Number of Qubits in Quantum Factoring](https://eprint.iacr.org/2024/222.pdf)
