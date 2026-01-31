# Quantum RSA Lab

RSA × 量子アルゴリズム実験のための専用リポジトリです。Shor アルゴリズムの理論整理から小規模実装、AWS Braket での実機検証、成果公開までを一気通貫で扱えるように構成しています。

## 🎯 このリポジトリで何ができるか

- **量子版 Shor アルゴリズム**: N=15, 21, 33 の因数分解を量子位相推定（QPE）+ QFT で実行
- **古典版 Shor アルゴリズム**: 同じロジックを古典的なループで実装（比較用）
- **統合インターフェース**: 1つの関数で量子/古典を切り替えて実行可能
- **拡張可能なアーキテクチャ**: レジストリパターンで複数の N 値に対応
- **ローカルシミュレータ**: Qiskit Aer で量子回路をシミュレート（実機不要）

## リポジトリ構成

```
quantum-rsa-lab/
├ docs/                # ロードマップ、調査メモ、設計資料
│ └ roadmap.md         # フェーズごとの実験計画
├ notebooks/           # Jupyter 実験ノート（Qiskit/Braket）
├ src/
│ └ quantum_rsa/
│   ├ __init__.py
│   ├ shor_demo.py     # Shor 統合実装（量子+古典）
│   └ modexp/          # N 専用の modular exponentiation 実装
│     ├ __init__.py    # レジストリシステム
│     ├ n15.py         # N=15 実装（最適化済み）
│     ├ n21.py         # N=21 実装（ユニタリ行列）
│     └ n33.py         # N=33 実装（ユニタリ行列）
├ pyproject.toml       # 依存管理
├ requirements.txt     # pip 用の依存リスト
└ README.md
```

## セットアップ

### 1. 必要なソフトウェア

- Python 3.10 以上
- pip または Poetry/uv

### 2. インストール手順

```bash
# リポジトリをクローン
git clone <repo-url>
cd quantum-rsa-lab

# 仮想環境を作成
python -m venv .venv
source .venv/bin/activate  
# Windows: .venv\Scripts\activate

# 依存パッケージをインストール
pip install -r requirements.txt

# 量子シミュレータをインストール（量子版を使う場合）
pip install qiskit-aer
```

> **Poetry を使う場合**: `pyproject.toml` を参照して `poetry install` を実行してください。

## 🚀 クイックスタート

### コマンドラインで実行

```bash
# 量子版と古典版の両方を実行（デモ）
python src/quantum_rsa/shor_demo.py
```

**実行結果例:**
```
======================================================================
Shor アルゴリズム - 古典 vs 量子
======================================================================

【古典版（ループで周期発見）】
  ✓ a= 7: N=15 = (3, 5) | 方法: classical_period_r=4
  ✓ a=11: N=15 = (5, 3) | 方法: classical_period_r=2
  ✓ a= 2: N=15 = (3, 5) | 方法: classical_period_r=4

【量子版（QPE + QFT）】
  ✓ a= 7: N=15 = (3, 5) | 位相=0.7500 | quantum_qpe_r=4_phase=0.7500
  ✓ a=11: N=15 = (5, 3) | 位相=0.5000 | quantum_qpe_r=2_phase=0.5000
  ✓ a= 2: N=15 = (3, 5) | 位相=0.5000 | quantum_qpe_r=2_phase=0.5000
```

### Python コードから使う

```python
from src.quantum_rsa.shor_demo import run_shor
from src.quantum_rsa.modexp import list_supported

# 実装済みの N を確認
print(f"実装済み: {list_supported()}")  # [15, 21, 33]

# 量子版で N=15 を因数分解
result = run_shor(number=15, base=7, method="quantum", shots=2048)

print(f"因数: {result.factors}")        # (3, 5)
print(f"周期: {result.period}")         # 4
print(f"位相: {result.measured_phase}") # 0.75
print(f"成功: {result.success}")        # True

# N=21 も実行可能
result_21 = run_shor(number=21, base=2, method="quantum", shots=2048)
print(f"N=21: {result_21.factors}")     # (3, 7)

# 古典版で比較
result_classical = run_shor(number=15, base=7, method="classical")
print(f"古典版の因数: {result_classical.factors}")  # (3, 5)
```

## 📚 Shor アルゴリズムの仕組み

### アルゴリズムの流れ

Shor アルゴリズムは、**因数分解を周期発見問題に帰着**させることで、量子コンピュータの並列性を活用します。

```
入力: 合成数 N（例: 15）
出力: N の因数（例: 3 と 5）

1. ランダムに a を選択（1 < a < N）
   例: a = 7

2. gcd(a, N) を計算
   → 1 でなければ因数発見（終了）

3. 【量子パート】周期 r を発見
   f(x) = a^x mod N の周期 r を求める
   → 量子位相推定（QPE）を使用

   古典版: ループで a^1, a^2, a^3... を計算
   量子版: 重ね合わせで全パターンを並列評価

4. 周期 r から因数を計算
   p = gcd(a^(r/2) - 1, N)
   q = gcd(a^(r/2) + 1, N)

   例: 7^(4/2) = 49 ≡ 4 (mod 15)
       gcd(4-1, 15) = gcd(3, 15) = 3
       gcd(4+1, 15) = gcd(5, 15) = 5
       → N = 3 × 5
```

### 量子版の核心：量子位相推定（QPE）

```
量子回路の構成（N=15 の場合）:

- 8量子ビット: カウントレジスタ（位相情報を記録）
- 4量子ビット: 作業レジスタ（15 < 2^4 なので）

手順:
1. カウントビットを |+⟩ 状態に初期化（Hadamard ゲート）
2. 作業ビットを |1⟩ に初期化
3. 制御付きユニタリ演算 U^(2^j) を適用
   U|x⟩ = |ax mod 15⟩ を SWAP/CNOT で実装
4. 逆量子フーリエ変換（QFT†）で位相を抽出
5. 測定結果から連分数展開で周期 r を復元
```

### 実装の詳細

#### 制御付き mod-exp 回路 (`c_amod15`)

N=15 に対する `a^(2^j) mod 15` を量子ゲートで実装。

**例: a=7 の場合**
- 7^1 = 7 mod 15
- 7^2 = 49 = 4 mod 15
- 7^4 = 2401 = 1 mod 15 → **周期 r=4**

これを SWAP ゲートと CNOT ゲートの組み合わせで表現：
```python
U.swap(2, 3)
U.swap(1, 3)
U.swap(2, 3)
U.cx(3, 2)  # 制御NOT
U.cx(3, 1)
U.cx(3, 0)
```

#### 位相から周期への変換

測定結果（例: `01000000` = 64）から位相を計算：
```python
phase = 64 / 256 = 0.25
```

連分数展開で周期を推定：
```python
frac = Fraction(0.25).limit_denominator(15)
# → 1/4
r = frac.denominator  # → 4
```

## 📁 プロジェクト構成とロードマップ

このプロジェクトは、Shor アルゴリズムによる RSA 因数分解を、理論から実装・実機検証まで段階的に進めることを目的としています。

詳細な実験計画は [`docs/roadmap.md`](docs/roadmap.md) を参照してください。

### 実験フェーズ

1. **Phase 1**: 理論と基礎実装（N=15 での動作検証）✓ 完了
2. **Phase 2**: スケールアップと最適化（N=21, 33, 35 への拡張）← 現在ここ
3. **Phase 3**: 実機検証（AWS Braket での実行）
4. **Phase 4**: 回路最適化（ゲート削減、ハイブリッド手法）
5. **Phase 5**: 発展的トピック（VQF, QAOA, PQC との比較）

### 実装状況

| N | 因数分解 | 量子ビット | 実装方式 | 状態 |
|---|---------|-----------|---------|------|
| 15 | 3 × 5 | 12 (8+4) | 最適化SWAP/CNOT | ✓ 完了 |
| 21 | 3 × 7 | 13 (8+5) | ユニタリ行列 | ✓ 完了 |
| 33 | 3 × 11 | 14 (8+6) | ユニタリ行列 | ✓ 完了 |
| 35+ | ... | ... | - | 今後追加予定 |

**N を大きくするには？**

現在の実装（ユニタリ行列方式）は簡単に拡張できますが、N が大きくなると課題も見えてきます：

- N=51 (6 qubits): 64×64 行列 → まだ軽い
- N=85 (7 qubits): 128×128 行列 → 実行可能
- N=255 (8 qubits): 256×256 行列 → このあたりから重くなりそう

より大きな N を扱うには、N=15 のような最適化されたゲート実装や、別のアプローチ（modular addition/multiplication を組み合わせる方法など）が必要になってくるかもしれません。Phase 2 で実験予定です。

## 🔬 技術スタック

- **量子フレームワーク**: Qiskit 1.2.4
- **シミュレータ**: Qiskit Aer 0.17+
- **可視化**: Matplotlib 3.9+
- **データ処理**: Pandas 2.2+, NumPy 1.26+
- **開発環境**: JupyterLab 4.2+
- **（今後）実機実行**: AWS Braket SDK

## 📖 参考文献・関連研究

- **Shor's Algorithm (1994)**: 因数分解の量子アルゴリズムの原論文
- **Gidney & Ekerå (2021)**: RSA-2048 分解に必要なリソース見積もり
- **Yan et al. (2022)**: 48bit 合成数の実機分解実験
- **Nielsen & Chuang**: "Quantum Computation and Quantum Information" 教科書

## 🤝 コントリビューション

現在は個人実験プロジェクトですが、以下のような貢献を歓迎します：

- N=21, 33 などへの拡張実装
- 回路最適化のアイデア
- AWS Braket 実機での実行結果
- 可視化・ドキュメントの改善

## ライセンス

MIT License（予定）
