# Quantum RSA Lab

本プロジェクトは、Shorアルゴリズムを実際の量子コンピュータ上でどこまで実行できるかを、回路複雑性と成功率の関係に着目して実験的に検証することを目的としています。

小さな合成数（N=21, 35）の因数分解を対象に、異なる基数選択・回路構成での比較実験を行いました。

## 主な実験

### N=21 の因数分解
- 5量子ビット、Margolus gate（近似Toffoli）を使った最適化回路
- Skosana & Tame (2021) の論文実装を再現
- Rigetti Ankaa-3 で実行

### N=35 の因数分解
- 基数 a=6 を選択すると周期 r=2 となり、3量子ビット・2Qゲート2個という極めてコンパクトな実装が可能
- 同一デバイス上で a=6 (r=2) と a=8 (r=4) を比較実験し、回路複雑度が成功率に与える影響を定量評価
  - a=6 (3 qubits, 2 2Q-gates): 成功率 96.6%
  - a=8 (5 qubits, 8 2Q-gates): 成功率 47.6%

## 実験結果

成功率は、理論的に期待される位相ビンに対応する測定結果の確率質量として定義しています。

| 実験 | デバイス | 量子ビット | 2Qゲート | 成功率 |
|------|----------|-----------|----------|--------|
| N=21, a=4 | Ankaa-3 | 5 | 15 | ~60% |
| N=35, a=6 | Ankaa-3 | 3 | 2 | 96.6% |
| N=35, a=8 | Ankaa-3 | 5 | 8 | 47.6% |

詳細なレポートは [docs/reports/](docs/reports/) を参照してください。

> **Note**: 実機での結果はデバイスの状態やキャリブレーションに依存します。本実験は絶対的な性能保証ではなく、定性的な傾向の実証を目的としています。

## プロジェクト構成

```
quantum-rsa-lab/
├── src/quantum_rsa/
│   ├── shor_demo.py          # Shor アルゴリズム実装
│   ├── modexp/               # N別の最適化回路
│   │   ├── n15.py            # N=15（シミュレータ用）
│   │   ├── n21_optimized.py  # N=21（実機用）
│   │   ├── n35_a6_optimized.py  # N=35, a=6
│   │   └── n35_a8_optimized.py  # N=35, a=8
│   └── backends/             # バックエンド接続
├── docs/
│   ├── references/           # 参考資料（論文要約、アルゴリズム解説）
│   └── reports/              # 実験レポート
├── scripts/                  # 実行スクリプト
└── tests/                    # テスト
```

## セットアップ

```bash
git clone <repo-url>
cd quantum-rsa-lab

python -m venv .venv
source .venv/bin/activate

pip install -e .
```

## 実行方法

### シミュレータで試す

```python
from quantum_rsa.shor_demo import run_shor

result = run_shor(15, base=7, method="quantum", shots=1024)
print(result.factors)  # (3, 5)
```

### 実機で実行（AWS Braket）

AWS Braket の認証設定が必要です。詳細は [docs/references/hardware/real-hardware-guide.md](docs/references/hardware/real-hardware-guide.md) を参照。

## 学習リソース

Shorアルゴリズムの理論や回路実装について学びたい場合は [docs/references/](docs/references/) を参照してください。論文要約、アルゴリズム解説、Margolus gate の実装詳細などをまとめています。

## 参考文献

- Skosana & Tame, "Demonstration of Shor's factoring algorithm for N=21 on IBM quantum processors", Scientific Reports 11, 16599 (2021)
- Pelofske et al., "Practical Challenges in Executing Shor's Algorithm on Existing Quantum Platforms", arXiv:2512.15330 (2024)
- Shor, "Algorithms for quantum computation: discrete logarithms and factoring", FOCS (1994)

## ライセンス

MIT License
