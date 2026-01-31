# 実機実行ガイド

> **注**: 本リポジトリのN=35実験ではAWS Braket（Rigetti Ankaa-3）のみを使用しているため、本ガイドではAWSの導入手順のみ記載しています。

> **注意**: 本ガイドのデバイス名・価格・無料枠条件は執筆時点（2026年1月）のものです。最新情報は各クラウドベンダーの公式ページを参照してください。

> **免責**: 本リポジトリのShor実装は教育・実証目的のcompiled circuitを含み、大規模Nに対する汎用RSA破りを目的とするものではありません。

## 概要

AWS Braketを使用して、Shorのアルゴリズムを実際の量子ハードウェアで実行する方法。

## AWS Braket

### セットアップ

```bash
# AWS Braket SDKをインストール
pip install amazon-braket-sdk

# またはプロジェクトと一緒にインストール
pip install -e ".[aws]"
```

### AWS認証情報の設定

```bash
# 方法A: AWS CLI
aws configure

# 方法B: 環境変数
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

### 利用可能なデバイス

| デバイス | タイプ | 量子ビット | 忠実度 | コスト |
|---------|--------|-----------|--------|--------|
| **IonQ Aria** | イオントラップ | 25 | 99.4% 2Q | $0.03/shot + $0.01/gate |
| **IonQ Forte** | イオントラップ | 32 | 高 | $0.03/shot + $0.01/gate |
| **Rigetti Ankaa-3** | 超伝導 | 84 | 99.5% 2Q | $0.00035/shot + $0.00035/gate |
| **IQM Garnet** | 超伝導 | 20 | 良好 | $0.00145/shot |

### 使用方法

```python
from quantum_rsa.backends import AWSBraketBackend
from quantum_rsa.algorithms import QuantumShor
from quantum_rsa.modexp import get_config

# バックエンドを初期化
backend = AWSBraketBackend(device="ionq_aria")

# 回路を構築
shor = QuantumShor()
config = get_config(15)
circuit = shor._construct_circuit(7, 15, config)

# 実機で実行
counts = backend.run(circuit, shots=100)
print(counts)
```

### 推奨アプローチ

```python
# 1. まずAWSシミュレータでテスト（安価）
backend_sim = AWSBraketBackend(device="sv1")
counts = backend_sim.run(circuit, shots=1000)

# 2. その後、少ないショット数で実機実行
backend_real = AWSBraketBackend(device="ionq_aria")
counts = backend_real.run(circuit, shots=100)
```

---

## 重要な注意点

### 1. 回路最適化

ユニタリ行列実装はトランスパイル後に多くのゲートを生成する傾向がある:

```python
from qiskit import transpile

# トランスパイル前
print(f"Original: {circuit.depth()} depth")

# 実機用にトランスパイル後
qc_transpiled = transpile(circuit, backend, optimization_level=3)
print(f"Transpiled: {qc_transpiled.depth()} depth")
```

### 2. N=15から始める

実機はノイズが多い。最小のケースから始めるのが良い:

```python
# 推奨: 小さい数から始める
result = run_shor(15, method='quantum', shots=1000)

# 大きなNには最適化回路が必要な場合がある
result = run_shor(77, method='quantum', shots=1000)
```

### 3. ノイズを想定する

実機の結果には通常ノイズが含まれる:

```
シミュレータ: {'0000': 256, '0100': 256, '1000': 256, '1100': 256}
実機:        {'0000': 180, '0100': 210, '1000': 195, '1100': 220, '0001': 45, ...}
```

エラー緩和の使用:
```python
# 今後対応予定: エラー緩和サポート
```

### 4. コストに注意

**AWS Braket:**
- IonQ: 1回あたり約$3-10（100 shots）
- Rigetti: 1回あたり約$0.05（100 shots）

---

## クイックスタートスクリプト

```python
#!/usr/bin/env python3
"""実機でShorのアルゴリズムを実行"""

from quantum_rsa.runner import run_shor

# AWS Braket
# pip install amazon-braket-sdk
# aws configure

from quantum_rsa.backends import AWSBraketBackend
backend = AWSBraketBackend(device="sv1")  # まずシミュレータで

# N=15で実行
from quantum_rsa.algorithms import QuantumShor
from quantum_rsa.modexp import get_config

shor = QuantumShor()
config = get_config(15)
circuit = shor._construct_circuit(7, 15, config)

counts = backend.run(circuit, shots=1000)
print("結果:", counts)
```

---

## トラブルシューティング

### "Custom unitary gates must be decomposed"

modexpはユニタリ行列を使用。先にトランスパイルが必要:

```python
from qiskit import transpile

# 基本ゲートに分解
basis_gates = ['h', 'cx', 'rz', 'rx', 'ry', 'x', 'y', 'z']
circuit_decomposed = transpile(circuit, basis_gates=basis_gates, optimization_level=3)
```

### "Results are too noisy"

現在のNISQデバイスはエラー率が高い。対処法:
- ショット数を増やす（平均化を増やす）
- エラー緩和技術を使用
- 回路を簡素化（深度を減らす）
