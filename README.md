# Quantum Crypto Lab

量子暗号の**攻撃と防御**の両面を実装・実験するプロジェクトです。

```
┌────────────────────────────────────────────────────────────┐
│  ATTACK (量子アルゴリズム)     │  DEFENSE (耐量子暗号)      │
│  ───────────────────────────  │  ─────────────────────     │
│  Shor's Algorithm             │  ML-KEM (FIPS 203)         │
│  → RSA/ECDSA を破る            │  → 量子耐性のある鍵交換     │
│                               │                            │
│  Grover's Algorithm           │  ML-DSA (FIPS 204)         │
│  → 対称鍵暗号の鍵探索          │  → 量子耐性のある署名       │
└────────────────────────────────────────────────────────────┘
```

## クイックスタート

```bash
# セットアップ
git clone <repo-url>
cd quantum-crypto-lab
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

# Shorアルゴリズム（N=15の因数分解）
python -c "from quantum_crypto import run_shor; print(run_shor(15).factors)"
# Output: (3, 5)

# Groverアルゴリズム（4ビット探索）
python -c "from quantum_crypto.grover import GroverSearch, create_marking_oracle
o = create_marking_oracle(4, [7]); g = GroverSearch(4, o, 1)
print(g.run_simulation(100).measured_state)"
# Output: 7

# PQC鍵交換（ML-KEM-768）
python -c "from quantum_crypto.pqc import ml_kem_full_exchange
r = ml_kem_full_exchange(768); print(f'Secret: {r.shared_secret.hex()[:16]}...')"
```

---

## 実装内容

### 1. Shor's Algorithm（因数分解）

RSA暗号の基盤となる因数分解問題を量子コンピュータで解きます。

| N | 基数a | 周期r | 量子ビット | 2Qゲート | 成功率 | デバイス |
|---|-------|-------|-----------|----------|--------|----------|
| 15 | 7 | 4 | 4 | 6 | ~85% | Simulator |
| 21 | 4 | 3 | 5 | 15 | ~60% | Ankaa-3 |
| 35 | 6 | 2 | 3 | 2 | **96.6%** | Ankaa-3 |
| 35 | 8 | 4 | 5 | 8 | 47.6% | Ankaa-3 |
| 91 | 8 | 4 | 5 | 8 | **75.3%** | Ankaa-3 |
| 143 | 34 | 4 | 5 | 8 | **74.2%** | Ankaa-3 |
| 185 | 6 | 4 | 5 | 8 | **73.3%** | Ankaa-3 |

```python
from quantum_crypto import run_shor

# シミュレータ
result = run_shor(15, backend_type='simulator', shots=1000)
print(result.factors)  # (3, 5)

# AWS Braket実機
result = run_shor(35, backend_type='braket', shots=100)
```

### 2. Grover's Algorithm（鍵探索）

対称鍵暗号への攻撃をデモします。古典的なO(N)探索をO(√N)に高速化。

```python
from quantum_crypto.grover import GroverCipherAttack, toy_encrypt

# 4ビットトイ暗号への攻撃
key = 5
ciphertext = toy_encrypt(plaintext=3, key=key, use_sbox=False)

attack = GroverCipherAttack(plaintext=3, ciphertext=ciphertext, key_bits=4)
result = attack.run_simulation(shots=1000)
print(f"Recovered key: {result.found_key}")  # 5
```

| 鍵長 | 古典探索 | Grover探索 | 高速化 |
|------|---------|-----------|--------|
| 4-bit | 16 | 3 | 5.3x |
| 128-bit | 2^128 | 2^64 | 2^64 x |

### 3. Post-Quantum Cryptography（耐量子暗号）

NIST標準の耐量子暗号アルゴリズムを実装。

```python
from quantum_crypto.pqc import ml_kem_keygen, ml_kem_encapsulate, ml_kem_decapsulate

# ML-KEM鍵交換
pub, sec = ml_kem_keygen(768)
ct, shared_bob = ml_kem_encapsulate(pub, 768)
shared_alice = ml_kem_decapsulate(sec, ct, 768)
assert shared_alice == shared_bob
```

```python
from quantum_crypto.pqc import ml_dsa_keygen, ml_dsa_sign, ml_dsa_verify

# ML-DSAデジタル署名
pub, sec = ml_dsa_keygen(65)
sig = ml_dsa_sign(sec, b"message", 65)
valid = ml_dsa_verify(pub, b"message", sig, 65)  # True
```

| アルゴリズム | 用途 | 公開鍵 | 署名/暗号文 |
|-------------|------|--------|------------|
| ML-KEM-768 | 鍵交換 | 1,184 B | 1,088 B |
| ML-DSA-65 | 署名 | 1,952 B | 3,309 B |

---

## プロジェクト構成

```
quantum-crypto-lab/
├── src/quantum_crypto/
│   ├── algorithms/          # Shorアルゴリズム
│   ├── modexp/              # N別の最適化回路
│   │   ├── n15.py, n21.py, n33.py, n35_*.py
│   ├── grover/              # Groverアルゴリズム
│   │   ├── core.py          # メイン実装
│   │   ├── oracles.py       # オラクル構築
│   │   └── toy_cipher.py    # トイ暗号攻撃
│   ├── pqc/                 # 耐量子暗号
│   │   ├── kem.py           # ML-KEM
│   │   ├── signatures.py    # ML-DSA
│   │   ├── hybrid.py        # ハイブリッド暗号
│   │   └── benchmark.py     # パフォーマンス計測
│   ├── backends/            # 量子バックエンド
│   └── runner.py            # 実行インターフェース
├── scripts/
│   ├── run_shor_experiment.py
│   ├── run_grover_attack.py
│   └── run_pqc_demo.py
├── docs/
│   ├── quantum-crypto-implementation.md  # 詳細解説
│   ├── references/          # 参考資料
│   └── reports/             # 実験レポート
└── tests/
```

---

## セットアップ

```bash
# 基本（Shor + Grover）
pip install -e .

# PQC機能を含む
pip install -e ".[pqc]"

# AWS Braket実機対応
pip install -e ".[hardware]"

# 全機能
pip install -e ".[all]"
```

### liboqs（macOS）

PQC機能にはliboqs共有ライブラリが必要：

```bash
git clone --depth 1 https://github.com/open-quantum-safe/liboqs.git
cd liboqs && mkdir build && cd build
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/opt/homebrew ..
make -j$(sysctl -n hw.ncpu)
cp lib/liboqs*.dylib /opt/homebrew/lib/
```

---

## デモスクリプト

```bash
# Shorアルゴリズム
python scripts/run_shor_experiment.py --n 15 --backend simulator

# Grover攻撃
python scripts/run_grover_attack.py --key-bits 4 --shots 1000

# PQCデモ
python scripts/run_pqc_demo.py --quick
python scripts/run_pqc_demo.py --benchmark
```

---

## ドキュメント

- [実装詳細解説](docs/quantum-crypto-implementation.md) - アルゴリズムの詳細と使用例
- [N=35最適基数分析](docs/n35-optimal-base-analysis.md) - 基数選択の影響分析
- [参考資料](docs/references/) - 論文要約、アルゴリズム解説

---

## 参考文献

### Quantum Algorithms
- Shor (1994), "Algorithms for quantum computation: discrete logarithms and factoring"
- Grover (1996), "A fast quantum mechanical algorithm for database search"
- Skosana & Tame (2021), "Demonstration of Shor's factoring algorithm for N=21"

### Post-Quantum Cryptography
- NIST FIPS 203: ML-KEM (2024)
- NIST FIPS 204: ML-DSA (2024)
- NIST FIPS 205: SLH-DSA (2024)

---

## ライセンス

MIT License
