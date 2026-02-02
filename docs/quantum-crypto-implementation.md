# Quantum Crypto Lab - 実装解説

本ドキュメントでは、このリポジトリで実装した量子暗号関連のアルゴリズムと実験について解説します。

## 概要

このリポジトリは**量子暗号の攻撃と防御の両面**を実装しています：

| カテゴリ | アルゴリズム | 対象 | 目的 |
|---------|-------------|------|------|
| **攻撃** | Shor's Algorithm | RSA/楕円曲線暗号 | 公開鍵暗号を破る |
| **攻撃** | Grover's Algorithm | 対称鍵暗号 | 鍵探索を高速化 |
| **防御** | ML-KEM (FIPS 203) | 鍵交換 | 量子耐性のある鍵カプセル化 |
| **防御** | ML-DSA (FIPS 204) | デジタル署名 | 量子耐性のある署名 |

```
┌─────────────────────────────────────────────────────────────┐
│                    量子暗号の全体像                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ATTACK (量子コンピュータによる攻撃)                         │
│   ┌─────────────────┐    ┌─────────────────┐               │
│   │  Shor's Algo    │    │  Grover's Algo  │               │
│   │  ─────────────  │    │  ─────────────  │               │
│   │  RSA/ECDSA破り   │    │  AES鍵探索      │               │
│   │  指数的高速化    │    │  2次高速化      │               │
│   └─────────────────┘    └─────────────────┘               │
│          ↓ 脅威                   ↓ 脅威                    │
│   ┌─────────────────┐    ┌─────────────────┐               │
│   │  公開鍵暗号      │    │  対称鍵暗号      │               │
│   │  (RSA, ECDSA)   │    │  (AES-128)      │               │
│   └─────────────────┘    └─────────────────┘               │
│          ↓ 対策                   ↓ 対策                    │
│   ┌─────────────────┐    ┌─────────────────┐               │
│   │  ML-KEM/ML-DSA  │    │  AES-256        │               │
│   │  (格子暗号)      │    │  (鍵長2倍)      │               │
│   └─────────────────┘    └─────────────────┘               │
│                                                             │
│   DEFENSE (耐量子暗号)                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Shor's Algorithm（ショアのアルゴリズム）

### 1.1 概要

Shorのアルゴリズムは、整数の因数分解を**指数関数的に高速化**する量子アルゴリズムです。

| 計算量 | 古典計算機 | 量子計算機 |
|--------|-----------|-----------|
| N桁の因数分解 | O(exp(N^(1/3))) | O(N³) |
| RSA-2048 | 宇宙の年齢以上 | 数時間（理論上） |

### 1.2 アルゴリズムの流れ

```
1. ランダムな基数 a を選択 (1 < a < N, gcd(a,N) = 1)
2. 量子位相推定で f(x) = a^x mod N の周期 r を求める
3. r が偶数なら gcd(a^(r/2) ± 1, N) で因数を得る
```

### 1.3 実装詳細

```
src/quantum_crypto/
├── algorithms/
│   └── shor.py          # メインアルゴリズム
├── modexp/              # 各Nに最適化されたmod-exp回路
│   ├── n15.py           # N=15 (教育用)
│   ├── n21.py           # N=21 (Margolus gate使用)
│   ├── n33.py           # N=33
│   └── n35_a6_optimized.py  # N=35, a=6 (最小回路)
└── runner.py            # 実行インターフェース
```

### 1.4 実験結果

| N | 基数a | 周期r | 量子ビット | 2Qゲート | 成功率 |
|---|-------|-------|-----------|----------|--------|
| 15 | 7 | 4 | 4 | 6 | ~85% |
| 21 | 4 | 3 | 5 | 15 | ~60% |
| 35 | 6 | 2 | 3 | 2 | **96.6%** |
| 35 | 8 | 4 | 5 | 8 | 47.6% |
| 91 | 8 | 4 | 5 | 8 | **75.3%** |
| 143 | 34 | 4 | 5 | 8 | **74.2%** |
| 185 | 6 | 4 | 5 | 8 | **73.3%** |

**重要な知見**:
- 周期rが小さいほど回路が単純になり、NISQデバイスでの成功率が大幅に向上します
- r=4 の場合、同一回路構造で異なるNを因数分解可能（encoded orbit の再利用）
- N=143 は Shor型 order-finding + 超伝導デバイスでの実機実験として新規性あり

### 1.5 使用例

```python
from quantum_crypto import run_shor

# シミュレータで実行
result = run_shor(15, backend_type='simulator', shots=1000)
print(f"Factors of 15: {result.factors}")  # (3, 5)

# AWS Braket実機で実行
result = run_shor(35, backend_type='braket', shots=100,
                  braket_device='arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3')
```

---

## 2. Grover's Algorithm（グローバーのアルゴリズム）

### 2.1 概要

Groverのアルゴリズムは、構造のないデータベースの探索を**2次的に高速化**します。

| 探索空間 | 古典計算機 | 量子計算機 |
|---------|-----------|-----------|
| N個の要素 | O(N) | O(√N) |
| AES-128鍵 | 2^128 回 | 2^64 回 |
| AES-256鍵 | 2^256 回 | 2^128 回 |

**注意**: Shorとは異なり、Groverは対称鍵暗号を「破る」のではなく「弱める」だけです。
AES-256を使えば、量子コンピュータに対しても128ビット相当の安全性を維持できます。

### 2.2 アルゴリズムの流れ

```
1. 全状態の均等重ね合わせを作成: |ψ⟩ = H^⊗n|0⟩
2. 以下をO(√N)回繰り返す:
   a. オラクル: 解の状態の位相を反転
   b. 拡散演算子: 平均周りの振幅反転
3. 測定して解を得る
```

### 2.3 実装詳細

```
src/quantum_crypto/grover/
├── core.py        # GroverSearchクラス（メインアルゴリズム）
├── oracles.py     # オラクル構築ユーティリティ
└── toy_cipher.py  # トイ暗号に対するGrover攻撃デモ
```

### 2.4 トイ暗号攻撃

NISQデバイスで実行可能なデモとして、4ビットのトイ暗号（XORベース）に対する
Grover攻撃を実装しています。

```
暗号化: ciphertext = plaintext XOR key
攻撃目標: 既知の(plaintext, ciphertext)ペアから秘密鍵を復元
```

| 鍵長 | 探索空間 | 古典探索 | Grover探索 | 高速化 |
|------|---------|---------|-----------|--------|
| 4ビット | 16 | 16回 | 3回 | 5.3x |
| 8ビット | 256 | 256回 | 12回 | 21x |
| 128ビット | 2^128 | 2^128 | 2^64 | 2^64 x |

### 2.5 使用例

```python
from quantum_crypto.grover import GroverSearch, create_marking_oracle

# 基本的な探索（4量子ビット空間で7を探索）
oracle = create_marking_oracle(4, [7])
grover = GroverSearch(n_qubits=4, oracle=oracle, n_solutions=1)
result = grover.run_simulation(shots=1000)
print(f"Found: {result.measured_state}")  # 7

# トイ暗号攻撃
from quantum_crypto.grover import GroverCipherAttack, toy_encrypt

key = 5
plaintext = 3
ciphertext = toy_encrypt(plaintext, key, use_sbox=False)

attack = GroverCipherAttack(plaintext, ciphertext, key_bits=4)
result = attack.run_simulation(shots=1000)
print(f"Recovered key: {result.found_key}")  # 5
```

---

## 3. Post-Quantum Cryptography（耐量子暗号）

### 3.1 概要

量子コンピュータの脅威に対抗するため、NISTは2024年に新しい暗号標準を策定しました：

| 標準 | アルゴリズム | 用途 | 基盤問題 |
|------|-------------|------|---------|
| FIPS 203 | ML-KEM | 鍵カプセル化 | Module-LWE |
| FIPS 204 | ML-DSA | デジタル署名 | Module-LWE |
| FIPS 205 | SLH-DSA | デジタル署名 | ハッシュベース |

これらは格子問題（Lattice Problem）に基づいており、量子コンピュータでも
効率的に解けないと考えられています。

### 3.2 ML-KEM（鍵カプセル化）

ML-KEMは、RSAやECDHの代替となる鍵交換メカニズムです。

| レベル | アルゴリズム | 公開鍵サイズ | 暗号文サイズ | セキュリティ |
|--------|-------------|-------------|-------------|-------------|
| 1 | ML-KEM-512 | 800 bytes | 768 bytes | AES-128相当 |
| 3 | ML-KEM-768 | 1,184 bytes | 1,088 bytes | AES-192相当 |
| 5 | ML-KEM-1024 | 1,568 bytes | 1,568 bytes | AES-256相当 |

```python
from quantum_crypto.pqc import ml_kem_keygen, ml_kem_encapsulate, ml_kem_decapsulate

# Alice: 鍵ペア生成
public_key, secret_key = ml_kem_keygen(768)

# Bob: 共有秘密をカプセル化
ciphertext, shared_secret_bob = ml_kem_encapsulate(public_key, 768)

# Alice: 共有秘密を復元
shared_secret_alice = ml_kem_decapsulate(secret_key, ciphertext, 768)

assert shared_secret_alice == shared_secret_bob  # 一致！
```

### 3.3 ML-DSA（デジタル署名）

ML-DSAは、RSAやECDSAの代替となるデジタル署名スキームです。

| レベル | アルゴリズム | 公開鍵サイズ | 署名サイズ | セキュリティ |
|--------|-------------|-------------|-----------|-------------|
| 2 | ML-DSA-44 | 1,312 bytes | 2,420 bytes | AES-128相当 |
| 3 | ML-DSA-65 | 1,952 bytes | 3,309 bytes | AES-192相当 |
| 5 | ML-DSA-87 | 2,592 bytes | 4,627 bytes | AES-256相当 |

```python
from quantum_crypto.pqc import ml_dsa_keygen, ml_dsa_sign, ml_dsa_verify

# 鍵ペア生成
public_key, secret_key = ml_dsa_keygen(65)

# 署名
message = b"Important document"
signature = ml_dsa_sign(secret_key, message, 65)

# 検証
is_valid = ml_dsa_verify(public_key, message, signature, 65)
print(f"Signature valid: {is_valid}")  # True
```

### 3.4 ハイブリッド暗号

移行期間中は、古典暗号と耐量子暗号を組み合わせた**ハイブリッド方式**が推奨されます。

```
┌───────────────────────────────────────────────────┐
│              ハイブリッドKEM                        │
│  ┌─────────────┐    ┌─────────────┐              │
│  │   X25519    │ + │  ML-KEM-768 │              │
│  │  (古典ECDH) │    │   (PQC)     │              │
│  └─────────────┘    └─────────────┘              │
│         ↓                  ↓                     │
│    shared_1           shared_2                   │
│         └───────┬───────┘                        │
│                 ↓                                │
│         HKDF(shared_1 || shared_2)               │
│                 ↓                                │
│           最終共有秘密                             │
└───────────────────────────────────────────────────┘
```

```python
from quantum_crypto.pqc import HybridKEM

hybrid = HybridKEM(ml_kem_level=768)

# Alice: 鍵ペア生成
alice_keys = hybrid.generate_keypair()

# Bob: カプセル化
encapsulation = hybrid.encapsulate(alice_keys)
bob_secret = encapsulation.combined_secret

# Alice: デカプセル化
alice_secret = hybrid.decapsulate(alice_keys, encapsulation)

assert alice_secret == bob_secret  # 一致！
```

### 3.5 実装詳細

```
src/quantum_crypto/pqc/
├── __init__.py    # パブリックAPI
├── kem.py         # ML-KEM実装
├── signatures.py  # ML-DSA実装
├── hybrid.py      # ハイブリッド暗号
├── benchmark.py   # パフォーマンス計測
└── utils.py       # ユーティリティ
```

---

## 4. 依存関係とセットアップ

### 4.1 基本インストール

```bash
# 基本パッケージ（Shor + Grover）
pip install -e .

# PQC機能を含む
pip install -e ".[pqc]"

# AWS Braket実機対応
pip install -e ".[hardware]"

# 全機能
pip install -e ".[all]"
```

### 4.2 liboqs共有ライブラリ（macOS）

PQC機能にはliboqsの共有ライブラリが必要です：

```bash
# ソースからビルド（共有ライブラリ有効）
git clone --depth 1 https://github.com/open-quantum-safe/liboqs.git
cd liboqs && mkdir build && cd build
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/opt/homebrew ..
make -j$(sysctl -n hw.ncpu)
cp lib/liboqs*.dylib /opt/homebrew/lib/
```

---

## 5. スクリプト

### 5.1 Shor実験

```bash
# シミュレータでN=15を因数分解
python scripts/run_shor_experiment.py --n 15 --backend simulator

# AWS Braket実機でN=35を因数分解
python scripts/run_shor_experiment.py --n 35 --backend braket --shots 100
```

### 5.2 Grover攻撃デモ

```bash
# 基本デモ
python scripts/run_grover_attack.py

# 4ビット暗号攻撃
python scripts/run_grover_attack.py --key-bits 4 --shots 1000

# 統計分析（10回試行）
python scripts/run_grover_attack.py --stats
```

### 5.3 PQCデモ

```bash
# 基本デモ（ML-KEM + ML-DSA）
python scripts/run_pqc_demo.py

# ハイブリッド暗号デモ
python scripts/run_pqc_demo.py --hybrid

# パフォーマンスベンチマーク
python scripts/run_pqc_demo.py --benchmark
```

---

## 6. 量子脅威タイムライン

```
2024  ├─ NIST PQC標準策定（FIPS 203, 204, 205）
      │
2025  ├─ 移行期間開始
      │  - ハイブリッド暗号の採用推奨
      │  - 長期秘密データの保護に注意
      │
2030  ├─ 量子優位性の可能性
      │  - 1000+論理量子ビット
      │  - RSA-2048への実質的脅威
      │
2035+ ├─ 暗号学的に関連する量子コンピュータ(CRQC)
         - RSA/ECDSAの完全な破壊が可能に
```

**「Harvest Now, Decrypt Later」攻撃**:
現在暗号化された通信を保存しておき、将来の量子コンピュータで解読する攻撃。
長期間の機密性が必要なデータは、今すぐPQCへの移行を検討すべきです。

---

## 7. 参考文献

### Shor's Algorithm
- Shor, P. W. (1994). "Algorithms for quantum computation"
- Skosana & Tame (2021). "Demonstration of Shor's factoring algorithm for N=21"

### Grover's Algorithm
- Grover, L. K. (1996). "A fast quantum mechanical algorithm for database search"

### Post-Quantum Cryptography
- NIST FIPS 203: ML-KEM (Module-Lattice-Based Key-Encapsulation Mechanism)
- NIST FIPS 204: ML-DSA (Module-Lattice-Based Digital Signature Algorithm)
- NIST FIPS 205: SLH-DSA (Stateless Hash-Based Digital Signature Algorithm)

---

## 8. ライセンス

MIT License
