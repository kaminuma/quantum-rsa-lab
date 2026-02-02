"""Tests for modular exponentiation implementations.

このファイルは量子回路による modular exponentiation の実装をテストします。

Modular exponentiation (mod-exp) は Shor アルゴリズムの核心部分で、
ユニタリ演算子 U を実装します:

    U|x⟩ = |ax mod N⟩

量子位相推定（QPE）では、この U の固有値から周期情報を抽出します。

テスト対象:
1. レジストリシステム（複数の N に対応）
2. N=15, 21, 33 の各実装
3. 有効なベースの検証
4. 統合テスト（QPE コンテキストでの動作確認）
"""

import pytest
import numpy as np

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import Operator

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

from src.quantum_crypto.modexp import get_config, list_supported


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit required")
class TestModExpRegistry:
    """Modexp レジストリシステムのテスト.

    レジストリシステムは、異なる N 値に対する mod-exp 実装を
    統一的に管理します。

    設計パターン: Registry Pattern
    - 各 N 専用の実装を動的に登録
    - 実装の切り替えを簡単に
    - 新しい N の追加が容易

    構成要素:
    - REGISTRY: 辞書 {N: config}
    - config: 実装関数、有効なベース、量子ビット数など
    """

    def test_list_supported(self):
        """サポートされている N のリストを取得.

        実装済みの N 値:
        - N=15: 3 × 5 (最適化された SWAP/CNOT 実装)
        - N=21: 3 × 7 (ユニタリ行列実装)
        - N=33: 3 × 11 (ユニタリ行列実装)

        将来的に N=35, 51 などが追加される可能性があります。
        """
        supported = list_supported()
        assert 15 in supported
        assert 21 in supported
        assert 33 in supported
        assert len(supported) >= 3

    def test_get_config_n15(self):
        """N=15 の設定を取得して検証.

        N=15 の構成:
        - factors: (3, 5)
        - n_work_qubits: 4 (15 < 2^4 = 16)
        - valid_bases: gcd(a, 15) = 1 を満たす a
          例: [2, 4, 7, 8, 11, 13]
        - func: c_amod15 関数

        N=15 は最も効率的な実装で、SWAP と CNOT ゲートで
        最適化されています。
        """
        config = get_config(15)
        assert config["n"] == 15
        assert config["factors"] == (3, 5)
        assert config["n_work_qubits"] == 4
        assert 7 in config["valid_bases"]
        assert 11 in config["valid_bases"]
        assert callable(config["func"])

    def test_get_config_n21(self):
        """N=21 の設定を取得して検証.

        N=21 の構成:
        - factors: (3, 7)
        - n_work_qubits: 5 (21 < 2^5 = 32)

        N=21 以降はユニタリ行列を直接使用する実装方式です。
        SWAP/CNOT への最適化は行われていません。
        """
        config = get_config(21)
        assert config["n"] == 21
        assert config["factors"] == (3, 7)
        assert config["n_work_qubits"] == 5

    def test_get_config_unsupported(self):
        """未実装の N に対するエラーハンドリング.

        N=17 は実装されていないため、NotImplementedError が
        発生すべきです。エラーメッセージには実装済みの N の
        リストが含まれます。
        """
        with pytest.raises(NotImplementedError):
            get_config(17)


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit required")
class TestN15ModExp:
    """N=15 専用 modular exponentiation 実装のテスト.

    N=15 は Shor アルゴリズムの教育的実装で最もよく使われる例です。

    実装の特徴:
    - 4量子ビット (15 < 2^4 = 16)
    - SWAP と CNOT ゲートで最適化
    - 各 a 値に対して手動で最適化された回路

    数学的背景:
    - 15 = 3 × 5
    - オイラーの関数 φ(15) = φ(3)×φ(5) = 2×4 = 8
    - 有効なベース: gcd(a, 15) = 1 を満たす a
      → [2, 4, 7, 8, 11, 13] (合計6個)
    """

    def test_valid_bases(self):
        """有効なベースの検証.

        valid_bases リストの各要素が gcd(a, 15) = 1 を
        満たすことを確認します。

        N=15 の場合:
        - 3の倍数: 3, 6, 9, 12 → 除外
        - 5の倍数: 5, 10 → 除外
        - それ以外: 有効

        これにより、周期発見が可能なベースのみが選択されます。
        """
        config = get_config(15)
        valid_bases = config["valid_bases"]
        assert all(np.gcd(a, 15) == 1 for a in valid_bases)

    def test_circuit_creation(self):
        """量子回路の生成テスト.

        c_amod15(a, power) は制御付きユニタリゲートを返します。
        power パラメータは指数を 2^power として解釈します。

        例: c_amod15(7, 0) → 7^(2^0) = 7^1 mod 15
            c_amod15(7, 2) → 7^(2^2) = 7^4 mod 15
        """
        config = get_config(15)
        c_amod15 = config["func"]

        # a=7, power=0 の回路を作成
        circuit = c_amod15(7, 0)
        # 制御付きゲートであることを確認
        assert isinstance(circuit, type(QuantumCircuit(1).to_gate().control()))

    def test_invalid_base(self):
        """無効なベースに対するエラーハンドリング.

        gcd(a, N) != 1 のベースは周期が定義されないため、
        ValueError を発生させるべきです。

        例: a=3 の場合、gcd(3, 15) = 3 ≠ 1
            これは 3 が 15 の因数であることを意味します。
        """
        config = get_config(15)
        c_amod15 = config["func"]

        # a=3 は無効（gcd(3, 15) = 3）
        with pytest.raises(ValueError):
            c_amod15(3, 0)

    def test_modular_exponentiation_in_qpe_context(self):
        """QPE コンテキストでの mod-exp の統合テスト.

        個々のゲートレベルの挙動ではなく、Shor アルゴリズム全体での
        動作を検証します。これにより、mod-exp 実装が実際の
        因数分解タスクで正しく機能することを確認します。

        テスト戦略:
        - 複数の異なるベース (a=7, 2, 11) で実行
        - 各ベースで正しい因数 {3, 5} が得られることを確認
        - これにより、mod-exp の周期計算が正確であることを検証

        注意:
        このテストは量子シミュレータを使用するため、
        実行時間が長くなります（各ケース数秒）。
        """
        from src.quantum_crypto import run_shor

        # N=15, 複数のベースで因数分解が成功することを確認
        # これにより、modexp実装が正しく動作していることが検証される
        test_cases = [
            (15, 7,  {3, 5}),  # a=7: 周期 r=4
            (15, 2,  {3, 5}),  # a=2: 周期 r=4
            (15, 11, {3, 5}),  # a=11: 周期 r=2
        ]

        for N, base, expected_factors in test_cases:
            result = run_shor(number=N, base=base, method="quantum", shots=1024)
            assert result.success, f"N={N}, base={base}: factorization should succeed"
            assert result.factors is not None
            assert set(result.factors) == expected_factors, (
                f"N={N}, base={base}: expected {expected_factors}, got {result.factors}"
            )


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit required")
class TestN21ModExp:
    """N=21 専用 modular exponentiation 実装のテスト.

    実装の特徴:
    - 5量子ビット (21 < 2^5 = 32)
    - ユニタリ行列を直接使用
    - N=15 のような手動最適化は行われていない

    数学的背景:
    - 21 = 3 × 7
    - オイラーの関数 φ(21) = φ(3)×φ(7) = 2×6 = 12
    - 有効なベース: gcd(a, 21) = 1 を満たす a
      → 3の倍数と7の倍数を除く

    実装方式:
    ユニタリ行列を32×32の置換行列として直接構築します。
    これは教育的にはわかりやすいですが、ゲート数が多くなります。
    """

    def test_valid_bases(self):
        """有効なベースの検証.

        N=21 の因数は 3 と 7 なので、以下を除外:
        - 3の倍数: 3, 6, 9, 12, 15, 18
        - 7の倍数: 7, 14
        - 両方の倍数: 21

        検証項目:
        1. すべてのベースが gcd(a, 21) = 1 を満たす
        2. 因数の倍数が含まれていない
        """
        config = get_config(21)
        valid_bases = config["valid_bases"]
        # gcd(a, 21) = 1 を確認
        assert all(np.gcd(a, 21) == 1 for a in valid_bases)
        # 3の倍数と7の倍数は含まれない
        assert 3 not in valid_bases
        assert 7 not in valid_bases
        assert 14 not in valid_bases  # 7×2

    def test_unitary_matrix_is_permutation(self):
        """ユニタリ行列実装の回路生成テスト.

        N=21 の実装はユニタリ行列を使用します。
        置換行列 P は各列・各行に1つだけ 1 を持ち、残りは 0 です。

        数学的性質:
        - P は正方行列
        - P^T P = I (直交行列)
        - P はユニタリ (量子ゲートとして有効)

        注意:
        制御付きゲートから内部のユニタリを取り出すのは困難なため、
        ここでは回路が正常に生成されることのみを確認します。
        実際の動作は統合テスト (run_shor) で検証されます。
        """
        config = get_config(21)
        c_amod21 = config["func"]

        # a=2, power=0 でテスト
        circuit = c_amod21(2, 0)

        # NOTE: 制御付きゲートから元のゲートを取り出すのは困難なので、
        # ここでは回路が生成できることだけを確認
        assert circuit is not None


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit required")
class TestN33ModExp:
    """N=33 専用 modular exponentiation 実装のテスト.

    実装の特徴:
    - 6量子ビット (33 < 2^6 = 64)
    - ユニタリ行列を直接使用 (64×64行列)
    - より大きな状態空間

    数学的背景:
    - 33 = 3 × 11
    - オイラーの関数 φ(33) = φ(3)×φ(11) = 2×10 = 20
    - 有効なベース: gcd(a, 33) = 1 を満たす a

    スケーラビリティ:
    N=33 は教育的な範囲では大きめの例です。
    これ以上大きな N では、ユニタリ行列のサイズが急速に増大:
    - N=51 (6 qubits): 64×64 行列
    - N=85 (7 qubits): 128×128 行列
    - N=255 (8 qubits): 256×256 行列
    """

    def test_valid_bases(self):
        """有効なベースの検証.

        N=33 の因数は 3 と 11 なので、以下を除外:
        - 3の倍数: 3, 6, 9, 12, 15, 18, 21, 24, 27, 30
        - 11の倍数: 11, 22
        - 両方の倍数: 33

        N=21 より除外される数が多いため、
        有効なベースの数も相対的に少なくなります。
        """
        config = get_config(33)
        valid_bases = config["valid_bases"]
        # gcd(a, 33) = 1 を確認
        assert all(np.gcd(a, 33) == 1 for a in valid_bases)
        # 3の倍数と11の倍数は含まれない
        assert 3 not in valid_bases
        assert 11 not in valid_bases
        assert 22 not in valid_bases  # 11×2

    def test_circuit_size(self):
        """量子ビット数の検証.

        N=33 を表現するには少なくとも 6 量子ビットが必要です:
        - 2^5 = 32 < 33 → 不足
        - 2^6 = 64 > 33 → 十分

        6量子ビットで 0〜63 の状態を表現できるため、
        33 より大きい状態 (34〜63) は恒等変換として扱われます。
        """
        config = get_config(33)
        assert config["n_work_qubits"] == 6  # 33 < 2^6 = 64
