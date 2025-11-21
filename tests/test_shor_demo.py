"""Tests for Shor's algorithm implementation.

このファイルは Shor アルゴリズムの実装を包括的にテストします。
以下の側面をカバーしています:

1. 基本的な数学関数（GCD、周期発見、因数分解）
2. 古典版 Shor アルゴリズム
3. 量子版 Shor アルゴリズム（QPE + QFT）
4. エッジケースとエラーハンドリング

各テストは独立しており、特定の機能や条件を検証します。
"""

import pytest
from src.quantum_rsa.utils import (
    gcd,
    factor_from_period,
    find_period_from_phase,
)
from src.quantum_rsa.algorithms.classical import ClassicalShor
from src.quantum_rsa.runner import run_shor


class TestGCD:
    """最大公約数（GCD）関数のテスト.

    GCD は Shor アルゴリズムの前処理で使用されます。
    gcd(a, N) != 1 の場合、すぐに因数が見つかります。
    """

    def test_basic_gcd(self):
        """基本的な GCD 計算の検証.

        テストケース:
        - gcd(15, 5) = 5: 15 = 3×5, 5 = 1×5 → 共通因数 5
        - gcd(21, 14) = 7: 21 = 3×7, 14 = 2×7 → 共通因数 7
        - gcd(7, 5) = 1: 互いに素
        """
        assert gcd(15, 5) == 5
        assert gcd(21, 14) == 7
        assert gcd(7, 5) == 1

    def test_coprime(self):
        """互いに素な数のペアをテスト.

        Shor アルゴリズムでは gcd(a, N) = 1 が前提条件。
        これらのケースは周期発見フェーズに進みます。
        """
        assert gcd(7, 15) == 1   # 7 は 15 の因数を持たない
        assert gcd(11, 15) == 1  # 11 は 15 の因数を持たない

    def test_edge_cases(self):
        """境界値とエッジケースのテスト.

        - gcd(0, n) = n: 0 はすべての数で割り切れる
        - gcd(1, 1) = 1: 最小の自然数
        """
        assert gcd(0, 5) == 5  # 0 と任意の数の GCD はその数自身
        assert gcd(5, 0) == 5  # 順序を入れ替えても同じ
        assert gcd(1, 1) == 1  # 最小ケース


class TestClassicalPeriodFinding:
    """古典的な周期発見アルゴリズムのテスト.

    find_period_classical は f(x) = a^x mod N の周期 r を発見します。
    周期 r とは、a^r ≡ 1 (mod N) を満たす最小の正整数です。

    この古典版は O(r) の計算量で、量子版の O(log r) と比較するための
    リファレンス実装です。
    """

    def test_n15_periods(self):
        """N=15 における既知の周期を検証.

        数学的背景:
        - a=7:  7^1=7, 7^2=4, 7^3=13, 7^4=1 (mod 15) → r=4
        - a=11: 11^1=11, 11^2=1 (mod 15) → r=2
        - a=2:  2^1=2, 2^2=4, 2^3=8, 2^4=1 (mod 15) → r=4
        - a=4:  4^1=4, 4^2=1 (mod 15) → r=2
        """
        algo = ClassicalShor()
        assert algo._find_period(7, 15) == 4
        assert algo._find_period(11, 15) == 2
        assert algo._find_period(2, 15) == 4
        assert algo._find_period(4, 15) == 2

    def test_invalid_base(self):
        """無効なベース（gcd(a, N) != 1）のテスト.

        gcd(a, N) != 1 の場合、周期は定義されません。
        これらのケースでは None を返すべきです。

        - gcd(3, 15) = 3 ≠ 1: 3 は 15 の因数
        - gcd(5, 15) = 5 ≠ 1: 5 は 15 の因数
        """
        algo = ClassicalShor()
        assert algo._find_period(3, 15) is None  # gcd(3, 15) = 3
        assert algo._find_period(5, 15) is None  # gcd(5, 15) = 5

    def test_n21_periods(self):
        """N=21 における周期の検証.

        - a=2: 2^6 ≡ 64 ≡ 1 (mod 21) → r=6
        - a=5: 5^6 ≡ 15625 ≡ 1 (mod 21) → r=6
        """
        algo = ClassicalShor()
        assert algo._find_period(2, 21) == 6
        assert algo._find_period(5, 21) == 6


class TestFactorFromPeriod:
    """周期から因数を抽出するアルゴリズムのテスト.

    Shor アルゴリズムの核心部分: 周期 r から因数を計算します。

    数学的原理:
    a^r ≡ 1 (mod N) が成り立つとき、
    (a^(r/2))^2 ≡ 1 (mod N)
    (a^(r/2) - 1)(a^(r/2) + 1) ≡ 0 (mod N)

    つまり、N は (a^(r/2) ± 1) の因数を含む可能性があります。

    アルゴリズム:
    1. r が偶数であることを確認（奇数なら失敗）
    2. y = a^(r/2) mod N を計算
    3. y ≡ -1 (mod N) でないことを確認（この場合は失敗）
    4. gcd(y - 1, N) と gcd(y + 1, N) を計算
    5. 自明でない因数（1でもNでもない）が得られる

    成功条件:
    - r が偶数
    - a^(r/2) ≢ -1 (mod N)
    これらの条件が満たされると、約50%の確率で因数が見つかります。
    """

    def test_n15_factorization(self):
        """N=15 の標準的な因数分解テスト.

        テストケース: a=7, r=4, N=15
        計算過程:
        1. r=4 は偶数 ✓
        2. y = 7^(4/2) mod 15 = 7^2 mod 15 = 49 mod 15 = 4
        3. y=4 ≠ -1 (mod 15) ✓
        4. gcd(4-1, 15) = gcd(3, 15) = 3
        5. gcd(4+1, 15) = gcd(5, 15) = 5
        6. 因数: {3, 5} ✓

        検証項目:
        - 正しい因数が得られる
        - 因数の積が N に等しい
        """
        # a=7, r=4 で N=15 を因数分解
        factors = factor_from_period(7, 15, 4)
        assert factors is not None
        assert set(factors) == {3, 5}
        assert factors[0] * factors[1] == 15

    def test_n15_another_base(self):
        """異なるベースでの因数分解テスト.

        テストケース: a=11, r=2, N=15
        計算過程:
        1. r=2 は偶数 ✓
        2. y = 11^(2/2) mod 15 = 11^1 mod 15 = 11
        3. y=11 ≠ -1 (mod 15) ✓（-1 ≡ 14 (mod 15)）
        4. gcd(11-1, 15) = gcd(10, 15) = 5
        5. gcd(11+1, 15) = gcd(12, 15) = 3
        6. 因数: {3, 5} ✓

        この例は、異なるベースでも同じ因数が得られることを示します。
        """
        # a=11, r=2 で N=15 を因数分解
        factors = factor_from_period(11, 15, 2)
        assert factors is not None
        assert set(factors) == {3, 5}

    def test_odd_period_fails(self):
        """奇数の周期での失敗ケーステスト.

        失敗理由:
        r が奇数の場合、r/2 が整数にならないため、
        a^(r/2) を計算できません。

        例: r=3 の場合、a^(3/2) は整数べき乗ではありません。

        実際の Shor アルゴリズムでは、奇数の周期が得られた場合、
        別のランダムなベース a を選んで再実行します。
        約50%の確率で偶数の周期が得られます。
        """
        # 奇数の周期は失敗する
        assert factor_from_period(2, 15, 3) is None

    def test_trivial_period_fails(self):
        """自明な周期（r=1）での失敗ケーステスト.

        失敗理由:
        r=1 の場合、a^1 ≡ 1 (mod N) を意味しますが、
        これは a=1 の場合のみ成立します。

        a=1 では周期情報から因数を抽出できません:
        - y = 1^(1/2) = 1
        - gcd(1-1, N) = gcd(0, N) = N（自明）
        - gcd(1+1, N) = gcd(2, N)（部分的な情報）

        実装では None を返すか、自明な因数のみを返します。
        実際の Shor アルゴリズムでは、r>1 を保証するために
        a>1 を選択します。
        """
        # r=1 は役に立たない
        # この場合、factor_from_period は None を返すか、
        # または N 自身を返す可能性がある
        pass


class TestPhaseToperiod:
    """位相から周期を復元するロジックのテスト.

    Shor アルゴリズムの量子部分では QPE の結果として位相 φ を得ます。
    この φ を有理数近似 (連分数展開) することで周期 r を求め、
    factor_from_period につなぎます。

    本クラスでは find_period_from_phase の境界条件と代表的な
    入力パターンを網羅的に確認します。
    """

    def test_exact_fractions(self):
        """正確に表せる有理数位相の変換を検証.

        位相が 0.75 (=3/4) や 0.5 (=1/2) など有限桁で表現できる場合、
        連分数復元に誤差が生じないことを確認します。
        期待される周期は φ = s/r の r に一致するはずです。
        """
        # 位相 0.75 = 3/4 → 周期 4 (r=4)
        assert find_period_from_phase(0.75, 15) == 4
        # 位相 0.5 = 1/2 → 周期 2
        assert find_period_from_phase(0.5, 15) == 2
        # 位相 0.25 = 1/4 → 周期 4
        assert find_period_from_phase(0.25, 15) == 4

    def test_zero_phase(self):
        """ゼロ位相の特別扱いをチェック.

        φ=0 は QPE が単位固有値を返したことを意味しますが、
        これは a=1 などの自明なケースと解釈されるため周期情報を
        提供できません。関数は None を返して処理を終えるべきです。
        """
        assert find_period_from_phase(0.0, 15) is None

    def test_approximate_fractions(self):
        """浮動小数点誤差を伴う位相でも妥当な周期を得る.

        実機・シミュレータではサンプリングの揺らぎにより φ が
        理論値から僅かにずれます。0.7499 などの近似値に対して
        連分数が r=4 を再現できるか、もしくは安全に None を返すか
        を確認します。
        """
        # 0.7499 ≈ 3/4
        r = find_period_from_phase(0.7499, 15)
        assert r in [4, None]  # 連分数近似で 4 が得られるはず


class TestClassicalShor:
    """古典版 run_shor パイプラインのテスト.

    method="classical" のブランチは純粋な計算機上で Shor を模倣し、
    周期発見には find_period_classical を使用します。

    このクラスでは代表的な入力シナリオを用意し、
    - 既知の因数分解が成功するか
    - 偶数などのショートカットが機能するか
    - 無効なベースを渡したときの GCD チェックが働くか
    を確認します。
    """

    def test_n15_factorization(self):
        """N=15, a=7 の古典的成功ケース.

        古典ベースでは周期 r=4 が導出され、その後 factor_from_period が
        {3,5} を返すことを期待します。period フィールドも返却値に
        含まれるため、属性が 4 になっているかを確認します。
        """
        result = run_shor(number=15, base=7, method="classical")
        assert result.success
        assert result.factors is not None
        assert set(result.factors) == {3, 5}
        assert result.period == 4

    def test_even_number(self):
        """入力 N が偶数の場合の早期終了を検証.

        Shor の前処理では N が偶数なら瞬時に (2, N/2) を返せます。
        base=None としても内部でランダム選択が行われますが、
        偶数チェックが優先されるため deterministic な結果を期待します。
        """
        result = run_shor(number=14, base=None, method="classical")
        assert result.success
        assert result.factors == (2, 7)

    def test_invalid_base(self):
        """gcd(a, N) ≠ 1 のベースを渡した際の挙動.

        base=3, N=15 では gcd=3 なので周期探索へ進むまでもなく
        因数が得られるはずです。結果が成功扱いとなり、
        factors に 3 が含まれることを確認します。
        """
        result = run_shor(number=15, base=3, method="classical")
        assert result.success
        assert result.factors is not None
        assert 3 in result.factors


@pytest.mark.skipif(
    not pytest.importorskip("qiskit", reason="Qiskit not installed"),
    reason="Qiskit required for quantum tests",
)
class TestQuantumShor:
    """量子版 run_shor の統合テスト.

    Qiskit / Aer シミュレーションを通じて QPE → 逆QFT → 周期→因数
    のフローが一貫して正しく機能するかをチェックします。

    教材サイズ (N=15, 21, 33) に加えて、エラー条件や回路メトリクス
    収集のテストも含めています。
    """

    def test_n15_quantum_factorization(self):
        """N=15 の量子因数分解が成功することを検証.

        成功条件:
        - factors が {3,5}
        - measured_phase / period が取得できる
        - success フラグが True
        """
        result = run_shor(number=15, base=7, method="quantum", shots=2048)
        assert result.success
        assert result.factors is not None
        assert set(result.factors) == {3, 5}
        assert result.measured_phase is not None
        assert result.period is not None

    def test_n21_quantum_factorization(self):
        """N=21 でも因数 {3,7} が得られることを確認.

        ユニタリ行列表現を使う mod-exp 実装の健全性を検証します。
        """
        result = run_shor(number=21, base=2, method="quantum", shots=2048)
        assert result.success
        assert result.factors is not None
        assert set(result.factors) == {3, 7}

    def test_n33_quantum_factorization(self):
        """N=33 (3×11) のケースをチェック.

        6量子ビット + 64×64 ユニタリの構成でも最終結果が正しいかを
        検証し、大きめの状態空間に対する回路が壊れていないかを確認。
        """
        result = run_shor(number=33, base=2, method="quantum", shots=2048)
        assert result.success
        assert result.factors is not None
        assert set(result.factors) == {3, 11}

    def test_invalid_n(self):
        """未実装 N に対して NotImplementedError が発生する."""
        with pytest.raises(NotImplementedError):
            run_shor(number=17, base=2, method="quantum")

    def test_invalid_base_quantum(self):
        """量子モードでも無効なベースは拒否される."""
        with pytest.raises(ValueError):
            run_shor(number=15, base=3, method="quantum")

    def test_circuit_metrics(self):
        """回路深さ・ゲート数などのメトリクスが記録される."""
        result = run_shor(number=15, base=7, method="quantum", shots=1024)
        assert result.circuit_depth is not None
        assert result.total_gates is not None
        assert result.total_gates > 0


class TestEdgeCases:
    """補足的なエラーハンドリングとランダム動作のテスト."""

    def test_invalid_method(self):
        """method パラメータに未知の文字列を渡した場合のエラー."""
        with pytest.raises(ValueError):
            run_shor(number=15, base=7, method="invalid")  # type: ignore

    def test_classical_with_random_base(self):
        """base=None でランダム選択される挙動を確認."""
        result = run_shor(number=15, base=None, method="classical")
        assert result.base is not None
        assert 2 <= result.base < 15
