"""Tests for experiment logging utilities.

このファイルは実験データの収集と集計機能をテストします。
主な機能:

1. sweep_shot_counts: パラメータスイープ実験の実行とデータ収集
2. summarize_success: 成功率の集計
3. QuantumRunSetting: 実験設定の管理

これらの機能は、量子アルゴリズムの性能評価やノイズの影響分析に使用されます。
"""

import pytest
import pandas as pd

try:
    from qiskit_aer import AerSimulator

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

from src.quantum_crypto.experiment_logging import (
    QuantumRunSetting,
    sweep_shot_counts,
    summarize_success,
)


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit required")
class TestSweepShotCounts:
    """ショット数スイープ機能のテスト.

    sweep_shot_counts は異なるショット数で複数回実験を実行し、
    結果を pandas DataFrame として収集します。

    用途:
    - ショット数と成功率の関係を調査
    - ノイズモデルの影響を評価
    - 統計的な信頼性を確保するための繰り返し実験
    """

    def test_basic_sweep(self):
        """基本的なスイープ機能のテスト.

        テスト内容:
        - 2つのショット数 [512, 1024] で実行
        - 各条件を2回繰り返し (repeats=2)
        - 合計 2×2=4 回の実験データを収集

        検証項目:
        - DataFrame が正しく生成される
        - 必要なカラムが含まれている
        - パラメータが正しく記録されている
        """
        df = sweep_shot_counts(
            number=15,
            base=7,
            shots_list=[512, 1024],
            n_count=8,
            repeats=2,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4  # 2 shots × 2 repeats
        assert "success" in df.columns
        assert "shots" in df.columns
        assert "period" in df.columns
        assert df["number"].iloc[0] == 15
        assert df["base"].iloc[0] == 7

    def test_multiple_settings(self):
        """複数の実験設定でのスイープテスト.

        テスト内容:
        - 2つの異なる設定で実行
          1. ideal: ノイズなし
          2. test: シードを固定（再現性のため）

        用途:
        - ノイズモデルの比較
        - エラー軽減手法の評価
        - 異なるシミュレータ設定での比較
        """
        settings = [
            QuantumRunSetting(label="ideal"),
            QuantumRunSetting(label="test", simulator_options={"seed_simulator": 42}),
        ]

        df = sweep_shot_counts(
            number=15,
            base=7,
            shots_list=[512],
            n_count=8,
            repeats=1,
            settings=settings,
        )

        assert len(df) == 2  # 2 settings × 1 shots × 1 repeat
        assert set(df["label"]) == {"ideal", "test"}

    def test_default_settings(self):
        """デフォルト設定でのスイープテスト.

        設定を指定しない場合、"ideal" ラベルで
        ノイズなしのシミュレーションが実行されます。
        """
        df = sweep_shot_counts(
            number=15,
            base=7,
            shots_list=[512],
            n_count=8,
            repeats=1,
        )

        assert len(df) == 1
        assert df["label"].iloc[0] == "ideal"


class TestSummarizeSuccess:
    """成功率集計機能のテスト.

    summarize_success は複数回の実験結果を集計し、
    各条件（ラベル、ショット数）ごとの成功率を計算します。

    用途:
    - 実験結果の統計的分析
    - 条件間の比較
    - 論文やレポートのためのデータ整形
    """

    def test_summarize(self):
        """成功率の正しい計算を検証.

        テストデータ:
        - ideal (ノイズなし):
          - shots=512: 2回中2回成功 → 成功率 1.0 (100%)
          - shots=1024: 1回中1回成功 → 成功率 1.0
        - noisy (ノイズあり):
          - shots=512: 2回中1回成功 → 成功率 0.5 (50%)
          - shots=1024: 1回中1回成功 → 成功率 1.0

        検証内容:
        - グループ化が正しく行われる (label × shots)
        - 成功率が平均値として正しく計算される
        - 出力が DataFrame 形式である
        """
        # サンプルデータを作成
        data = {
            "label": ["ideal", "ideal", "ideal", "noisy", "noisy", "noisy"],
            "shots": [512, 512, 1024, 512, 512, 1024],
            "success": [True, True, True, True, False, True],
        }
        df = pd.DataFrame(data)

        summary = summarize_success(df)

        assert isinstance(summary, pd.DataFrame)
        assert "success_rate" in summary.columns
        assert len(summary) == 4  # 2 labels × 2 shot counts

        # ideal, shots=512 の成功率は 1.0 (2/2)
        ideal_512 = summary[(summary["label"] == "ideal") & (summary["shots"] == 512)]
        assert ideal_512["success_rate"].iloc[0] == 1.0

        # noisy, shots=512 の成功率は 0.5 (1/2)
        noisy_512 = summary[(summary["label"] == "noisy") & (summary["shots"] == 512)]
        assert noisy_512["success_rate"].iloc[0] == 0.5

    def test_empty_dataframe(self):
        """空の DataFrame が正しく処理されることを確認.

        エッジケース: データがない場合でもエラーにならず、
        空の DataFrame を返すべきです。
        """
        df = pd.DataFrame()
        summary = summarize_success(df)
        assert summary.empty


class TestQuantumRunSetting:
    """QuantumRunSetting データクラスのテスト.

    QuantumRunSetting は実験設定をカプセル化します:
    - label: 実験条件の名前 (例: "ideal", "noisy", "mitigated")
    - noise_model: ノイズモデル（オプション）
    - apply_readout_mitigation: 測定誤差補正の有効/無効
    - meas_calibration_shots: キャリブレーション用のショット数
    - simulator_options: AerSimulator への追加オプション

    frozen=True により、作成後の変更を防止しています（イミュータブル）。
    """

    def test_default_values(self):
        """デフォルト値の検証.

        必須パラメータ (label) のみ指定した場合、
        他のパラメータは適切なデフォルト値を持つべきです。

        デフォルト値:
        - noise_model: None (ノイズなし)
        - apply_readout_mitigation: False (補正なし)
        - meas_calibration_shots: 2048
        """
        setting = QuantumRunSetting(label="test")
        assert setting.label == "test"
        assert setting.noise_model is None
        assert setting.apply_readout_mitigation is False
        assert setting.meas_calibration_shots == 2048

    def test_custom_values(self):
        """カスタム値の設定を検証.

        すべてのパラメータをカスタマイズできることを確認。
        特に測定誤差補正（readout mitigation）の設定が正しく反映されるか。
        """
        setting = QuantumRunSetting(
            label="custom",
            apply_readout_mitigation=True,
            meas_calibration_shots=4096,
        )
        assert setting.apply_readout_mitigation is True
        assert setting.meas_calibration_shots == 4096

    def test_immutable(self):
        """イミュータブル性の検証.

        frozen=True により、作成後の属性変更が禁止されています。
        これにより、実験設定の意図しない変更を防ぎます。

        試みられた変更は AttributeError を発生させるべきです。
        """
        setting = QuantumRunSetting(label="test")
        with pytest.raises(AttributeError):
            setting.label = "changed"  # type: ignore
