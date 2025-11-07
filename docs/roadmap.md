# 実験ロードマップ

このプロジェクトは、Shor アルゴリズムによる RSA 因数分解を、理論から実装・実機検証まで段階的に進めることを目的としています。

## Phase 1: 理論と基礎実装

**目標**: Shor アルゴリズムの数理と実装の基礎を確立

- RSA と古典因数分解アルゴリズムの復習
- Shor アルゴリズムの構成要素（量子位相推定、mod-exp、連分数展開）の理解
- N=15 での量子回路実装と動作検証
- 古典版との比較実装

**成果物**: 動作する実装、アルゴリズム解説ドキュメント

## Phase 2: スケールアップと最適化

**目標**: より大きな N への拡張と性能評価

- N=21, 33, 35, 77 への段階的拡張
- ショット数・成功率・ゲート数の測定と可視化
  - `src/quantum_rsa/experiment_logging.py` で DataFrame ロギング & `notebooks/shots_vs_success.ipynb` で可視化済み
- ノイズモデルの導入と誤り緩和技術の検証
  - Aer の depolarizing + readout ノイズ、readout mitigation ワークフローを追加（ideal/noisy/noisy+mitigated 比較）
  - 現状は `apply_readout_mitigation=True` のたびに `2**n_count` 個のキャリブレーション回路を毎回生成 → (n_count, noise_model) ごとに再利用できるようキャッシュ仕組みを追加予定
- 古典アルゴリズムとのベンチマーク比較

**成果物**: 性能データ、グラフ、Jupyter ノートブック

## Phase 3: 実機検証

**目標**: クラウド量子コンピュータでの実行

- AWS Braket 環境構築
- SV1 シミュレータでの回路検証
- QPU（Rigetti/IQM/IonQ）での実行とコスト測定
- 実機特有のノイズ・エラー特性の分析

**成果物**: 実機実行ログ、コスト分析、成功率データ

## Phase 4: 回路最適化

**目標**: より効率的な量子回路の設計

- mod-exp 回路のゲート削減
- 量子/古典ハイブリッドアプローチの検討
- 既存研究（Gidney, Yan, Ekerå 等）との比較
- 限られた量子ビットでの最大 N の探索

**成果物**: 最適化された回路、比較表、技術レポート

## Phase 5: 発展的トピック

**目標**: 関連技術への展開

- 他の量子因数分解手法（VQF, QAOA, DCQF）の検証
- ポスト量子暗号（Kyber, Dilithium）との比較研究
- ハイブリッド量子古典最適化の探索

**成果物**: 調査レポート、実験結果

---

## 参考文献

- **Shor (1994)**: "Algorithms for quantum computation: discrete logarithms and factoring"
- **Gidney & Ekerå (2021)**: "How to factor 2048 bit RSA integers in 8 hours using 20 million noisy qubits"
- **Yan et al. (2022)**: "Factoring integers with sublinear resources on a superconducting quantum processor"
