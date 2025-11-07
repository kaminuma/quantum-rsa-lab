"""Experiment utilities for collecting Shor run metrics into pandas DataFrames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from .shor_demo import DEFAULT_N_COUNT, run_shor

try:  # pragma: no cover - typing 補助用
    from qiskit_aer.noise import NoiseModel
except ImportError:  # pragma: no cover
    NoiseModel = object  # type: ignore[misc,assignment]


@dataclass(frozen=True)
class QuantumRunSetting:
    """Conditions for a sweep (noise / mitigation)."""

    label: str
    noise_model: NoiseModel | None = None
    apply_readout_mitigation: bool = False
    meas_calibration_shots: int = 2048
    simulator_options: Optional[dict] = None


def sweep_shot_counts(
    number: int,
    base: int,
    shots_list: Sequence[int],
    n_count: int = DEFAULT_N_COUNT,
    repeats: int = 1,
    settings: Optional[Sequence[QuantumRunSetting]] = None,
    default_simulator_options: Optional[dict] = None,
) -> pd.DataFrame:
    """Run quantum Shor repeatedly and collect metrics into a DataFrame.

    Parameters
    ----------
    number : int
        Composite number to factor.
    base : int
        Base "a" used in modular exponentiation.
    shots_list : Sequence[int]
        Shot counts evaluated for the sweep.
    n_count : int
        Number of counting qubits for QPE precision.
    repeats : int
        How many times to repeat each (setting, shots) pair.
    settings : Sequence[QuantumRunSetting]
        Noise / mitigation scenarios. Defaults to [ideal].
    default_simulator_options : dict
        Global AerSimulator options (merged with per-setting overrides).
    """

    if settings is None:
        settings = (QuantumRunSetting(label="ideal"),)

    records: list[dict] = []

    for setting in settings:
        for shots in shots_list:
            for repeat in range(repeats):
                simulator_options = _merge_dicts(default_simulator_options, setting.simulator_options)
                result = run_shor(
                    number=number,
                    base=base,
                    method="quantum",
                    shots=shots,
                    n_count=n_count,
                    noise_model=setting.noise_model,
                    apply_readout_mitigation=setting.apply_readout_mitigation,
                    meas_calibration_shots=setting.meas_calibration_shots,
                    simulator_options=simulator_options,
                )

                records.append(
                    {
                        "label": setting.label,
                        "number": number,
                        "base": result.base,
                        "shots": shots,
                        "repeat": repeat,
                        "success": result.success,
                        "period": result.period,
                        "measured_phase": result.measured_phase,
                        "method": result.method,
                        "n_count": n_count,
                        "noise_model_name": result.noise_model_name,
                        "mitigated": result.mitigated_counts is not None,
                        "circuit_depth": result.circuit_depth,
                        "total_gates": result.total_gates,
                        "two_qubit_gates": result.two_qubit_gates,
                        "counts": result.counts,
                        "mitigated_counts": result.mitigated_counts,
                    }
                )

    return pd.DataFrame.from_records(records)


def summarize_success(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate success probability per label/shot pair."""

    if df.empty:
        return df

    summary = (
        df.groupby(["label", "shots"], as_index=False)["success"].mean().rename(columns={"success": "success_rate"})
    )
    return summary


def _merge_dicts(base: Optional[dict], extra: Optional[dict]) -> Optional[dict]:
    if base is None and extra is None:
        return None
    merged = dict(base or {})
    if extra:
        merged.update(extra)
    return merged
