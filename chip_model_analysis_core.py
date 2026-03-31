from __future__ import annotations

import numpy as np
import pandas as pd


def engagement_angle(radial_doc_mm: float, tool_diameter_mm: float) -> float:
    ae = min(max(radial_doc_mm, 1e-9), tool_diameter_mm)
    return float(np.arccos(1.0 - 2.0 * ae / tool_diameter_mm))


def chip_metrics_base(
    tool_diameter_mm: float,
    flutes: int,
    spindle_rpm: float,
    feed_mm_min: float,
    axial_doc_mm: float,
    radial_doc_mm: float,
) -> dict[str, float]:
    phi = engagement_angle(radial_doc_mm, tool_diameter_mm)
    fz = feed_mm_min / (spindle_rpm * flutes)
    h_max = fz * np.sin(phi)
    h_mean = fz * (1.0 - np.cos(phi)) / max(phi, 1e-9)
    mrr = feed_mm_min * axial_doc_mm * radial_doc_mm
    chip_contact_length = 0.5 * tool_diameter_mm * phi
    return {
        "fz_mm_per_tooth": fz,
        "h_max_mm": h_max,
        "h_mean_mm": h_mean,
        "mrr_mm3_per_min": mrr,
        "entry_angle_rad": phi,
        "chip_contact_length_mm": chip_contact_length,
    }


def stability_and_chatter(
    axial_doc_mm: float,
    radial_doc_mm: float,
    entry_angle_rad: float,
    perturb_freq_hz: float,
    spindle_rpm: float,
    flutes: int,
    h_mean_eff_mm: float,
    material: dict[str, float],
    machine: dict[str, float],
) -> dict[str, float | bool]:
    fn = machine["natural_freq_hz"]
    zeta = machine["damping_ratio"]
    k_n_per_mm = machine["stiffness_n_per_um"] * 1000.0
    ktc = material["ktc_n_per_mm2"]

    r = perturb_freq_hz / max(fn, 1e-9)
    dyn_gain = 1.0 / np.sqrt((1.0 - r**2) ** 2 + (2.0 * zeta * r) ** 2)
    dyn_gain = min(dyn_gain, 4.0)

    tooth_pass_hz = spindle_rpm * flutes / 60.0
    tooth_ratio = perturb_freq_hz / max(tooth_pass_hz, 1e-9)
    regen_factor = 1.0 + 0.35 * np.exp(-((tooth_ratio - 1.0) / 0.25) ** 2)

    directional = max(np.sin(entry_angle_rad), 0.12)
    ap_crit_mm = (2.0 * zeta * k_n_per_mm) / max(ktc * directional, 1e-9)

    h_ref = material["h_ref_mm"]
    chatter_risk = (axial_doc_mm / max(ap_crit_mm, 1e-9)) * (h_mean_eff_mm / max(h_ref, 1e-9)) * dyn_gain * regen_factor

    return {
        "dynamic_gain": dyn_gain,
        "tooth_pass_hz": tooth_pass_hz,
        "regen_factor": regen_factor,
        "ap_crit_mm": ap_crit_mm,
        "chatter_risk_index": chatter_risk,
        "is_stable": chatter_risk <= machine["risk_threshold"],
    }


def chip_metrics_with_frequency(
    tool_diameter_mm: float,
    flutes: int,
    spindle_rpm: float,
    feed_mm_min: float,
    axial_doc_mm: float,
    radial_doc_mm: float,
    perturb_amp_mm: float,
    perturb_freq_hz: float,
    material: dict[str, float],
    machine: dict[str, float],
) -> dict[str, float | bool]:
    base = chip_metrics_base(tool_diameter_mm, flutes, spindle_rpm, feed_mm_min, axial_doc_mm, radial_doc_mm)

    vf = feed_mm_min / 60.0
    f_c = machine["servo_cutoff_hz"]
    servo_gain = 1.0 / np.sqrt(1.0 + (perturb_freq_hz / max(f_c, 1e-9)) ** 2)
    amp_eff_mm = perturb_amp_mm * servo_gain

    v_osc_peak = 2.0 * np.pi * perturb_freq_hz * amp_eff_mm
    v_eff = np.sqrt(vf**2 + (v_osc_peak / np.sqrt(2.0)) ** 2)
    tooth_rate = spindle_rpm * flutes / 60.0
    fz_eff = v_eff / max(tooth_rate, 1e-9)

    phi = base["entry_angle_rad"]
    h_max_eff = fz_eff * np.sin(phi)
    h_mean_eff = fz_eff * (1.0 - np.cos(phi)) / max(phi, 1e-9)

    wavelength_mm = vf / max(perturb_freq_hz, 1e-9)
    chip_break_length_est_mm = wavelength_mm / 2.0
    chip_segment_length_est_mm = min(base["chip_contact_length_mm"], chip_break_length_est_mm)
    chip_size_proxy_mm2 = h_mean_eff * chip_segment_length_est_mm

    stab = stability_and_chatter(
        axial_doc_mm=axial_doc_mm,
        radial_doc_mm=radial_doc_mm,
        entry_angle_rad=phi,
        perturb_freq_hz=perturb_freq_hz,
        spindle_rpm=spindle_rpm,
        flutes=flutes,
        h_mean_eff_mm=h_mean_eff,
        material=material,
        machine=machine,
    )

    return {
        **base,
        **stab,
        "perturb_freq_hz": perturb_freq_hz,
        "perturb_amp_mm": perturb_amp_mm,
        "servo_gain": servo_gain,
        "amp_eff_mm": amp_eff_mm,
        "v_eff_mm_s": v_eff,
        "fz_eff_mm_per_tooth": fz_eff,
        "h_max_eff_mm": h_max_eff,
        "h_mean_eff_mm": h_mean_eff,
        "wavelength_mm": wavelength_mm,
        "chip_break_length_est_mm": chip_break_length_est_mm,
        "chip_segment_length_est_mm": chip_segment_length_est_mm,
        "chip_size_proxy_mm2": chip_size_proxy_mm2,
    }


def sweep_one(var_name: str, values: np.ndarray, baseline: dict[str, float]) -> pd.DataFrame:
    rows = []
    for v in values:
        p = dict(baseline)
        p[var_name] = float(v)
        rows.append({var_name: v, **chip_metrics_base(**p)})
    return pd.DataFrame(rows)


def path_length_multiplier(amplitude_mm: float, spatial_freq_cyc_per_mm: float, samples: int = 3000) -> float:
    if amplitude_mm <= 0.0 or spatial_freq_cyc_per_mm <= 0.0:
        return 1.0
    b = 2.0 * np.pi * amplitude_mm * spatial_freq_cyc_per_mm
    theta = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    return float(np.mean(np.sqrt(1.0 + (b * np.cos(theta)) ** 2)))


def chipload_time_series(
    base_process: dict[str, float],
    amp_mm: float,
    spatial_freq_cyc_per_mm: float,
    path_length_mm: float = 140.0,
    n: int = 3000,
) -> pd.DataFrame:
    phi = engagement_angle(base_process["radial_doc_mm"], base_process["tool_diameter_mm"])
    geom_factor = (1.0 - np.cos(phi)) / max(phi, 1e-9)

    vf = base_process["feed_mm_min"] / 60.0
    tooth_rate = base_process["spindle_rpm"] * base_process["flutes"] / 60.0

    s = np.linspace(0.0, path_length_mm, n)
    t = s / max(vf, 1e-9)

    dyds = 2.0 * np.pi * amp_mm * spatial_freq_cyc_per_mm * np.cos(2.0 * np.pi * spatial_freq_cyc_per_mm * s)
    v_base = np.full_like(s, vf)
    v_pert = vf * np.sqrt(1.0 + dyds**2)

    fz_base = v_base / max(tooth_rate, 1e-9)
    fz_pert = v_pert / max(tooth_rate, 1e-9)
    h_base = fz_base * geom_factor
    h_pert = fz_pert * geom_factor

    return pd.DataFrame(
        {
            "time_s": t,
            "path_s_mm": s,
            "h_mean_base_mm": h_base,
            "h_mean_pert_mm": h_pert,
            "dh_mm": h_pert - h_base,
            "dh_um": (h_pert - h_base) * 1000.0,
            "chipload_ratio": h_pert / np.maximum(h_base, 1e-12),
        }
    )


def feed_for_target_h_mean(target_h_mean_mm: float, process: dict[str, float]) -> float:
    phi = engagement_angle(process["radial_doc_mm"], process["tool_diameter_mm"])
    geom_factor = (1.0 - np.cos(phi)) / max(phi, 1e-9)
    fz_req = target_h_mean_mm / max(geom_factor, 1e-9)
    return fz_req * process["spindle_rpm"] * process["flutes"]


def feed_for_target_h_mean_with_perturbation(
    target_h_mean_mm: float, process: dict[str, float], amp_mm: float, spatial_freq_cyc_per_mm: float
) -> tuple[float, float]:
    kappa = np.sqrt(1.0 + 0.5 * (2.0 * np.pi * amp_mm * spatial_freq_cyc_per_mm) ** 2)
    target_base = target_h_mean_mm / max(kappa, 1e-9)
    return feed_for_target_h_mean(target_base, process), float(kappa)


def pareto_mask_minimize(df: pd.DataFrame, objective_cols: list[str]) -> np.ndarray:
    vals = df[objective_cols].to_numpy(dtype=float)
    n = vals.shape[0]
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        vi = vals[i]
        dominates_i = np.all(vals <= vi, axis=1) & np.any(vals < vi, axis=1)
        if np.any(dominates_i):
            is_pareto[i] = False
            continue
        dominated_by_i = np.all(vi <= vals, axis=1) & np.any(vi < vals, axis=1)
        dominated_by_i[i] = False
        is_pareto[dominated_by_i] = False
    return is_pareto
