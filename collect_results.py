from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.optimize import least_squares
from scipy.signal import lfilter

import io_utils

from simnibs.neurosimnibs.neurosim.simulation.waveform import Waveform, WaveformType

global C_TMS
C_TMS = ("c_tms_30", "c_tms_60", "c_tms_120")
global C_TMS_WAVEFORMS
C_TMS_WAVEFORMS = {w.lower(): Waveform(getattr(WaveformType, w.upper())) for w in C_TMS}


def threshold_string(d):
    return f"diameter-{d:.2f}"


def est_time_constant(white):
    # durations
    pws = [0.03, 0.06, 0.12]
    wvfrms = [w.e_field_magnitude for w in C_TMS_WAVEFORMS.values()]
    times = C_TMS_WAVEFORMS[C_TMS[0]].time  # all times are the same so just use 30
    fs = 1e3 / (times[1] - times[0])  # sampling frequency in Hz

    wvfrms = np.array(wvfrms).T
    # thresholds
    # mt = [white[f"threshold:{i}.ap.smooth.6"] for i in C_TMS]
    tau_m0 = 200e-6
    tau_m_lb = 10e-6
    tau_m_ub = 5e-2
    mt_b0 = 30
    mt_b_lb = 1
    mt_b_ub = 5000

    def mt_mtcalc(param, pws, fs):
        tau_m, mt_b = param
        n_pws = len(pws)
        mt_mt_vals = np.zeros(n_pws)
        b = np.array([1, 1]) / (1 + 2 * tau_m * fs)
        a = np.array([1, (1 - 2 * tau_m * fs) / (1 + 2 * tau_m * fs)])
        for i in range(n_pws):
            filtered = lfilter(b, a, wvfrms[:, i])
            mt_mt_vals[i] = mt_b / mt[i] / np.max(filtered)
        return mt_mt_vals

    def residuals(param):
        return mt_mtcalc(param, pws, fs) - np.ones_like(mt)

    # Initial guess and bounds
    x0 = [tau_m0, mt_b0]
    bounds = ([tau_m_lb, mt_b_lb], [tau_m_ub, mt_b_ub])

    print("Estimating time constants...")

    taus, rbs, resnorms = [], [], []
    for i in range(white.n_points):
        if i % 1000 == 0:
            print(f"{i} of {white.n_points}")
        mt = [white[f"threshold:{w}.pa.smooth.6"][i] for w in C_TMS]

        # Curve fitting using least squares
        result = least_squares(residuals, x0, bounds=bounds, method="trf")

        tau = result.x[0] * 1e6  # convert seconds to microseconds
        rb = result.x[1]
        resnorm = np.sum(result.fun**2)
        taus.append(tau)
        rbs.append(rb)
        resnorms.append(resnorm)

    return taus, rbs, resnorms


if __name__ == "__main__":
    FIG_DIR = Path(
        "/mnt/projects/INN/jesper/nobackup/projects/white_matter_axons/domain_M1/simulation_02-02/figures/"
    )

    # Load data
    data_dir = Path(
        "/mnt/projects/INN/jesper/nobackup/projects/white_matter_axons/domain_M1"
    )
    surf_dir = data_dir / "surfaces_domain"

    # wm = cortech.Surface.from_file(surf_dir / "domain-white.vtk")
    # gm = cortech.Surface.from_file(surf_dir / "domain-gray.vtk")
    # white = pv.make_tri_mesh(wm.vertices, wm.faces)
    # gray = pv.make_tri_mesh(gm.vertices, gm.faces)

    white = pv.read(surf_dir / "domain-white.vtk")
    gray = pv.read(surf_dir / "domain-gray.vtk")

    offset = 0.2
    cond_ratio = 0.2
    settings_str = "-".join([str(i).replace(".", "") for i in (offset, cond_ratio)])

    simulation_dir = data_dir / f"simulation_{settings_str}"
    neuron_dir = simulation_dir / "neuron_simulations" / "M1"

    points, vertex_data, line_data = io_utils.projections_load_all(
        simulation_dir / "projections.h5"
    )

    orig_index = line_data["original_index"]  # [line_data["valid_projection"]]

    WAVEFORMS = ["biphasic", "monophasic", "c_tms_30", "c_tms_60", "c_tms_120"]
    directions = ["ap", "pa"]
    diameters = [1, 2, 3, 4, 5, 6]
    conds = ["smooth", "default"]

    # THRESHOLDS
    clims = {}
    for w in WAVEFORMS:
        for fd in directions:
            for cond in conds:
                _name = f"waveform-{w}_direction-{fd}_cond-{cond}"
                res_dir = neuron_dir / _name
                if not res_dir.exists():
                    continue

                print(f"{w:10s} {fd:10s} {cond:10s}")
                mb = pv.MultiBlock(
                    [pv.read(i) for i in sorted(res_dir.glob("projection*.vtp"))]
                )
                for d in diameters:
                    # name = outstr.format(waveform=w, direction=fd, diameter=d)
                    name = f"threshold:{w}.{fd}.{cond}.{d}"
                    if f"{threshold_string(d):s}:threshold" not in mb[0].cell_data:
                        continue
                    print(f"diameter: {d}")
                    thresholds = np.array(
                        [i[f"{threshold_string(d):s}:threshold"][0] for i in mb]
                    )

                    v = np.full(white.n_points, np.nan)  # , dtype=np.float32)
                    v[orig_index] = thresholds
                    white[name] = v

                    name = f"ap-distance:{w}.{fd}.{cond}.{d}"
                    ap_distance = np.array(
                        [i[f"{threshold_string(d):s}:ap-distance"][0] for i in mb]
                    )
                    v = np.full(white.n_points, np.nan)  # , dtype=np.float32)
                    v[orig_index] = ap_distance
                    white[name] = v

                    name = f"ap-index:{w}.{fd}.{cond}.{d}"
                    ap_distance = np.array(
                        [i[f"{threshold_string(d):s}:ap-index"].argmax() for i in mb]
                    )
                    v = np.full(white.n_points, np.nan)  # , dtype=np.float32)
                    v[orig_index] = ap_distance
                    white[name] = v

    # THRESHOLD AP/PA COMPARISON
    for d in diameters:
        white[f"difference:monophasic.ap-pa.smooth.{d}"] = np.log10(
            white[f"threshold:monophasic.ap.smooth.{d}"]
            / white[f"threshold:monophasic.pa.smooth.{d}"]
        )

    # THRESHOLD AP/PA COMPARISON - biphasic
    for d in (2,):
        white[f"difference:biphasic.ap-pa.smooth.{d}"] = np.log10(
            white[f"threshold:biphasic.ap.smooth.{d}"]
            / white[f"threshold:biphasic.pa.smooth.{d}"]
        )

    # CONDUCTIVITIES: SMOOTH vs. DEFAULT
    white["difference:monophasic.ap.default-smooth.2"] = (
        white["threshold:monophasic.ap.default.2"]
        - white["threshold:monophasic.ap.smooth.2"]
    ) / (
        white["threshold:monophasic.ap.default.2"]
        + white["threshold:monophasic.ap.smooth.2"]
    )

    white["difference:monophasic.pa.default-smooth.2"] = (
        white["threshold:monophasic.pa.default.2"]
        - white["threshold:monophasic.pa.smooth.2"]
    ) / (
        white["threshold:monophasic.pa.default.2"]
        + white["threshold:monophasic.pa.smooth.2"]
    )

    # Interpolate E field magnitude to point of maximum curvature
    for cond in conds:
        mesh = pv.read(data_dir / f"domain_efield_{cond}_cond.vtu")

        max_curv_idx = np.ma.MaskedArray(
            vertex_data["curv"], ~vertex_data["valid_point"]
        ).argmax(1)
        max_curv_point = points[np.arange(len(points)), max_curv_idx]
        max_curv_magnE = pv.PolyData(max_curv_point).interpolate(mesh)

        v = np.full(white.n_points, np.nan)
        v[orig_index] = max_curv_magnE["magnE"]
        white[f"magnE:{cond}"] = v

    white["difference:magnE:default-smooth"] = (
        white["magnE:default"] - white["magnE:smooth"]
    )  # / (white["magnE:default"] + white["magnE:smooth"])

    not_nan = ~np.isnan(white["threshold:monophasic.ap.smooth.2"])
    white_incl_nan = white.copy()
    white = white.threshold(
        scalars="threshold:monophasic.ap.default.2", all_scalars=True
    )

    tau, rb, resnorm = est_time_constant(white)

    white["time_constant:threshold:c_tms.pa.smooth.6"] = tau
    white["rheobase:threshold:c_tms.pa.smooth.6"] = rb
    white["residual:threshold:c_tms.pa.smooth.6"] = resnorm

    print(f"saving {FIG_DIR / 'white.vtk'}")
    white.save(FIG_DIR / "white.vtk")

    print(f"saving {FIG_DIR / 'white_incl_nan.vtk'}")
    white_incl_nan.save(FIG_DIR / "white_incl_nan.vtk")
