import argparse
from pathlib import Path
import tempfile
import time
from neuron import h
import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

from simnibs.neurosimnibs.neurosim.neuron_cell import NeuronCell, SectionType
from simnibs.neurosimnibs.neurosim.mechanisms_util import load_mechanisms
from simnibs.neurosimnibs.neurosim.simulation.waveform import Waveform, WaveformType
from simnibs.neurosimnibs.neurosim.simulation.threshold_factor_simulation import (
    ThresholdFactorSimulation,
)

import sys

sys.path.append("/home/jesperdn/repositories/project-long_range_axon_projections")

import io_utils
from torge_script.neuron_builder import Section, NeuronBuilder

global PATH_NEURON_DATASETS
PATH_NEURON_DATASETS = Path("/mrhome/jesperdn/repositories/neuron_datasets")


def crop_efield_mesh(efield_vtk, simulation_dir):
    """Crop the whole-head electrical field to the domain of interest."""
    bb = pv.read(simulation_dir / "laplace_field.vtm").bounds
    sim_mesh = pv.read(efield_vtk)

    sim_mesh = sim_mesh.cell_data_to_point_data()
    sim_mesh = sim_mesh.clip_box(
        [bb[0] - 3, bb[1] + 3, bb[2] - 3, bb[3] + 3, bb[4] - 6, bb[5] + 6],
        crinkle=True,
        invert=False,
    )
    sim_mesh = sim_mesh.threshold(
        value=[1, 2], scalars="tag", invert=False, all_scalars=False
    )
    return sim_mesh


# CROP E-FIELDS TO DOMAIN...

# subject_dir = Path("/mnt/projects/INN/bungert_revisited/subject_5")
# subject_sim = subject_dir / "Experiments" / "Simulations"
# domain_dir = Path(
#     "/mnt/projects/INN/jesper/nobackup/projects/white_matter_axons/domain_M1"
# )
# simulation_dir = domain_dir / "simulation_02-02"

# res = crop_efield_mesh(subject_sim / "fdi_min_threshold" / f"{subject_dir.name}_ROI_P1_LH_M1_TMS_1-0001_MagVenture_MC-B70_dir.vtk"ame, simulation_dir)
# res.save(domain_dir / "domain_efield_smooth_cond.vtu")

# res = crop_efield_mesh(
#     subject_sim / "fdi_min_threshold_isotropic" / f"{subject_dir.name}_ROI_P1_LH_M1_TMS_1-0001_MagVenture_MC-B70_scalar.vtk", simulation_dir
# )
# res.save(domain_dir / "domain_efield_default_cond.vtu")


def make_myelinated_axon(
    axon,
    initial_unmyelinated_segment_length,
    cap_unmyelinated_segment_length,
    axon_diameter,
):
    internode_length_mm = (axon_diameter * 100) / 1000
    node_length_mm = 1 / 1000
    points = axon.points
    # Find the point of maximum curvature/minimum bend radius
    radius = axon["radius"]
    invalid = np.isnan(radius)
    index_of_mid_bend = np.ma.MaskedArray(radius, invalid).argmin()

    s = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    d_from_start = s.copy().tolist()

    d_to_mid_bend = d_from_start[index_of_mid_bend]
    myelinated_length_to_bend = d_to_mid_bend - initial_unmyelinated_segment_length
    remainder_myelinated_length = myelinated_length_to_bend % (
        node_length_mm + internode_length_mm
    )
    corrected_initial_unmyelinated_segment_length = (
        initial_unmyelinated_segment_length + remainder_myelinated_length
    )

    section_types = []

    # unmyelinated initial segment
    myelin_start = s[
        np.where((s > corrected_initial_unmyelinated_segment_length))[0][0]
    ]
    myelin_end = s[np.where(s > np.max(s) - cap_unmyelinated_segment_length)[0][0]]

    for distance in s[(s <= myelin_start)]:
        section_types.append(SectionType.PASSIVE_AXON)

    d_from_start.append(myelin_start)
    section_types.append(SectionType.PASSIVE_AXON)

    # myelinated part
    alternating_type = [SectionType.NODE_OF_RANVIER, SectionType.MYELINATED_AXON]
    alternating_length = [node_length_mm, internode_length_mm]

    current_distance = myelin_start
    alternating_index = 0
    while current_distance < myelin_end:
        next_distance = current_distance + alternating_length[alternating_index]

        for distance in s[(s <= next_distance) & (s > current_distance)]:
            section_types.append(alternating_type[alternating_index])

        d_from_start.append(next_distance)
        section_types.append(alternating_type[alternating_index])

        alternating_index = (alternating_index + 1) % 2
        current_distance = next_distance

    # unmyelinated end segment
    # distance_from_start.append(unmyelinated_segment_start)
    # section_types.append(SectionType.AXON)
    for distance in s[s >= current_distance]:
        section_types.append(SectionType.PASSIVE_AXON)

    d_from_start.sort()
    idx = np.searchsorted(s, d_from_start, side="right") - 1
    idx = np.clip(idx, 0, len(points) - 2)
    t = (d_from_start - s[idx]) / (s[idx + 1] - s[idx])
    path = (1 - t)[:, None] * points[idx] + t[:, None] * points[idx + 1]

    sections = []
    current_points = []
    current_section_type = section_types[0]
    parent = None
    for point, section_type in zip(path, section_types):
        if current_section_type != section_type:
            sections.append(
                Section(
                    current_section_type,
                    parent,
                    points=np.array(current_points),
                    diameter=np.full((len(current_points)), axon_diameter),
                )
            )
            parent = sections[-1]
            current_points = [] if parent is None else [parent.points[-1]]
            current_section_type = section_type
        current_points.append(point * 1000)

    sections.append(
        Section(
            current_section_type,
            parent,
            points=np.array(current_points),
            diameter=np.full((len(current_points)), axon_diameter),
        )
    )
    return sections


def sample_field(points, mesh, scalar):
    sampled = pv.PolyData(points).sample(mesh)

    if not np.all(sampled["vtkValidPointMask"]):
        tree = KDTree(mesh.points)
        _, idx = tree.query(points[sampled["vtkValidPointMask"] == 0])
        sampled[scalar][sampled["vtkValidPointMask"] == 0] = mesh[scalar][idx]

    return sampled[scalar]


def simulate_axon(axon, sim_mesh, diameter, field_direction, waveform_type):
    load_mechanisms(
        PATH_NEURON_DATASETS / "BBRatToHuman" / "Aberra_BrainStim_2020" / "mechanisms"
    )
    waveform = Waveform(WaveformType[waveform_type.upper()])

    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        sections = make_myelinated_axon(axon, 1.0, 1.0, diameter)

        bp_path = (
            PATH_NEURON_DATASETS
            / "BBRatToHuman"
            / "biophysics"
            / "L5_TTPC2_cADpyr.json"
        )
        NeuronBuilder(sections).save(f.name, biophysics_path=str(bp_path))

        neuron = NeuronCell(f.name)
        neuron.chunk_sizes[SectionType.AXON] = 5
        neuron.chunk_sizes[SectionType.MYELINATED_AXON] = 5
        neuron.chunk_sizes[SectionType.NODE_OF_RANVIER] = 5
        neuron.load()

        # MagVenture coils are optimized for biphasic pulses, for monophasic
        # pulses the e field needs to be inverted
        print(f"Field direction {field_direction}")
        print(f"Inverting field : {field_direction == 'pa'}")
        e_field = sample_field(
            neuron.get_segment_coordinates_flat() / 1000, sim_mesh, "e_brain"
        )
        e_field = -e_field if field_direction == "pa" else e_field

        # labels = []
        # diam = []
        # for section, section_type in zip(neuron.sections, neuron.section_types):
        #    for segment in section:
        #        labels.append(section_type)
        #        diam.append(segment.diam)
        # poly = pv.PolyData(neuron.get_segment_coordinates_flat() / 1000)
        # poly["E"] = e_field
        # poly["labels"] = labels
        # poly["diam"] = diam
        # poly["AP"] = np.full(poly.n_points, np.inf)

        sim = ThresholdFactorSimulation(neuron, waveform)
        sim.attach()
        sim.apply_e_field(e_field)
        threshold, ids, threshold_time = sim.find_threshold_factor(400)  # or higher
        sim.detach()

        ids = np.array(ids, dtype=np.int64)
        threshold_time = np.array(threshold_time)

        min_time_ids = ids[threshold_time == np.min(threshold_time)]
        i = 0
        ap_distances = []
        for section in neuron.sections:
            for segment in section:
                if i in min_time_ids:
                    ap_distances.append(h.distance(neuron.sections[0](0), segment))
                i += 1
        neuron.unload()

    # print(threshold, np.min(ap_distances), np.max(ap_distances), np.median(ap_distances))

    # print(neuron.get_segment_coordinates_flat()[int(ids[0])] / 1000)
    # poly["AP"][np.array(ids, dtype=np.int64)] = time
    # poly.save(f"/mrhome/torgehw/Documents/Projects/bungert_revisited/subject_5/Axons/axon_test_e_{angle}.vtk")
    # print("saved")

    return threshold, np.median(ap_distances)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subject", type=int, help="The subject number (default: 5)")
    parser.add_argument("index", type=int)
    parser.add_argument(
        "field-direction", choices=["ap", "pa"], help="Direction of the field"
    )
    parser.add_argument(
        "waveform", type=str, help="Waveform string (e.g., monophasic, biphasic)"
    )
    parser.add_argument("cond", type=str, help="Conductivity (e.g., smooth, default)")
    args = parser.parse_args()

    print("Settings")
    print("----------------")
    for k, v in vars(args).items():
        print(k, v)
    print("----------------")

    OFFSET = "02"  # offset from wm/gm interface
    COND_RATIO = "02"  # gm/wm conductivity ratio
    # diameters = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    diameters = (2.0,)
    # diameters = (6.0,)

    bungert_dir = Path("/mnt/projects/INN/bungert_revisited")
    domain_dir = Path(
        "/mnt/projects/INN/jesper/nobackup/projects/white_matter_axons/domain_M1"
    )
    simu_dir = domain_dir / f"simulation_{OFFSET}-{COND_RATIO}"
    out_dir = Path("/mnt/scratch/personal/jesperdn/neuron_simulations/M1")

    field_dir = getattr(args, "field-direction")

    subject = f"subject_{args.subject}"
    subject_dir = bungert_dir / subject
    results_dir = (
        out_dir
        / f"waveform-{args.waveform.lower()}_direction-{field_dir}_cond-{args.cond}"  # cond-{args.domain_efield}
    )
    filename_projections = simu_dir / "projections.h5"

    time.sleep(np.random.rand())  # jitter
    if not results_dir.exists():
        results_dir.mkdir(parents=True)

    print("Loading axon data")
    points, vd, ld = io_utils.projections_load_single(filename_projections, args.index)
    axon = pv.MultipleLines(points)
    axon.point_data.update(vd)
    axon.cell_data.update(ld)

    tmp = np.zeros(axon.n_points, bool)
    if axon["valid_projection"]:
        print("Projection is valid")
        print("Loading E-field")
        sim_mesh = pv.read(domain_dir / f"domain_efield_{args.cond}_cond.vtu")
        for diameter in diameters:
            print(f"Simulation : {diameter:.2f}")
            threshold, ap_distance = simulate_axon(
                axon, sim_mesh, diameter, field_dir, args.waveform
            )
            print(f"Threshold   : {threshold:.2f}")
            print(f"AP distance : {ap_distance:.2f}")

            # Estimate the segment where the action potential initialized
            diffs = np.concatenate(
                [[0.0], np.linalg.norm(np.diff(axon.points, axis=0), axis=1)]
            )
            index = np.abs(np.cumsum(diffs) - ap_distance * 0.001).argmin()
            tmp[index] = True

            axon[f"diameter-{diameter:.2f}:ap-distance"] = ap_distance
            axon[f"diameter-{diameter:.2f}:ap-index"] = tmp
            axon[f"diameter-{diameter:.2f}:threshold"] = threshold

    else:
        print("Projection is not valid (setting NaNs)")
        for diameter in diameters:
            axon[f"diameter-{diameter:.2f}:ap-distance"] = np.nan
            axon[f"diameter-{diameter:.2f}:ap-index"] = tmp
            axon[f"diameter-{diameter:.2f}:threshold"] = np.nan

    f = results_dir / f"projection_{args.index:05d}.vtp"
    axon.save(f)
    print(f)

    # roi_indexes_result = []
    # roi_threshold_result = []
    # roi_ap_distance_result = []
    # with Pool(processes=args.processes) as pool:
    #     for res in tqdm(
    #         pool.imap_unordered(generate_axons_at_angle_func, ),
    #         total=len(surface_roi_indexes),
    #     ):
    #         roi_indexes_result.append(res[0])
    #         roi_threshold_result.append(res[1])
    #         roi_ap_distance_result.append(res[2])

    # for i, angle in enumerate(range(0, 360, 20)):
    #     parallel_surface[f"threshold_{angle}"] = np.zeros(parallel_surface.n_points)
    #     parallel_surface[f"threshold_{angle}"][np.array(roi_indexes_result)] = np.array(
    #         roi_threshold_result
    #     )[:, i]

    #     parallel_surface[f"ap_distance_{angle}"] = np.zeros(parallel_surface.n_points)
    #     parallel_surface[f"ap_distance_{angle}"][np.array(roi_indexes_result)] = (
    #         np.array(roi_ap_distance_result)[:, i]
    #     )
