from pathlib import Path
import nibabel as nib
import numpy as np

import cortech
import simnibs
from simnibs.utils.transformations import cross_subject_map


def make_roi_surface(m2m_dir, f_mask, surface_type):
    surface_roi = simnibs.RegionOfInterest()
    surface_roi.subpath = m2m_dir
    surface_roi.method = "surface"
    surface_roi.surface_type = surface_type
    # path to mask in fsaverage space
    surface_roi.mask_path = f_mask
    surface_roi.mask_space = "fs_avg_lh"
    surface_roi._prepare()
    roi = surface_roi._mesh
    roi.add_node_field(surface_roi._mask, "ROI")
    # roi = roi.crop_mesh(nodes=np.where(roi.field["ROI"].value)[0] + 1)
    return roi


def make_roi_surface_custom(m2m_dir, file_mask_label, hemi):
    morph = cross_subject_map(
        "fsaverage", m2m_dir, hemi=hemi, project_kwargs=dict(method="nearest")
    )
    indices = nib.freesurfer.read_label(file_mask_label)
    mask_fsaverage = np.zeros(morph[hemi]._mapping_matrix.shape[1], dtype=int)
    mask_fsaverage[indices] = 1
    return morph[hemi].resample(mask_fsaverage).astype(bool)


def initialize_surface_roi():
    bungert_dir = Path("/mnt/projects/INN/bungert_revisited")
    output_dir = Path("/mnt/projects/INN/jesper/nobackup/projects/white_matter_axons")
    # output_dir = Path("/mnt/scratch/personal/jesperdn/subject_5")

    m2m_dir = Path(bungert_dir / "subject_5" / "m2m_subject_5")

    f_mask = bungert_dir / "scripts" / "resources" / "P1_LH_M1.label"

    roi_mask = make_roi_surface_custom(m2m_dir, f_mask, "lh")
    np.save(output_dir / "roi_surface_mask.npy", roi_mask)

    wm = cortech.Surface.from_file(m2m_dir / "surfaces" / "lh.white.gii")
    gm = cortech.Surface.from_file(m2m_dir / "surfaces" / "lh.pial.gii")

    wm = wm.remove_vertices(~roi_mask)
    gm = gm.remove_vertices(~roi_mask)

    wm.save(output_dir / "roi_surface_wm.vtk")
    gm.save(output_dir / "roi_surface_gm.vtk")

    return wm, gm
