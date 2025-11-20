from pathlib import Path

import scipy.ndimage
import simnibs
from simnibs import ElementTags
from simnibs.utils.transformations import cross_subject_map

from scipy.spatial import KDTree
import numpy as np
import nibabel as nib
import cortech


def fix_roi_image(img):
    # dilate the image roi a bit so avoid removing tetrahedra at the boundary
    x = img.get_fdata()

    for i in range(1):
        x = scipy.ndimage.binary_dilation(x)
        labels, num = scipy.ndimage.label(x)
        x = labels == 1
        x = x

    return nib.Nifti1Image(x.astype(np.uint8), img.affine)


def make_roi_volume(f_mesh, f_mask):
    roi = simnibs.RegionOfInterest()
    roi.method = "volume"
    # roi.subpath = "/mnt/projects/INN/bungert_revisited/subject_5/m2m_subject_5"
    roi.mesh = f_mesh
    roi.mask_space = "subject"
    roi.mask_path = f_mask
    # roi.mask_type = "node"
    roi.tissues = [ElementTags.WM, ElementTags.GM]
    return roi.get_roi_mesh()


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

if __name__ == "__main__":
    bungert_dir = Path("/mnt/projects/INN/bungert_revisited/subject_5")
    scratch_dir = Path("/mnt/scratch/personal/jesperdn/subject_5")

    img = nib.load(bungert_dir / "m2m_subject_5" / "ROI_P1_LH_M1/P1_LH_M1_depth.nii.gz")
    # img = fix_roi_image(img)
    img.to_filename(scratch_dir / "ROI.nii.gz")

    # Volume ROI
    f_mesh = (
        bungert_dir
        / "m2m_subject_5"
        / "ROI_P1_LH_M1"
        / "cond_wm_volume_normalized"
        / "cond_wm_volume_normalized_smoothed.msh"
    )
    # f_mesh = bungert_dir / "m2m_subject_5" / "subject_5.msh"
    f_mask = scratch_dir / "ROI.nii.gz"
    roi = make_roi_volume(f_mesh, f_mask)
    roi.save(str(scratch_dir / "roi_volume.msh"))
    roi.to_multiblock().save(scratch_dir / "roi_volume.vtm")
