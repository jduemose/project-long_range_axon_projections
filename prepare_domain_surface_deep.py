from pathlib import Path

import numpy as np
import torch

from brainnet.mesh.topology import DeepSurferTopology
import cortech


def shrink_surface(
    s,
    smooth_kwargs: list[dict] | None = None,
    decoupling_amount: float = 0.1,
    do_relax: bool = True,
    do_decouple: bool = True,
):
    """Shrink surface while doing tangential relaxation to avoid vertices
    piling up where the surface is shrunk a lot. The topology remains
    unchanged.
    """
    if smooth_kwargs is None:
        smooth_kwargs = [
            dict(time=1.0, n_iter=5, curv_threshold=0.2, ball_radius=0.5),
            dict(time=1.0, n_iter=5, curv_threshold=0.1, ball_radius=0.5),
            dict(time=1.0, n_iter=5, curv_threshold=0.05, ball_radius=0.5),
            dict(time=1.0, n_iter=5, curv_threshold=0.025, ball_radius=0.5),
            dict(time=1.0, n_iter=5, curv_threshold=0.0, ball_radius=0.5),
        ]

    for i, kw in enumerate(smooth_kwargs, 1):
        print(f"Iteration :: {i} of {len(smooth_kwargs)}")
        print(">> Smoothing")
        n_iter = kw.pop("n_iter")
        for j in range(1, n_iter + 1):
            print(f">> {j} of {n_iter}")
            s = s.smooth_shape_by_curvature_threshold(**kw)
            s = s.tangential_relaxation(n_iter=10)
            s = s.smooth_angle_and_area(n_iter=10, use_delaunay_flips=False)
            s = s.smooth_taubin(n_iter=5)

        if do_decouple:
            print(">> Decoupling")
            if i < len(smooth_kwargs):
                curv = s.compute_interpolated_corrected_curvatures()
                n = s.compute_vertex_normals()
                mask = curv.H <= 0.0
                s.vertices[mask] = s.vertices[mask] - decoupling_amount * n[mask]
                s = s.smooth_taubin(n_iter=10)
    return s


def remesh_for_shrink(s, roi, id_of_v, id_of_v_orig, remesh_kwargs):
    remesh_faces = np.flatnonzero(roi)
    s, vmap_, fmap = s.isotropic_remeshing_with_id(
        remesh_faces=remesh_faces, protect_constraints=True, **remesh_kwargs
    )
    roi = (fmap == -1) | np.isin(fmap, remesh_faces)
    mask = np.isin(vmap_, id_of_v)
    id_of_v = np.flatnonzero(mask)
    id_of_v_orig = vmap_[id_of_v]

    remesh_faces = np.flatnonzero(~roi)
    s, vmap_, fmap = s.isotropic_remeshing_with_id(
        remesh_faces=remesh_faces, protect_constraints=True, **remesh_kwargs
    )
    roi = ~((fmap == -1) | np.isin(fmap, remesh_faces))
    id_of_v = np.flatnonzero(np.isin(vmap_, id_of_v))
    id_of_v_orig = vmap_[id_of_v]

    return s, roi, id_of_v, id_of_v_orig


def shrink_surface_with_remeshing(
    s,
    remesh_faces,
    smooth_kwargs: list[dict] | None = None,
    remesh_kwargs: dict | None = None,
    decoupling_amount: float = 0.1,
    do_relax: bool = True,
    do_remesh: bool = False,
    do_decouple: bool = True,
):
    """Keep track of the ROI/non-ROI and the vertices at the edge while remeshing."""
    if smooth_kwargs is None:
        smooth_kwargs = [
            dict(time=1.0, n_iter=5, curv_threshold=0.2, ball_radius=0.5),
            dict(time=1.0, n_iter=5, curv_threshold=0.1, ball_radius=0.5),
            dict(time=1.0, n_iter=5, curv_threshold=0.05, ball_radius=0.5),
            dict(time=1.0, n_iter=5, curv_threshold=0.025, ball_radius=0.5),
            dict(time=1.0, n_iter=5, curv_threshold=0.0, ball_radius=0.5),
        ]

    id_of_v = border_v.copy()
    id_of_v_orig = id_of_v.copy()
    roi = np.zeros(s.n_faces, bool)
    roi[remesh_faces] = True

    if remesh_kwargs is None:
        remesh_kwargs = dict(
            target_edge_length=2 * s.compute_edge_norm().mean(), n_iter=5
        )

    s, roi, id_of_v, id_of_v_orig = remesh_for_shrink(
        s, roi, id_of_v, id_of_v_orig, remesh_kwargs
    )

    for i, kw in enumerate(smooth_kwargs, 1):
        print(f"Iteration :: {i} of {len(smooth_kwargs)}", flush=True)

        print(">> Smoothing", flush=True)
        s = s.smooth_shape_by_curvature_threshold(**kw)
        if do_remesh:
            print(">> Remeshing", flush=True)
            s, roi, id_of_v, id_of_v_orig = remesh_for_shrink(
                s, roi, id_of_v, id_of_v_orig, remesh_kwargs
            )

        if do_decouple:
            if i < len(smooth_kwargs):
                curv = s.compute_interpolated_corrected_curvatures()
                n = s.compute_vertex_normals()
                mask = curv.H <= 0.0
                s.vertices[mask] = s.vertices[mask] - decoupling_amount * n[mask]
                s = s.smooth_taubin(n_iter=10)

    return s, roi, id_of_v


def make_deep_surface():
    bungert_dir = Path("/mnt/projects/INN/bungert_revisited")
    data_dir = Path("/mnt/projects/INN/jesper/nobackup/projects/white_matter_axons")

    m2m_dir = Path(bungert_dir / "subject_5" / "m2m_subject_5")
    wm = cortech.Surface.from_file(m2m_dir / "surfaces" / "lh.white.gii")

    # subsample wm to speed up shrinking
    faces = np.load(data_dir / "deepsurfer_faces.npz")
    wm5 = cortech.Surface(wm.vertices[: faces["5"].max() + 1], faces["5"])

    s = shrink_surface(wm5, decoupling_amount=0.25)

    # upsample to full resolution
    t = DeepSurferTopology.recursive_subdivision(5)[-1]
    v = t.subdivide_vertices(torch.tensor(s.vertices)[None])
    deep = cortech.Surface(v.squeeze(), wm.faces)

    # deep = cortech.Surface.from_file("inner_surface_upsampled.vtk")
    roi_mask = np.load(data_dir / "roi_surface_mask.npy")
    deep = deep.remove_vertices(~roi_mask)
    deep.save(data_dir / "roi_surface_deep.vtk")

    # x = np.unique(wm.find_border_edges())
    # roi_vertices = np.flatnonzero()
    # border_v = roi_vertices[x]
    # remesh_faces = np.isin(wm.faces, roi_vertices).all(1)
    # s = shrink_surface_with_remeshing(wm, )
