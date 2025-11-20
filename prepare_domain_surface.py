import functools
from pathlib import Path

import numpy as np

import pyvista as pv
import scipy.spatial

import cortech
import cortech.cgal.mesh_3


def build_multiblock(v, f, cells, fmap, cmap):
    mb = pv.MultiBlock()

    mb["TRIANGLE"] = pv.make_tri_mesh(v, f)
    mb["TRIANGLE"]["map"] = fmap
    mb["TETRAHEDRON"] = pv.UnstructuredGrid(
        np.concatenate((np.full((len(cells), 1), 4), cells), axis=1).ravel(),
        np.full(len(cells), pv.CellType.TETRA, dtype=np.uint8),
        v,
    )
    mb["TETRAHEDRON"]["map"] = cmap
    return mb


def make_mesh3_complex(
    surfaces,
    indicent_subdomains,
    cell_size=None,
    edge_size=None,
    facet_distance=None,
    facet_size=None,
    factor=1.0,
):
    mean_edge_length = np.mean([s.compute_edge_norm().mean() for s in surfaces])
    mean_edge_length = mean_edge_length * factor
    mesh_kw = dict(
        cell_radius_edge_ratio=2,
        cell_size=cell_size or 2 * mean_edge_length,
        edge_size=edge_size or mean_edge_length,
        facet_angle=25,
        facet_distance=facet_distance or 0.01,
        facet_size=facet_size or 2 * mean_edge_length,
    )

    return cortech.cgal.mesh_3.make_mesh_complex(
        [s.vertices for s in surfaces],
        [s.faces for s in surfaces],
        indicent_subdomains,
        **mesh_kw,
    )


def remesh_patch(s, vmap, fmap, fmap_id, target_edge_length, n_iter=10):
    # remesh only the patch defined by `fmap_id`
    faces_to_remesh = np.flatnonzero(fmap == fmap_id)
    sr, v_id, f_id = s.isotropic_remeshing_with_id(
        target_edge_length, n_iter, faces_to_remesh, protect_constraints=True
    )
    # CGAL modifies the order of the vertices, so sort the vertices as before
    # the remeshing (sorting of new vertices is arbitrary)
    asort = np.argsort(v_id)
    v_id_sorted = v_id[asort]
    sr.vertices = sr.vertices[asort]
    sr.faces = np.argsort(asort)[sr.faces]

    vmap = vmap[v_id_sorted]
    vmap[v_id_sorted == -1] = fmap_id
    fmap = fmap[f_id]
    fmap[f_id == -1] = fmap_id

    return sr, vmap, fmap


def get_fair_indices(s, vmap, vmap_id, vmap_id_no_touch=None, nn=3):
    # get the indices of the vertices to fair from `vmap_id`
    v_to_fair = np.flatnonzero(vmap == vmap_id)
    if nn > 0:
        knn, kr = s.k_ring_neighbors(nn, v_to_fair, which="vertices")
        # remesh sides plus a little more
        v_to_fair = np.unique(np.concatenate(knn))

    if vmap_id_no_touch is not None:
        no_touch = np.flatnonzero(vmap == vmap_id_no_touch)
        v_to_fair = np.setdiff1d(v_to_fair, no_touch)
    return v_to_fair


def join_conformed_surfaces(s0, s1, pi0=0, pi1=1, pi2=2):
    # join s0 and s1 on corresponding vertices, i.e., triangulation of of s0
    # and s1 should be the same!
    s0_e = s0.find_border_edges()
    s1_e = s0_e + s0.n_vertices
    nv0 = s0.n_vertices
    nf0 = s0.n_faces

    # join
    join = cortech.Surface(
        np.concatenate([s0.vertices, s1.vertices], 0),
        # reverse s0 faces so that the orientation is consistent
        np.concatenate([s0.faces[:, ::-1], s1.faces + nv0]),
    )

    # build the faces connecting the two surfaces
    connecting_faces = np.concatenate(
        [
            np.stack([s0_e[:, 0], s1_e[:, 0], s0_e[:, 1]], -1),
            np.stack([s1_e[:, 0], s1_e[:, 1], s0_e[:, 1]], -1),
        ],
    )
    join.faces = np.concatenate([join.faces, connecting_faces])

    # vertex map
    vmap = np.zeros(join.n_vertices, int)
    vmap[:nv0] = pi0  # s0
    vmap[nv0:] = pi1  # s1

    # face map
    fmap = np.zeros(join.n_faces, int)
    fmap[:nf0] = pi0  # s0
    fmap[nf0 : 2 * nf0] = pi1  # s1
    fmap[2 * nf0 :] = pi2  # connecting faces

    return join, vmap, fmap


def remove_spikes_from_edges(s):
    # remove triangles whose vertices are all edge vertices. Otherwise, we get
    # self-intersections after fairing the outside of the domain
    e = s.find_border_edges()
    remove_faces = np.isin(s.faces, e).all(1)
    return s.remove_faces(remove_faces)


def smooth_patch(s, face_mask, n_iter=20, nn=2):
    # Gaussian smoothing of a patch
    tmp = s.remove_faces(face_mask)
    kept = np.unique(s.faces[~face_mask])
    edges = np.unique(tmp.find_border_edges())
    knn, _ = tmp.k_ring_neighbors(nn, edges, which="vertices")
    constrained_v = np.setdiff1d(
        np.arange(tmp.n_vertices), np.setdiff1d(np.unique(np.concatenate(knn)), edges)
    )
    ss = tmp.copy()
    for _ in range(n_iter):
        ss = ss.smooth_gaussian()
        ss.vertices[constrained_v] = tmp.vertices[constrained_v]

    assert len(ss.self_intersections()) == 0

    out = s.copy()
    out.vertices[kept] = ss.vertices
    return out


def smooth_compartments(vol, vmap):
    # separate taubin smoothing of wm and gm compartments
    vol["wm"] = vol["wm"].smooth_taubin(n_iter=10)
    vol["gm"].vertices[vmap["gm"] == 0] = vol["wm"].vertices[vmap["wm"] == 1]

    vol["gm"] = vol["gm"].smooth_taubin(n_iter=10)
    vol["wm"].vertices[vmap["wm"] == 1] = vol["gm"].vertices[vmap["gm"] == 0]
    return vol


def stitch_and_fair_unstitch(vol, vmap, fmap, pi0=1, pi1=0, pif=2, nn=3):
    """Join and stitch s0 and s1, then fair and unstitch again recovering the
    original surfaces (but now faired).


    Parameters
    ----------
    s0, s1

    fi0, fi1
        The faces
    pi0, pi1
        Indices of the overlapping patch on surfaces 0 and 1.
    pif
        Indices of the patch to fair.


    """
    s0 = vol["wm"]
    s1 = vol["gm"]
    fmap0 = fmap["wm"]
    fmap1 = fmap["gm"]
    vmap0 = vmap["wm"]
    vmap1 = vmap["gm"]

    # Make a vertex map for the concatenate surface where overlapping patches
    # are labeled to the to-be-fair patch.
    tmp0 = vmap0.copy()
    tmp0[tmp0 == pi0] = pif
    tmp1 = vmap1.copy()
    tmp1[tmp1 == pi1] = pif
    vmap_c = np.concat([tmp0, tmp1])

    cat = cortech.Surface(
        np.concatenate([s0.vertices, s1.vertices]),
        np.concatenate(
            [s0.faces[fmap0 != pi0], s1.faces[fmap1 != pi1] + s0.n_vertices]
        ),
    )
    # vertex indices of the concatenated surface
    vi_c = np.arange(s0.n_vertices + s1.n_vertices)
    # removes "duplicate" vertices at borders
    cat_stitch, vi_cs, _ = cat.stitch_borders()
    # removes unused vertices (not used by any faces). Needed before fair
    cat_stitch_prune = cat_stitch.prune()

    # Keep track of vertex correspondences

    # after stitch
    vi_rm_cs = np.setdiff1d(vi_c, vi_cs)

    # after stitch and prune
    vi_csp = vi_cs[np.unique(cat_stitch.faces)]  # used vertices
    vmap_csp = vmap_c[vi_csp]

    # now find the kept points corresponding to those which were removing
    # during stitch_borders
    tree = scipy.spatial.KDTree(cat_stitch.vertices)
    distance, index = tree.query(cat.vertices[vi_rm_cs])
    s = np.argsort(index)
    np.isclose(distance, 0.0).all()

    vi_fair = get_fair_indices(cat_stitch_prune, vmap_csp, pif, nn=nn)
    faired = cat_stitch_prune.fair(vi_fair)

    vi_cspf = vi_csp[vi_fair]
    # `vi_fair_border` are the indices of the border vertices. We move the
    # vertices removed by the stitching to this position as well
    vi_fair_border = vi_fair[np.isin(vi_cspf, vi_cs[index[s]])]

    is_s0 = np.concat(
        [np.ones(s0.n_vertices, dtype=bool), np.zeros(s1.n_vertices, dtype=bool)]
    )

    cat.vertices[vi_cspf] = faired.vertices[vi_fair]
    cat.vertices[vi_rm_cs[s]] = faired.vertices[vi_fair_border]

    # Split concatenated surface
    s0_faired = cortech.Surface(cat.vertices[is_s0], s0.faces.copy())
    s1_faired = cortech.Surface(cat.vertices[~is_s0], s1.faces.copy())

    s0_faired = smooth_patch(s0_faired, fmap0 != pi0)
    s1_faired.vertices[vmap1 == pi1] = s0_faired.vertices[vmap0 == pi0]

    return dict(wm=s0_faired, gm=s1_faired)


def fair_compartments_individually(vol, vmap):
    # Fair deep/wm
    vi_fair = get_fair_indices(vol["wm"], vmap["wm"], 2, vmap_id_no_touch=0, nn=4)
    vol["wm"] = vol["wm"].fair(vi_fair)
    vol["gm"].vertices[vmap["gm"] == 0] = vol["wm"].vertices[vmap["wm"] == 1]

    vi_fair = get_fair_indices(vol["gm"], vmap["gm"], 2, vmap_id_no_touch=0, nn=4)
    vol["gm"] = vol["gm"].fair(vi_fair)
    return vol, vmap


def save_volumes(
    d, vol, vmap, fmap, suffix=None, include_self_intersections: bool = False
):
    suffix = "" if suffix is None else "_" + suffix
    for k in vol:
        scalars = dict(vmap=vmap[k], fmap=fmap[k])
        if include_self_intersections:
            x = np.zeros(vol[k].n_faces)
            x[np.unique(vol[k].self_intersections())] = 1
            scalars["si"] = x
            print(f"SIF: {k} = {x.sum()}")
        vol[k].save(d / f"vol_{k}{suffix}.vtk", scalars=scalars)


def make_domain():
    # inferior surface : 0 # vol["wm"] this is deep; vol["gm"] this is WM
    # superior surface : 1
    # sides            : 2
    id_sides = 2
    data_dir = Path("/mnt/projects/INN/jesper/nobackup/projects/white_matter_axons")

    # the triangulations are all the same
    wm = cortech.Surface.from_file(data_dir / "roi_surface_wm.vtk")
    gm = cortech.Surface.from_file(data_dir / "roi_surface_gm.vtk")
    deep = cortech.Surface.from_file(data_dir / "roi_surface_deep.vtk")

    target_edge_length = (
        0.5 * wm.compute_edge_norm().mean() + 0.5 * gm.compute_edge_norm().mean()
    )

    remesh_sides = functools.partial(
        remesh_patch, fmap_id=id_sides, target_edge_length=target_edge_length
    )

    # remove triangles where all vertices are part of the ROI edge
    deep = remove_spikes_from_edges(deep)
    wm = remove_spikes_from_edges(wm)
    gm = remove_spikes_from_edges(gm)

    vol = {}
    vmap = {}
    fmap = {}

    vol["wm"], vmap["wm"], fmap["wm"] = join_conformed_surfaces(deep, wm)
    vol["gm"], vmap["gm"], fmap["gm"] = join_conformed_surfaces(wm, gm)

    save_volumes(data_dir, vol, vmap, fmap, suffix="0")

    for k in vol:
        vol[k], vmap[k], fmap[k] = remesh_sides(vol[k], vmap[k], fmap[k])
    save_volumes(data_dir, vol, vmap, fmap, suffix="1")

    # Remesh deep surface
    # vol["wm"], vmap["wm"], fmap["wm"] = remesh_patch(
    #     vol["wm"], vmap["wm"], fmap["wm"], 0, target_edge_length
    # )

    # Fair the deep/side transition in wm and side/gm transition in gm
    vol, vmap = fair_compartments_individually(vol, vmap)
    for k in vol:
        vol[k], vmap[k], fmap[k] = remesh_sides(vol[k], vmap[k], fmap[k])
    save_volumes(data_dir, vol, vmap, fmap, suffix="2")

    # Fair the sides and transitions
    vol = stitch_and_fair_unstitch(vol, vmap, fmap, pi0=1, pi1=0, pif=2, nn=2)
    for k in vol:
        vol[k], vmap[k], fmap[k] = remesh_sides(vol[k], vmap[k], fmap[k])

    vol = smooth_compartments(vol, vmap)
    vol = stitch_and_fair_unstitch(vol, vmap, fmap, pi0=1, pi1=0, pif=2, nn=2)
    for k in vol:
        vol[k], vmap[k], fmap[k] = remesh_patch(
            vol[k], vmap[k], fmap[k], id_sides, target_edge_length
        )
    save_volumes(data_dir, vol, vmap, fmap, suffix="final")

    assert all(v.self_intersections().size == 0 for v in vol.values()), (
        "self-intersections in surfaces!"
    )

    # Volume meshing

    deep_and_sides = vol["wm"].remove_faces(fmap["wm"] == 1)
    gray_and_sides = vol["gm"].remove_faces(fmap["gm"] == 0)

    white_sides = vol["wm"].remove_faces(fmap["wm"] != id_sides)
    gray_sides = vol["gm"].remove_faces(fmap["gm"] != id_sides)
    deep = vol["wm"].remove_faces(fmap["wm"] != 0)
    white = vol["wm"].remove_faces(fmap["wm"] != 1)
    gray = vol["gm"].remove_faces(fmap["gm"] != 1)

    surfaces = [deep_and_sides, white, gray_and_sides]
    incident_subdomains = [(1, 0), (1, 2), (2, 0)]
    v, f, c, fid, cid = make_mesh3_complex(surfaces, incident_subdomains)

    np.savez(
        data_dir / "mesh_complex.npz",
        vertices=v,
        faces=f,
        cells=c,
        face_id=fid,
        cell_id=cid,
    )

    deep.save(data_dir / "domain-deep.vtk")
    gray.save(data_dir / "domain-gray.vtk")
    gray_sides.save(data_dir / "domain-gray-sides.vtk")
    white.save(data_dir / "domain-white.vtk")
    white_sides.save(data_dir / "domain-white-sides.vtk")

    mb = build_multiblock(v, f, c, fid, cid)
    mb.save(data_dir / "domain.vtm")

    return v, f, c, fid, cid
