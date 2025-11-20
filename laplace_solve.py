from pathlib import Path
import time

import numpy as np
import scipy.spatial
import pyvista as pv

import cortech

import simnibs

# from simnibs.simulation.fem import KSPSolver
from simnibs.mesh_tools import cython_msh, mesh_io
from simnibs.utils.mesh_element_properties import ElementTypes

from .polynomial import StackedVectorPolynomial


"""

OTHER APPROACHES

2D Laplace in GM; then control points
    https://iopscience.iop.org/article/10.1088/1741-2552/ab8ccf

Tracing first eigenvector of DTI data
    https://www.sciencedirect.com/science/article/pii/S1935861X13002969?pes=vor&utm_source=iopp&getft_integrator=iopp#ec1


Not sure how these do it
    https://www.sciencedirect.com/science/article/pii/S1094715921017761?pes=vor&utm_source=iopp&getft_integrator=iopp

(apply bending on realistic neuronal models?)
    https://www.nature.com/articles/srep27353.pdf?utm_source=iopp&getft_integrator=iopp

"""


def compute_primary_depth_dir(deep_wm, gm):
    # dti = mesh_io.read(
    #     bungert_dir
    #     / "m2m_subject_5"
    #     / "ROI_P1_LH_M1"
    #     / "cond_wm_volume_normalized"
    #     / "cond_wm_volume_normalized_smoothed.msh"
    # )
    # bone = head.crop_mesh(tags=ElementTags.COMPACT_BONE_TH_SURFACE)

    depth_vector = gm.vertices - deep_wm.vertices
    res = np.linalg.svd(depth_vector, full_matrices=False)
    primary_depth_dir = res.Vh[0]
    # make sure it points from skull and into the brain
    neg = depth_vector @ primary_depth_dir < 0.0
    if neg.sum() / neg.size > 0.5:
        primary_depth_dir *= -1.0
    return primary_depth_dir


def prepare_field(mesh, boundary_indices: dict, boundary_values: dict, cond=None):
    """Solve a PDE where a certain potential is set on the white and pial
    surfaces.
    """
    assert np.all(mesh.elm.elm_type == ElementTypes.TETRAHEDRON)

    v_idx = np.concatenate([boundary_indices[k] for k in boundary_indices]) + 1
    v_val = np.concatenate(
        [
            np.full(boundary_indices[k].size, boundary_values[k])
            for k in boundary_indices
        ]
    )

    # Define boundary conditions
    dirichlet = simnibs.simulation.fem.DirichletBC(v_idx, v_val)
    #     np.concatenate((boundary_indices["inner"], boundary_indices["outer"])) + 1,
    #     np.concatenate(
    #         (
    #             np.full(boundary_indices["inner"].size, boundary_values["inner"]),
    #             np.full(boundary_indices["outer"].size, boundary_values["outer"]),
    #         )
    #     ),
    # )

    print("Setting up system")
    if cond is None:
        # the actual value of cond is irrelevant
        cond = simnibs.simulation.sim_struct.SimuList(mesh).cond2elmdata()
        # cond.value[mesh.elm.tag1 == 2] = 100.0
        cond.value[:] = 1.0
        print("Setting all conductivities to 1.0!")
    else:
        print("Using provided conductivities")
        # cond = mesh.field["cond_smoothed"]

    laplace_eq = simnibs.simulation.fem.FEMSystem(mesh, cond, dirichlet, store_G=True)
    print("Solving equation")
    potential = laplace_eq.solve()
    # potential = np.clip(potential, boundary_values["inner"], boundary_values["outer"])

    print("Computing E field")
    potential_elm = potential[mesh.elm.node_number_list - 1]
    E_elm = -np.sum(laplace_eq._G * potential_elm[..., None], 1)

    # SPR interpolation matrix
    print("Interpolating E field to vertices")
    M = mesh.interp_matrix(
        mesh.nodes.node_coord, out_fill="nearest", th_indices=None, element_wise=True
    )
    E = M @ E_elm
    E_mag = np.linalg.norm(E, axis=1)
    is_valid = E_mag.squeeze() > 1e-8

    print("E magnitude (minimum)", E_mag.min())
    mesh.nodedata += [
        mesh_io.NodeData(potential, "V", mesh),
        mesh_io.NodeData(E, "E", mesh),
        mesh_io.NodeData(is_valid, "valid", mesh),
    ]
    # mesh.elmdata = []

    return mesh


def prepare_for_tetrahedron_with_points(mesh):
    indices_tetra = mesh.elm.tetrahedra
    nodes_tetra = np.array(mesh.nodes[mesh.elm[indices_tetra]], float)
    th_baricenters = nodes_tetra.mean(1)

    # Calculate a few things we will use later
    _, faces_tetra, adjacency_list = mesh.elm.get_faces(indices_tetra)
    faces_tetra = np.array(faces_tetra, dtype=int)
    adjacency_list = np.array(adjacency_list, dtype=int)

    kdtree = scipy.spatial.KDTree(th_baricenters)

    return faces_tetra, nodes_tetra, adjacency_list, kdtree, indices_tetra


def tetrahedron_with_points(
    points, faces_tetra, nodes_tetra, adjacency_list, indices_tetra, init_tetra
):
    tetra_index = cython_msh.find_tetrahedron_with_points(
        np.array(points, float), nodes_tetra, init_tetra, faces_tetra, adjacency_list
    )

    # calculate baricentric coordinates
    inside = tetra_index != -1

    M = np.transpose(
        nodes_tetra[tetra_index[inside], :3]
        - nodes_tetra[tetra_index[inside], 3, None],
        (0, 2, 1),
    )
    baricentric = np.zeros((len(points), 4), dtype=float)
    A = M
    b = points[inside] - nodes_tetra[tetra_index[inside], 3]
    baricentric[inside, :3] = np.linalg.solve(A, b[..., None]).squeeze()
    baricentric[inside, 3] = 1 - np.sum(baricentric[inside], axis=1)

    # Return indices
    tetra_index[inside] = indices_tetra[tetra_index[inside]]

    return tetra_index - 1, baricentric


def integrate_field(
    mesh,
    seed_points,
    V_stepsize: float,
    h_max=0.1,
    max_iter: int = 1000,
    verbose: bool = False,
):
    """_summary_

    Parameters
    ----------
    mesh : _type_
        _description_
    seed_points : _type_
        _description_
    V_stepsize: float,

    h_max : float, optional
        maximum stepsize in mm.

    Returns
    -------
    _type_
        _description_
    """

    # is_valid = mesh.field["valid"].value

    # collect necessary quantities

    # Vertices
    V = mesh.field["V"].value
    E = mesh.field["E"].value
    E_mag = np.linalg.norm(E, axis=1)
    N = np.divide(E, E_mag[:, None], where=E_mag[:, None] > 0)  # normalize E

    # Elements
    # For linear interpolation within elements)
    faces = mesh.elm.node_number_list - 1
    V_elm = V[faces]
    N_elm = N[faces]
    E_mag_elm = E_mag[faces]

    # intialize the random walk to tetrahedron with closest baricenter
    # subsequent iterations use the previously found tetrahedron at starting
    # point
    t0 = time.perf_counter()
    faces_tetra, nodes_tetra, adjacency_list, kdtree, indices_tetra = (
        prepare_for_tetrahedron_with_points(mesh)
    )
    t1 = time.perf_counter()
    print(f"{'Initializing':40s} {t1 - t0:5.2f} s")

    # valid_seed_points = is_valid[seed_points]
    # valid_gm = seed_points[valid_seed_points]
    # thickness = np.zeros(valid_gm.size)

    # START
    # -----

    # y = V[valid_gm]
    # pos = mesh.nodes.node_coord[valid_gm]

    # Initialize
    # Starting position for walking algorithm: the closest baricenter
    _, tetra_index = np.array(kdtree.query(seed_points), int)

    tetra_index, coo_bari = tetrahedron_with_points(
        seed_points,
        faces_tetra,
        nodes_tetra,
        adjacency_list,
        indices_tetra,
        tetra_index,
    )
    valid = tetra_index >= 0
    n_valid = valid.sum()
    tetra_index = tetra_index[valid]
    coo_bari = coo_bari[valid]
    print(
        f"valid source points : {n_valid} of {len(valid)} ({n_valid / len(valid) * 100.0: 6.2f} %)"
    )

    pos = seed_points[valid]
    y = np.sum(V[mesh.elm.node_number_list[tetra_index]] * coo_bari, 1)
    thickness = np.zeros(n_valid)
    is_in_domain = np.ones(n_valid, dtype=bool)
    is_in_domain_masked = np.ones(n_valid, dtype=bool)

    # is_in_domain = np.ones(valid_gm.size, dtype=bool)
    # is_in_domain_masked = np.ones(valid_gm.size, dtype=bool)

    traces = [pos.copy()]
    field_value = [y.copy()]
    n_iter = np.zeros(n_valid, int)

    for i in range(1, max_iter):
        # Find tetrahedron in which each point is located (index is zero-based!)
        tetra_index, coo_bari = tetrahedron_with_points(
            pos[is_in_domain],
            faces_tetra,
            nodes_tetra,
            adjacency_list,
            indices_tetra,
            tetra_index[is_in_domain_masked],
        )
        is_in_domain_masked = tetra_index >= 0
        is_in_domain[is_in_domain] = is_in_domain_masked

        if not is_in_domain_masked.any():
            reason = "no more valid points"
            i -= 1
            break

        # Determine step direction
        dydt = np.sum(
            N_elm[tetra_index[is_in_domain_masked]]
            * coo_bari[is_in_domain_masked, :, None],
            1,
        )
        dydt_norm = np.linalg.norm(dydt, axis=1, keepdims=True)
        dydt = np.divide(dydt, dydt_norm, where=dydt_norm > 0)  # check zeros...

        # Determine step size

        # update h for next iteration: sample |E| at current position to
        # determine step size
        E_mag_sample = np.sum(
            E_mag_elm[tetra_index[is_in_domain_masked]] * coo_bari[is_in_domain_masked],
            1,
        )
        h = np.minimum(h_max, V_stepsize / E_mag_sample)

        # The Euler step
        pos_next = pos[is_in_domain] + h[:, None] * dydt

        pos[is_in_domain] = pos_next

        # idx = thickness[still_valid]>target_frac
        # y_prev[still_valid][idx] + y[still_valid][idx]

        # Accept move and update thickness for points which are still inside
        # the domain

        # NOTE
        # For points that move outside of the domain at the current iteration,
        # we could calculate the exact point where the field line crosses the
        # mesh but perhaps that is not really worth it given that the lines
        # seem to terminate at >97% thickness

        # check if we are roughly stepping equally fast at each seed point
        y[is_in_domain] = np.sum(
            V_elm[tetra_index[is_in_domain_masked]] * coo_bari[is_in_domain_masked], 1
        )

        # REMOVE; only for diagnostics...
        if verbose:
            print(
                (
                    f"{i:3d} : {y[is_in_domain].min():10.3f} "
                    f"{y[is_in_domain].mean():10.3f} "
                    f"{y[is_in_domain].max():10.3f} "
                    f" {is_in_domain.sum() / len(is_in_domain):10.3f}"
                )
            )

        traces.append(pos.copy())
        field_value.append(y.copy())
        thickness[is_in_domain] += h  # add thickness at this step
        n_iter[is_in_domain] += 1
    else:
        reason = "max iter reached"

    print("Terminating after iteration", i)
    print(f"Reason: {reason}")

    traces = np.array(traces)
    field_value = np.array(field_value)

    print(f"{'Trace field lines':40s} {time.perf_counter() - t1:5.2f} s")

    return traces, field_value, thickness, n_iter, valid


def sample_points_on_intersection(gm):
    origin = [-40, -1.3, 58.7]
    normal = [0.71, -0.22, 0.67]

    # origin = [-30, -3.4, 65.4]
    # normal = [0.81, -0.18, 0.56]

    # pd = pv.PolyData(mesh.nodes.node_coord)
    # pd_proj = pd.project_points_to_plane(origin, normal)
    # pd_proj.points - pd.points

    # mesh_io.make_surface_mesh(mesh.nodes.node_coord, f + 1).to_multiblock().save(
    #     root_dir / "subject_5_ROI_surface.vtm"
    # )
    # supfaces = f[sup[f].all(1)]

    # u = np.unique(f)
    # v = mesh.nodes.node_coord[u]
    # reind = np.zeros(f.max() + 1, dtype=int)
    # reind[u] = np.arange(len(u))
    # f = reind[f]

    plane = pv.Plane(origin, normal, i_size=50, j_size=50)
    plane = plane.triangulate()

    outline = pv.make_tri_mesh(gm.vertices, gm.faces)

    intersection, split1, split2 = outline.intersection(plane, False, False)
    # intersection.save(root_dir / "source_points.vtk")

    # plotter = pv.Plotter()
    # plotter.add_mesh(outline)
    # # plotter.add_mesh(plane, color="red")
    # plotter.add_mesh(intersection, color="red")
    # plotter.show()

    return intersection


# filename_domain = "/home/jesperdn/nobackup/laplace_torge/laplace_roi.msh"
# filename_domain_field = "/home/jesperdn/nobackup/laplace_torge/domain.vtm"
# filename_lines = "/home/jesperdn/nobackup/laplace_torge/line_traces.vtm"

# mesh = mesh_io.read(filename_domain)
# mesh = mesh.crop_mesh(elm_type=ElementTypes.TETRAHEDRON)
# # Remove tetrahedra that does not belong to the domain (flat potential)
# is_deep = mesh.field["potential"].value == 2
# is_not_all_deep = np.flatnonzero(~is_deep[mesh.elm.node_number_list - 1].all(1))
# mesh = mesh.crop_mesh(elements=is_not_all_deep + 1)

# depth = mesh.nodedata[0].value[mesh.field["potential"].value == 1]


# # Boundary conditions
# boundary_indices = dict(
#     inner=np.where(mesh.field["potential"].value == 2)[0],  # Deep
#     outer=np.where(mesh.field["potential"].value == 1)[0],  # WM surface
# )
# boundary_values = dict(inner=0, outer=1000 + 1000 / depth)

# seed_points = boundary_indices["outer"]
# # target gradient in V
# V_stepsize = 0.01 * np.abs(boundary_values["outer"] - boundary_values["inner"])

# # Solve
# mesh = prepare_field(mesh, boundary_indices, boundary_values)
# sampled_p, sampled_v, thickness, n_iter, valid_seed_points = integrate_field(
#     mesh, seed_points, V_stepsize
# )

# mb = mesh.to_multiblock()
# mb.save(filename_domain_field)

# indices = range(1000)
# lines = traces_as_multiblock(sampled_p, sampled_v, thickness, n_iter, indices)
# lines.save(filename_lines)


def generate_conductivity_map(mesh, wm, wm_dist_cutoff=0.0, value=0.25):
    """_summary_

    Parameters
    ----------
    mesh : _type_
        _description_
    wm : _type_
        _description_
    wm_dist_cutoff : float, optional
        Negative (positive) values move the cutoff *into* white (gray) matter.
    value : float, optional
        _description_, by default 0.25

    Returns
    -------
    _type_
        _description_
    """
    tree = scipy.spatial.KDTree(wm.vertices)
    distance, _ = tree.query(mesh.nodes.node_coord)

    # Sign the distance
    elm_wm = mesh.elm.tag1 == 1
    # elm_gm = mesh.elm.tag1 == 2
    wm_vertices = np.unique(mesh.elm.node_number_list[elm_wm]) - 1
    # gm_vertices = np.unique(mesh.elm.node_number_list[elm_gm])
    distance[wm_vertices] *= -1.0
    # distance[wm_vertices - 1] = dist

    cond = np.ones(mesh.nodes.nr)
    close = distance <= wm_dist_cutoff
    cond[close] = value
    # Map to elements
    cond = cond[mesh.elm.node_number_list - 1].mean(1)
    return mesh_io.ElementData(cond, "conductivity")

    # cond = m.field["cond_smoothed"].value.reshape(-1, 3, 3)
    # res = np.linalg.eigh(cond)
    # evals = res.eigenvalues
    # evecs = res.eigenvectors
    # evals /= evals.sum(1, keepdims=True)
    # mu = evals.mean(1, keepdims=True)
    # FA = np.sqrt(3.0 / 2.0 * np.sum((evals - mu) ** 2, 1) / evals.sum(1))
    # # s = evals.sum(1, keepdims=True)
    # # is_ani = ~np.isclose(np.diff(evals, 1), 0).all(1)
    # # evals[is_ani, -1] = evals[is_ani, -1] * 10.0
    # mask = FA > 0.025
    # evals[mask] = evals[mask] / (10 * FA[mask][:, None])
    # # evals *= s
    # cond_new = evecs @ np.transpose((evals[:, None] * evecs), (0, 2, 1))

    # cond_new = np.array(
    #     [
    #         [0.26401833, 0.02487021, -0.44010695],
    #         [0.02487021, 0.00234274, -0.04145754],
    #         [-0.44010695, -0.04145754, 0.73363892],
    #     ]
    # )
    # cond_new = cond_new.ravel()
    # x = np.zeros_like(m.field["cond_smoothed"].value)
    # x[m.elm.tag1 == 1] = cond_new
    # x[m.elm.tag1 == 2] = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
    # cond_new = x

    # m.elmdata.append(mesh_io.ElementData(cond_new.reshape(-1, 9), "cond_new"))
    # m.elmdata.append(mesh_io.ElementData(FA, "FA"))


def compute_curvatures_and_radii(sampled_p, n_iter, number_of_points=3):
    assert number_of_points % 2 == 1
    c = number_of_points // 2
    insert_start_offset = c
    insert_end_offset = c - 1

    sampled_p_t = sampled_p.transpose(1, 0, 2)
    # view = np.transpose(view, (1, 0, 3, 2))
    curv = np.full(sampled_p_t.shape[:2], np.nan)
    radii = np.full(sampled_p_t.shape[:2], np.nan)

    for i, (iterations, y) in enumerate(zip(n_iter, sampled_p_t)):
        if iterations < number_of_points:
            continue
        yw = np.lib.stride_tricks.sliding_window_view(
            y[: iterations + 1], number_of_points, axis=0
        )
        yw = yw.swapaxes(1, 2)
        x = np.linalg.norm(yw - yw[:, [0]], axis=2)
        x /= x[:, [-1]]

        # l01 = np.linalg.norm(yw[:, 1] - yw[:, 0], axis=1)
        # l12 = np.linalg.norm(yw[:, 1] - yw[:, 2], axis=1)
        # x = l01 / (l01 + l12)
        # x = np.stack((np.zeros_like(x), x, np.ones_like(x)), axis=1)

        p = StackedVectorPolynomial()
        p.fit(x, yw, deg=2)
        k = p.compute_curvature(x[:, c])
        curv[i, insert_start_offset : iterations - (insert_end_offset)] = k
        radii[i, insert_start_offset : iterations - (insert_end_offset)] = 1.0 / k

    return curv, radii


def orthogonalize_a_wrt_b(a, b):
    """

    a,b :
        (n, 3)
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    pa_on_b = np.sum(a * b_norm, 1, keepdims=True) * b_norm
    return a - pa_on_b


def normalize_01(arr):
    amin = arr.min()
    amax = arr.max()
    return (arr - amin) / (amax - amin)


def solve_laplace():
    bungert_dir = Path("/mnt/projects/INN/bungert_revisited")
    m2m_dir = Path(bungert_dir / "subject_5" / "m2m_subject_5")
    data_dir = Path("/mnt/projects/INN/jesper/nobackup/projects/white_matter_axons")

    f_domain = data_dir / "mesh_complex.npz"
    f_domain_field = data_dir / "domain_laplace.msh"

    deep_wm = cortech.Surface.from_file(data_dir / "domain-deep.vtk")
    gm_sides = cortech.Surface.from_file(data_dir / "domain-gray-sides.vtk")
    wm_sides = cortech.Surface.from_file(data_dir / "domain-white-sides.vtk")
    wm = cortech.Surface.from_file(data_dir / "domain-white.vtk")
    gm = cortech.Surface.from_file(data_dir / "domain-gray.vtk")

    # Domain
    # ---------------------------
    domain_data = np.load(f_domain)

    nodes = mesh_io.Nodes(domain_data["vertices"])
    elements = mesh_io.Elements(tetrahedra=domain_data["cells"] + 1)
    domain = mesh_io.Msh(nodes, elements)
    domain.elm.tag1 = domain_data["cell_id"]

    index = np.unique(domain_data["faces"])

    surface_vertices = domain.nodes.node_coord[index]

    cat_s = (wm, gm, deep_wm, wm_sides, gm_sides)
    cat_v = np.concat([s.vertices for s in cat_s])
    cat_id = np.concat([np.full(s.n_vertices, i) for i, s in enumerate(cat_s)])
    tree = scipy.spatial.KDTree(cat_v)
    d, i = tree.query(surface_vertices)
    label = cat_id[i]
    domain.node_data.append(label)

    boundary_indices = dict(
        deep=index[label == 2],
        gray=index[label == 1],
        # white=index[label == 0],
    )
    boundary_values = dict(deep=0, white=500, gray=1000)

    # filename_domain = root_dir / "subject_5_ROI.msh"
    # filename_domain_field = root_dir / f"subject_5_ROI_{suffix}.vtm"
    # filename_lines = root_dir / f"subject_5_ROI_line_traces_{suffix}.vtm"

    # seed_points = pv.read(root_dir / "source_points.vtk").points

    cond = generate_conductivity_map(domain, wm, 0.2, 0.25)
    domain.elmdata.append(cond)

    # Solve
    mesh = prepare_field(domain, boundary_indices, boundary_values, cond)
    mesh.save(f_domain_field)
    mesh.to_multiblock().save(f_domain_field.with_suffix(".vtm"))


def trace_field_lines():
    data_dir = Path("/mnt/projects/INN/jesper/nobackup/projects/white_matter_axons")

    f_domain_field = data_dir / "domain_laplace.msh"
    f_field_data = data_dir / "domain_field_data.npz"

    domain = mesh_io.read(f_domain_field)

    deep_wm = cortech.Surface.from_file(data_dir / "domain-deep.vtk")
    # wm = cortech.Surface.from_file(data_dir / "domain-white.vtk")
    gm = cortech.Surface.from_file(data_dir / "domain-gray.vtk")
    edge_vertices = np.unique(gm.find_border_edges())

    # Find the primary depth direction: vector pointing from GM to deep WM
    primary_depth_dir = compute_primary_depth_dir(deep_wm, gm)

    prune, _ = gm.k_ring_neighbors(3, edge_vertices)
    prune = np.unique(np.concat(prune))
    sources = gm.remove_vertices(prune)
    source_normals = sources.compute_vertex_normals()
    # Move a little inwards
    seed_points = sources.vertices - 0.5 * source_normals

    # vector pointing from deep WM to skull
    # primary_depth_dir = np.array([-0.51905445, -0.04680712, 0.85345859])
    depth_angle = np.degrees(np.acos(source_normals @ primary_depth_dir))

    # seed_points = mesh.nodes.node_coord[boundary_indices["outer"]]
    # seed_points = gm.points[select_on_surface_eroded]
    # normals = gm.compute_normals(cell_normals=False, consistent_normals=False)[
    #     "Normals"
    # ][select_on_surface_eroded]
    # seed_points_moved = seed_points - 0.5 * normals

    # indices = np.random.choice(np.arange(len(seed_points)), 1000)
    # seed_points = seed_points[indices]

    # target gradient in V
    V_stepsize = 0.01 * np.abs(domain.field["V"].max() - domain.field["V"].min())
    # V_stepsize = 0.01
    # V_stepsize = 0.1

    # , h_max=1.0, # for normalized vector fields
    points, potentials, thickness, n_iter, valid_seed_points = integrate_field(
        domain, seed_points, V_stepsize, verbose=True
    )
    curv, radii = compute_curvatures_and_radii(points, n_iter)
    # radii *= 1e3  # mm to um

    depth_angle = depth_angle[valid_seed_points]

    np.savez(
        f_field_data,
        curv=curv,
        depth_angle=depth_angle,
        n_iter=n_iter,
        points=points,
        potentials=potentials,
        radii=radii,
        thickness=thickness,
        valid_seed_points=valid_seed_points,
    )


"""


def compute_vertex_adjacency(connectivity, include_self: bool = False):
    #Make sparse adjacency matrix for vertices with connections `tris`.

    n_vertices = connectivity.max() + 1

    pairs = list(itertools.combinations(np.arange(connectivity.shape[1]), 2))
    row_ind = np.concatenate([connectivity[:, i] for p in pairs for i in p])
    col_ind = np.concatenate([connectivity[:, i] for p in pairs for i in p[::-1]])

    data = np.ones_like(row_ind)
    A = scipy.sparse.csr_array(
        (data, (row_ind, col_ind)), shape=(n_vertices, n_vertices)
    )
    A.data[:] = 1.0

    if include_self:
        A = A.tolil()
        A.setdiag(1)
        A = A.tocsr()

    A.sum_duplicates()  # ensure canocical format

    return A


def compute_degree_matrix(A):
    return scipy.sparse.diags_array(A.sum(1))


def compute_laplacian_matrix(A, D):
    return D - A

    mesh = mesh_io.read(filename_domain)
    mesh = mesh.crop_mesh(elm_type=ElementTypes.TETRAHEDRON)
    # Remove tetrahedra that does not belong to the domain (flat potential)

    # head = mesh_io.read(m2m_dir / "subject_5.msh")

    # set superficial...
    is_super = np.flatnonzero(mesh.field["superficial"].value.astype(bool))

    gm = pv.read(root_dir / "subject_5_ROI_gray_surface.vtm")
    gm = gm["TRIANGLE"]["WM"]
    gm = gm.connectivity("largest")

    A = compute_vertex_adjacency(gm.faces.reshape(-1, 4)[:, 1:])
    D = compute_degree_matrix(A)

    tree = scipy.spatial.KDTree(gm.points)
    distance, index = tree.query(mesh.nodes.node_coord)
    on_surface = np.isclose(distance, 0.0)
    select_on_mesh = np.intersect1d(np.flatnonzero(on_surface), is_super)
    select_on_surface = index[select_on_mesh]
    mesh.field["superficial"].value[:] = 0.0
    mesh.field["superficial"].value[select_on_mesh] = 1.0

    # erode "surface" points to get a little bit away from the edge of the
    # surface domain
    b = np.zeros(gm.n_points)
    b[select_on_surface] = 1.0
    gm["on_surface"] = b
    for _ in range(5):
        b = np.squeeze(A @ b == D.data).astype(float)
    select_on_surface_eroded = np.flatnonzero(b)
    gm["on_surface_eroded"] = b
    # gm.save("test.vtk")

    is_deep = mesh.field["deep"].value.astype(bool)
    is_super = mesh.field["superficial"].value.astype(bool)

    is_not_all_deep = np.flatnonzero(~is_deep[mesh.elm.node_number_list - 1].all(1))
    mesh = mesh.crop_mesh(elements=is_not_all_deep + 1)
    is_not_all_super = np.flatnonzero(~is_super[mesh.elm.node_number_list - 1].all(1))
    mesh = mesh.crop_mesh(elements=is_not_all_super + 1)

    # Boundary conditions
    boundary_indices = dict(
        inner=np.where(mesh.field["deep"].value == 1)[0],  # Deep
        outer=np.where(mesh.field["superficial"].value == 1)[0],  # GM surface
        # white=np.where(mesh.field["white"].value == 1)[0],  # WM surface
    )
    boundary_values = dict(inner=0, white=500, outer=1000)  # , white=100


    # skull depth and surface depth fields...

    grad = simnibs.simulation.fem._gradient_operator(mesh)
    M = mesh.interp_matrix(
        mesh.nodes.node_coord, out_fill="nearest", th_indices=None, element_wise=True
    )

    skull_depth = normalize_01(mesh.field["depth"].value)
    skull_depth_elm = skull_depth[mesh.elm.node_number_list - 1]
    skull_depth_field_elm = -np.sum(grad * skull_depth_elm[..., None], 1)
    skull_depth_field = M @ skull_depth_field_elm

    surface_depth = normalize_01(mesh.field["surface depth"].value)
    surface_depth_elm = surface_depth[mesh.elm.node_number_list - 1]
    surface_depth_field_elm = -np.sum(grad * surface_depth_elm[..., None], 1)
    surface_depth_field = M @ surface_depth_field_elm

    # fraction of surface depth...
    p_wm = 0.10
    p_gm = 0.90

    gm_v = np.unique(mesh.elm.node_number_list[mesh.elm.get_tags(2)]) - 1
    wm_v = np.unique(mesh.elm.node_number_list[mesh.elm.get_tags(1)]) - 1

    interface = np.intersect1d(wm_v, gm_v)

    # combined_depth = np.zeros_like(surface_depth)
    # combined_depth[gm_v] += p_gm * surface_depth[gm_v] + (1 - p_gm) + skull_depth[gm_v]
    # combined_depth[wm_v] += p_wm * surface_depth[wm_v] + (1 - p_wm) + skull_depth[wm_v]
    # combined_depth[interface] /= 2
    # combined_depth = normalize_01(combined_depth)
    # combined_depth_raw = combined_depth.copy()

    A = compute_vertex_adjacency(mesh.elm.node_number_list - 1)
    D = compute_degree_matrix(A)
    L = compute_laplacian_matrix(A, D)
    # D_inv_sqrt = D.copy()
    # D_inv_sqrt.data = np.sqrt(1/D_inv_sqrt.data)
    # LL = D_inv_sqrt @ L @ D_inv_sqrt

    interface_with_neighbors = np.unique(A[interface].indices)
    not_interface = np.setdiff1d(np.arange(mesh.nodes.nr), interface_with_neighbors)

    elements = np.where(
        np.isin(mesh.elm.node_number_list - 1, interface_with_neighbors).all(1)
    )[0]
    mesh_interface = mesh.crop_mesh(elements=elements + 1)
    A = compute_vertex_adjacency(mesh_interface.elm.node_number_list - 1)
    D = compute_degree_matrix(A)
    L = compute_laplacian_matrix(A, D)

    # LL = L.tolil()fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(
    *yw[:, 1].T, marker=".", c=1e3 * r, cmap="viridis", vmin=0.0, vmax=5000.0
)
fig.colorbar(sc, ax=ax, label="radius (um)")
fig.show()
    # for i in not_interface:
    #     LL.rows[i] = []
    #     LL.data[i] = []
    # LL = LL.T
    # for i in not_interface:
    #     LL.rows[i] = []
    #     LL.data[i] = []
    # LL = LL.T
    # LL = LL.tocsr()

    # L = LL

    lmdb = 1.0
    dt = 0.2
    # matA = scipy.sparse.eye_array(A.shape[0]) - lmdb * dt * L
    matA = scipy.sparse.eye_array(A.shape[0]) + lmdb * dt * L

    A.data = matA.data.astype(np.float32)
    A.indices = matA.indices.astype(np.int32)
    A.indptr = matA.indptr.astype(np.int32)

    solver = KSPSolver(A, "cg", "hypre")

    # for _ in range(2):
    #     combined_depth[interface_with_neighbors] = solver.solve(
    #         combined_depth[interface_with_neighbors]
    #     )

    # combined_depth = p * surface_depth + (1 - p) + skull_depth
    # combined_depth_elm = combined_depth[mesh.elm.node_number_list - 1]
    # combined_depth_field_elm = np.sum(grad * combined_depth_elm[..., None], 1)
    # combined_depth_field = M @ combined_depth_field_elm

    # gm_v = np.unique(mesh.elm.node_number_list[mesh.elm.get_tags(2)]) - 1
    # wm_v = np.unique(mesh.elm.node_number_list[mesh.elm.get_tags(1)]) - 1

    nsurfdf = surface_depth_field / np.linalg.norm(
        surface_depth_field, axis=1, keepdims=True
    )
    nskulldf = skull_depth_field / np.linalg.norm(
        skull_depth_field, axis=1, keepdims=True
    )

    combined_depth_field = np.zeros_like(surface_depth_field)
    combined_depth_field[gm_v] += p_gm * nsurfdf[gm_v] + (1 - p_gm) * nskulldf[gm_v]
    combined_depth_field[wm_v] += p_wm * nsurfdf[wm_v] + (1 - p_wm) * nskulldf[wm_v]
    # combined_depth[interface] /= 2
    combined_depth_field /= np.linalg.norm(combined_depth_field, axis=1, keepdims=True)

    combined_depth_field *= -1.0

    for _ in range(1):
        combined_depth_field[interface_with_neighbors] = solver.solve(
            combined_depth_field[interface_with_neighbors]
        )

    combined_depth = np.ones_like(surface_depth)

    # mesh.nodedata = [
    #     mesh_io.NodeData(combined_depth, "V", mesh),
    #     # mesh_io.NodeData(combined_depth_raw, "V (raw)", mesh),
    #     mesh_io.NodeData(combined_depth_field, "E", mesh),
    #     mesh_io.NodeData(np.ones(len(combined_depth), dtype=bool), "valid", mesh),
    # ]

    # combined_depth_field = p * surface_depth_field + (1 - p) * skull_depth_field
    # mesh.nodedata.append(
    #     simnibs.mesh_io.NodeData(skull_depth_field, "skull depth field")
    # )
    # mesh.nodedata.append(
    #     simnibs.mesh_io.NodeData(surface_depth_field, "surface depth field")
    # )
    # mesh.nodedata[-1] = simnibs.mesh_io.NodeData(
    #     combined_depth_field, "combined depth field"
    # )
    # mesh.nodedata += [
    #     mesh_io.NodeData(combined_depth, "V", mesh),
    #     mesh_io.NodeData(combined_depth_field, "E", mesh),
    #     mesh_io.NodeData(np.ones(len(combined_depth), dtype=bool), "valid", mesh),
    # ]

    # mesh.to_multiblock().save(root_dir / "test.vtm")
"""
