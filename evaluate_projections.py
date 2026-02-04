from pathlib import Path

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import scipy

import io_utils

import cortech
import pyvista as pv

plt.style.use("default")
plt.style.use("science")


def get_indices(how, value, points, vertex_data, line_data):
    n = len(points)
    match how:
        case "random":
            rng = np.random.default_rng(seed=1234)
            indices = rng.choice(np.arange(n), value)
        case "thickness-smaller-than":
            indices = np.flatnonzero(thickness < value)
        case "":
            indices = np.flatnonzero((min_bend_radii > 2) & (min_bend_radii < 3))
            indices = np.flatnonzero(max_bend_curv > 1.0)
            indices = np.flatnonzero((depth_angle > 88) & (depth_angle < 92))
            indices = np.flatnonzero(depth_angle > 120)
            indices = np.flatnonzero(min_bend_radii < 0.75)
            indices = valid_seed_points[:100]
            indices = np.arange(valid_seed_points.sum())
        case "start-in-box":
            lower_left, upper_right = value
            indices = np.flatnonzero(
                (points[:, 0] > lower_left) & (points[:, 0] < upper_right)
            )


def indices_from_planes_crop(origins, normals, p):
    """Crop point set from number of planes defined by (origin, normal) pairs.
    Points on the side of the negative normal are kept.

    Parameters
    ----------
    origins : _type_
        _description_
    normals : _type_
        _description_
    p : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    inside = np.ones(len(p), dtype=bool)
    for origin, normal in zip(origins, normals):
        inside[inside] &= np.vecdot(normal, (p[inside] - origin)) < 0.0
    return inside


origin0 = np.array([-31.0, -2.5, 62.0])
normal0 = np.array([0.7, -0.09, 0.70])
origin1 = origin0 - 2 * normal0
normal1 = -normal0
origins = [origin0, origin1]
normals = [normal0, normal1]

# i = indices_between_planes([origin1], [normal1], p)
i = indices_from_planes_crop(origins, normals, p)
print(f"Number of lines: {i.sum()}")

mb = io_utils.projections_as_multiblock(
    points, line_data["n_iter"], vertex_data, line_data, np.flatnonzero(i)
)
mb.save("test.vtm")

mb = io_utils.projections_as_multiblock(
    points, line_data["n_iter"], vertex_data, line_data, indices
)
mb.save("test_rand.vtm")


def subset_to_multiblock():
    print("Generating projections as multiblock")
    mb = io_utils.projections_as_multiblock(
        points, line_data["n_iter"], vertex_data, line_data
    )
    mb.save(f_field_lines)
    print(f"Wrote {f_field_lines}")


def make_alignment_bend_plot(depth_angle, curv):
    data = np.stack([depth_angle, curv])
    kernel = scipy.stats.gaussian_kde(data)
    z = kernel.evaluate(data)

    result = scipy.stats.linregress(depth_angle, curv)

    x = np.linspace(depth_angle.min(), depth_angle.max(), 2)
    y = result.intercept + result.slope * x

    def curv2radius(x):
        return 1 / x

    fig, ax = plt.subplots(1, 1)
    ax.scatter(*data, marker=".", c=z)
    ax.plot(x, y)
    # ax.set_xlim(-1.0)
    ax.set_xlabel("Angular alignment (degrees)")
    ax.set_ylim(0.1)
    ax.set_ylabel("Curvature (1/mm)")
    ax.set_title(f"r = {result.rvalue:.2f}")
    ax.grid(alpha=0.2)
    secaxy = ax.secondary_yaxis("right", functions=(curv2radius, curv2radius))
    secaxy.set_yticks(np.round(curv2radius(ax.get_yticks()[1:-1]), 2))
    secaxy.set_ylabel("Bend radius (mm)")

    return fig, ax


def make_alignment_on_surface(gm):
    x = np.full(wm.n_points, np.nan)
    valid_radii = ~min_bend_radii.mask
    x[select_on_surface[valid_seed_points][valid_radii]] = min_bend_radii.data[
        valid_radii
    ]
    # x[select_on_surface] = 1.0
    gm.plot(scalars=x, clim=[0.5, 3])


data_dir = Path(
    "/mnt/projects/INN/jesper/nobackup/projects/white_matter_axons/domain_M1"
)

offset = 0.2
cond_ratio = 0.2
settings_str = "-".join([str(i).replace(".", "") for i in (offset, cond_ratio)])

simulation_dir = data_dir / f"simulation_{settings_str}"

points, vertex_data, line_data = io_utils.projections_load_all(
    simulation_dir / "projections.h5"
)
# points, vertex_data, line_data = load_single_projection(simulation_dir / "domain_projections.h5", 123)

# fig, ax = make_alignment_bend_plot(
#     depth_angle[~max_bend_curv.mask], max_bend_curv.data[~max_bend_curv.mask]
# )

valid = line_data["valid_projection"]
depth_angle = line_data["depth_angle"][valid]
curv_max = line_data["curv_max"][valid]

mean_curv_li = 1 / 1.217
mean_curv_cottaar = 1 / 1.410

box_bend = np.array([2.0, 0.8])
box_curv = 1 / box_bend
box_angle = np.array([70.0, 90.0])
rect = Rectangle(
    (box_angle[0], box_curv[0]),
    box_angle[1] - box_angle[0],
    box_curv[1] - box_curv[0],
    alpha=0.5,
    fill=False,
    edgecolor="red",
)

fig, ax = make_alignment_bend_plot(depth_angle, curv_max)
ax.set_ylim(0.05, 2.0)
ax.plot(box_angle, [mean_curv_li, mean_curv_li], c="r")
ax.plot(box_angle, [mean_curv_cottaar, mean_curv_cottaar], c="r")
ax.add_artist(rect)
ax.set_title(f"conductivity ratio (gray/white) = {cond_ratio}")
fig.savefig("curvature_vs_alignment.png", transparent=True)

# cottaar (2018) width parameter converted to bend radius
#   width param     bend radius
#   0.25             710 um
#   0.50            1410 um
#   0.75            2120 um

# Li S et al. 2015. Single-Neuron Reconstruction of the Macaque Primary Motor
# Cortex Reveals the Diversity of Neuronal Morphology.
#   range 788.7 - 1560.8 [mean 1217.3±210.4] um

# H01 dataset
#   812.4555, 732.964 um


fig.show()


results_dir = Path(
    "/mnt/scratch/personal/jesperdn/neuron_simulations/M1/waveform-monophasic_direction-ap"
)
data_dir = Path(
    "/mnt/projects/INN/jesper/nobackup/projects/white_matter_axons/domain_M1"
)
surf_dir = data_dir / "surfaces_domain"

domain = pv.read(data_dir / "simulation_02-02" / "laplace_field.vtm")


wm = cortech.Surface.from_file(surf_dir / "domain-white.vtk")

edge_vertices = np.unique(wm.find_border_edges())
prune, _ = wm.k_ring_neighbors(3, edge_vertices)
prune = np.unique(np.concat(prune))
wm = wm.remove_vertices(prune)


white = pv.make_tri_mesh(wm.vertices, wm.faces)

gm = cortech.Surface.from_file(surf_dir / "domain-gray.vtk")
gray = pv.make_tri_mesh(gm.vertices, gm.faces)
deep_wm = cortech.Surface.from_file(surf_dir / "domain-deep.vtk")
deep_white = pv.make_tri_mesh(deep_wm.vertices, deep_wm.faces)

depth = np.linalg.norm(gm.vertices - deep_wm.vertices, axis=1)


points_flat = points.reshape(-1, 3)
cells = np.arange(points_flat.shape[0]).reshape(points.shape[:2])
cells = np.concatenate((np.full((cells.shape[0], 1), points.shape[1]), cells), 1)
cells = cells.ravel()
projections = pv.PolyData(points_flat, lines=cells)
for k, v in vertex_data.items():
    projections[k] = v.ravel()
for k, v in line_data.items():
    projections[k] = v


mb = pv.MultiBlock([pv.read(i) for i in sorted(results_dir.glob("projection*.vtp"))])
# mb.save(result_dir / "projection_all.vtm")


def dia_str(d):
    return f"diameter-{d:.2f}"


contour_percentile = 10
orig_index = np.array([i["original_index"][0] for i in mb])
contours = pv.MultiBlock()
for d in [1, 2, 3, 4, 5, 6]:
    thresholds = np.array([i[f"{dia_str(d):s}:threshold"][0] for i in mb])

    v = np.full(wm.n_vertices, np.nan)
    v[orig_index] = thresholds
    white[dia_str(d)] = v

    contour_val = np.percentile(v[~np.isnan(v)], contour_percentile)
    print(f"{contour_percentile:d}th percentile : {contour_val:.2f}")
    contours[dia_str(d)] = white.contour(isosurfaces=[contour_val], scalars=dia_str(d))

v = np.full(wm.n_vertices, np.nan)
v[orig_index] = np.array([i["radius_min"][0] for i in mb])
white["radius_min"] = v
v = np.full(wm.n_vertices, np.nan)
v[orig_index] = np.array([i["curv_max"][0] for i in mb])
white["curv_max"] = v


plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(
    white,
    copy_mesh=True,
    scalars="curv_max",
    # scalars=np.log(white["radius_min"]),
    # cmap="viridis_r",
    # clim=[0, 1000],
    # clim=[0,np.percentile(white[dia_str(d)][~np.isnan(white[dia_str(d)])],50)],
    show_scalar_bar=True,
    smooth_shading=True,
)
plotter.show()


d = 2

plotter = pv.Plotter(off_screen=True)

# First row: view from side 1
plotter.add_mesh(
    white,
    copy_mesh=True,
    scalars=dia_str(d),
    cmap="viridis_r",
    # clim=[0, 1000],
    clim=[0, np.percentile(white[dia_str(d)][~np.isnan(white[dia_str(d)])], 50)],
    show_scalar_bar=True,
    smooth_shading=True,
)
# plotter.camera_position = "xz"
# plotter.camera.azimuth = -50
# plotter.camera.roll -= 20
# plotter.camera.elevation = 20
# plotter.add_mesh(contours[dia_str(d)], color="black", line_width=2, render_lines_as_tubes=True)
plotter.show()

max_curv_coord = np.array([i.points[np.nan_to_num(i["curv"]).argmax()] for i in mb])
max_curv_depth = np.linalg.norm(max_curv_coord - deep_wm.vertices[orig_index], axis=1)

depth_angle = np.array([i["depth_angle"][0] for i in mb])
curv_max = np.array([i["curv_max"][0] for i in mb])
radius_min = np.array([i["radius_min"][0] for i in mb])
depth0 = depth[orig_index]

t = np.array([i[f"{dia_str(d):s}:threshold"][0] for i in mb])

white_curv_max = np.full(white.n_points, np.nan)
white_curv_max[orig_index] = curv_max
white.plot(scalars=white_curv_max)


d = 4
fig, ax = plt.subplots(1, 2, sharey=True)
ax[0].scatter(depth0, np.log(t), c=curv_max, marker=".")
ax[0].set_xlabel("Distance to deep wm")
ax[0].set_ylabel("log( threshold )")
ax[1].scatter(curv_max, np.log(t), c=depth0, marker=".")
ax[1].set_xlabel("max curvature")

for d in [1, 2, 3, 4, 5, 6]:
    print(d)
    for proj in mb:
        diffs = np.concatenate(
            [[0.0], np.linalg.norm(np.diff(proj.points, axis=0), axis=1)]
        )
        distances = np.cumsum(diffs)
        index = np.abs(distances - proj[f"{dia_str(d):s}:ap-distance"] * 0.001).argmin()
        # tmp = np.full(proj.n_points, 1e5)
        # tmp[index] = proj[f"{dia_str(d):s}:threshold"].squeeze()
        tmp = np.zeros(proj.n_points, bool)
        tmp[index] = True
        proj[f"{dia_str(d):s}:index"] = tmp

x = [i for i in mb if i[f"{dia_str(d):s}:threshold"] < 317]
x = pv.MultiBlock(x)
x.save("test.vtm")

mb[500:1000].save("test.vtm")

# plotter.background_color = "pink"
img_1 = plotter.screenshot(return_img=True, scale=4, transparent_background=True)

# Second row: view from opposite side
plotter.camera_position = "xz"
plotter.camera.azimuth = 140
plotter.camera.roll += 15
plotter.camera.elevation = 50
img_2 = plotter.screenshot(return_img=True, scale=4, transparent_background=True)
screenshots.append(crop_img(img_1))
screenshots.append(crop_img(img_2))
