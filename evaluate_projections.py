from pathlib import Path

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import scipy

import io_utils


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


p = points[:, 0]

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
    ax.set_ylabel("Mean curvature (1/mm)")
    ax.set_title(f"r = {result.rvalue}")
    ax.grid(alpha=0.2)
    secaxy = ax.secondary_yaxis("right", functions=(curv2radius, curv2radius))
    secaxy.set_yticks(np.round(curv2radius(ax.get_yticks()[1:-1]), 2))
    secaxy.set_ylabel("Bend radius (mm)")

    return fig, ax


def make_alignment_on_surface(gm):
    x = np.full(gm.n_points, np.nan)
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
valid = line_data["valid"]
depth_angle = line_data["depth_angle"][valid]
curv_max = line_data["curv_max"][valid]
fig, ax = make_alignment_bend_plot(depth_angle, curv_max)
ax.set_ylim(0.05, 2.0)
ax.add_artist(rect)
ax.plot(box_angle, [mean_curv_li, mean_curv_li], c="r")
ax.plot(box_angle, [mean_curv_cottaar, mean_curv_cottaar], c="r")
fig.suptitle(f"cond ratio = {cond_ratio}")


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

fig.show()

from pathlib import Path
import cortech
import numpy as np
import pyvista as pv


results_dir = Path(
    "/mnt/scratch/personal/jesperdn/neuron_simulations/M1/waveform-monophasic_direction-ap"
)
data_dir = Path(
    "/mnt/projects/INN/jesper/nobackup/projects/white_matter_axons/domain_M1"
)
surf_dir = data_dir / "surfaces_domain"

wm = cortech.Surface.from_file(surf_dir / "domain-white.vtk")
# edge_vertices = np.unique(wm.find_border_edges())
# prune, _ = wm.k_ring_neighbors(3, edge_vertices)
# prune = np.unique(np.concat(prune))
# source_points_indices = np.setdiff1d(np.arange(wm.n_vertices), prune)
white = pv.make_tri_mesh(wm.vertices, wm.faces)

gm = cortech.Surface.from_file(surf_dir / "domain-gray.vtk")
deep_wm = cortech.Surface.from_file(surf_dir / "domain-deep.vtk")

depth = np.linalg.norm(gm.vertices - deep_wm.vertices, axis=1)


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


import os
import matplotlib
from matplotlib import pyplot as plt
import pyvista as pv
import numpy as np
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm


def crop_img(img):
    bg = img[0, 0, :]

    non_bg = (img[:, :, :3] != bg[:3]).any(axis=2) & (img[:, :, 3] > 0)

    if not non_bg.any():
        cropped = img
    else:
        ys, xs = np.where(non_bg)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        pad = 50
        y_min = max(y_min - pad, 0)
        y_max = min(y_max + pad, img.shape[0] - 1)
        x_min = max(x_min - pad, 0)
        x_max = min(x_max + pad, img.shape[1] - 1)

        cropped = img[y_min : y_max + 1, x_min : x_max + 1]

    return cropped


def create_cw_around_zero(data):
    vmin, vmax = data.min(), data.max()
    vcenter = 0
    # Compute relative position of white in [0,1]
    white_pos = (vcenter - vmin) / (vmax - vmin)
    white_pos = np.clip(white_pos, 0, 1)  # make sure it's in [0,1]

    # Create custom colormap
    cdict = {
        "red": [
            (0.0, 0.0, 0.0),  # blue at 0
            (white_pos, 1.0, 1.0),  # white at vcenter
            (1.0, 1.0, 1.0),
        ],  # red at max
        "green": [(0.0, 0.0, 0.0), (white_pos, 1.0, 1.0), (1.0, 0.0, 0.0)],
        "blue": [(0.0, 1.0, 1.0), (white_pos, 1.0, 1.0), (1.0, 0.0, 0.0)],
    }

    return LinearSegmentedColormap("BlueWhiteRed", cdict)


waveform_type = "monophasic"
subject = f"subject_5"
diameter = 1.0

base_path = f"/mrhome/torgehw/Documents/Projects/bungert_revisited/{subject}/Axons"
base_result_path = os.path.join(
    base_path, f"axon_bend_{parallel_surface}_{target_surface_distance}_{bend_radius}"
)
myelinated_axons_path = os.path.join(base_result_path, f"myelinated_axon_bends")

parallel_surf: pv.PolyData = pv.read(
    os.path.join(base_path, f"{parallel_surface}.vtk")
).extract_surface()
parallel_surf_E: pv.PolyData = pv.read(
    os.path.join(base_path, f"{parallel_surface}_{target_surface_distance}_E_.vtk")
)

parallel_surf["E"] = parallel_surf_E["E"]

parallel_surf_roi: pv.PolyData = parallel_surf.threshold(
    1, scalars="ROI", all_scalars=True
).extract_surface()


wm = pv.read(
    "/mnt/projects/INN/jesper/nobackup/projects/white_matter_axons/domain_M1/surfaces_roi/roi_surface_wm.vtk"
)


plotter = pv.Plotter(off_screen=True)

threshold_value = np.percentile(parallel_surf_roi[scalars], 5)
print(threshold_value)
contours = parallel_surf_roi.contour(isosurfaces=[threshold_value], scalars=scalars)

# First row: view from side 1
plotter.add_mesh(
    parallel_surf_roi,
    copy_mesh=True,
    scalars=scalars,
    cmap=cmap,
    clim=clim,
    show_scalar_bar=False,
    smooth_shading=True,
)
if enabled_percentile:
    plotter.add_mesh(contours, color="black", line_width=10, render_lines_as_tubes=True)

plotter.camera_position = "xz"
plotter.camera.azimuth = -50
plotter.camera.roll -= 20
plotter.camera.elevation = 20
plotter.background_color = "pink"
img_1 = plotter.screenshot(return_img=True, scale=4, transparent_background=True)

# Second row: view from opposite side
plotter.camera_position = "xz"
plotter.camera.azimuth = 140
plotter.camera.roll += 15
plotter.camera.elevation = 50
img_2 = plotter.screenshot(return_img=True, scale=4, transparent_background=True)
screenshots.append(crop_img(img_1))
screenshots.append(crop_img(img_2))


for inverted in [0, 1]:
    direction = "P-A" if inverted else "A-P"

    parallel_surf: pv.PolyData = pv.read(
        os.path.join(
            base_result_path,
            f"axon_bend_{parallel_surface}_thresholds_{target_surface_distance}_{bend_radius}_{diam}_{waveform_type}_{direction}.vtk",
        )
    )
    threshold_wm_surf_roi: pv.PolyData = parallel_surf.threshold(
        1, scalars="ROI", all_scalars=True
    ).extract_surface()
    thresholds = np.zeros((threshold_wm_surf_roi.n_points, 18))
    for i, angle in enumerate(range(0, 360, 20)):
        thresholds[:, i] = threshold_wm_surf_roi[f"threshold_{angle}"]

    threshold_min = thresholds.min(axis=1)
    threshold_variance = np.std(thresholds, axis=1) / np.mean(thresholds, axis=1)

    parallel_surf_roi.point_data[f"{direction}_Min"] = threshold_min
    parallel_surf_roi.point_data[f"{direction}_Var"] = threshold_variance

parallel_surf_roi.point_data[f"diff"] = np.log10(
    parallel_surf_roi.point_data[f"A-P_Min"]
    / (parallel_surf_roi.point_data[f"P-A_Min"])
)
diff_cm = create_cw_around_zero(parallel_surf_roi.point_data[f"diff"])

# Define what to plot
scalars_list = ["A-P_Min", "A-P_Var", "P-A_Min", "P-A_Var", "diff", "E"]

clims = [
    [90, 1000],
    [0, 2.5],
    [90, 1000],
    [0, 2.5],
    [
        np.min(parallel_surf_roi.point_data[f"diff"]),
        np.max(parallel_surf_roi.point_data[f"diff"]),
    ],
    [0, np.max(np.linalg.norm(parallel_surf_roi.point_data[f"E"], axis=1))],
]
print("A", np.max(np.linalg.norm(parallel_surf_roi.point_data[f"E"], axis=1)))

base_cmap = matplotlib.colormaps["viridis_r"]
my_colors = base_cmap(np.linspace(0, 1, 1024))
my_colors[-1] = np.array([0.7, 0.7, 0.7, 1.0])
my_cmap = LinearSegmentedColormap.from_list("my_viridis_r", my_colors)
my_cmap.set_over(np.array([0.7, 0.7, 0.7, 1.0]))

colormaps = [my_cmap, "plasma", my_cmap, "plasma", "bwr", "turbo"]
enable_percentile = [True, False, True, False, False, False]

screenshots = []
for j, (scalars, clim, cmap, enabled_percentile) in enumerate(
    zip(scalars_list, clims, colormaps, enable_percentile)
):
    plotter = pv.Plotter(off_screen=True)

    threshold_value = np.percentile(parallel_surf_roi[scalars], 5)
    print(threshold_value)
    contours = parallel_surf_roi.contour(isosurfaces=[threshold_value], scalars=scalars)

    # First row: view from side 1
    plotter.add_mesh(
        parallel_surf_roi,
        copy_mesh=True,
        scalars=scalars,
        cmap=cmap,
        clim=clim,
        show_scalar_bar=False,
        smooth_shading=True,
    )
    if enabled_percentile:
        plotter.add_mesh(
            contours, color="black", line_width=10, render_lines_as_tubes=True
        )

    plotter.camera_position = "xz"
    plotter.camera.azimuth = -50
    plotter.camera.roll -= 20
    plotter.camera.elevation = 20
    plotter.background_color = "pink"
    img_1 = plotter.screenshot(return_img=True, scale=4, transparent_background=True)

    # Second row: view from opposite side
    plotter.camera_position = "xz"
    plotter.camera.azimuth = 140
    plotter.camera.roll += 15
    plotter.camera.elevation = 50
    img_2 = plotter.screenshot(return_img=True, scale=4, transparent_background=True)
    screenshots.append(crop_img(img_1))
    screenshots.append(crop_img(img_2))

fig, axs = plt.subplots(
    nrows=3,
    ncols=4,
    layout="constrained",
    figsize=(6.5, 6.5 * np.sqrt(2) * 0.4),
)
# ----------------------------------------------------------------
axs[0, 0].set_title("A-P")
axs[0, 2].set_title("P-A")

axs[0, 0].imshow(screenshots[0], interpolation="none")
axs[0, 0].axis("off")

axs[0, 1].imshow(screenshots[1], interpolation="none")
axs[0, 1].axis("off")

axs[0, 2].imshow(screenshots[4], interpolation="none")
axs[0, 2].axis("off")

axs[0, 3].imshow(screenshots[5], interpolation="none")
axs[0, 3].axis("off")

mappable = plt.cm.ScalarMappable(
    norm=Normalize(vmin=clims[0][0], vmax=clims[0][1]), cmap=colormaps[0]
)
ticks = [clims[2][0], 250, 500, 750, clims[2][1]]

cbar = fig.colorbar(
    mappable,
    ax=[axs[0, 0], axs[0, 1], axs[0, 2], axs[0, 3]],
    extend="max",
    ticks=ticks,
    label="threshold \nin A/μs",
    aspect=15,
)

cbar.ax.axhline(125, c="black")

# ----------------------------------------------------------------

axs[1, 0].imshow(screenshots[2], interpolation="none")
axs[1, 0].axis("off")

axs[1, 1].imshow(screenshots[3], interpolation="none")
axs[1, 1].axis("off")

axs[1, 2].imshow(screenshots[6], interpolation="none")
axs[1, 2].axis("off")

axs[1, 3].imshow(screenshots[7], interpolation="none")
axs[1, 3].axis("off")

mappable = plt.cm.ScalarMappable(
    norm=Normalize(vmin=clims[1][0], vmax=clims[1][1]), cmap=colormaps[1]
)
ticks = [0, 0.5, 1, 1.5, 2, 2.5]

cbar = fig.colorbar(
    mappable,
    ax=[axs[1, 0], axs[1, 1], axs[1, 2], axs[1, 3]],
    ticks=ticks,
    label="coefficient \nof variation",
    aspect=15,
)


# ----------------------------------------------------------------
axs[2, 0].imshow(screenshots[8], interpolation="none")
axs[2, 0].axis("off")

axs[2, 1].imshow(screenshots[9], interpolation="none")
axs[2, 1].axis("off")

mappable = plt.cm.ScalarMappable(
    norm=LogNorm(vmin=10 ** clims[4][0], vmax=10 ** clims[4][1]), cmap=colormaps[4]
)

cbar = fig.colorbar(
    mappable,
    ax=axs[2, 1],
    label=r"$\log_{10}\!\left(\dfrac{\text{A-P}}{\text{P-A}}\right)$",
    aspect=15,
)

cbar.ax.set_yticks([0.25, 1, 4])
cbar.ax.set_yticklabels([r"$\dfrac{1}{4}$", "1", "4"])

# ----------------------------------------------------------------
axs[2, 2].imshow(screenshots[10], interpolation="none")
axs[2, 2].axis("off")

axs[2, 3].imshow(screenshots[11], interpolation="none")
axs[2, 3].axis("off")

mappable = plt.cm.ScalarMappable(
    norm=Normalize(vmin=clims[5][0], vmax=clims[5][1]), cmap=colormaps[5]
)

cbar = fig.colorbar(
    mappable,
    ax=[axs[2, 0], axs[2, 1], axs[2, 2], axs[2, 3]],
    ticks=[0, 0.5, 1, 1.5, 2, 2.35],
    label="|E| at 1 A/μs",
    aspect=15,
)

# Save screenshot
# plotter.show(screenshot="threshold_surface_summary.png")
# plotter.screenshot(f"visualization/figures/axon_bend_{parallel_surface}_thresholds_{target_surface_distance}_{bend_radius}_{diam}_{waveform_type}_P-A_A-P_summary.png", scale=4)

plt.savefig(
    f"visualization/figures/axon_bend_{parallel_surface}_thresholds_{target_surface_distance}_{bend_radius}_{diam}_{waveform_type}_P-A_A-P_summary.svg"
)
