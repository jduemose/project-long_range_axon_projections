from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import cortech

import io_utils

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

# projections
offset = 0.2
cond_ratio = 0.2
settings_str = "-".join([str(i).replace(".", "") for i in (offset, cond_ratio)])

simulation_dir = data_dir / f"simulation_{settings_str}"

points, vertex_data, line_data = io_utils.projections_load_all(
    simulation_dir / "projections.h5"
)
points_flat = points.reshape(-1, 3)
cells = np.arange(points_flat.shape[0]).reshape(points.shape[:2])
cells = np.concatenate((np.full((cells.shape[0], 1), points.shape[1]), cells), 1)
cells = cells.ravel()
projections = pv.PolyData(points_flat, lines=cells)
for k, v in vertex_data.items():
    projections[k] = v.ravel()
for k, v in line_data.items():
    projections[k] = v


normal = np.array([0.78, -0.36, 0.51])
origin = white.points.mean(0)
# origin = normal + origin
camera_zoom = 1.6

# manually estimated: plot, adjust camera, then print p.camera_position
camera_position = [
    (-13.401142267493304, -45.859029897825636, 138.6450724227779),
    (-40.73885909887744, 1.9747993883642434, 63.81320946369287),
    (-0.7795130826116279, 0.35828915953879525, 0.5137978514885997),
]

window_size = tuple(2 * i for i in (1024, 1024))

images = {}

# surfaces (full)
p = pv.Plotter(off_screen=True, window_size=window_size)
p.add_mesh(deep_white, color="blue", style="wireframe")
p.add_mesh(white, color="white")
p.add_mesh(gray, color="gray")
p.add_mesh(gray.slice(normal, origin), line_width=2.0)
# p.view_vector(camera_vector)
p.camera_position = camera_position
p.zoom_camera(camera_zoom * 0.8)
images["ROI"] = p.screenshot(transparent_background=True, return_img=True)
# p.show()

# surfaces (cut)
p = pv.Plotter(off_screen=True, window_size=window_size)
p.add_mesh(deep_white.clip(normal, origin), color="blue", style="wireframe")
p.add_mesh(white.clip(normal, origin), color="white")
p.add_mesh(gray.clip(normal, origin), color="gray")
p.camera_position = camera_position
p.zoom_camera(camera_zoom)
images["ROI (cut)"] = p.screenshot(transparent_background=True, return_img=True)
# p.show()

p = pv.Plotter(off_screen=True, window_size=window_size)
p.add_mesh(
    domain.clip(normal, origin, crinkle=True),
    show_edges=True,
    scalars="conductivity",
    show_scalar_bar=False,
)
p.camera_position = camera_position
p.zoom_camera(camera_zoom)
images["Conductivity"] = p.screenshot(transparent_background=True, return_img=True)
# p.show()

p = pv.Plotter(off_screen=True, window_size=window_size)
p.add_mesh(
    domain.clip(normal, origin),
    scalars="V",
    colormap="RdBu",
    show_scalar_bar=False,
    # scalar_bar_args=dict(vertical=True),
)
# p.add_mesh(
#     domain.slice(normal, origin).generic_filter("contour", [np.arange(100,1000,100)], "V"),
# )
p.add_mesh(white.slice(normal, origin), line_width=2.0)
# E vectors
# E = domain.slice(normal, origin)

# # for name in ("WM", "GM"):
# #     proj = (E["TETRAHEDRON"][name]["E"] @ normal)[:,None] * normal
# #     E["TETRAHEDRON"][name]["E_orth"] = E["TETRAHEDRON"][name]["E"] - proj
# # E.set_active_scalars("E_orth")

# p.add_mesh(
#     E.generic_filter("glyph", scale=False, geom=pv.Arrow(scale=0.75)),
#     color="black",
#     opacity=0.35,
# )
p.camera_position = camera_position
p.zoom_camera(camera_zoom)
images["Solution"] = p.screenshot(transparent_background=True, return_img=True)
# p.show()

p = pv.Plotter(off_screen=True, window_size=window_size)
p.add_mesh(white.clip(normal, origin), color="white")
p.add_mesh(gray.clip(normal, origin), color="gray")
p.add_mesh(deep_white.clip(normal, origin), color="blue", style="wireframe")
p.add_mesh(
    projections.clip(normal, origin, crinkle=True).clip(-normal, origin - normal * 2),
    scalars="curv",
    style="wireframe",
    line_width=1.0,
    show_scalar_bar=False,
)
p.camera_position = camera_position
p.zoom_camera(camera_zoom)
images["Streamlines"] = p.screenshot(transparent_background=True, return_img=True)
# p.show()

# p = pv.Plotter()
# # p.add_mesh(white.clip(normal, origin), color="white")
# # p.add_mesh(gray.clip(normal, origin), color="gray")
# p.add_mesh(deep_white.clip(normal, origin), color="blue")
# p.add_mesh(
#     projections.clip(normal, origin+normal * 4).clip(-normal, origin - normal * 4),
#     scalars="curv",
#     style="wireframe",
# )
# p.camera_position = camera_position_top_down
# p.show()

for k, v in images.items():
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(v)
    ax.axis("off")
    fig.savefig(
        f"pipeline_{k}.png",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.0,
        dpi=600,
    )


# layout with plt

size = ff.get_figsize("double", 1.5, subplots=(1, len(images)))
fig, axes = plt.subplots(1, len(images), layout="constrained", figsize=size)
for ax, k, label in zip(axes, images, range(97, 97 + len(images))):
    ax.imshow(images[k])
    # ax.set_title(k)
    ax.set_xlabel(f"({chr(label)}) {k}")
    # ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    # remove all spines (box around the plot)
    for spine in ax.spines.values():
        spine.set_visible(False)
fig.savefig("streamline_method.png", dpi=300, transparent=True, bbox_inches="tight")


from matplotlib.gridspec import GridSpec

size = ff.get_figsize("onehalf", 1.0, subplots=(4, 2))
fig = plt.figure(layout="constrained", figsize=size)
gs = GridSpec(4, 2, figure=fig)  # 2 rows, 3 columns

ax0 = fig.add_subplot(gs[0, 0])  # row 0, col 0
ax1 = fig.add_subplot(gs[0, 1])  # row 0, col 1
# Bottom row: subplot spanning all 3 columns
ax2 = fig.add_subplot(gs[1, 0])  # row 1, all columns
ax3 = fig.add_subplot(gs[1, 1])  # row 1, all columns
ax4 = fig.add_subplot(gs[2:, :])  # row 1, all columns
axes = [ax0, ax1, ax2, ax3, ax4]

for ax, k, label in zip(axes, images, range(97, 97 + len(images))):
    ax.imshow(images[k])
    # ax.set_title(k)
    ax.set_xlabel(f"({chr(label)}) {k}")
    # ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    # remove all spines (box around the plot)
    for spine in ax.spines.values():
        spine.set_visible(False)
fig.savefig("streamline_method_big.png", dpi=300, transparent=True, bbox_inches="tight")
