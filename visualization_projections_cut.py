from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import io_utils

offset = 0.2
cond_ratio = 0.2
settings_str = "-".join([str(i).replace(".", "") for i in (offset, cond_ratio)])

data_dir = Path(
    "/mnt/projects/INN/jesper/nobackup/projects/white_matter_axons/domain_M1"
)
simulation_dir = data_dir / f"simulation_{settings_str}"
surface_roi_dir = data_dir / "surfaces_roi"

# Full domain
domain = pv.read(
    "/mnt/projects/INN/bungert_revisited/subject_5/Experiments/Simulations/fdi_min_threshold/subject_5_ROI_P1_LH_M1_TMS_1-0001_MagVenture_MC-B70_dir.vtk"
)

# Projections
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


# Surfaces
deep = pv.read(surface_roi_dir / "roi_surface_deep.vtk")
white = pv.read(surface_roi_dir / "roi_surface_wm.vtk")
gray = pv.read(surface_roi_dir / "roi_surface_gm.vtk")

# cutting plane
origin = np.array([-37.21, -1.82, 63.08])
normal = np.array([0.54, -0.29, 0.79])

p = points[:, 0]

d = -normal @ origin
signed_distance = np.vecdot(normal, p) + d
denom = np.sum(normal**2)  # == 1.0
# projected points
p_prime = p - normal[None] * signed_distance[:, None] / denom

mask = (signed_distance > 0) & (signed_distance < 0.55)

domain_slice = domain.slice(normal, 0.5 * normal + origin)
x = domain_slice.extract_cells(np.isin(domain_slice["tag"], (1, 2)))
y = domain_slice.extract_cells(np.isin(domain_slice["tag"], (1001, 1002)))

# Figure

cam_pos = camera_position = [
    (-13.4866789893013, -12.119022776260568, 103.57870949135645),
    (-39.23045376440518, -1.0692484019338417, 66.17381167503468),
    (-0.8067734312828665, 0.09467504293018057, 0.5832265998925618),
]

p = pv.Plotter(off_screen=True)
kw = dict(color="black", render_lines_as_tubes=True, line_width=14.0)
# p.add_mesh(deep.slice(normal, 0.5 * normal + origin), **kw)
# p.add_mesh(white.slice(normal, 0.5 * normal + origin), **kw)
# p.add_mesh(gray.slice(normal, 0.5 * normal + origin), **kw)
p.add_mesh(x, color="white")
p.add_mesh(y, **kw)
p.camera_position = cam_pos
# p.show()
img_bg = p.screenshot(
    transparent_background=True, return_img=True, window_size=(6000, 6540)
)

p = pv.Plotter(off_screen=True)
p.add_mesh(
    projections.extract_cells(mask),
    render_lines_as_tubes=True,
    line_width=20.0,
    color=(170, 0, 0),
    show_scalar_bar=False,
)
p.camera_position = cam_pos
img_fg = p.screenshot(
    transparent_background=True, return_img=True, window_size=(6000, 6540)
)

fig, ax = plt.subplots(1, 1, constrained_layout=True)
ax.imshow(img_bg)
ax.imshow(img_fg)
ax.axis("off")
fig.savefig(
    "/mnt/projects/INN/jesper/nobackup/projects/white_matter_axons/domain_M1/simulation_02-02/figures/smooth_axon_projections.png",
    pad_inches=0,
    bbox_inches="tight",
    dpi=600,
)
