from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import scipy


def traces_as_multiblock(
    points,
    n_iter,
    indices,
    scalars_all: dict | None = None,
    scalars_single: dict | None = None,
):
    mb = pv.MultiBlock()
    for i in indices:
        p = pv.MultipleLines(points=points[: n_iter[i] + 1, i])
        if scalars_all is not None:
            for k, v in scalars_all.items():
                p[k] = v[i, : n_iter[i] + 1]
        if scalars_single is not None:
            for k, v in scalars_single.items():
                p[k] = v[i]
        mb[f"Line {i}"] = p
    return mb


def make_alignment_bend_plot(depth_angle, bend_radii):
    data = np.stack([depth_angle, bend_radii])
    kernel = scipy.stats.gaussian_kde(data)
    z = kernel.evaluate(data)

    result = scipy.stats.linregress(depth_angle, bend_radii)

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


data_dir = Path("/mnt/projects/INN/jesper/nobackup/projects/white_matter_axons")

f_field_lines = data_dir / "domain_field_lines.vtm"
f_field_data = data_dir / "domain_field_data.npz"

data = np.load(f_field_data)

short_projections = 5

mask = thickness < short_projections
invalid = np.isnan(radii)
invalid[mask] = True

r = np.ma.MaskedArray(radii, invalid)
c = np.ma.MaskedArray(curv, invalid)
min_bend_radii = r.min(1)
max_bend_curv = r.max(1)

min_bend_radii = r[:, 10:].min(1)
max_bend_curv = c[:, 10:].max(1)
# min_bend_radii = np.percentile(r, 1.0, axis=1)

mask = thickness > 5

fig = make_alignment_bend_plot(depth_angle[mask], min_bend_radii[mask])

valid = ~min_bend_radii.mask
fig = make_alignment_bend_plot(depth_angle[valid], min_bend_radii.data[valid])

fig, ax = make_alignment_bend_plot(depth_angle[valid], max_bend_curv.data[valid])
ax.set_ylabel("max mean curvature (1/mm)")


def save_field_lines(points, thickness, n_iter, curv, radii):
    # indices = range(1000)
    indices = np.random.choice(np.arange(len(thickness)), 400)
    indices = np.flatnonzero(thickness < 5)
    indices = np.flatnonzero((min_bend_radii > 2) & (min_bend_radii < 3))
    indices = np.flatnonzero(max_bend_curv > 1.5)
    indices = np.flatnonzero((depth_angle > 88) & (depth_angle < 92))
    indices = np.flatnonzero(depth_angle > 120)
    # indices = np.flatnonzero(min_bend_radii < 0.75)
    # indices = valid_seed_points[:100]
    # indices = np.arange(valid_seed_points.sum())
    lines = traces_as_multiblock(
        points,
        n_iter,
        indices,
        scalars_all=dict(radius=radii, curv=curv),
        scalars_single=dict(angle=depth_angle, n_iter=n_iter, thickness=thickness),
    )
    lines.save(f_field_lines)
