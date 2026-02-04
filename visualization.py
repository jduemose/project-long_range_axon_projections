from pathlib import Path
import sys

import matplotlib
from matplotlib import pyplot as plt
import pyvista as pv
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

import cortech

plt.style.use("default")

sys.path.append("/mnt/projects/INN/bungert_revisited/scripts/visualization")
from plot_settings import markers, colors

from matplotlib import rcParams

rcParams.update(
    {
        ## Font
        # "font.family": "serif",             # or "Times New Roman", "STIX", "DejaVu Serif"
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        # Figure
        "figure.figsize": (12, 10),
        "figure.dpi": 400,
        # Lines & markers
        "lines.linewidth": 1.5,
        "lines.markersize": 3,
        "lines.markeredgewidth": 0.9,
        "lines.markerfacecolor": "none",
        "lines.markeredgecolor": "black",
        # Axes
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.8,
        # "axes.spines.top": False,
        # "axes.spines.right": False,
        # Ticks
        # "xtick.direction": "in",
        # "ytick.direction": "in",
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        # Legend
        # "legend.frameon": False,
    }
)


# COLOR MAPS

base_cmap = matplotlib.colormaps["viridis_r"]
my_colors = base_cmap(np.linspace(0, 1, 1024))
my_colors[-1] = np.array([0.7, 0.7, 0.7, 1.0])
my_cmap = LinearSegmentedColormap.from_list("my_viridis_r", my_colors)
my_cmap.set_over(np.array([0.7, 0.7, 0.7, 1.0]))

color_min = np.array([1.0, 0.0, 0.0, 1.0])
color_max = np.array([1.0, 1.0, 0.0, 1.0])
base_cmap = matplotlib.colormaps["BuPu"]
my_colors = base_cmap(np.linspace(0, 1, 1024))
my_colors[0] = color_min
my_colors[-1] = color_max
my_cmap2 = LinearSegmentedColormap.from_list("my_BuPu", my_colors)
my_cmap2.set_under(color_min)
my_cmap2.set_over(color_max)

color_min = np.array([1.0, 0.0, 0.0, 1.0])
color_max = np.array([1.0, 1.0, 0.0, 1.0])
base_cmap = matplotlib.colormaps["tab10"]
my_colors = base_cmap(np.linspace(0, 1, 1024))
my_colors[0] = color_min
my_colors[-1] = color_max
my_cmap3 = LinearSegmentedColormap.from_list("my_tab10", my_colors)
my_cmap3.set_under(color_min)
my_cmap3.set_over(color_max)


global COLORMAPS
COLORMAPS = dict(
    threshold=my_cmap,
    difference="bwr",
    magnE="turbo",
    time_constant=my_cmap3,
    rheobase=my_cmap2,
    residual="Reds",
)


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


def make_screenshots(white, cases, clims, log_scale=False, contours_percentile=None):
    screenshots = {}
    for name in cases:
        cmap = COLORMAPS[name.split(":")[0]]
        plotter = pv.Plotter(off_screen=True)

        plotter.add_mesh(
            white,
            copy_mesh=True,
            scalars=name,
            cmap=cmap,
            clim=clims,
            show_scalar_bar=False,
            smooth_shading=True,
            log_scale=log_scale,
        )
        if contours_percentile is not None:
            plotter.add_mesh(
                contours_percentile[name],
                color="black",
                show_scalar_bar=False,
                line_width=15,
                render_lines_as_tubes=True,
            )

        # First row     : view from side 1
        # Second row    : view from opposite side
        for i, options in enumerate([("xz", -50, -20, 20), ("xz", 140, 15, 50)]):
            plotter.camera_position = options[0]
            plotter.camera.azimuth = options[1]
            plotter.camera.roll += options[2]
            plotter.camera.elevation = options[3]
            # plotter.background_color = "pink"
            img = plotter.screenshot(
                return_img=True, scale=4, transparent_background=True
            )
            screenshots[(name, i)] = crop_img(img)
    return screenshots


def make_contours(white, cases):
    # THRESHOLD CONTOURS
    contour_percentile = 5

    contours = dict(threshold=None, difference=None, magnE=None, difference_magnE=None)
    contours_val = dict(
        threshold={}, difference=None, magnE=None, difference_magnE=None
    )

    contours["threshold"] = pv.MultiBlock()
    for k in cases["threshold"]:
        v = white[k]
        val = np.percentile(v[~np.isnan(v)], contour_percentile)
        print(f"{k} : {contour_percentile:d}th percentile : {val:.2f}")
        contours["threshold"][k] = white.contour(isosurfaces=[val], scalars=k)
        contours_val["threshold"][k] = val

    return contours, contours_val


def make_arrows_ap_view(ax):
    # A-P view annotations for A,P,M,L
    ann = ax.annotate(
        "",
        xy=(0.5, 0.4),
        xycoords="axes fraction",
        xytext=(1, 0.1),
        textcoords="axes fraction",
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="<->", shrinkA=0, shrinkB=0),
        annotation_clip=False,
        zorder=0,
    )
    ann.set_in_layout(False)
    ann = ax.text(0.8, 0, "P", transform=ax.transAxes)
    ann.set_in_layout(False)
    ann = ax.annotate(
        "",
        xy=(0.5, 0.4),
        xycoords="axes fraction",
        # xytext=(-0.125, 0.775),
        xytext=[-0.09375, 0.75625],
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="<->", shrinkA=0, shrinkB=0),
        annotation_clip=False,
        zorder=0,
    )
    ann.set_in_layout(False)
    ann = ax.text(0, 0.8, "A", transform=ax.transAxes)
    ann.set_in_layout(False)
    ann = ax.annotate(
        "",
        xy=(0.4, 0.4),
        xycoords="axes fraction",
        xytext=(0.9, 1.0),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="<->", shrinkA=0, shrinkB=0),
        annotation_clip=False,
        zorder=0,
    )
    ann.set_in_layout(False)
    ann = ax.text(0.9, 0.8, "M", transform=ax.transAxes)
    ann.set_in_layout(False)
    ann = ax.annotate(
        "",
        xy=(0.4, 0.4),
        xycoords="axes fraction",
        xytext=(0.025, -0.05),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="<->", shrinkA=0, shrinkB=0),
        annotation_clip=False,
        zorder=0,
    )
    ann.set_in_layout(False)
    ann = ax.text(0, 0.1, "L", transform=ax.transAxes)
    ann.set_in_layout(False)


def make_arrows_pa_view(ax):
    # P-A view annotations for A,P,M,L
    ann = ax.annotate(
        "",
        xy=(0.7, 0.4),
        xycoords="axes fraction",
        xytext=(0.925, -0.2),
        textcoords="axes fraction",
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="<->", shrinkA=0, shrinkB=0),
        annotation_clip=False,
        zorder=0,
    )
    ann.set_in_layout(False)
    ann = ax.text(0.75, -0.2, "A", transform=ax.transAxes)
    ann.set_in_layout(False)
    ann = ax.annotate(
        "",
        xy=(0.7, 0.4),
        xycoords="axes fraction",
        xytext=(0.4, 1.2),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="<->", shrinkA=0, shrinkB=0),
        annotation_clip=False,
        zorder=0,
    )
    ann.set_in_layout(False)
    ann = ax.text(0.5, 1.05, "P", transform=ax.transAxes)
    ann.set_in_layout(False)
    ann = ax.annotate(
        "",
        xy=(0.7, 0.4),
        xycoords="axes fraction",
        # xytext=(1.159375, 0.6296875),
        xytext=[1.0675, 0.58375],
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="<->", shrinkA=0, shrinkB=0),
        annotation_clip=False,
        zorder=0,
    )
    ann.set_in_layout(False)
    ann = ax.text(1.0, 0.35, "L", transform=ax.transAxes)
    ann.set_in_layout(False)
    ann = ax.annotate(
        "",
        xy=(0.7, 0.4),
        xycoords="axes fraction",
        xytext=(0.0, 0.05),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="<->", shrinkA=0, shrinkB=0),
        annotation_clip=False,
        zorder=0,
    )
    ann.set_in_layout(False)
    ann = ax.text(0, 0.15, "M", transform=ax.transAxes)
    ann.set_in_layout(False)


# Define some stuff and read surfaces
# -----------------------------------------------------------------------------
WAVEFORMS = ["monophasic", "c_tms_30", "c_tms_60", "c_tms_120"]
directions = ["ap", "pa"]
diameters = [1, 2, 3, 4, 5, 6]
conds = ["smooth", "default"]

domain_dir = Path(
    "/mnt/projects/INN/jesper/nobackup/projects/white_matter_axons/domain_M1"
)

FIG_DIR = Path(domain_dir / "simulation_02-02" / "figures")

white = pv.read(FIG_DIR / "white.vtk")
white_incl_nan = pv.read(FIG_DIR / "white_incl_nan.vtk")

imshow_kw = dict(interpolation="none", zorder=5)


# FIGURE 1: thresholds
# -----------------------------------------------------------------------------
cases = dict(
    threshold=["threshold:biphasic.ap.smooth.2", "threshold:monophasic.pa.smooth.2"],
    difference=["difference:biphasic.ap-pa.smooth.2"],
    magnE=["magnE:smooth"],
)
clims = dict(
    threshold=[200, 5000],
    difference=[
        white[cases["difference"][0]].min(),
        white[cases["difference"][0]].max(),
    ],
    magnE=[
        0.0,
        white[cases["magnE"][0]].max(),
    ],
)
ticks = dict(
    threshold=[250, 1000, 2500, 5000],
    difference=[0.25, 1.0, 4.0],
    magnE=[0, 100, 200, 100 * clims["magnE"][1]],
)
contours, contours_val = make_contours(white, cases)


images = {
    k: make_screenshots(white, cases[k], clims[k], False, contours[k]) for k in cases
}


fig, axes = plt.subplots(
    nrows=2,
    ncols=4,
    layout="constrained",
    figsize=(6.5, 6.5 * np.sqrt(2) * 0.4 * 2 / 3),
)
axes[0, 0].imshow(
    images["threshold"]["threshold:monophasic.ap.smooth.2", 0], **imshow_kw
)
axes[0, 1].imshow(
    images["threshold"]["threshold:monophasic.ap.smooth.2", 1], **imshow_kw
)
axes[0, 2].imshow(
    images["threshold"]["threshold:monophasic.pa.smooth.2", 0], **imshow_kw
)
axes[0, 3].imshow(
    images["threshold"]["threshold:monophasic.pa.smooth.2", 1], **imshow_kw
)

axes[1, 0].imshow(
    images["difference"]["difference:monophasic.ap-pa.smooth.2", 0],
    **imshow_kw,
)
axes[1, 1].imshow(
    images["difference"]["difference:monophasic.ap-pa.smooth.2", 1],
    **imshow_kw,
)
axes[1, 2].imshow(images["magnE"]["magnE:smooth", 0], **imshow_kw)
axes[1, 3].imshow(images["magnE"]["magnE:smooth", 1], **imshow_kw)

# remove ticks and spines
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# COLORBARS

# threshold
mappable = plt.cm.ScalarMappable(
    norm=Normalize(*clims["threshold"]), cmap=COLORMAPS["threshold"]
)
# ticks =
cbar = fig.colorbar(
    mappable,
    ax=axes[0, 2:],
    extend="max",
    ticks=ticks["threshold"],
    # format="{x:.2e}",
    label="Threshold\nin A/μs",
    aspect=15,
)
cbar.ax.minorticks_off()
cbar.ax.axhline(np.mean(tuple(contours_val["threshold"].values())), c="black")

# difference
mappable = plt.cm.ScalarMappable(
    norm=LogNorm(10 ** clims["difference"][0], 10 ** clims["difference"][1]),
    cmap=COLORMAPS["difference"],
)
cbar = fig.colorbar(
    mappable,
    ax=axes[1, 1],
    ticks=ticks["difference"],
    label=r"$\dfrac{\text{A-P}}{\text{P-A}}$",
    aspect=15,
)
cbar.ax.set_yticklabels([r"$\dfrac{1}{4}$", "1", "4"])
cbar.ax.set_in_layout(False)

# magnitude
mappable = plt.cm.ScalarMappable(
    norm=Normalize(clims["magnE"][0], 100 * clims["magnE"][1]), cmap=COLORMAPS["magnE"]
)
cbar = fig.colorbar(
    mappable,
    ax=axes[1, 2:],
    ticks=ticks["magnE"],
    format="{x:<3.0f}",
    label="|E| in V/m\nat 100 A/μs",
    aspect=15,
)
cbar.ax.minorticks_off()

# SUBPLOT TITLES
ann = axes[0, 0].text(
    0.9, 1, "A-P", ha="left", va="bottom", fontsize=14, transform=axes[0, 0].transAxes
)
ann.set_in_layout(False)
ann = axes[0, 2].text(
    0.9, 1, "P-A", ha="left", va="bottom", fontsize=14, transform=axes[0, 2].transAxes
)
ann.set_in_layout(False)

axes[0, 0].set_title(" ")
axes[0, 2].set_title(" ")

ann = axes[0, 0].text(
    -0.1, 1, "A", transform=axes[0, 0].transAxes, ha="left", fontsize=14
)
ann.set_in_layout(False)
ann = axes[1, 0].text(
    -0.1, 1, "B", transform=axes[1, 0].transAxes, ha="left", fontsize=14
)
ann.set_in_layout(False)
ann = axes[1, 2].text(
    -0.1, 1, "C", transform=axes[1, 2].transAxes, ha="left", fontsize=14
)
ann.set_in_layout(False)

make_arrows_ap_view(axes[0, 0])
make_arrows_pa_view(axes[0, 3])

fig.show()

print("saving smooth_bend_threshold")
fig.savefig(FIG_DIR / "smooth_bend_threshold.svg", bbox_inches="tight")
fig.savefig(FIG_DIR / "smooth_bend_threshold.png", bbox_inches="tight", dpi=600)

# Figure 2: smooth vs. default conductivities
# -----------------------------------------------------------------------------

cases = dict(
    threshold=[
        "threshold:monophasic.pa.smooth.2",
        "threshold:monophasic.pa.default.2",
    ],
    difference=[
        "difference:monophasic.pa.default-smooth.2",
    ],
    magnE=["magnE:smooth", "magnE:default"],
    difference_magnE=["difference:magnE:default-smooth"],
)

clims = dict(
    threshold=[200, 5000],
    difference=[-1, 1],
    #     np.min(
    #         [
    #             np.percentile(white[c][~np.isnan(white[c])], 1.0)
    #             for c in cases["difference"]
    #         ]
    #     ),
    #     np.max(
    #         [
    #             np.percentile(white[c][~np.isnan(white[c])], 99.0)
    #             for c in cases["difference"]
    #         ]
    #     ),
    # ],
    magnE=[
        0.0,
        np.max([white[c][~np.isnan(white[c])].max() for c in cases["magnE"]]),
    ],
    # difference_magnE=[-0.25, 0.25],
    difference_magnE=[
        np.min(
            [white[c][~np.isnan(white[c])].min() for c in cases["difference_magnE"]]
        ),
        np.max(
            [white[c][~np.isnan(white[c])].max() for c in cases["difference_magnE"]]
        ),
    ],
)
ticks = dict(
    threshold=[250, 1000, 2500, 5000],
    difference=[0.25, 1.0, 4.0],
    magnE=[0, 0.5, 1, 1.5, 2, clims["magnE"][1]],
    difference_magnE=[-0.15, 0.0, 0.15, 0.25],
)

contours, contours_val = make_contours(white, cases)


images = {k: make_screenshots(white, cases[k], clims[k], contours[k]) for k in cases}


fig, axes = plt.subplots(
    nrows=6,
    ncols=2,
    layout="constrained",
    figsize=(6.5, 6.5 * np.sqrt(2)),
)
axes[0, 0].imshow(
    images["threshold"]["threshold:monophasic.pa.smooth.2", 0], **imshow_kw
)
axes[0, 1].imshow(
    images["threshold"]["threshold:monophasic.pa.smooth.2", 1], **imshow_kw
)
axes[1, 0].imshow(
    images["threshold"]["threshold:monophasic.pa.default.2", 0], **imshow_kw
)
axes[1, 1].imshow(
    images["threshold"]["threshold:monophasic.pa.default.2", 1], **imshow_kw
)
axes[2, 0].imshow(
    images["difference"]["difference:monophasic.pa.default-smooth.2", 0],
    **imshow_kw,
)
axes[2, 1].imshow(
    images["difference"]["difference:monophasic.pa.default-smooth.2", 1],
    **imshow_kw,
)
axes[3, 0].imshow(images["magnE"]["magnE:smooth", 0], **imshow_kw)
axes[3, 1].imshow(images["magnE"]["magnE:smooth", 1], **imshow_kw)
axes[4, 0].imshow(images["magnE"]["magnE:default", 0], **imshow_kw)
axes[4, 1].imshow(images["magnE"]["magnE:default", 1], **imshow_kw)
axes[5, 0].imshow(
    images["difference_magnE"]["difference:magnE:default-smooth", 0],
    **imshow_kw,
)
axes[5, 1].imshow(
    images["difference_magnE"]["difference:magnE:default-smooth", 1],
    **imshow_kw,
)

# remove ticka and spines
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# axes[0, 0].set_title("A-P")
# axes[0, 2].set_title("P-A")
# axes[0, 0].set_ylabel("Threshold (2 μm)")
# axes[1, 0].set_ylabel("A-P/P-A")
# axes[1, 2].set_ylabel("|E|")

# COLORBARS

# threshold
mappable = plt.cm.ScalarMappable(
    norm=Normalize(*clims["threshold"]), cmap=COLORMAPS["threshold"]
)
# ticks =
cbar = fig.colorbar(
    mappable,
    ax=axes[:2],
    extend="max",
    ticks=ticks["threshold"],
    # format="{x:.2e}",
    label="Threshold in A/μs",
    shrink=0.7,
    aspect=15 * 2.0 * 0.7,
)
cbar.ax.minorticks_off()
cbar.ax.axhline(np.mean(tuple(contours_val["threshold"].values())), c="black")

# difference
mappable = plt.cm.ScalarMappable(
    norm=Normalize(*clims["difference"]), cmap=COLORMAPS["difference"]
)

cbar = fig.colorbar(
    mappable,
    ax=axes[2],
    # ticks=ticks["difference"],
    format="{x:4.2f}",
    label=r"$\dfrac{T_\text{iso}-T_\text{aniso}}{T_\text{iso}+T_\text{aniso}}$",
    aspect=15,
)
cbar.ax.minorticks_off()

# magnitude
mappable = plt.cm.ScalarMappable(
    norm=Normalize(*clims["magnE"]), cmap=COLORMAPS["magnE"]
)
cbar = fig.colorbar(
    mappable,
    ax=axes[3:5],
    ticks=ticks["magnE"],
    format="{x:4.2f}",
    label=r"$\left\vert \text{E} \right\vert $ at 1 A/μs",
    shrink=0.7,
    aspect=15 * 2 * 0.7,
)
cbar.ax.minorticks_off()

# magnitude difference
mappable = plt.cm.ScalarMappable(
    norm=Normalize(*clims["difference_magnE"]), cmap=COLORMAPS["difference"]
)
cbar = fig.colorbar(
    mappable,
    ax=axes[5],
    ticks=ticks["difference_magnE"],
    format="{x:4.2f}",
    label=r"$\left\Vert E_\text{iso} \right\Vert- \left\Vert E_\text{aniso} \right\Vert$",
    # label=r"$(e-d)/(e+d)$",
    aspect=15,
)
cbar.ax.minorticks_off()

fig.show()
print("Saving smooth_bend_conductivity_comparison")
fig.savefig(FIG_DIR / "smooth_bend_conductivity_comparison.svg")
fig.savefig(FIG_DIR / "smooth_bend_conductivity_comparison.png", dpi=600)


# FIGURE 3. Line plots
# -----------------------------------------------------------------------------

roi_mask = np.load(domain_dir / "surfaces_roi" / "roi_surface_mask.npy")
roi_mask = np.concat((roi_mask, np.zeros_like(roi_mask)))

ba_on_surface = pv.read(
    "/mnt/projects/INN/bungert_revisited/subject_5/m2m_subject_5/BA/white_roi_BA.vtk"
)

s = cortech.Surface.from_file(domain_dir / "surfaces_roi" / "roi_surface_wm.vtk")
e = s.find_border_edges()
remove_faces = np.isin(s.faces, e).all(1)
original_roi_index = np.unique(s.faces[~remove_faces])

# roi_mask[
#     np.flatnonzero(roi_mask)[
#         np.setdiff1d(np.arange(roi_mask.sum()), original_roi_index)
#     ]
# ] = False
# roi_mask[np.flatnonzero(roi_mask)[~not_nan]] = False

ba_on_surface["roi"] = roi_mask

ba_on_surface_roi = ba_on_surface.threshold(
    True, scalars="roi", all_scalars=True
).extract_surface()
ba_6_mask = ba_on_surface_roi["BA_map"] == 1
ba_4a_mask = ba_on_surface_roi["BA_map"] == 2
ba_4b_mask = ba_on_surface_roi["BA_map"] == 3
masks = [m[original_roi_index] for m in [ba_6_mask, ba_4a_mask, ba_4b_mask]]

mask_titles = [
    r"$\mathregular{BA6_{HAND}}$",
    r"$\mathregular{BA4a_{HAND}}$",
    r"$\mathregular{BA4p_{HAND}}$",
]

thresholds = np.array(
    [
        [
            white_incl_nan[f"threshold:monophasic.{direction}.smooth.{d}"]
            for direction in directions
        ]
        for d in diameters
    ]
)

fig, axs = plt.subplots(
    nrows=3,
    ncols=2,
    layout="constrained",
    figsize=(6.5, 6.5 * np.sqrt(2) * 0.75),
    sharex=True,
    sharey=True,
)
axs[0, 0].set_title("A-P")
axs[0, 1].set_title("P-A")
axs[0, 0].set_xticks(diameters)
axs[0, 1].set_xticks(diameters)
axs[-1, 0].set_xlabel("axon diameter in μm")
axs[-1, 1].set_xlabel("axon diameter in μm")

for i, mask_title in zip(range(3), mask_titles):
    axs[i, 0].set_ylabel("threshold in A/μs")
    x = axs[i, 0].twinx()
    x.yaxis.set_label_position("left")
    x.spines["left"].set_position(("axes", -0.45))
    x.spines["left"].set_visible(False)
    x.set_yticks([])
    x.set_ylabel(mask_title, rotation=0, size=10, ha="center", va="center")

percentiles = [0.0, 5, 10, 20, 50]
for ax_idx, (mask, mask_title) in enumerate(zip(masks, mask_titles)):
    for color_idx, percentile in enumerate(percentiles):
        for ax_2_idx, direction in enumerate(["A-P", "P-A"]):
            percentile_threshold = [
                np.percentile(
                    thresholds[i, ax_2_idx, mask][
                        ~np.isnan(thresholds[i, ax_2_idx, mask])
                    ],
                    percentile,
                )
                for i in range(thresholds.shape[0])
            ]
            axs[ax_idx, ax_2_idx].plot(
                diameters,
                percentile_threshold,
                color=colors[color_idx],
                marker=markers[0],
                zorder=5,
            )
            axs[ax_idx, ax_2_idx].set_yscale("log")
            axs[ax_idx, ax_2_idx].set_xlim(1 - 0.15, 6.15)
            axs[ax_idx, ax_2_idx].hlines(
                0.3844197463989258 * 155.3,
                *axs[ax_idx, ax_2_idx].get_xlim(),
                color="black",
                linestyle="--",
                lw=1,
            )
            axs[ax_idx, ax_2_idx].hlines(
                155.3,
                *axs[ax_idx, ax_2_idx].get_xlim(),
                color="black",
                linestyle="--",
                lw=1,
            )

            axs[ax_idx, ax_2_idx].set_ylim(30, 20000)

color_labels = [
    "Min",
    "5th %",
    "10th %",
    "20th %",
    "50th % (median)",
]

color_handles = [
    Line2D([0], [0], color=color, linestyle="-", label=name)
    for color, name in zip(colors, color_labels)
]
leg = fig.legend(handles=color_handles, loc="outside lower center", ncol=5)

for ax in axs.flat:
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.5)

fig.show()

print("Saving smooth_bend_activation_percentiles")
fig.savefig(FIG_DIR / "smooth_bend_activation_percentiles.svg")
fig.savefig(FIG_DIR / "smooth_bend_activation_percentiles.png", dpi=600)


# Figure 4. Time constants
# -----------------------------------------------------------------------------

k = "threshold:c_tms.pa.smooth.6"

cases = dict(
    time_constant=[f"time_constant:{k}"],
    rheobase=[f"rheobase:{k}"],
    residual=[f"residual:{k}"],
)
clims = dict(
    # time_constant=np.percentile(white[f"time_constant:{k}"], (0, 100)),
    time_constant=(0, 1000),
    # rheobase=np.percentile(white[f"rheobase:{k}"], (5, 95)),
    rheobase=(10, 500),
    # residual=np.percentile(white[f"residual:{k}"], (0, 100)),
    residual=(1e-4, 5e-2),
)
ticks = dict(
    time_constant=None,
    rheobase=None,
    residual=None,
)
logscale = dict(time_constant=False, rheobase=True, residual=True)
# contours, contours_val = make_contours(white, cases)

contours = {k: None for k in cases}


images = {
    k: make_screenshots(white, cases[k], clims[k], logscale[k], contours[k])
    for k in cases
}

fig, axes = plt.subplots(
    nrows=3,
    ncols=2,
    layout="constrained",
    figsize=(6.5, 6.5 * np.sqrt(2) * 0.4 * 2),
)
axes[0, 0].imshow(images["time_constant"][f"time_constant:{k}", 0], **imshow_kw)
axes[0, 1].imshow(images["time_constant"][f"time_constant:{k}", 1], **imshow_kw)
axes[1, 0].imshow(images["rheobase"][f"rheobase:{k}", 0], **imshow_kw)
axes[1, 1].imshow(images["rheobase"][f"rheobase:{k}", 1], **imshow_kw)
axes[2, 0].imshow(images["residual"][f"residual:{k}", 0], **imshow_kw)
axes[2, 1].imshow(images["residual"][f"residual:{k}", 1], **imshow_kw)

axes[0, 0].set_ylabel("Time constant")
axes[1, 0].set_ylabel("Rheobase")
axes[2, 0].set_ylabel("Residual")

# remove ticks and spines
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# time constant
mappable = plt.cm.ScalarMappable(
    norm=Normalize(*clims["time_constant"]), cmap=COLORMAPS["time_constant"]
)
cbar = fig.colorbar(
    mappable,
    ax=axes[0],
    # ticks=[100, 500, 1000],
    extend="max",
    # format="{x:4d}",
    label="μs",
    aspect=15,
)
# cbar.set_ticks([100, 500, 1000],labels=list(map(str,[100, 500, 1000])))  # After creation

# rheobase
mappable = plt.cm.ScalarMappable(
    norm=LogNorm(*clims["rheobase"]), cmap=COLORMAPS["rheobase"]
)
cbar = fig.colorbar(
    mappable,
    ax=axes[1],
    # ticks=[1, 10, 100, 500],
    extend="both",
    # format="{x:3d}",
    label="A/μs",
    aspect=15,
)

# residual
mappable = plt.cm.ScalarMappable(
    norm=LogNorm(*clims["residual"]), cmap=COLORMAPS["residual"]
)
cbar = fig.colorbar(
    mappable,
    ax=axes[2],
    # ticks=ticks["difference"],
    # format="{x:4.2f}",
    aspect=15,
)

fig.show()

print("Saving smooth_bend_time_constant")
fig.savefig(FIG_DIR / "smooth_bend_time_constant.svg")
fig.savefig(FIG_DIR / "smooth_bend_time_constant.png", dpi=600)


# FIGURE 5. Thresholds for biphasic TMS pulse (like figure 1)
# -----------------------------------------------------------------------------
cases = dict(
    threshold=["threshold:biphasic.ap.smooth.2", "threshold:biphasic.pa.smooth.2"],
    difference=["difference:biphasic.ap-pa.smooth.2"],
    magnE=["magnE:smooth"],
)
clims = dict(
    threshold=[200, 5000],
    difference=[
        white[cases["difference"][0]].min(),
        white[cases["difference"][0]].max(),
    ],
    magnE=[
        0.0,
        white[cases["magnE"][0]].max(),
    ],
)
ticks = dict(
    threshold=[250, 1000, 2500, 5000],
    # difference=[0.25, 1.0, 4.0],
    difference=[1 / 1.35, 1.0, 1.35],
    magnE=[0, 100, 200, 100 * clims["magnE"][1]],
)
contours, contours_val = make_contours(white, cases)


images = {
    k: make_screenshots(white, cases[k], clims[k], False, contours[k]) for k in cases
}


fig, axes = plt.subplots(
    nrows=2,
    ncols=4,
    layout="constrained",
    figsize=(6.5, 6.5 * np.sqrt(2) * 0.4 * 2 / 3),
)
axes[0, 0].imshow(images["threshold"]["threshold:biphasic.ap.smooth.2", 0], **imshow_kw)
axes[0, 1].imshow(images["threshold"]["threshold:biphasic.ap.smooth.2", 1], **imshow_kw)
axes[0, 2].imshow(images["threshold"]["threshold:biphasic.pa.smooth.2", 0], **imshow_kw)
axes[0, 3].imshow(images["threshold"]["threshold:biphasic.pa.smooth.2", 1], **imshow_kw)

axes[1, 0].imshow(
    images["difference"]["difference:biphasic.ap-pa.smooth.2", 0],
    **imshow_kw,
)
axes[1, 1].imshow(
    images["difference"]["difference:biphasic.ap-pa.smooth.2", 1],
    **imshow_kw,
)
axes[1, 2].imshow(images["magnE"]["magnE:smooth", 0], **imshow_kw)
axes[1, 3].imshow(images["magnE"]["magnE:smooth", 1], **imshow_kw)

# remove ticks and spines
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# COLORBARS

# threshold
mappable = plt.cm.ScalarMappable(
    norm=Normalize(*clims["threshold"]), cmap=COLORMAPS["threshold"]
)
cbar = fig.colorbar(
    mappable,
    ax=axes[0, 2:],
    extend="max",
    ticks=ticks["threshold"],
    # format="{x:.2e}",
    label="Threshold\nin A/μs",
    aspect=15,
)
cbar.ax.minorticks_off()
cbar.ax.axhline(np.mean(tuple(contours_val["threshold"].values())), c="black")

# difference
mappable = plt.cm.ScalarMappable(
    norm=LogNorm(10 ** clims["difference"][0], 10 ** clims["difference"][1]),
    cmap=COLORMAPS["difference"],
)
cbar = fig.colorbar(
    mappable,
    ax=axes[1, 1],
    ticks=ticks["difference"],
    label=r"$\dfrac{\text{AP-PA}}{\text{PA-AP}}$",
    aspect=15,
)
cbar.ax.set_yticklabels([r"$\dfrac{1}{1.35}$", "1", "1.35"])
cbar.ax.yaxis.set_minor_formatter(plt.NullFormatter())
cbar.ax.set_in_layout(False)

# magnitude
mappable = plt.cm.ScalarMappable(
    norm=Normalize(clims["magnE"][0], 100 * clims["magnE"][1]), cmap=COLORMAPS["magnE"]
)
cbar = fig.colorbar(
    mappable,
    ax=axes[1, 2:],
    ticks=ticks["magnE"],
    format="{x:<3.0f}",
    label="|E| in V/m\nat 100 A/μs",
    aspect=15,
)
cbar.ax.minorticks_off()

# SUBPLOT TITLES
ann = axes[0, 0].text(
    0.775,
    1,
    "AP-PA",
    ha="left",
    va="bottom",
    fontsize=14,
    transform=axes[0, 0].transAxes,
)
ann.set_in_layout(False)
ann = axes[0, 2].text(
    0.775,
    1,
    "PA-AP",
    ha="left",
    va="bottom",
    fontsize=14,
    transform=axes[0, 2].transAxes,
)
ann.set_in_layout(False)

axes[0, 0].set_title(" ")
axes[0, 2].set_title(" ")

ann = axes[0, 0].text(
    -0.1, 1, "A", transform=axes[0, 0].transAxes, ha="left", fontsize=14
)
ann.set_in_layout(False)
ann = axes[1, 0].text(
    -0.1, 1, "B", transform=axes[1, 0].transAxes, ha="left", fontsize=14
)
ann.set_in_layout(False)
ann = axes[1, 2].text(
    -0.1, 1, "C", transform=axes[1, 2].transAxes, ha="left", fontsize=14
)
ann.set_in_layout(False)

make_arrows_ap_view(axes[0, 0])
make_arrows_pa_view(axes[0, 3])

fig.show()

print("saving smooth_bend_threshold_biphasic")
fig.savefig(FIG_DIR / "smooth_bend_threshold_biphasic.svg", bbox_inches="tight")
fig.savefig(
    FIG_DIR / "smooth_bend_threshold_biphasic.png", bbox_inches="tight", dpi=600
)
