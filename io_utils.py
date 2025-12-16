import h5py
import numpy as np
import pyvista as pv


def projections_as_multiblock(points, n_iter, vertex_data, line_data, indices=None):
    n = len(points)
    indices = np.arange(n) if indices is None else indices

    lines = {}
    for i in indices:
        p = pv.MultipleLines(points=points[i, : n_iter[i] + 1])
        if vertex_data is not None:
            p.point_data.update(
                {k: v[i, : n_iter[i] + 1] for k, v in vertex_data.items()}
            )
        if line_data is not None:
            p.cell_data.update({k: v[i] for k, v in line_data.items()})
        lines[str(i)] = p

    return pv.MultiBlock(lines)


def projections_as_hdf5(filename, points, vertex_data, line_data):
    with h5py.File(filename, "w") as f:
        f.create_dataset("points", data=points)

        gr_vd = f.create_group("vertex_data")
        for k, v in vertex_data.items():
            gr_vd.create_dataset(k, data=v)

        gr_ld = f.create_group("line_data")
        for k, v in line_data.items():
            gr_ld.create_dataset(k, data=v)


def projections_load_single(filename, index):
    with h5py.File(filename, "r") as f:
        n = f["line_data"]["n_iter"][index]
        points = f["points"][index, : n + 1]
        vertex_data = {k: ds[index, : n + 1] for k, ds in f["vertex_data"].items()}
        line_data = {k: ds[index] for k, ds in f["line_data"].items()}
    return points, vertex_data, line_data


def projections_load_all(filename):
    with h5py.File(filename, "r") as f:
        points = f["points"][:]
        vertex_data = {k: ds[:] for k, ds in f["vertex_data"].items()}
        line_data = {k: ds[:] for k, ds in f["line_data"].items()}
    return points, vertex_data, line_data
