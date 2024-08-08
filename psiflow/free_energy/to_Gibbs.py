from ase import Atoms
import psiflow
from psiflow.data.dataset import Dataset, write_frames
from parsl.dataflow.futures import AppFuture
from psiflow.geometry import Geometry
from psiflow.utils.apps import _running_average, _calculate_hist
from parsl.app.app import join_app, python_app
import numpy as np
from ase.units import kB
import typeguard


@typeguard.typechecked
def _get_average_cell(NPT_geos: list[Geometry]) -> np.ndarray:
    ave_cell = np.mean([geo.cell for geo in NPT_geos], axis=0)
    return ave_cell


get_average_cell = python_app(_get_average_cell, executors=["default_threads"])


@typeguard.typechecked
def _get_scaled_geo(geo: Geometry, ave_cell: np.ndarray) -> Geometry:
    atoms = Atoms(
        numbers=geo.per_atom.numbers[:],
        positions=geo.per_atom.positions[:, :],
        pbc=True,
        cell=geo.cell,
    )
    atoms.set_cell(ave_cell, scale_atoms=True)
    return Geometry.from_atoms(atoms)


get_scaled_geo = python_app(_get_scaled_geo, executors=["default_threads"])


@join_app
@typeguard.typechecked
def _get_scaled_data(
    input_geos: list[Geometry],
    ave_cell: np.ndarray,
    outputs: list = []
) -> AppFuture:
    scaled_geos_lst = []
    for geo in input_geos:
        scaled_geos_lst.append(get_scaled_geo(geo, ave_cell))
    return write_frames(*scaled_geos_lst, outputs=[outputs[0]])


@typeguard.typechecked
def get_scaled_data(dataset: Dataset, *args) -> Dataset:
    extxyz = _get_scaled_data(
        dataset.geometries(),
        *args,
        outputs=[psiflow.context().new_file("data_", ".xyz")],
    ).outputs[0]
    return Dataset(None, extxyz)


@typeguard.typechecked
def _to_Gibbs_correction(
    cell: np.ndarray,
    NPT_geos: list[Geometry],
    temperature: float,
    pressure: float,
    bin_size: float = 0.1,  # bin size in Angstrom^3
    window: int = 7,
) -> float:
    """
    Calculates the normalized Gibbs correction term.
    G(P,T)/(k_B*T) = F(V,T)/(k_B*T) + correction_term_norm
    correction_term_norm = P*V/(k_B*T) + log(prob(V|P,T))

    Args:
        cell (np.ndarray): The cell of the NVT system.
        NPT_geos (list[Geometry]): List of geometries at different NPT steps.
        temperature (float): The temperature of the system.
        pressure (float): The pressure of the system.
        bin_size (float, optional): The bin size in Angstrom^3 for
                                    histogram calculation. Defaults to 0.1.
        window (int, optional): The window size for running average.
                                Defaults to 7.

    Returns:
        float: The normalized Gibbs correction term.

    """
    vol_lst = [np.linalg.det(geo.cell) for geo in NPT_geos]
    hist, center_bin = _calculate_hist(vol_lst, bin_size)
    run_hist = _running_average(hist, window)
    run_bin = _running_average(center_bin, window)
    vol = np.linalg.det(cell)
    min_diff = np.abs(vol - run_bin[0])
    hist_opt = run_hist[0]
    for hist_val, bin_val in zip(run_hist, run_bin):
        diff_vol = np.abs(vol - bin_val)
        if diff_vol <= min_diff:
            hist_opt = hist_val
            min_diff = diff_vol
    return pressure * vol / (kB*temperature) + np.log(hist_opt)


to_Gibbs_correction = python_app(
    _to_Gibbs_correction, executors=["default_threads"]
)
