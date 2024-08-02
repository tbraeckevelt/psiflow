from psiflow.sampling.walker import replica_exchange
from psiflow.sampling import sample
from psiflow.free_energy.integration import Integration
import typeguard
from typing import Optional
from psiflow.data import Dataset
from psiflow.hamiltonians import Hamiltonian


import numpy as np


@typeguard.typechecked
def create_temperatures(
    tem_min: float,
    tem_max: float,
    num_tem: int,
) -> np.ndarray:
    """
    Create an array of temperatures using exponential spacing.

    Parameters:
    tem_min (float): The minimum temperature.
    tem_max (float): The maximum temperature.
    num_tem (int): The number of temperatures to generate.

    Returns:
    numpy.ndarray: An array of temperatures.
    """
    return tem_min * np.exp(np.linspace(0.0, np.log(tem_max/tem_min), num_tem))


@typeguard.typechecked
def perform_integration(
    hamiltonian: Hamiltonian,
    input_data: Dataset,
    temperatures: Optional[np.ndarray] = None,
    delta_hamil: Optional[float] = None,
    delta_coefficients: Optional[float] = None,
    pressure: Optional[float] = None,
    initialize_by: str = "quench",
    natoms: Optional[int] = None,
    nblocks: int = 4,
    npara: int = 1,
    timestep: float = 0.5,
    trial_frequency: int = 10,
    steps: int = 100,
    step: int = 1,
    calibration: int = 10,
) -> Integration:
    """
    Combination of all steps necessary to perform thermodynamic integration.

    Args:
        hamiltonian: The Hamiltonian object used for the integration.
        input_data: The input data for creating walkers.
        temperatures: The temperatures for the integration (default: None).
        delta_hamil: The delta Hamiltonian for the integration (default: None).
        delta_coefficients: The delta coefficients for the integration
                            (default: None).
        pressure: The pressure for the integration (default: None).
        initialize_by: The initialization method for walkers
                        (default: "quench").
        natoms: The number of atoms (default: None). If not provided,
                it will be inferred from the input data via a result call,
                so futures will get resolved.
        nblocks: The number of blocks to determine the error (default: 4).
        npara: The number of parallel walkers per setting (default: 1).
        timestep: The timestep for the walkers (default: 0.5).
        trial_frequency: The trial frequency for replica exchange
                        (default: 10).
        steps: The total number of steps to sample each walker (default: 100).
        step: The step size for the walker (default: 1).
        calibration: The calibration value for the integration (default: 10).

    Returns:
        The Integration object containing the integration outputs.
    """
    integration = Integration(
        hamiltonian,
        temperatures=temperatures,
        delta_hamiltonian=delta_hamil,
        delta_coefficients=delta_coefficients,
        pressure=pressure,
        natoms=natoms,
        nblocks=nblocks,
        npara=npara,
    )
    walkers = integration.create_walkers(
        input_data, timestep=timestep, initialize_by=initialize_by
    )
    for j in range(npara):
        replica_exchange(
            walkers[j:: npara],
            trial_frequency=trial_frequency
        )
    integration.outputs = sample(
        walkers,
        steps=steps,
        step=step,
        start=calibration,
    )
    integration.compute_gradients()
    return integration
