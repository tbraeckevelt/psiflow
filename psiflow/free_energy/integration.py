from __future__ import annotations  # necessary for type-guarding class methods

from typing import Optional, Union

import numpy as np
import typeguard
from ase.units import bar, kB
from parsl.app.app import python_app

from psiflow.data import Dataset
from psiflow.hamiltonians import Hamiltonian, Zero
from psiflow.sampling import SimulationOutput, Walker, sample
from psiflow.sampling.walker import quench, randomize
from psiflow.utils.apps import multiply, concatenate, take_mean, compute_sum


def cumtrapz(x, y, xbegin=None, xend=None):
    int_sum = 0
    com_integral = np.zeros(len(x), dtype=float)
    for i in range(len(x)):
        if i == 0:
            if xbegin is None:
                dx = 0
            else:
                dx = x[i] - xbegin
        else:
            dx = (x[i] - x[i - 1]) / 2
        if i == len(x) - 1:
            if xend is None:
                dx += 0
            else:
                dx += xend - x[i]
        else:
            dx += (x[i + 1] - x[i]) / 2
        int_sum += dx * y[i]
        com_integral[i] = int_sum
    return com_integral


def cumtrapz_error(x, yerr, xbegin=None, xend=None):
    error_sum = 0
    com_integral_error = np.zeros(len(x), dtype=float)
    for i in range(len(x)):
        if i == 0:
            if xbegin is None:
                dx = 0
            else:
                dx = x[i] - xbegin
        else:
            dx = (x[i] - x[i - 1]) / 2
        if i == len(x) - 1:
            if xend is None:
                dx += 0
            else:
                dx += xend - x[i]
        else:
            dx += (x[i + 1] - x[i]) / 2
        error_sum += (dx * yerr[i]) ** 2
        com_integral_error[i] = np.sqrt(error_sum)
    return com_integral_error


@typeguard.typechecked
def _integrate(
    x: np.ndarray,
    *args: float,
    xbegin: Optional[float] = None,
    xend: Optional[float] = None,
) -> np.ndarray:
    assert len(args) == len(x)
    y = np.array(args, dtype=float)
    return cumtrapz(x, y, xbegin=xbegin, xend=xend)


integrate = python_app(_integrate, executors=["default_threads"])


@typeguard.typechecked
def _integrate_error(
    x: np.ndarray,
    *args: float,
    xbegin: Optional[float] = None,
    xend: Optional[float] = None,
) -> np.ndarray:
    assert len(args) == len(x)
    yerr = np.array(args, dtype=float)
    return cumtrapz_error(x, yerr, xbegin=xbegin, xend=xend)


integrate_error = python_app(_integrate_error, executors=["default_threads"])


def _take_error(input_array: np.ndarray, nblocks: int) -> float:
    assert nblocks <= len(
        input_array
    ), "nblocks must not be greater than the length of the input array."
    assert nblocks > 1, "nblocks must be greater than 1."
    blocks = np.array_split(input_array, nblocks)
    errors = np.std(
        [np.mean(block) for block in blocks]
    ) / np.sqrt(nblocks - 1)
    return errors


take_error = python_app(_take_error, executors=["default_threads"])


@typeguard.typechecked
class ThermodynamicState_own:
    temperature: float
    natoms: int
    pressure: Optional[float]
    mass: Optional[float]

    def __init__(
        self,
        temperature: float,
        natoms: int,
        pressure: Optional[float],
        mass: Optional[float],
        nblocks: int = 4,
    ):
        self.temperature = temperature
        self.natoms = natoms
        self.pressure = pressure
        self.mass = mass
        self.nblocks = nblocks

        self.gradients = {
            "temperature": None,
            "delta": None,
            "pressure": None,
            "mass": None,
        }

        self.gradients_error = {
            "temperature": None,
            "delta": None,
            "pressure": None,
            "mass": None,
        }

    def gradient(
        self,
        outputs: list[SimulationOutput],
        delta_hamiltonian: Optional[Hamiltonian] = None,
    ):

        self.temperature_gradient(outputs)
        self.delta_gradient(outputs, delta_hamiltonian)
        if self.mass is not None:
            self.mass_gradient(outputs)

    def temperature_gradient(self, outputs: list[SimulationOutput]):
        output_e = [o["potential{electronvolt}"] for o in outputs]
        energies = concatenate(*output_e)
        _energy = take_mean(energies)
        _energy_error = take_error(energies, self.nblocks)
        if self.pressure is not None:  # use enthalpy
            output_v = [o["volume{angstrom3}"] for o in outputs]
            volumes = concatenate(*output_v)
            pv = multiply(take_mean(volumes), 10 * bar * self.pressure)
            pv_error = multiply(
                take_error(volumes, self.nblocks), 10 * bar * self.pressure
            )
            _energy = compute_sum(_energy, pv)
            _energy_error = compute_sum(_energy_error, pv_error)

        # grad_u = < - u / kBT**2 >
        # grad_k = < - E_kin > / kBT**2 >
        gradient_u = multiply(_energy, (-1.0) / (kB * self.temperature**2))
        gradient_k = (-1.0) * (3 * self.natoms - 3) / (2 * self.temperature)
        self.gradients["temperature"] = compute_sum(gradient_u, gradient_k)

        gradient_u_error = multiply(
            _energy_error, (1.0) / (kB * self.temperature**2)
        )
        self.gradients_error["temperature"] = gradient_u_error

    def delta_gradient(
        self, outputs: list[SimulationOutput], delta_hamiltonian: Hamiltonian
    ):
        output_e = [o.get_energy(delta_hamiltonian) for o in outputs]
        energies = concatenate(*output_e)
        self.gradients["delta"] = multiply(
            take_mean(energies), 1 / (kB * self.temperature)
        )
        self.gradients_error["delta"] = multiply(
            take_error(energies, self.nblocks), 1 / (kB * self.temperature)
        )

    def mass_gradient(self, outputs):
        raise NotImplementedError


@typeguard.typechecked
class Integration_own:
    def __init__(
        self,
        hamiltonian: Hamiltonian,
        temperatures: Union[list[float], np.ndarray],
        delta_hamiltonian: Optional[Hamiltonian] = None,
        delta_coefficients: Union[list[float], np.ndarray, None] = None,
        pressure: Optional[float] = None,
        natoms: Optional[int] = None,
        nblocks: int = 4,
        npara: int = 1,
    ):
        self.hamiltonian = hamiltonian
        self.temperatures = np.array(temperatures, dtype=float)
        if delta_hamiltonian is not None:
            assert delta_coefficients is not None
            self.delta_hamiltonian = delta_hamiltonian
            self.delta_coefficients = np.array(delta_coefficients, dtype=float)
        else:
            self.delta_coefficients = np.array([0.0])
            self.delta_hamiltonian = Zero()
        self.pressure = pressure
        self.natoms = natoms
        self.nblocks = nblocks
        self.npara = npara

        assert len(np.unique(self.temperatures)) == len(self.temperatures)
        unique_deltas = np.unique(self.delta_coefficients)
        assert len(unique_deltas) == len(self.delta_coefficients)

        self.states = []  # length: ndeltas * ntemperatures
        self.walkers = []  # length: ndeltas * ntemperatures * npara
        self.outputs = []  # length: ndeltas * ntemperatures * npara

    def create_walkers(
        self,
        dataset: Dataset,
        initialize_by: str = "quench",
        **walker_kwargs,
    ) -> list[Walker]:
        if self.natoms is None:
            self.natoms = len(dataset[0].result())
        for delta in self.delta_coefficients:
            for T in self.temperatures:
                hamiltonian = self.hamiltonian + delta * self.delta_hamiltonian
                for _ in range(self.npara):
                    walker = Walker(
                        dataset[0],  # do quench later
                        hamiltonian,
                        temperature=T,
                        **walker_kwargs,
                    )
                    self.walkers.append(walker)
                state = ThermodynamicState_own(
                    temperature=T,
                    natoms=self.natoms,
                    pressure=self.pressure,
                    mass=None,
                    nblocks=self.nblocks,
                )
                self.states.append(state)

        # initialize walkers
        if initialize_by == "quench":
            quench(self.walkers, dataset)
        elif initialize_by == "shuffle":
            randomize(self.walkers, dataset)
        else:
            raise ValueError("unknown initialization")
        return self.walkers

    def sample(self, **sampling_kwargs):
        self.outputs[:] = sample(
            self.walkers,
            **sampling_kwargs,
        )

    def compute_gradients(self):
        for i, state in enumerate(self.states):
            outputs = self.outputs[i * self.npara: (i + 1) * self.npara]
            state.gradient(outputs, delta_hamiltonian=self.delta_hamiltonian)

    def along_delta(self, temperature: Optional[float] = None):
        if temperature is None:
            assert self.ntemperatures == 1
            temperature = self.temperatures[0]
        index = np.where(self.temperatures == temperature)[0][0]
        assert self.temperatures[index] == temperature
        N = self.ntemperatures
        states = [self.states[N * i + index] for i in range(self.ndeltas)]

        # do integration
        x = self.delta_coefficients
        y = [state.gradients["delta"] for state in states]
        e = [state.gradients_error["delta"] for state in states]
        f = integrate(
            x, *y, xbegin=0.0, xend=1.0
        )  # assure that the integration starts at 0.0 and ends at 1.0
        f_error = integrate_error(
            x, *e, xbegin=0.0, xend=1.0
        )  # assure that the integration starts at 0.0 and ends at 1.0
        return f, f_error
        # return multiply(f, kB * temperature)

    def along_temperature(self, delta_coefficient: Optional[float] = None):
        if delta_coefficient is None:
            assert self.ndeltas == 1
            delta_coefficient = self.delta_coefficients[0]
        index = np.where(self.delta_coefficients == delta_coefficient)[0][0]
        assert self.delta_coefficients[index] == delta_coefficient
        N = self.ntemperatures
        states = [
            self.states[N * index + i] for i in range(self.ntemperatures)
        ]

        # do integration
        x = self.temperatures
        y = [state.gradients["temperature"] for state in states]
        e = [state.gradients_error["temperature"] for state in states]
        f = integrate(x, *y)
        f_error = integrate_error(x, *e)
        return f, f_error
        # return multiply(f, kB * self.temperatures)

    @property
    def ntemperatures(self):
        return len(self.temperatures)

    @property
    def ndeltas(self):
        return len(self.delta_coefficients)
