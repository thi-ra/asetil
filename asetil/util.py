import numpy as np
from ase import units


def calc_thermal_de_broglie(molecular_weight, beta):
    """
    Calculate the thermal de Broglie wavelength in Ang.
    molecular_weight: float
        Molecular weight in g/mol.
    beta: float
        Inverse temperature in 1/eV.
    """

    # constants
    meter_to_ang = 1e10  # m -> Ang.
    Nav = units._Nav  # mol^-1
    h_planck = units._hplanck  # J s

    # convert units
    molecular_weight *= 1e-3  # kg/mol
    mass = molecular_weight / Nav  # kg
    beta = (
        beta / units.J
    )  # 1/J, "* units.J" represents "convert eV to J", so "/ units.J" represents "convert 1/ev to 1/J"

    # thermal de Broglie wavelength
    dbl = h_planck / np.sqrt(2 * np.pi * mass / beta) * meter_to_ang  # Ang.
    return dbl
