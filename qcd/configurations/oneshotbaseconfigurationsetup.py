from . import SetupConfiguration
from typing import Dict, List, Tuple
import numpy as np


class OneShotSetupConfiguration(SetupConfiguration):
    """ Representation of the One Shot Setup configuration """

    def __init__(self, setup: Dict) -> None:
        self._points_theta = setup['points_theta']
        if self._points_theta is None:
            raise ValueError('points_theta is required')
        self._points_phase = setup['points_phase']
        if self._points_phase is None:
            raise ValueError('points_phase is required')
        self._attenuation_factors = setup['attenuation_factors']
        if self._attenuation_factors is None:
            raise ValueError('attenuation_factors is required')
        self._angles_rx = setup['angles_rx']
        if self._angles_rx is None:
            raise ValueError('angles_rx is required')
        self._angles_ry = setup['angles_ry']
        if self._angles_ry is None:
            raise ValueError('angles_ry is required')

        self._angles_theta_interval = (0, np.pi / 2)
        self._angles_phase_interval = (0, 2 * np.pi)
        self._angles_rx_interval = (0, 2 * np.pi)
        self._angles_ry_interval = (0, 2 * np.pi)
        self._angles_theta = np.mgrid[0:np.pi / 2:self._points_theta * 1j]
        self._angles_phase = np.mgrid[0:2 * np.pi:self._points_phase * 1j]
        self._attenuation_angles = list(map(lambda attenuation_factor: np.arcsin(
            np.sqrt(attenuation_factor)), self._attenuation_factors))

    @property
    def points_theta(self) -> List[int]:
        return self._points_theta

    @property
    def points_phase(self) -> List[float]:
        return self._points_phase

    @property
    def angles_theta(self) -> List[float]:
        return self._angles_theta

    @property
    def angles_phase(self) -> List[float]:
        return self._angles_phase

    @property
    def angles_rx(self) -> List[float]:
        return self._angles_rx

    @property
    def angles_ry(self) -> List[float]:
        return self._angles_ry

    @property
    def attenuation_factors(self) -> List[float]:
        return self._attenuation_factors

    @property
    def attenuation_angles(self) -> List[float]:
        return self._attenuation_angles

    @property
    def angles_theta_interval(self) -> Tuple[float, float]:
        return self._angles_theta_interval

    @property
    def angles_phase_interval(self) -> Tuple[float, float]:
        return self._angles_phase_interval

    @property
    def angles_rx_interval(self) -> Tuple[float, float]:
        return self._angles_rx_interval

    @property
    def angles_ry_interval(self) -> Tuple[float, float]:
        return self._angles_ry_interval
