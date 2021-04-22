from . import OneShotEntangledUniversalConfiguration
from ..typings.dicts import OneShotEntangledFullUniversalConfigurationDict
from typing import Dict


class OneShotEntangledFullUniversalConfiguration(OneShotEntangledUniversalConfiguration):
    """ Definition for One Shot Entangled Universal channel configuration """

    def __init__(self, configuration: OneShotEntangledFullUniversalConfigurationDict) -> None:
        self._input_theta0 = configuration['input_theta0']
        self._input_phi0 = configuration['input_phi0']
        self._input_lambda0 = configuration['input_lambda0']
        self._input_theta1 = configuration['input_theta1']
        self._input_phi1 = configuration['input_phi1']
        self._input_lambda1 = configuration['input_lambda1']
        self._input_theta2 = configuration['input_theta2']
        self._input_phi2 = configuration['input_phi2']
        self._input_lambda2 = configuration['input_lambda2']
        self._input_theta3 = configuration['input_theta3']
        self._input_phi3 = configuration['input_phi3']
        self._input_lambda3 = configuration['input_lambda3']
        self._input_theta4 = configuration['input_theta4']
        self._input_phi4 = configuration['input_phi4']
        self._input_lambda4 = configuration['input_lambda4']
        self._input_theta5 = configuration['input_theta5']
        self._input_phi5 = configuration['input_phi5']
        self._input_lambda5 = configuration['input_lambda5']
        self._input_theta6 = configuration['input_theta6']
        self._input_phi6 = configuration['input_phi6']
        self._input_lambda6 = configuration['input_lambda6']
        self._input_theta7 = configuration['input_theta7']
        self._input_phi7 = configuration['input_phi7']
        self._input_lambda7 = configuration['input_lambda7']

        self._theta0 = configuration['theta0']
        self._phi0 = configuration['phi0']
        self._lambda0 = configuration['lambda0']
        self._theta1 = configuration['theta1']
        self._phi1 = configuration['phi1']
        self._lambda1 = configuration['lambda1']
        self._theta2 = configuration['theta2']
        self._phi2 = configuration['phi2']
        self._lambda2 = configuration['lambda2']
        self._theta3 = configuration['theta3']
        self._phi3 = configuration['phi3']
        self._lambda3 = configuration['lambda3']
        self._theta4 = configuration['theta4']
        self._phi4 = configuration['phi4']
        self._lambda4 = configuration['lambda4']
        self._theta5 = configuration['theta5']
        self._phi5 = configuration['phi5']
        self._lambda5 = configuration['lambda5']
        self._theta6 = configuration['theta6']
        self._phi6 = configuration['phi6']
        self._lambda6 = configuration['lambda6']
        self._theta7 = configuration['theta7']
        self._phi7 = configuration['phi7']
        self._lambda7 = configuration['lambda7']

        super().__init__(configuration)

    @property
    def input_theta0(self) -> float:
        return self._input_theta0

    @property
    def input_phi0(self) -> float:
        return self._input_phi0

    @property
    def input_lambda0(self) -> float:
        return self._input_lambda0

    @property
    def input_theta1(self) -> float:
        return self._input_theta1

    @property
    def input_phi1(self) -> float:
        return self._input_phi1

    @property
    def input_lambda1(self) -> float:
        return self._input_lambda1

    @property
    def input_theta2(self) -> float:
        return self._input_theta2

    @property
    def input_phi2(self) -> float:
        return self._input_phi2

    @property
    def input_lambda2(self) -> float:
        return self._input_lambda2

    @property
    def input_theta3(self) -> float:
        return self._input_theta3

    @property
    def input_phi3(self) -> float:
        return self._input_phi3

    @property
    def input_lambda3(self) -> float:
        return self._input_lambda3

    @property
    def input_theta4(self) -> float:
        return self._input_theta4

    @property
    def input_phi4(self) -> float:
        return self._input_phi4

    @property
    def input_lambda4(self) -> float:
        return self._input_lambda4

    @property
    def input_theta5(self) -> float:
        return self._input_theta5

    @property
    def input_phi5(self) -> float:
        return self._input_phi5

    @property
    def input_lambda5(self) -> float:
        return self._input_lambda5

    @property
    def input_theta6(self) -> float:
        return self._input_theta6

    @property
    def input_phi6(self) -> float:
        return self._input_phi6

    @property
    def input_lambda6(self) -> float:
        return self._input_lambda6

    @property
    def input_theta7(self) -> float:
        return self._input_theta7

    @property
    def input_phi7(self) -> float:
        return self._input_phi7

    @property
    def input_lambda7(self) -> float:
        return self._input_lambda7

    @property
    def theta0(self) -> float:
        return self._theta0

    @property
    def phi0(self) -> float:
        return self._phi0

    @property
    def lambda0(self) -> float:
        return self._lambda0

    @property
    def theta1(self) -> float:
        return self._theta1

    @property
    def phi1(self) -> float:
        return self._phi1

    @property
    def lambda1(self) -> float:
        return self._lambda1

    @property
    def theta2(self) -> float:
        return self._theta2

    @property
    def phi2(self) -> float:
        return self._phi2

    @property
    def lambda2(self) -> float:
        return self._lambda2

    @property
    def theta3(self) -> float:
        return self._theta3

    @property
    def phi3(self) -> float:
        return self._phi3

    @property
    def lambda3(self) -> float:
        return self._lambda3

    @property
    def theta4(self) -> float:
        return self._theta4

    @property
    def phi4(self) -> float:
        return self._phi4

    @property
    def lambda4(self) -> float:
        return self._lambda4

    @property
    def theta5(self) -> float:
        return self._theta5

    @property
    def phi5(self) -> float:
        return self._phi5

    @property
    def lambda5(self) -> float:
        return self._lambda5

    @property
    def theta6(self) -> float:
        return self._theta6

    @property
    def phi6(self) -> float:
        return self._phi6

    @property
    def lambda6(self) -> float:
        return self._lambda6

    @property
    def theta7(self) -> float:
        return self._theta7

    @property
    def phi7(self) -> float:
        return self._phi7

    @property
    def lambda7(self) -> float:
        return self._lambda7

    def to_dict(self) -> Dict:
        return {'angle_rx_input0': self._angle_rx_input0,
                'angle_ry_input0': self._angle_ry_input0,
                'angle_rx_input1': self._angle_rx_input1,
                'angle_ry_input1': self._angle_ry_input1,
                'input_theta0': self._input_theta0,
                'input_phi0': self._input_phi0,
                'input_lambda0': self._input_lambda0,
                'input_theta1': self._input_theta1,
                'input_phi1': self._input_phi1,
                'input_lambda1': self._input_lambda1,
                'input_theta2': self._input_theta2,
                'input_phi2': self._input_phi2,
                'input_lambda2': self._input_lambda2,
                'input_theta3': self._input_theta3,
                'input_phi3': self._input_phi3,
                'input_lambda3': self._input_lambda3,
                'input_theta4': self._input_theta4,
                'input_phi4': self._input_phi4,
                'input_lambda4': self._input_lambda4,
                'input_theta5': self._input_theta5,
                'input_phi5': self._input_phi5,
                'input_lambda5': self._input_lambda5,
                'input_theta6': self._input_theta6,
                'input_phi6': self._input_phi6,
                'input_lambda6': self._input_lambda6,
                'input_theta7': self._input_theta7,
                'input_phi7': self._input_phi7,
                'input_lambda7': self._input_lambda7,
                'theta0': self._theta0,
                'phi0': self._phi0,
                'lambda0': self._lambda0,
                'theta1': self._theta1,
                'phi1': self._phi1,
                'lambda1': self._lambda1,
                'theta2': self._theta2,
                'phi2': self._phi2,
                'lambda2': self._lambda2,
                'theta3': self._theta3,
                'phi3': self._phi3,
                'lambda3': self._lambda3,
                'theta4': self._theta4,
                'phi4': self._phi4,
                'lambda4': self._lambda4,
                'theta5': self._theta5,
                'phi5': self._phi5,
                'lambda5': self._lambda5,
                'theta6': self._theta6,
                'phi6': self._phi6,
                'lambda6': self._lambda6,
                'theta7': self._theta7,
                'phi7': self._phi7,
                'lambda7': self._lambda7,
                'eta_group': self._eta_group
                }
