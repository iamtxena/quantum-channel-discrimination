from qcd.configurations.configuration import ChannelConfiguration
from qcd.configurations import OneShotConfiguration, OneShotEntangledUniversalConfiguration
from . import OneShotEntangledFullInputCircuit
from typing import List, Tuple, cast
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


class OneShotEntangledUniversalCircuit(OneShotEntangledFullInputCircuit):
    """ Representation of the One Shot Entangled Channel circuit """

    def _create_one_circuit_without_measurement(self,
                                                configuration: ChannelConfiguration,
                                                eta: float) -> Tuple[ClassicalRegister, QuantumCircuit]:
        """ Creates one circuit from a given  configuration and eta """
        configuration = cast(OneShotEntangledUniversalConfiguration, configuration)
        qreg_q = QuantumRegister(3, 'q')
        creg_c = ClassicalRegister(2, 'c')

        circuit = QuantumCircuit(qreg_q, creg_c)
        circuit.rx(configuration.angle_rx_input1, qreg_q[1])
        circuit.ry(configuration.angle_ry_input1, qreg_q[1])
        circuit.rx(configuration.angle_rx_input0, qreg_q[0])
        circuit.ry(configuration.angle_ry_input0, qreg_q[0])
        circuit.cx(qreg_q[0], qreg_q[1])
        circuit.reset(qreg_q[2])
        circuit.cry(2 * eta, qreg_q[1], qreg_q[2])
        circuit.cx(qreg_q[2], qreg_q[1])
        circuit.barrier()
        circuit.u3(configuration.theta0, configuration.phi0, configuration.lambda0, qreg_q[0])
        circuit.u3(configuration.theta1, configuration.phi1, configuration.lambda1, qreg_q[1])
        circuit.cx(qreg_q[1], qreg_q[0])
        circuit.u3(configuration.theta2, configuration.phi2, configuration.lambda2, qreg_q[0])
        circuit.u3(configuration.theta3, configuration.phi3, configuration.lambda3, qreg_q[1])
        circuit.cx(qreg_q[0], qreg_q[1])
        circuit.u3(configuration.theta4, configuration.phi4, configuration.lambda4, qreg_q[0])
        circuit.u3(configuration.theta5, configuration.phi5, configuration.lambda5, qreg_q[1])
        circuit.cx(qreg_q[1], qreg_q[0])
        circuit.u3(configuration.theta6, configuration.phi6, configuration.lambda6, qreg_q[0])
        circuit.u3(configuration.theta7, configuration.phi7, configuration.lambda7, qreg_q[1])
        return creg_c, circuit

    def _create_one_configuration(self,
                                  configuration: ChannelConfiguration,
                                  eta_group: List[float]) -> OneShotConfiguration:
        """ Creates a specific configuration setting a specific eta group """
        return OneShotEntangledUniversalConfiguration({
            'angle_rx_input0': cast(OneShotEntangledUniversalConfiguration, configuration).angle_rx_input0,
            'angle_ry_input0': cast(OneShotEntangledUniversalConfiguration, configuration).angle_ry_input0,
            'angle_rx_input1': cast(OneShotEntangledUniversalConfiguration, configuration).angle_rx_input1,
            'angle_ry_input1': cast(OneShotEntangledUniversalConfiguration, configuration).angle_ry_input1,
            'theta0': cast(OneShotEntangledUniversalConfiguration, configuration).theta0,
            'phi0': cast(OneShotEntangledUniversalConfiguration, configuration).phi0,
            'lambda0': cast(OneShotEntangledUniversalConfiguration, configuration).lambda0,
            'theta1': cast(OneShotEntangledUniversalConfiguration, configuration).theta1,
            'phi1': cast(OneShotEntangledUniversalConfiguration, configuration).phi1,
            'lambda1': cast(OneShotEntangledUniversalConfiguration, configuration).lambda1,
            'theta2': cast(OneShotEntangledUniversalConfiguration, configuration).theta2,
            'phi2': cast(OneShotEntangledUniversalConfiguration, configuration).phi2,
            'lambda2': cast(OneShotEntangledUniversalConfiguration, configuration).lambda2,
            'theta3': cast(OneShotEntangledUniversalConfiguration, configuration).theta3,
            'phi3': cast(OneShotEntangledUniversalConfiguration, configuration).phi3,
            'lambda3': cast(OneShotEntangledUniversalConfiguration, configuration).lambda3,
            'theta4': cast(OneShotEntangledUniversalConfiguration, configuration).theta4,
            'phi4': cast(OneShotEntangledUniversalConfiguration, configuration).phi4,
            'lambda4': cast(OneShotEntangledUniversalConfiguration, configuration).lambda4,
            'theta5': cast(OneShotEntangledUniversalConfiguration, configuration).theta5,
            'phi5': cast(OneShotEntangledUniversalConfiguration, configuration).phi5,
            'lambda5': cast(OneShotEntangledUniversalConfiguration, configuration).lambda5,
            'theta6': cast(OneShotEntangledUniversalConfiguration, configuration).theta6,
            'phi6': cast(OneShotEntangledUniversalConfiguration, configuration).phi6,
            'lambda6': cast(OneShotEntangledUniversalConfiguration, configuration).lambda6,
            'theta7': cast(OneShotEntangledUniversalConfiguration, configuration).theta7,
            'phi7': cast(OneShotEntangledUniversalConfiguration, configuration).phi7,
            'lambda7': cast(OneShotEntangledUniversalConfiguration, configuration).lambda7,
            'eta_group': eta_group
        })
