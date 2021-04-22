from qcd.configurations.configuration import ChannelConfiguration
from qcd.configurations import OneShotConfiguration, OneShotEntangledFullInputConfiguration
from . import OneShotEntangledCircuit
from typing import List, cast
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


class OneShotEntangledFullInputCircuit(OneShotEntangledCircuit):
    """ Representation of the One Shot Entangled Channel circuit """

    def _create_one_circuit(self,
                            configuration: ChannelConfiguration,
                            eta: float) -> QuantumCircuit:
        """ Creates one circuit from a given  configuration and eta """
        configuration = cast(OneShotEntangledFullInputConfiguration, configuration)
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
        circuit.cx(qreg_q[0], qreg_q[1])
        circuit.rx(configuration.angle_rx1, qreg_q[1])
        circuit.ry(configuration.angle_ry1, qreg_q[1])
        circuit.rx(configuration.angle_rx0, qreg_q[0])
        circuit.ry(configuration.angle_ry0, qreg_q[0])
        circuit.measure([0, 1], creg_c)
        return circuit

    def _create_one_configuration(self,
                                  configuration: ChannelConfiguration,
                                  eta_group: List[float]) -> OneShotConfiguration:
        """ Creates a specific configuration setting a specific eta group """
        return OneShotEntangledFullInputConfiguration({
            'angle_rx_input0': cast(OneShotEntangledFullInputConfiguration, configuration).angle_rx_input0,
            'angle_ry_input0': cast(OneShotEntangledFullInputConfiguration, configuration).angle_ry_input0,
            'angle_rx_input1': cast(OneShotEntangledFullInputConfiguration, configuration).angle_rx_input1,
            'angle_ry_input1': cast(OneShotEntangledFullInputConfiguration, configuration).angle_ry_input1,
            'angle_rx0': cast(OneShotEntangledFullInputConfiguration, configuration).angle_rx0,
            'angle_ry0': cast(OneShotEntangledFullInputConfiguration, configuration).angle_ry0,
            'angle_rx1': cast(OneShotEntangledFullInputConfiguration, configuration).angle_rx1,
            'angle_ry1': cast(OneShotEntangledFullInputConfiguration, configuration).angle_ry1,
            'eta_group': eta_group
        })
