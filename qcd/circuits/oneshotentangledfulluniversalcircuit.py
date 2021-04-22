from qcd.configurations.configuration import ChannelConfiguration
from qcd.configurations import OneShotConfiguration, OneShotEntangledFullUniversalConfiguration
from . import OneShotEntangledUniversalCircuit
from typing import List, cast
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


class OneShotEntangledFullUniversalCircuit(OneShotEntangledUniversalCircuit):
    """ Representation of the One Shot Entangled Channel circuit """

    def _create_one_circuit(self,
                            configuration: ChannelConfiguration,
                            eta: float) -> QuantumCircuit:
        """ Creates one circuit from a given  configuration and eta """
        configuration = cast(OneShotEntangledFullUniversalConfiguration, configuration)
        qreg_q = QuantumRegister(3, 'q')
        creg_c = ClassicalRegister(2, 'c')

        circuit = QuantumCircuit(qreg_q, creg_c)
        circuit.u3(configuration.input_theta0, configuration.input_phi0, configuration.input_lambda0, qreg_q[0])
        circuit.u3(configuration.input_theta1, configuration.input_phi1, configuration.input_lambda1, qreg_q[1])
        circuit.cx(qreg_q[1], qreg_q[0])
        circuit.u3(configuration.input_theta2, configuration.input_phi2, configuration.input_lambda2, qreg_q[0])
        circuit.u3(configuration.input_theta3, configuration.input_phi3, configuration.input_lambda3, qreg_q[1])
        circuit.cx(qreg_q[0], qreg_q[1])
        circuit.u3(configuration.input_theta4, configuration.input_phi4, configuration.input_lambda4, qreg_q[0])
        circuit.u3(configuration.input_theta5, configuration.input_phi5, configuration.input_lambda5, qreg_q[1])
        circuit.cx(qreg_q[1], qreg_q[0])
        circuit.u3(configuration.input_theta6, configuration.input_phi6, configuration.input_lambda6, qreg_q[0])
        circuit.u3(configuration.input_theta7, configuration.input_phi7, configuration.input_lambda7, qreg_q[1])
        circuit.reset(qreg_q[2])
        circuit.barrier()
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
        circuit.measure([0, 1], creg_c)
        return circuit

    def _create_one_configuration(self,
                                  configuration: ChannelConfiguration,
                                  eta_group: List[float]) -> OneShotConfiguration:
        """ Creates a specific configuration setting a specific eta group """
        return OneShotEntangledFullUniversalConfiguration({
            'input_theta0': cast(OneShotEntangledFullUniversalConfiguration, configuration).input_theta0,
            'input_phi0': cast(OneShotEntangledFullUniversalConfiguration, configuration).input_phi0,
            'input_lambda0': cast(OneShotEntangledFullUniversalConfiguration, configuration).input_lambda0,
            'input_theta1': cast(OneShotEntangledFullUniversalConfiguration, configuration).input_theta1,
            'input_phi1': cast(OneShotEntangledFullUniversalConfiguration, configuration).input_phi1,
            'input_lambda1': cast(OneShotEntangledFullUniversalConfiguration, configuration).input_lambda1,
            'input_theta2': cast(OneShotEntangledFullUniversalConfiguration, configuration).input_theta2,
            'input_phi2': cast(OneShotEntangledFullUniversalConfiguration, configuration).input_phi2,
            'input_lambda2': cast(OneShotEntangledFullUniversalConfiguration, configuration).input_lambda2,
            'input_theta3': cast(OneShotEntangledFullUniversalConfiguration, configuration).input_theta3,
            'input_phi3': cast(OneShotEntangledFullUniversalConfiguration, configuration).input_phi3,
            'input_lambda3': cast(OneShotEntangledFullUniversalConfiguration, configuration).input_lambda3,
            'input_theta4': cast(OneShotEntangledFullUniversalConfiguration, configuration).input_theta4,
            'input_phi4': cast(OneShotEntangledFullUniversalConfiguration, configuration).input_phi4,
            'input_lambda4': cast(OneShotEntangledFullUniversalConfiguration, configuration).input_lambda4,
            'input_theta5': cast(OneShotEntangledFullUniversalConfiguration, configuration).input_theta5,
            'input_phi5': cast(OneShotEntangledFullUniversalConfiguration, configuration).input_phi5,
            'input_lambda5': cast(OneShotEntangledFullUniversalConfiguration, configuration).input_lambda5,
            'input_theta6': cast(OneShotEntangledFullUniversalConfiguration, configuration).input_theta6,
            'input_phi6': cast(OneShotEntangledFullUniversalConfiguration, configuration).input_phi6,
            'input_lambda6': cast(OneShotEntangledFullUniversalConfiguration, configuration).input_lambda6,
            'input_theta7': cast(OneShotEntangledFullUniversalConfiguration, configuration).input_theta7,
            'input_phi7': cast(OneShotEntangledFullUniversalConfiguration, configuration).input_phi7,
            'input_lambda7': cast(OneShotEntangledFullUniversalConfiguration, configuration).input_lambda7,
            'theta0': cast(OneShotEntangledFullUniversalConfiguration, configuration).theta0,
            'phi0': cast(OneShotEntangledFullUniversalConfiguration, configuration).phi0,
            'lambda0': cast(OneShotEntangledFullUniversalConfiguration, configuration).lambda0,
            'theta1': cast(OneShotEntangledFullUniversalConfiguration, configuration).theta1,
            'phi1': cast(OneShotEntangledFullUniversalConfiguration, configuration).phi1,
            'lambda1': cast(OneShotEntangledFullUniversalConfiguration, configuration).lambda1,
            'theta2': cast(OneShotEntangledFullUniversalConfiguration, configuration).theta2,
            'phi2': cast(OneShotEntangledFullUniversalConfiguration, configuration).phi2,
            'lambda2': cast(OneShotEntangledFullUniversalConfiguration, configuration).lambda2,
            'theta3': cast(OneShotEntangledFullUniversalConfiguration, configuration).theta3,
            'phi3': cast(OneShotEntangledFullUniversalConfiguration, configuration).phi3,
            'lambda3': cast(OneShotEntangledFullUniversalConfiguration, configuration).lambda3,
            'theta4': cast(OneShotEntangledFullUniversalConfiguration, configuration).theta4,
            'phi4': cast(OneShotEntangledFullUniversalConfiguration, configuration).phi4,
            'lambda4': cast(OneShotEntangledFullUniversalConfiguration, configuration).lambda4,
            'theta5': cast(OneShotEntangledFullUniversalConfiguration, configuration).theta5,
            'phi5': cast(OneShotEntangledFullUniversalConfiguration, configuration).phi5,
            'lambda5': cast(OneShotEntangledFullUniversalConfiguration, configuration).lambda5,
            'theta6': cast(OneShotEntangledFullUniversalConfiguration, configuration).theta6,
            'phi6': cast(OneShotEntangledFullUniversalConfiguration, configuration).phi6,
            'lambda6': cast(OneShotEntangledFullUniversalConfiguration, configuration).lambda6,
            'theta7': cast(OneShotEntangledFullUniversalConfiguration, configuration).theta7,
            'phi7': cast(OneShotEntangledFullUniversalConfiguration, configuration).phi7,
            'lambda7': cast(OneShotEntangledFullUniversalConfiguration, configuration).lambda7,
            'eta_group': eta_group
        })
