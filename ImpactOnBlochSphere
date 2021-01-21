from itertools import product, combinations
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute
from numpy import pi
from mpl_toolkits.mplot3d import Axes3D
from math import e
import matplotlib.pyplot as plt
import numpy as np
import math


# Create 2 qbits circuit and 1 output classical bit
qreg_q = QuantumRegister(2, 'q')
creg_c = ClassicalRegister(2, 'c')
creg_c = ClassicalRegister(1, 'c')

# angles shift from first parameter, to the second one, with jumps using the third parameter
# angles = np.arange(0, 2*pi, 2*pi/10)

# Quantum states to pass through the circuit
## We want to pass the Bloch sphere through it to see visually the transformation
## First we generate the angles which will help the draw the sphere
Theta = np.mgrid[0.00000001:pi:3j]
Phase = np.mgrid[0.00000001:2*pi-0.00000001:3j]
## As we have to provide the state values to initialize the qreg[0] we have to do a conversion
## from angles in the sphere to statevector amplitudes. These statevectors will be the combination of 
## Zero_Amplitude*|0> plus One_Amplitude*|1>
Range = len(Theta)*len(Phase)
Zero_Amplitude = [0]*Range
One_Amplitude = [0]*Range
print ('Preparing states to pass through the circuit')
for a in range(len(Theta)):
    for b in range(len(Phase)):
        Zero_Amplitude[a*len(Phase)+b] = math.cos(Theta[a]/2)
        One_Amplitude[a*len(Phase)+b] = math.sin(Theta[a]/2)*e**(1j*Phase[b])
        print("State ", a*len(Phase)+b," =", Zero_Amplitude[a*len(Phase)+b],"*|0> + ", One_Amplitude[a*len(Phase)+b], "*|1>")

# Use Aer's qasm_simulator
backend_sim = Aer.get_backend('qasm_simulator')

totalResults=[]
totalCounts=[]
totalCircuits=[]

# Apply initialisation operation to the 0th qubit
#qc.initialize(initial_state, 0) 

print("Defining the circuit")

print("Initializing the circuit")
for i in range(len(Zero_Amplitude)):
    # Create the circuit gates
    #print("Starting to compute the angle: ", math.degrees(theta))
    circuit = QuantumCircuit(qreg_q, creg_c)
    circuit.initialize([Zero_Amplitude[i],One_Amplitude[i]],qreg_q[0])
    print("Input State ", i," =", Zero_Amplitude[i],"*|0> + ", One_Amplitude[i], "*|1>")
    circuit.x(qreg_q[0])
    circuit.reset(qreg_q[1])
    circuit.cry(pi/4, qreg_q[0], qreg_q[1])
    circuit.cx(qreg_q[1], qreg_q[0])
    circuit.measure(qreg_q[0], creg_c[0])      
    totalCircuits.append(circuit)    # Initialize circuit with desired initial_state        
    job_sim = execute(circuit, backend_sim, shots=5000)
    results_sim = job_sim.result()
    totalResults.append(results_sim)
    counts = results_sim.get_counts(circuit)
    print(counts)
    totalCounts.append(counts)

print("printing the circuit")
totalCircuits[0].draw('mpl')

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("auto")

# draw cube
r = [-1, 1]
for s, e in combinations(np.array(list(product(r, r, r))), 2):
    if np.sum(np.abs(s-e)) == r[1]-r[0]:
        ax.plot3D(*zip(s, e), color="b")

# draw sphere of initial states
x_i = [0]*Range
y_i = [0]*Range
z_i = [0]*Range
for i in range(len(Zero_Amplitude)):
    Theta_i = 2*np.arccos(Zero_Amplitude[i])
    if np.sin(2*Theta_i)==0:
        Phase_i = 0
    elif One_Amplitude[i]==0:
        Phase_i = 0
    else:
        Phase_i = -1j*np.log(One_Amplitude[i]/np.sin(2*Theta_i))
    x_i[i] = np.sin(Theta_i)*np.cos(Phase_i)
    y_i[i] = np.sin(Theta_i)*np.sin(Phase_i)
    z_i[i] = np.cos(Theta_i)
ax.plot_wireframe(x_i, y_i, z_i, color="r")
# draw center
ax.scatter([0], [0], [0], color="g", s=100)

plt.show()
