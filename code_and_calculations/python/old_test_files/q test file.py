# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 22:39:54 2019

@author: jmstf
"""
import numpy as np
import pylab as scp
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as skl
import qiskit  as q
import qiskit.visualization as qvis
# Use Aer's qasm_simulator
simulator = q.Aer.get_backend('qasm_simulator')
# Create a Quantum Circuit acting on the q register
circuit=q.QuantumCircuit(2,2)
# Add a H gate on qubit 0
circuit.x(0)
circuit.h(0)
'''
# Add a CX (CNOT) gate on control qubit 0 and target qubit 1
circuit.cx(0, 1)
'''
print(circuit.draw())
# Map the quantum measurement to the classical bits
circuit.measure([0,1],[0,1])
# Execute the circuit on the qasm simulator
job = q.execute(circuit, simulator, shots=1000)
# Grab results from the job
result = job.result()
# Returns counts
counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:",counts)
print(circuit.draw())