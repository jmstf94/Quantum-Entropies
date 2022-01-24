# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:45:28 2020

@author: jmstf
"""
import numpy as np
from scipy.linalg import logm, expm
def density_matrix_of_psi(psi):
    """
    It takes the ket as an np array of N dimensions
    and returns the density matrix as an NxN np array ie matrix.
    staight forward
    """
    return np.outer(psi,np.conjugate(psi))
def partial_trace(rho, dims, axis=0):
    """
    Takes partial trace over the subsystem defined by 'axis'
    rho: a matrix
    dims: a list containing the dimension of each subsystem
    axis: the index of the subsytem to be traced out
    (We assume that each subsystem is square)
    """
    dims_ = np.array(dims)
    # Reshape the matrix into a tensor with the following shape:
    # [dim_0, dim_1, ..., dim_n, dim_0, dim_1, ..., dim_n]
    # Each subsystem gets one index for its row and another one for its column
    reshaped_rho = rho.reshape(np.concatenate((dims_, dims_), axis=None))

    # Move the subsystems to be traced towards the end
    reshaped_rho = np.moveaxis(reshaped_rho, axis, -1)
    reshaped_rho = np.moveaxis(reshaped_rho, len(dims)+axis-1, -1)

    # Trace over the very last row and column indices
    traced_out_rho = np.trace(reshaped_rho, axis1=-2, axis2=-1)

    # traced_out_rho is still in the shape of a tensor
    # Reshape back to a matrix
    dims_untraced = np.delete(dims_, axis)
    rho_dim = np.prod(dims_untraced)
    return traced_out_rho.reshape([rho_dim, rho_dim])
def vN_entropy(rho):
    mx=rho.dot(logm(rho))
    return -mx.trace()
psi_1=np.array([0,1/np.sqrt(2),-1j/np.sqrt(2),0])
psi_2=np.array([1/np.sqrt(2),0,0,1/np.sqrt(2)])
rho_1=np.outer(psi_1,np.conjugate(psi_1))
rho_2=np.outer(psi_2,np.conjugate(psi_2))
rho_1_A=partial_trace(rho_1,[2,2], axis=1)
rho_1_B=partial_trace(rho_1,[2,2], axis=0)
print(vN_entropy(rho_1_B))
