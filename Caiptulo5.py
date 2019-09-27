import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson


def ejercicio_5_4():
  # Crear la pista de carreras
    A = np.zeros(16, 16) 
    A[16, 4: 12] = 1 
    A[15, 4: 12] = 1 
    A[10: 14, 5: 12]= 1
    A[8: 9, 7: 12] = 1
    A[7, 7: 13] = 1 
    A[6, 8: 13] = 1 
    A[5, 8: 16] = 1 
    A[4, 8: 16] = 1 
    A[3, 9: 16] = 1 
    A[2, 11: 16] = 1

    Rt= A
    maxNPii,maxNPjj =tuple(Rt.shape)
    maxNVii = 6
    maxNVjj = 6

    maxNAii = 3
    maxNAjj = 3
    maxNStates = maxNPii*maxNPjj*maxNVii*maxNVjj

    maxNActions = maxNAii*maxNAjj




def init_unif_policy(MZ, maxNStates,maxNActions,maxNPii,maxNPjj,maxNVii,maxNVjj,maxNAii,maxNAjj):
    pol_pi = np.zeros(maxNStates, maxNActions)
    for si in range(maxNStates):
        pii, pjj=  tuple(np.unravel_index(si,[maxNPii, maxNPjj]))
        vii, vjj = tuple(np.unravel_index(si,[maxNVii, maxNVjj]))
        vii = vii - 1
        vjj = vjj - 1
        if (MZ(pii, pjj)!=1) :
            return









