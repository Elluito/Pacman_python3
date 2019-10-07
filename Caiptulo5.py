import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson


def ejercicio_5_12():
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
    maxNVii = 5
    maxNVjj = 5

    maxNAii = 3
    maxNAjj = 3
    maxNStates = maxNPii*maxNPjj*maxNVii*maxNVjj

    maxNActions = maxNAii*maxNAjj




def velState2PosActions(vstate,maxNVii,maxNVjj,maxNAii,maxNAjj):
    from itertools import product
    maxNActions = maxNAii* maxNAjj
    vii = vstate[0]
    vjj = vstate[1]
    possActs = np.ones(1,maxNActions)
    A={-1,0,1}
    B={-1,0,1}
    accions =product(A,B)

    accions= np.array(list(accions))
    pos_accions=[]
    for i in range(9):
        action = accions[,:]
        next_vel_state = vstate+action
        if list(next_vel_state)==[0,0]:
            continue





    if vii==vjj==0:
        raise  Exception("There can`t be velocity components 0")







def init_unif_policy(MZ, maxNStates,maxNActions,maxNPii,maxNPjj,maxNVii,maxNVjj,maxNAii,maxNAjj):
    pol_pi = np.zeros(maxNStates, maxNActions)
    for si in range(maxNStates):
        pii, pjj=  tuple(np.unravel_index(si,[maxNPii, maxNPjj]))
        vii, vjj = tuple(np.unravel_index(si,[maxNVii, maxNVjj]))
        vii = vii - 1
        vjj = vjj - 1
        if (MZ(pii, pjj)!=1) :
            continue
        if vii == 0 and vjj == 0:
            continue









