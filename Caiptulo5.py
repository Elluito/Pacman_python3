import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson
import random
GAMMA =0.99
def ejercicio_5_12():
  # Crear la pista de carreras
    A = np.zeros((16, 16))
    A[15, 3: 11] = 1
    A[14, 3: 11] = 1
    A[9: 14, 4: 11]= 1
    A[7: 9, 6: 11] = 1
    A[6, 6: 12] = 1
    A[5, 7: 12] = 1
    A[4, 7: 15] = 1
    A[3, 7: 15] = 1
    A[2, 8: 15] = 1
    A[1, 10: 15] = 1

    #
    # im = plt.imshow(A, cmap=plt.cm.RdBu, extent=(0, 16, 0,16),)
    #
    # plt.colorbar(im)
    #
    # plt.title('')
    #
    # plt.show()

    RT= A
    maxNPii,maxNPjj =tuple(RT.shape)
    maxNVii = 6
    maxNVjj = 6

    maxNAii = 3
    maxNAjj = 3
    maxNStates = maxNPii*maxNPjj*maxNVii*maxNVjj

    maxNActions = maxNAii*maxNAjj


    firstSARewSum = np.zeros((maxNStates, maxNActions))
    firstSARewCnt = np.zeros((maxNStates, maxNActions))



    # % enumerate
    # the
    # possible
    # starting
    # locations:
    posStarts = np.nonzero(RT[-1,:])
    nPosStarts = len(posStarts)

    # % initialize
    # our
    # policy:
    # pol_pi = np.zeros((maxNStates, maxNActions))
    pol_pi = init_unif_policy(RT, maxNStates, maxNActions, maxNPii, maxNPjj, maxNVii, maxNVjj, maxNAii, maxNAjj)
    Q =np.random.rand(maxNStates,maxNActions)

    EPISODES=200000
    for i in range(EPISODES):
      print(f"Episodio {i:d}")
      stateseen,act_taken,rew=gen_rt_episode(i,pol_pi,RT,posStarts,nPosStarts,maxNStates,maxNActions,maxNPii,maxNPjj,maxNVii,maxNVjj,maxNAii,maxNAjj)
      Q, firstSARewCnt, firstSARewSum = Estimar_Q(stateseen, act_taken, rew, firstSARewCnt, firstSARewSum, Q, maxNPii,maxNPjj, maxNVii, maxNVjj)

       # (C)  update  our  policy:
      pol_pi = rt_pol_mod(stateseen, Q, pol_pi, maxNPii, maxNPjj, maxNVii, maxNVjj, maxNAii, maxNAjj)


    Q[np.argwhere(Q==0)] = np.nan
    V = np.mean(Q, 1)
    V[np.argwhere(np.isnan(V))]= 0
    V = np.reshape(V,(maxNPii,maxNPjj,maxNVii,maxNVjj))
    V[np.argwhere(V == 0)] = np.nan
    V = np.nanmean(V,3)
    V = np.nanmean(V,2)
    V[np.argwhere(np.isnan(V))] = 0
    np.save("C:/Users/Luis Alfredo/Pictures/Camera Roll/V.npy",V)


    im = plt.imshow(V,cmap=plt.cm.Spectral, extent=(16, 0, 0, 16))

    plt.colorbar(im)

    plt.title('$V_{\pi}^*(s)$')

    plt.show()




def Estimar_Q(stateseen,act_taken,rew, firstSARewCnt,firstSARewSum,Q, maxNPii,maxNPjj,maxNVii,maxNVjj):

    # G = 0
    for si,state in enumerate(stateseen):
        pii,pjj,vii,vjj=tuple(state)

        # TODO este +1 en laS VELOCIDADES NO ME CONVENCE  pero lo dejo por si acaso
        staInd = np.ravel_multi_index( (pii, pjj, vii , vjj),[maxNPii, maxNPjj, maxNVii, maxNVjj])
        actInd = act_taken[si]
        firstSARewCnt[staInd, actInd] = firstSARewCnt[staInd, actInd] + 1
        firstSARewSum[staInd, actInd] = firstSARewSum[staInd, actInd] + rew
        # % Q(staInd, actInd) = firstSARewSum(staInd, actInd) / firstSARewCnt(staInd, actInd); % < -take
        # the
        # direct
        # average
        Q[staInd, actInd] = Q[staInd, actInd] + (1 / firstSARewCnt[staInd, actInd]) * (rew - Q[staInd, actInd])# % < -use incremental  averaging
        # Q(staInd, actInd) = Q(staInd, actInd) + alpha * (rew - Q(staInd, actInd)) #       # a
        # geometric
        # average
    return Q,firstSARewCnt,firstSARewSum


def rt_pol_mod(stateseen,Q, pol_pi, maxNPii,maxNPjj,maxNVii,maxNVjj,maxNAii,maxNAjj):
    eps = 0.1
    for si in stateseen:
        pii, pjj, vii, vjj = tuple(si)
        staInd = np.ravel_multi_index((pii, pjj, vii, vjj), [maxNPii, maxNPjj, maxNVii, maxNVjj])
        posAction = velState2PosActions([vii,vjj],maxNVii,maxNVjj,maxNAii,maxNAjj)
        findPosAction =np.nonzero(posAction)[0]
        temp= [int(e) for e in findPosAction]


        numChoices = len(findPosAction)
        greedyAct =np.where(Q[staInd,:] == np.max(Q[staInd,findPosAction]))[0][0]
        nonGreedyAct = list(set(temp)-{int(greedyAct)})

        # % perform  an   eps - soft  on - policy    MC   update:
        pol_pi[staInd,greedyAct] = 1 - eps + eps / numChoices
        pol_pi[staInd, nonGreedyAct] = eps / numChoices
        assert np.all(np.nonzero(pol_pi[staInd,:])[0]==np.nonzero(velState2PosActions([vii,vjj],maxNVii,maxNVjj,maxNAii,maxNAjj))),"posAction: "+str(posAction)+"  "+"pol_pi[staInd, :]"+str(pol_pi[staInd,:])
    return pol_pi






def gen_rt_episode(ei,pol_pi,RT,posStarts,nPosStarts,maxNStates,maxNActions,maxNPii,maxNPjj,maxNVii,maxNVjj,maxNAii,maxNAjj):
    rew = 0
    stateseen = []
    act_taken = []

      # ii = maxNPii
     # tmp = randperm(nStarts)
     # jj = posStarts(tmp(1));
    vii = 0
    vjj = 0
    pii = maxNPii-1
    pjj = posStarts[0][ei % nPosStarts ]
    vii = ei%maxNVii
    vjj = ei%maxNVjj
    if vii==vjj==0:
        if np.random.rand()>0.5:
            vii = np.mod(ei, maxNVii - 1) + 1
        else:
            vjj = np.mod(ei, maxNVjj - 1) + 1
    stateseen.append([pii, pjj, vii, vjj])
    finish = False
    while not finish: #% ~didWeFinish([pii, pjj, vii, vjj], maxNPjj) ) % take a step
         stInd  = np.ravel_multi_index(( pii, pjj, vii, vjj) ,[maxNPii, maxNPjj, maxNVii, maxNVjj])
         from itertools import product
         while sum(pol_pi[stInd,:])>1:
             pol_pi[stInd,:]=pol_pi[stInd,:]/sum(pol_pi[stInd,:])
         assert sum(pol_pi[stInd,:])<=1
         act_to_take = np.random.choice(range(maxNActions),p= pol_pi[stInd,:])
         posActions = velState2PosActions([vii,vjj],maxNVii,maxNVjj,maxNAii,maxNAjj)

         assert posActions[act_to_take]==1,"possAct: "+str(posActions)+"   " +"act_to_take " + str(act_to_take) +" pol_pi: " +str(pol_pi[stInd,:])
         act_taken.append(act_to_take)
         # A = {-1, 0, 1}
         # B = {-1, 0, 1}
         # accions = product(A, B)
         # accions = np.array(accions)

         aIndii, aIndjj = np.unravel_index( act_to_take,[maxNAii, maxNAjj])[0],np.unravel_index( act_to_take,[maxNAii, maxNAjj])[1]
         aii = aIndii-1
         ajj = aIndjj - 1
         # % the
         # specific
         # actions
         # to
         # take \ in {-1, 0, +1}
         #           % update
         # our
         # state
         # according
         # to
         # this
         # action and recieve
         # a
         # reward: \

         vii = vii + aii
         vjj = vjj + ajj
         if (vii < 0 or vii >= maxNVii):
            print([pii, pjj, vii - aii, vjj - ajj, aii, ajj])
            raise Exception( 'vii out of bounds' )
         if ( vjj < 0 or vjj >= maxNVjj):
            print([pii, pjj, vii-aii, vjj-ajj, aii, ajj])
            raise Exception( 'vjj out of bounds' )
         pii=pii-vii
         pjj=pjj+vjj
         if didWeFinish([pii,pjj],maxNPjj):
             finish=True
             rew+=100
         else:
             #Random step
             rndUp = 0
             rndRt = 0
             rand =np.random.rand()
             if (rand < 0.5) : # we have a random step
                if (rand < 0.5) :#% that is up
                    pii = pii - 1
                    if (pii > 0):
                        rndUp=1
                    else:
                        pii=pii+1
                else: #% that is right
                    pjj = pjj + 1
                if (pjj < maxNPjj + 1):
                    rndRt=1
                else:
                     pjj=pjj-1
             if onRT(pii , pjj,RT,maxNPii,maxNPjj):
                rew -=1
             else:
                    rew -= 5
                    # Dado que no estoy en la carretera me devuelvo a la linea de salida
                    pii = maxNPii-1
                    pjj = posStarts[0][ei % nPosStarts]

             stateseen.append([pii, pjj, vii, vjj])

    return stateseen, act_taken, rew










def didWeFinish(st, maxNPjj):
    # % DIDWEFINISH -
    # %
    pii = st[0]
    pjj = st[1]

    if (pjj >=15 and (pii<5 and pii>=1)):
        finishQ = 1
    else:
        finishQ = 0

    return finishQ


def onRT(pii,pjj,RT,maxNPii,maxNPjj):

    if not (1<=pii and pii<maxNPii):
        return False
    if not ((1<=pjj and pjj<maxNPjj)):
        return False
    if RT[pii,pjj]==1:
        return True
    else:
        return  False




def velState2PosActions(vstate,maxNVii,maxNVjj,maxNAii,maxNAjj):
    from itertools import product
    maxNActions = maxNAii* maxNAjj
    vii = vstate[0]
    vjj = vstate[1]
    if vii==vjj==0:
        raise  Exception("There can`t be velocity components [0,0]")
    possActs = np.ones((maxNAii,maxNAjj))
    A={-1,0,1}
    B={-1,0,1}
    accions =product(A,B)

    accions= np.array(list(accions))

    for i in range(maxNActions):
        action = accions[i,:]
        next_vel_state = vstate+action
        if list(next_vel_state)==[0,0]:
            possActs[(action + 1)[0],(action + 1)[1]] = 0
        if next_vel_state[0]==-1 or next_vel_state[1]== -1:
            possActs[(action + 1)[0],(action + 1)[1]] = 0
        if next_vel_state[0]== 6 or next_vel_state[1]==6:
            possActs[(action + 1)[0],(action + 1)[1]]= 0

    return possActs.ravel()






def init_unif_policy(MZ, maxNStates,maxNActions,maxNPii,maxNPjj,maxNVii,maxNVjj,maxNAii,maxNAjj):
    pol_pi = np.zeros((maxNStates, maxNActions))
    for si in range(maxNStates):
        pii, pjj,vii, vjj =  tuple(np.unravel_index(si,[maxNPii, maxNPjj,maxNVii, maxNVjj]))

        # vii = vii - 1
        # vjj = vjj - 1
        if (MZ[pii, pjj]!=1) :
            continue
        if vii == 0 and vjj == 0:
            continue
        pos_act = velState2PosActions([vii,vjj],maxNVii,maxNVjj,maxNAii,maxNAjj)
        uni_prob = pos_act/sum(pos_act)
        pol_pi[si,:] = uni_prob


    return pol_pi



ejercicio_5_12()



