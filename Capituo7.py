import numpy as np
from collections import namedtuple
from Capitulo6 import valid_actions_non_king_move,valid_actions,generate_episode,  estimarVapartirdeQ
transition=namedtuple("Transition",("S","A","R","next_S"))
import matplotlib.pyplot as plt


def Estimate_V(Q,s_begin,s_finish,wind_colums,max_ii,max_jj,EPISODES,n_TD=3,mode=1):
    # V = np.random.rand(max_ii, max_jj)
    V = np.zeros((max_ii, max_jj))
    V[s_finish[0], s_finish[1]] = 0
    for i in range(EPISODES):
        print(f"Episodes estimating: {i:d}")
        s = s_begin
        epsilon = 0.1
        alpha = 0.2
        gamma = 1
        actions = [[-1, 0], [0, -1], [1, 0], [0, 1]]
        storing_reward = []
        storing_next_state = []
        storing_next_state.append(s)
        storing_reward.append(0)
        tao = 0
        t = 0
        T = float("Inf")
        ya_entre = False


        while tao!=T-1:
            if t < T:
                    state_ravel = np.ravel_multi_index(s, [max_ii, max_jj])
                    pos_act = valid_actions_non_king_move(s,maxii=max_ii, maxjj=max_jj)

                    action = int(np.random.choice(range(len(Q[state_ravel, :])))) if epsilon > np.random.rand() else \
                    int(np.argmax(Q[state_ravel, :]))

                    assert isinstance(action, int), f"Action no es un int: " + str(action)

                    aii = actions[action][0]
                    ajj = actions[action][1]
                    # aii = aii - 1
                    next_s = None
                    delta_reward = 0
                    if action not in pos_act:
                        delta_reward += -1
                        # next_s = [next_s[0] - aii, next_s[1] - ajj]
                        if s[0] - wind_colums[s[1]] < 0:
                            next_s = [s[0], s[1]]
                        else:
                            next_s = [s[0] - wind_colums[s[1]], s[1]]
                    else:

                        if s[0] - wind_colums[s[1]] + aii < 0 and s[0] + aii >= 0:
                            next_s = [s[0] + aii, s[1] + ajj]
                        else:
                            next_s = [s[0] - wind_colums[s[1]] + aii, s[1] + ajj]


                    reward = 0 if next_s == s_finish else -1
                    reward += delta_reward

                    storing_reward.append(reward)
                    storing_next_state.append(next_s)
                    if next_s == s_finish and (not ya_entre):
                        ya_entre = True
                        T = t + 1
            tao = t-n_TD+1
            if tao>=0:
                    maximun = min(tao+n_TD,T)
                    G = 0
                    j = tao+1
                    for elem in storing_reward[tao+1:maximun+1]:

                        G += (gamma**(j-tao-1))*elem
                        j += 1
                    # print(f"prueba para incluir V: Tao {int(tao):d},n_TD {int(n_TD):d},T {T:0.3f}")
                    if tao+n_TD<T:

                        S_tao_n =storing_next_state[tao+n_TD]

                        G += (gamma**(n_TD))*V[S_tao_n[0],S_tao_n[1]]

                        # Aqui hago la actualización dependiendo del modo
                    if tao == T-1:
                        G+= storing_reward[tao]
                    if mode == 1:
                        S_tao = storing_next_state[tao]
                        print(f"G: {G:0.2f}")

                        # print(f"V(S_tao): {V[S_tao[0], S_tao[1]]:0.3f}")
                        # print(f"tao: {tao:d}")
                        # print(f"Error: {G - V[S_tao[0], S_tao[1]]:0.2f}")
                        # print("---------------------------")
                        print(f"S_tao: [{S_tao[0]:d},{S_tao[1]:d}]")
                        V[S_tao[0], S_tao[1]] += alpha * (G - V[S_tao[0], S_tao[1]])

                    # En este modo calulo esto con los delta de cada periodo
                    if mode == 2:
                        S_tao = storing_next_state[tao]
                        # Esto es reemplaza G-V(s_t) anterior
                        diff = 0
                        for i in range(tao+1,maximun+1):
                            s_next = storing_next_state[i]
                            actual_s = storing_next_state[i-1]
                            #Este es un Delta

                            diff += storing_reward[i]+gamma*V[s_next[0], s_next[1]]-V[actual_s[0],actual_s[1]]
                        if tao>T-2:
                            cosas=0
                            pass
                        # print(f"tao: {tao:d}")
                        # # print(f"S_tao: [{S_tao[0]:d},{S_tao[1]:d}]")`
                        # print(f"V(S_tao): {V[S_tao[0], S_tao[1]]:0.2f}")
                        # print(f"Delta: {diff:0.2f}")`
                        # if tao == T - 1:
                        #     G += storing_reward[tao]
                        V[S_tao[0], S_tao[1]] += alpha * (diff)
            t += 1
            s = next_s
    return V


def ejercicio_7_2(EPISODES):
    MAX_ii = 7
    MAX_jj = 10
    max_n_states = MAX_ii * MAX_jj
    MAX_A_ii = 1
    MAX_A_jj = 4

    # Esto es en caso de querer hacerlo con cliff


    # cliff = [[3, 1], [3, 10]]
    # s_ini = [3, 0]
    # s_end = [3, 11]

    wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    s_start = [3, 0]
    s_end = [3, 7]
    Q_sarsa = np.zeros((max_n_states, MAX_A_ii * MAX_A_jj))


    for i in range(EPISODES):
        print(f"Episode: {i:d}")
        trajectory = generate_episode(Q_sarsa, s_start, s_end,wind, MAX_A_ii,\
                                                                           MAX_A_jj, MAX_ii, MAX_jj)



    V_real=estimarVapartirdeQ(Q_sarsa,MAX_ii,MAX_jj)
    V1=Estimate_V(Q_sarsa,s_start,s_end,wind,max_ii=MAX_ii,max_jj=MAX_jj,EPISODES=5000,mode=1,n_TD=3)
    V2 = Estimate_V(Q_sarsa,s_start,s_end,wind,max_ii=MAX_ii,max_jj=MAX_jj,EPISODES=5000,mode=2,n_TD=3)
    print(f"MSE for V_real and V1: {float(np.linalg.norm(V_real-V1)):0.3f}")
    print(f"MSE for V_real and V2: {float(np.linalg.norm(V_real-V2)):0.3f}")
    fig, ax = plt.subplots()
    im = ax.imshow(V_real)
    plt.title("$V_{real}$")
    plt.colorbar(im)

    for i in range(V_real.shape[0]):
        for j in range(V_real   .shape[1]):
            text = ax.text(j, i, "%0.1f" % V_real[i, j],
                           ha="center", va="center", color="w")



    fig, ax = plt.subplots()
    im = ax.imshow(V1)
    plt.title("$V_{TD}$ vanila")
    plt.colorbar(im)

    for i in range(V1.shape[0]):
        for j in range(V1.shape[1]):
            text = ax.text(j, i, "%0.1f" % V1[i, j],
                           ha="center", va="center", color="w")



    fig, ax = plt.subplots()
    im = ax.imshow(V2)
    plt.title("$V_{TD}$ with V estimate error")
    plt.colorbar(im)

    for i in range(V2.shape[0]):
        for j in range(V2.shape[1]):
            text = ax.text(j, i, "%0.1f" % V2[i, j],
                           ha="center", va="center", color="w")
    plt.show()






ejercicio_7_2(5000)

