import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def exe_6_9(EPISODES):
    MAX_ii=7
    MAX_jj=10

    # % the
    # wind in each
    # column:
    wind = [0,0 ,0 ,1 ,1 ,1 ,2 ,2 ,1 ,0]
    MAX_ACTIONS = 9
    MAX_A_ii=3
    MAX_A_JJ=3
    max_n_states=MAX_ii*MAX_jj

    Q = np.zeros((max_n_states,MAX_A_ii*MAX_A_JJ))
    V = np.random.random((MAX_ii, MAX_jj))


    # % the
    # beginning and terminal
    # states( in matrix
    # notation):
    s_start = [3, 0]
    s_end = [3, 7]
    V[s_end[0], s_end[1]] = 0
    s_end_ravel =np.ravel_multi_index(s_end,[MAX_ii,MAX_jj])
    Q[s_end_ravel,:] = 0





    trajectory = None

    for i in range(EPISODES):
        # s_start = [int(np.random.randint(0,MAX_ii,1)),int(np.random.randint(0,MAX_jj,1))]
        trajectory = generate_episode(Q,V,s_start,s_end,wind,max_a_ii=MAX_A_ii,max_a_jj=MAX_A_JJ,max_ii=MAX_ii,max_jj=MAX_jj)

        np.save("datos\Q_%i"%EPISODES, V)
        np.save("datos\last_trajectory", np.array(trajectory))
        print(f"Episode {i:d}")


    V1 = Estimar_V(Q,s_start,s_end,wind,max_a_ii=MAX_A_ii,max_a_jj=MAX_A_JJ,max_ii=MAX_ii,max_jj=MAX_jj,EPISODES=5000)


    plt.figure()

    im = plt.imshow(V, cmap=plt.cm.Spectral)

    plt.colorbar(im)

    plt.title('$V_{\pi}^*(s)$')
    plt.savefig("datos/V.png")
    plt.close()
    plt.figure()

    im = plt.imshow(V1, cmap=plt.cm.Spectral)

    plt.colorbar(im)

    plt.title('$V_{\pi}^*(s)$')
    plt.savefig("datos/V1.png")
    plt.close()
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from matplotlib import animation
    # fig = plt.figure()
    # # creating a subplot
    # ax1 = fig.add_subplot(1, 1, 1)
    #
    # def animate(i,trajectory):
    #     s_start = [3, 0]
    #     s_end = [3, 7]
    #     MAX_ii = 7
    #     MAX_jj = 10
    #     wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    #
    #     maps = np.zeros((MAX_ii, MAX_jj))
    #
    #     ax1.clear()
    #     maps[s_start[0],s_start[1]]=12
    #     maps[s_end[0],s_end[1]]=50
    #     pos_actual= trajectory[i]
    #     for elem in wind :
    #         maps[:,elem]=elem
    #     maps[pos_actual[0],pos_actual[1]]=400
    #
    #     im = ax1.imshow(maps)
    #     return im
    #
    #
    #
    # im = ax1.imshow(np.zeros((7, 10)))
    # plt.colorbar(im)
    # plt.title('$S_t$')
    # ani = animation.FuncAnimation(fig, lambda i: animate(i, trajectory), interval=100)
    #
    # plt.show()


def valid_actions(state,max_a_ii,max_a_jj,maxii,maxjj):
    valid_actions=[]
    state = np.unravel_index(state,[maxii,maxjj])

    for i in range(max_a_ii*max_a_jj):
        aii,ajj= tuple(np.unravel_index(i,[max_a_ii,max_a_jj]))
        aii= aii-1
        ajj= ajj-1
        if state[0]+aii<0 or state[0]+aii>=maxii:
            continue
        if state[1]+ajj<0 or state[1]+ajj>=maxjj:
            continue
        if aii== 0 and ajj ==0:
            continue
        else:
            valid_actions.append(i)
    return valid_actions
def valid_actions_non_king_move(state,maxii,maxjj):
    actions = [[-1,0],[0,-1],[1,0],[0,1]]
    valid_actions= []
    for i,a in enumerate(actions):
        aii= a[0]
        ajj = a [1]
        if state[0] + aii < 0 or state[0] + aii >= maxii:
            continue
        if state[1] + ajj < 0 or state[1] + ajj >= maxjj:
            continue
        if aii == 0 and ajj == 0:
            continue
        else:
            valid_actions.append(i)
    return valid_actions

def generate_episode(Q,s_begin,s_finish,wind_colums,max_a_ii,max_a_jj,max_ii,max_jj):
    trajectory=[]
    actions = [[-1, 0], [0, -1], [1, 0], [0, 1]]
    s = s_begin
    trajectory.append(s)
    epsilon = 0.1
    alpha =0.5
    gamma = 1
    while s!= s_finish:
        state_ravel = np.ravel_multi_index(s, [max_ii, max_jj])

        aii = None
        ajj = None
        pos_act = None
        action = int(np.random.choice(range(len(Q[state_ravel, :])))) if epsilon > np.random.rand() else \
            int(np.argmax(Q[state_ravel, :]))

        assert isinstance(action, int), f"Action no es un int: " + str(action)
        if max_a_ii*max_a_jj==4:
            pos_act = valid_actions_non_king_move(s, maxii=max_ii, maxjj=max_jj)
            aii = actions[action][0]
            ajj = actions[action][1]
        else:
            pos_act = valid_actions(state_ravel, max_a_ii, max_a_jj, maxii=max_ii, maxjj=max_jj)
            aii, ajj = tuple(np.unravel_index(action, [max_a_ii, max_a_jj]))
            aii = aii - 1
            ajj = ajj - 1
        # aii = aii - 1
        next_s = None
        delta_reward = 0
        if action not in pos_act:
            delta_reward += -3
            # next_s = [next_s[0] - aii, next_s[1] - ajj]

            if s[0] - wind_colums[s[1]] < 0 :
                next_s = [s[0] , s[1] ]
            else:
                next_s = [s[0] - wind_colums[s[1]] , s[1]]
        else:

            if s[0] - wind_colums[s[1]] + aii < 0 and s[0] + aii >= 0:
                next_s = [s[0] + aii, s[1] + ajj]
            else:
                next_s = [s[0] - wind_colums[s[1]] + aii, s[1] + ajj]

        reward = 0 if next_s == s_finish else -1


        reward+=delta_reward


        next_state_ravel = np.ravel_multi_index(next_s, [max_ii, max_jj])
        # # pos_act = valid_actions_non_king_move(next_s, maxii=max_ii, maxjj=max_jj)
        #
        # action_2 = int(np.random.choice(range(len(Q[state_ravel, :])))) if epsilon > np.random.rand() else \
        #     int(np.argmax(Q[state_ravel, :]))
        #
        # assert isinstance(action_2, int), f"Action no es un int: " + str(action_2)
        # # perform the update

        Q[state_ravel,action]+= alpha*(reward+gamma*np.max(Q[next_state_ravel, :])-Q[state_ravel,action])
        # V[s[0], s[1]] = V[s[0], s[1]] + alpha * (reward + gamma * V[next_s[0], next_s[1]] - V[s[0], s[1]])
        trajectory.append(next_s)
        s = next_s

    return trajectory
def generate_epsisode_stocastic(Q,s_begin,s_finsih,wind_colums,max_a_ii,max_a_jj,max_ii,max_jj):
    p = 1/3

    trajectory = []

    s = s_begin
    trajectory.append(s)
    epsilon = 0.1
    alpha = 0.2
    gamma = 1
    while s != s_finsih:
        state_ravel = np.ravel_multi_index(s, [max_ii, max_jj])
        pos_act = valid_actions(state_ravel, max_a_ii, max_a_jj, maxii=max_ii, maxjj=max_jj)

        action = int(np.random.choice(pos_act, 1)) if epsilon > np.random.rand() else \
            int(np.where(Q[state_ravel, :] == np.max(Q[state_ravel, pos_act]))[0][0])

        assert isinstance(action, int), f"Action no es un int: " + str(action)

        aii, ajj = tuple(np.unravel_index(action, [max_a_ii, max_a_jj]))
        aii = aii - 1
        ajj = ajj - 1
        next_s = None
        wind_action = 0
        random_number = np.random.rand()
        if random_number<=p:
            wind_action = wind_colums[s[1]]
        if random_number>p and random_number<=p*2:
            wind_action = wind_colums[s[1]]+1
        if random_number>p*2:
            wind_action = wind_colums[s[1]]-1




        if (s[0] - wind_action+ aii < 0 or s[0] - wind_action+ aii>=max_ii  )and s[0] + aii >= 0:
            next_s = [s[0] + aii, s[1] + ajj]
        else:
            next_s = [s[0] - wind_action + aii, s[1] + ajj]

        assert np.all(np.array(next_s) >= 0), "El proximo estado tiene componente negativo negativo: " + str(next_s) + \
                                              f" mientras que debería estar entre las dimensiones [{max_ii:d},{max_jj:d}]" + "estado anterior: " + str(
            s) + f" acción tomada  [{aii:d},{ajj:d}]_"
        reward = 0 if next_s == s_finsih else -1

        next_state_ravel = np.ravel_multi_index(next_s, [max_ii, max_jj])
        pos_act = valid_actions(next_state_ravel, max_a_ii, max_a_jj, maxii=max_ii, maxjj=max_jj)

        action_2 = int(np.random.randint(0, max_a_ii * max_a_jj, 1)) if epsilon < np.random.rand() else \
            int(np.where(Q[next_state_ravel, :] == np.max(Q[next_state_ravel, pos_act]))[0][0])

        assert isinstance(action_2, int), f"Action no es un int: " + str(action_2)
        # perform the update

        Q[state_ravel, action] += alpha * (reward + gamma * Q[next_state_ravel, action_2] - Q[state_ravel, action])
        trajectory.append(next_s)
        s = next_s

    return trajectory

def Estimar_V(Q,s_begin,s_finish,wind_colums,max_a_ii,max_a_jj,max_ii,max_jj,EPISODES):
    V = np.random.random((max_ii, max_jj))

    V[s_finish[0],s_finish[1]] = 0
    s = s_begin
    epsilon = 0.1
    alpha = 0.2
    gamma = 1
    for i in range(EPISODES):
        while s != s_finish:
            state_ravel = np.ravel_multi_index(s, [max_ii, max_jj])
            pos_act = valid_actions(state_ravel, max_a_ii, max_a_jj, maxii=max_ii, maxjj=max_jj)

            action = int(np.random.choice(pos_act, 1)) if epsilon > np.random.rand() else \
                int(np.where(Q[state_ravel, :] == np.max(Q[state_ravel, pos_act]))[0][0])

            assert isinstance(action, int), f"Action no es un int: " + str(action)

            aii, ajj = tuple(np.unravel_index(action, [max_a_ii, max_a_jj]))
            aii = aii - 1
            ajj = ajj - 1
            next_s = None
            if s[0] - wind_colums[s[1]] + aii < 0 and s[0] + aii >= 0:
                next_s = [s[0] + aii, s[1] + ajj]
            else:
                next_s = [s[0] - wind_colums[s[1]] + aii, s[1] + ajj]

            assert np.all(np.array(next_s) >= 0), "El proximo estado tiene componente negativo negativo: " + str(next_s) + \
                                                  f" mientras que debería estar entre las dimensiones [{max_ii:d},{max_jj:d}]" + "estado anterior: " + str(
                s) + f" acción tomada  [{aii:d},{ajj:d}]_"
            reward = 0 if next_s == s_finish else -1



            V[s[0],s[1]]+= alpha*(reward +gamma*V[next_s[0],next_s[1]]-V[s[0],s[1]])
            s=next_s


    return V

def is_incliff(state,cliff):
    left_bound = cliff[0][1]
    right_bound = cliff[1][1]

    if state[0]==3 and (left_bound<=state[1] and state[1]<=right_bound):
        return True
    return False


def generar_episodio_cliff_Q_learning(Q,s_begin,s_finish,cliff,max_a_ii,max_a_jj,max_ii,max_jj):
    trajectory = []
    actions = [[-1, 0], [0, -1], [1, 0], [0, 1]]
    episode_reward = 0
    n=1
    s = s_begin
    trajectory.append(s)
    epsilon = 0.1
    alpha = 0.5
    gamma = 1

    s=s_begin



    while s!= s_finish:
        # print("state: "+str(s))
        state_ravel = np.ravel_multi_index(s, [max_ii, max_jj])
        pos_act = valid_actions_non_king_move(s, maxii=max_ii, maxjj=max_jj)

        action = int(np.random.choice(range(len(Q[state_ravel, :])))) if epsilon > np.random.rand() else \
            int(np.argmax(Q[state_ravel, :]))


        assert isinstance(action, int), f"Action no es un int: " + str(action)

        # aii, ajj = tuple(np.unravel_index(action, [max_a_ii, max_a_jj]))

        aii = actions[action][0]
        ajj = actions[action][1]
        next_s = [s[0] + aii, s[1] + ajj]
        reward = 0 if next_s == s_finish else -1
        delta_reward = 0
        if is_incliff(next_s, cliff):
            delta_reward = -99
        if action not in pos_act:
            delta_reward += -3
            next_s = [next_s[0] - aii, next_s[1] - ajj]

        reward+= delta_reward
        episode_reward+=reward
        n+=1
        next_state_ravel = np.ravel_multi_index(next_s, [max_ii, max_jj])

        Q[state_ravel, action] += alpha * (reward + gamma *np.max(Q[next_state_ravel, :])- Q[state_ravel, action])
        # print(f"Q(state,a): {Q[state_ravel, action]:0.3f}")
        if is_incliff(next_s,cliff):
            next_s=s_begin
        trajectory.append(next_s)
        s = next_s
    return trajectory,episode_reward

def generar_episodio_cliff_Sarsa(Q,s_begin,s_finish,cliff,max_a_ii,max_a_jj,max_ii,max_jj):
    trajectory = []
    episode_reward = 0
    actions = [[-1, 0], [0, -1], [1, 0], [0, 1]]
    n=1
    s = s_begin
    trajectory.append(s)
    epsilon = 0.1
    alpha = 0.5
    gamma = 1

    s = s_begin

    while s!=s_finish:
        state_ravel = np.ravel_multi_index(s, [max_ii, max_jj])
        pos_act =valid_actions_non_king_move(s, maxii=max_ii, maxjj=max_jj)


        action =int(np.random.choice(range(len(Q[state_ravel, :])))) if epsilon > np.random.rand() else \
            int(np.argmax(Q[state_ravel, :]))


        assert isinstance(action, int), f"Action no es un int: " + str(action)

        # aii, ajj = tuple(np.unravel_index(action, [max_a_ii, max_a_jj]))
        aii = actions[action][0]
        ajj = actions[action][1]
        # aii = aii - 1
        # ajj = ajj - 1
        next_s = [s[0] + aii, s[1] + ajj]

        reward = 0 if next_s == s_finish else -1
        delta_reward = 0
        if is_incliff(next_s, cliff):
            delta_reward = -99

        if action not in pos_act:
            delta_reward += -3
            next_s = [next_s[0] - aii, next_s[1] - ajj]

        reward += delta_reward
        episode_reward+= reward
        n+=1



        next_state_ravel = np.ravel_multi_index(next_s, [max_ii, max_jj])

        pos_act = valid_actions(next_state_ravel, max_a_ii, max_a_jj, maxii=max_ii, maxjj=max_jj)

        action_2 = int(np.random.choice(range(len(Q[state_ravel, :])))) if epsilon > np.random.rand() else \
            int(np.argmax(Q[state_ravel, :]))



        assert isinstance(action_2, int), f"Action no es un int: " + str(action_2)

        Q[state_ravel, action] += alpha * (reward + gamma * Q[next_state_ravel, action_2] - Q[state_ravel, action])
        if is_incliff(next_s, cliff):
            next_s =s_begin
        trajectory.append(next_s)
        s = next_s
    return trajectory,episode_reward




def exe_6_10(EPSIDOES):
    MAX_ii = 7
    MAX_jj = 10

    # % the
    # wind in each
    # column:
    wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    MAX_ACTIONS = 9
    MAX_A_ii = 3
    MAX_A_JJ = 3
    max_n_states = MAX_ii * MAX_jj

    Q = np.random.random((max_n_states, MAX_A_ii * MAX_A_JJ)) * -1 / 2
    # % the
    # beginning and terminal
    # states( in matrix
    # notation):
    s_start = [4, 1]
    s_end = [4, 8]
    s_end_ravel = np.ravel_multi_index(s_end, [MAX_ii, MAX_jj])
    Q[s_end_ravel, :] = 0

    fig = plt.figure()
    # creating a subplot
    ax1 = fig.add_subplot(1, 1, 1)


    def animate(i,trajectory):

        maps = np.zeros((MAX_ii, MAX_jj))

        ax1.clear()
        maps[s_start[0],s_start[1]]=12
        maps[s_end[0],s_end[1]]=50
        pos_actual= trajectory[i]
        maps[pos_actual[0],pos_actual[1]]=400
        for elem in wind :
            maps[:,elem]=elem
        im = ax1.imshow(maps, cmap=plt.cm.Spectral)
        return im
    trajectory = None
    for i in range(EPSIDOES):
        trajectory = generate_epsisode_stocastic(Q, s_start, s_end, wind, max_a_ii=MAX_A_ii, max_a_jj=MAX_A_JJ, max_ii=MAX_ii,
                                      max_jj=MAX_jj)

        print(f"Episode {i:d}")
    if len(trajectory) < 20:
        im = ax1.imshow(np.zeros((MAX_ii, MAX_jj)), cmap=plt.cm.Spectral)
        plt.colorbar(im)
        plt.title('$S(t)$')
        ani = animation.FuncAnimation(fig, lambda i: animate(i, trajectory), interval=100)


    V = np.mean(Q, 1)
    V = np.reshape(V, (MAX_ii, MAX_jj))

    im = plt.imshow(V, cmap=plt.cm.Spectral)

    plt.colorbar(im)

    plt.title('$V_{\pi}^*(s)$')

    plt.show()


def estimarVapartirdeQ(Q,MAX_ii,MAX_jj,epsilon=0.1):
    V= np.zeros((MAX_ii,MAX_jj))
    num_actions = Q.shape[1]
    for i in range(MAX_ii):
        for j in range(MAX_jj):
            state_ravel = np.ravel_multi_index([i,j],[MAX_ii,MAX_jj])
            pos_act= None
            if num_actions>4:
                pos_act = valid_actions(state_ravel, 3, 3, maxii=MAX_ii, maxjj=MAX_jj)
            elif num_actions==4:
                pos_act = valid_actions_non_king_move([i,j],MAX_ii,MAX_jj)
            action = None
            set_actions = set(np.where(Q[state_ravel, :] == np.max(Q[state_ravel, pos_act]))[0])
            result = list(set_actions.intersection(set(pos_act)))[0]
            # otros = list(set(range(9)).difference(set(result)))
            V[i,j] = (1-epsilon)*Q[state_ravel, result]+epsilon*(np.sum(Q[state_ravel,:]))
    return V




def example_6_6(episodes):

    MAX_ii = 4
    MAX_jj = 12
    max_n_states = MAX_ii*MAX_jj
    MAX_A_ii = 1
    MAX_A_jj = 4
    cliff = [[3,1],[3,10]]
    s_ini=[3,0]
    s_end=[3,11]
    Q_sarsa = np.zeros((max_n_states,MAX_A_ii*MAX_A_jj))
    Q_learning = np.zeros((max_n_states,MAX_A_ii*MAX_A_jj))

    rew_q_learnig =[]
    rew_sarsa = []
    for i in range(episodes):
        print(f"Episode: {i:d}")
        trajectory_qlearning,mean_reward_q_learning = generar_episodio_cliff_Q_learning(Q_learning,s_ini,s_end,cliff,MAX_A_ii,MAX_A_jj,MAX_ii,MAX_jj)
        trajectory_sarsa,mean_reward_sarsa = generar_episodio_cliff_Sarsa(Q_sarsa,s_ini,s_end,cliff,MAX_A_ii,MAX_A_jj,MAX_ii,MAX_jj)
        rew_q_learnig.append(mean_reward_q_learning)
        rew_sarsa.append(mean_reward_sarsa)

    fig = plt.figure()
    # creating a subplot
    ax1 = fig.add_subplot(1, 1, 1)
    def animate(i,trajectory,title,ax):
        MAX_ii = 4
        MAX_jj = 12
        maps=np.zeros((MAX_ii,MAX_jj))
        s_ini = [3, 0]
        s_end = [3, 11]
        cliff = [[3,1],[3,10]]
        ax.clear()
        maps[s_ini[0],s_ini[1]]=12
        maps[s_end[0],s_end[1]]=50
        maps[3, 1:11] = -10
        pos_actual= trajectory[i%len(trajectory)]

        maps[pos_actual[0],pos_actual[1]]=100

        im = ax.imshow(maps)
        plt.title('$S_t$ '+title)
        return im



    im = ax1.imshow(np.zeros((MAX_ii,MAX_jj )))
    plt.title('$S_t Q learning$')
    ani = animation.FuncAnimation(fig, lambda i: animate(i, trajectory_qlearning,"Q learning",ax1), interval=100)
    ani.save('datos/Qlearning_cliff.gif',writer="imagemagick",fps=10)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    im = ax1.imshow(np.zeros((MAX_ii,MAX_jj )))
    plt.title('$S_t$ S.A.R.S.A')
    ani = animation.FuncAnimation(fig, lambda i: animate(i, trajectory_sarsa,"S.A.R.S.A",ax1), interval=100)
    ani.save('datos/SARSA_cliff.gif',writer="imagemagick",fps=10)
    plt.figure()
    plt.plot(rew_q_learnig)
    plt.title("$R_t$")
    plt.plot(rew_sarsa)
    plt.ylabel("Mean reward")
    plt.xlabel("Episodes")
    plt.ylim(-100,0)
    plt.legend(["Q learning","S.A.R.S.A"])
    plt.savefig("datos/cliff_reward_comparison.png")

    plt.figure()
    V= estimarVapartirdeQ(Q_learning,MAX_ii,MAX_jj)
    fig,ax = plt.subplots()
    im = ax.imshow(V)
    plt.title("$v_\pi(S)$ Q learning")
    plt.colorbar(im)

    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            text = ax.text(j, i, "%0.1f"%V[i, j],
                           ha="center", va="center", color="w")
    fig.set_size_inches(11, 5)
    plt.savefig("datos/cliff_V_q_learning.png")
    plt.figure()
    V = estimarVapartirdeQ(Q_sarsa, MAX_ii, MAX_jj)
    fig, ax = plt.subplots()
    im = ax.imshow(V)

    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            text = ax.text(j, i, "%0.1f" % V[i, j],
                           ha="center", va="center", color="w")
    fig.set_size_inches(11, 5)
    plt.title("$V_\pi(s)$ S.A.R.S.A")
    plt.savefig("datos/cliff_v_SARSA.png")
    plt.show()
# example_6_6(500)