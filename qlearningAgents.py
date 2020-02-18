# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
import numpy as np
from copy import deepcopy as dpc
import matplotlib.pyplot as plt
import random as rnd
import pdb
import tensorflow as tf
from tensorflow import keras

import random, util, math
from collections import namedtuple

import pickle
import  tensorboard as tb
from segtree import SumSegmentTree, MinSegmentTree
from pacman import GameState
import inspect

# gpus = tf.config.experimental.list_physical_devices('GPU')

NORTH = 'North'
SOUTH = 'South'
EAST = 'East'
WEST = 'West'
STOP = 'Stop'
EPS_START = 0.4
EPS_END = 0.1
EPS_DECAY = 0.999970043
HEIGTH = 19
MAX_GUARDAR=500000
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
BATCH_SIZE = 128
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='alfredoavendano')
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)
# global  strategy

def flatten(X):
    '''
    Flatten a 3D array.

    Input
    X            A 3D array for lstm, where the array is sample x timesteps x features.

    Output
    flattened_X  A 2D array, sample x features.
    '''
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1] - 1), :]
    return (flattened_X)

def dar_pedazo_de_imagenstate(state:GameState,policy):
    pos_pacman =state.getPacmanPosition()

    imagen=dar_features(policy,state)
    lim_filas_der =imagen.shape[0]
    lim_filas_izq = 0
    lim_column_abajo =imagen.shape[1]
    lim_column_arriba = 0

    filas_plus = pos_pacman[0]+3 if pos_pacman[0]+2<imagen.shape[0] else lim_filas_der
    filas_minus = pos_pacman[0]-2 if pos_pacman[0]>=2 else lim_filas_izq
    colum_plus = pos_pacman[1]+3 if pos_pacman[1]+2<imagen.shape[1] else lim_column_abajo
    colum_minus = pos_pacman[1]-2 if pos_pacman[1]>=2 else lim_column_arriba

    pedazo = np.zeros((5,5))
    pedazo_imagen =imagen[filas_minus:filas_plus,colum_minus:colum_plus]
    for i in range(pedazo_imagen.shape[0]):
        for j in range(pedazo_imagen.shape[1]):
            pedazo[i,j]=pedazo_imagen[i,j]

    pedazo = np.ravel(pedazo)
    pedazo = np.append(pedazo,pos_pacman[0])
    pedazo = np.append (pedazo,pos_pacman[1])
    return pedazo

def dar_features(policy,state:GameState):
    if not policy.use_image:
        posicion_pacman = state.getPacmanPosition()
        posicion_fantasma = state.getGhostPosition(1)
        temp = np.nonzero(np.array(state.getFood().data))
        if state.data._win:
            posicion_comida = posición_pacman
        else:
            posicion_comida = (int(temp[0]),int(temp[1]))
        distancia_a_comida =np.linalg.norm(np.array(posición_pacman)-np.array(posicion_comida))
        res =[distancia_a_comida]+list(posición_pacman)+list(posición_fantasma)
        return res
    else:
        return  np.array(policy.mapeo_fn(str(state))).reshape(policy.height,policy.width,1)
class ReplayBuffer(object):
        def __init__(self, size):
            """Create Replay buffer.
            Parameters
            ----------
            size: int
                Max number of transitions to store in the buffer. When the buffer
                overflows the old memories are dropped.
            """
            self._storage = []
            self._maxsize = size
            self._next_idx = 0

        def __len__(self):
            return len(self._storage)

        def add(self, obs_t, action, reward, obs_tp1, done):
            obs_t = np.array(obs_t)
            obs_tp1 = np.array(obs_tp1)
            data = (obs_t,action, reward, obs_tp1, done)

            if self._next_idx >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data
            self._next_idx = (self._next_idx + 1) % self._maxsize

        def _encode_sample(self, idxes):
            obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
            for i in idxes:
                data = self._storage[i]
                obs_t, action, reward, obs_tp1, done = data
                obses_t.append(np.array(obs_t, copy=False))
                actions.append(np.array(action, copy=False))
                rewards.append(reward)
                obses_tp1.append(np.array(obs_tp1, copy=False))
                dones.append(done)
            return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

        def sample(self, batch_size):
            """Sample a batch of experiences.
            Parameters
            ----------
            batch_size: int
                How many transitions to sample.
            Returns
            -------
            obs_batch: np.array
                batch of observations
            act_batch: np.array
                batch of actions executed given obs_batch
            rew_batch: np.array
                rewards received as results of executing act_batch
            next_obs_batch: np.array
                next set of observations seen after executing act_batch
            done_mask: np.array
                done_mask[i] = 1 if executing act_batch[i] resulted in
                the end of an episode and 0 otherwise.
            """
            idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
            return self._encode_sample(idxes)
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

with strategy.scope():
    @tf.function
    def train_step(inputs):
            global policy, GLOBAL_BATCH_SIZE
            # print(policy)
            features,labels = inputs

            l = tf.keras.losses.Huber(reduction=keras.losses.Reduction.NONE)
            def compute_loss(labels,predictions):

                    # training=True is only needed if there are layers with different
                    # behavior during training versus inference (e.g. Dropout).
                    # logits = policy.model(features)

                    per_example_loss=l(y_true=labels,y_pred=predictions)

                    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

                # grads = tape.gradient(loss, policy.model.trainable_variables)
                # policy.optimizer.apply_gradients(list(zip(grads, policy.model.trainable_variables)))
                # return cross_entropy
            print("Features: "+str(features))
            with tf.GradientTape() as tape:
                predictions = policy.model(features, training=True)
                loss = compute_loss(labels, predictions)

            grads = tape.gradient(loss, policy.model.trainable_variables)
            policy.optimizer.apply_gradients(zip(grads, policy.model.trainable_variables))


            # print(per_example_losses)
            # mean_loss = policy.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)
            return loss
class Policy:
    __slots__ = ( 'width', 'height', 'dim_action', 'gamma','load_name','use_prior','use_image','model','memory','epsilon','escala','mapeo','state_space','priority','action_space','strategy','optimizer')

    def __init__(self, width, height, dim_action, gamma=0.9, load_name=None,use_prior =False,use_image =False):
        # tf.enable_eager_execution()
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        print('caller name:', calframe[1][3])
        print("INICIE POLICY ESTA VEZ")

        self.width = width
        self.height = height

        self.priority = use_prior
        if use_image:
            self.state_space = (self.height,self.width,1)
        else:
            self.state_space = (dim_action,)
        self.action_space = dim_action
        self.use_image = use_image
        self.gamma = gamma
        self.memory = ReplayMemory(10000)

        # self.priority_memory = PrioritizedReplayBuffer(10000,0.5)
        self.epsilon = EPS_START
        # self.pesos = np.ones(BATCH_SIZE, dtype=np.float32)



        self.mapeo = {"%": 10, "<": 30, ">": 30, "v": 30, "^": 30, ".": 150, "G": 90, " ":1,"o":10}
        self.escala = 255
        if self.use_image:

            #
            # global strategy
            #
            self.strategy =strategy

            with strategy.scope():
                self.model = keras.Sequential([
                    keras.layers.Conv2D(32, (3, 3),  input_shape=self.state_space),
                    keras.layers.BatchNormalization(),
                    keras.layers.Activation("relu"),
                    keras.layers.Conv2D(64, (3, 3),strides=[2,2],use_bias=False),
                    keras.layers.BatchNormalization(),
                    keras.layers.Activation("relu"),
                    keras.layers.Conv2D(64, (3, 3),use_bias=False),
                    keras.layers.BatchNormalization(),
                    keras.layers.Activation("relu"),
                    keras.layers.Flatten(),
                    keras.layers.Dense(7*7*64, activation=tf.nn.tanh, use_bias=False),
                    keras.layers.Dense(512, activation=tf.nn.tanh, use_bias=False),
                    keras.layers.Dense(self.action_space, activation="linear")])

                # if not use_prior:

                self.optimizer=keras.optimizers.RMSprop(learning_rate=0.0002,momentum=0.01)
                # self.model.compile(loss=tf.compat.v1.losses.huber_loss, optimizer=op)

        else:
            self.model = keras.Sequential([
                # keras.layers.Dense(128, activation=tf.nn.tanh, use_bias=False, input_shape=(self.height * self.width,)),
                keras.layers.Dense(32, activation=tf.nn.tanh, use_bias=False,input_shape=self.state_space),
                # keras.layers.Dropout(rate=0.6),
                keras.layers.Dense(self.action_space, activation="linear")])
            self.model.compile(loss=lambda y_t,y_pred: self.func(y_pred=y_pred,y_true=y_t),optimizer=tf.train.RMSPropOptimizer(0.01))
            if not use_prior:
                self.model.compile(loss="mse",optimizer=tf.train.RMSPropOptimizer(0.01))


        if load_name is not None: self.model = keras.models.load_model(load_name)






        # Episode policy and reward history

    # @tf.function
    def func(self,y_true, y_pred):
        errors = tf.pow(tf.reduce_sum(y_true- y_pred, axis=1), 2)
        print(self.pesos)

        loss = tf.reduce_mean(tf.multiply(self.pesos, errors))
        return loss

    def load_Model(self, load_name=None):
        self.model = keras.models.load_model(load_name)


    def saveModel(self, name):
        self.model.save('models/' + name + '.h5')

    def mapeo_fn(self, state):
        filas = state.split("\n")

        imagen = np.zeros((self.height, self.width))
        for i in range(self.height):
            for j in range(self.width):
                imagen[i, j] = self.mapeo[filas[i][j]] / 255

        return imagen.reshape((-1, 1))


    # @tf.function
    def update_policy(self,agent,callbacks=[],log_dir=""):
        if not self.priority:
            # print(gpus)

            # with self.strategy.scope():

                if len(self.memory) < BATCH_SIZE:
                            return

                transitions = self.memory.sample(BATCH_SIZE)
                # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
                # detailed explanation). This converts batch-array of Transitions
                # to Transition of batch-arrays.

                shape = [-1]
                shape.extend(self.state_space)
                batch = Transition(*zip(*transitions))
                state_batch = batch.state
                state_batch = np.array(state_batch, dtype=np.float64).reshape(shape )
                action_batch = np.array([list(range(len(batch.action))),list(batch.action)]).transpose()
                reward_batch = np.array(batch.reward)
                reward_batch = (reward_batch-np.mean(reward_batch))/(np.std(reward_batch)+0.001)


                # Compute a mask of non-final states and concatenate the batch elements
                # (a final state would've been the one after which simulation ended)
                non_final_mask = np.array((tuple(map(lambda s: not s.data._lose and not s.data._win, batch.next_state))),
                                          dtype=np.int)
                non_final_mask = np.nonzero(non_final_mask)[0]
                non_final_next_states = [s for s in batch.next_state
                                         if not s.data._lose and not s.data._win]
                next_state_values = np.zeros([BATCH_SIZE],dtype =float)
                non_final_next_states = list(map(lambda s : dar_features(self,s), non_final_next_states))
                non_final_next_states = np.array(non_final_next_states, dtype=np.float64).reshape(shape)
                next_state_values[non_final_mask] = np.max(np.array(self.model.predict_on_batch([non_final_next_states])),axis=1)
                q_update = (reward_batch+ self.gamma * next_state_values)
                q_values = np.array(self.model.predict_on_batch([state_batch]))
                q_values[action_batch[:,0],action_batch[:,1]] = q_update
                # strategy = self.strategy
                global GLOBAL_BATCH_SIZE
                GLOBAL_BATCH_SIZE = int(BATCH_SIZE/ strategy.num_replicas_in_sync)
                print(f"GLOBAL BATCH SIZE:{GLOBAL_BATCH_SIZE:d}")
                print(f"Number of replicas: {strategy.num_replicas_in_sync}")
                X = tf.data.Dataset.from_tensors(state_batch)
                print("X:"+str(X))
                y = tf.data.Dataset.from_tensors(q_values)
                print("Y:"+str(y))
                dataset = tf.data.Dataset.zip((X,y))
                print("Dataset:"+str(dataset))
                batched_data = dataset.batch(GLOBAL_BATCH_SIZE,drop_remainder=True)
                print("batched dataset:"+str(batched_data))
                print("Lista del batched dataset "+str(list(batched_data.as_numpy_iterator())))
                prob_dataset = tf.data.Dataset.from_tensor_slices((state_batch,q_values))
                print("Probando con from_tensor_slices:"+str(prob_dataset))
                batchd_prob = prob_dataset.batch(GLOBAL_BATCH_SIZE)
                print("Batched prob_dataset:"+str(batchd_prob))
                print("lista del slices batched:"+str(batchd_prob.as_numpy_iterator()))


                dist_dataset = strategy.experimental_distribute_dataset(batched_data)
                indexes=[range(GLOBAL_BATCH_SIZE),range(GLOBAL_BATCH_SIZE,2*GLOBAL_BATCH_SIZE),range(2*GLOBAL_BATCH_SIZE,3*GLOBAL_BATCH_SIZE),range(3*GLOBAL_BATCH_SIZE,4*GLOBAL_BATCH_SIZE),range(4*GLOBAL_BATCH_SIZE,5*GLOBAL_BATCH_SIZE),range(5*GLOBAL_BATCH_SIZE,6*GLOBAL_BATCH_SIZE),range(6*GLOBAL_BATCH_SIZE,7*GLOBAL_BATCH_SIZE),range(7*GLOBAL_BATCH_SIZE,8*GLOBAL_BATCH_SIZE)]
                global policy
                policy = self

                with strategy.scope():

                    @tf.function
                    def distributed_train_step(dataset_inputs):
                        # tf.distribute.get_replica_context().merge_all()
                        per_replica_losses = strategy.experimental_run_v2(train_step,args=(dataset_inputs,))
                        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)

                    for epoch in range(2):
                        total_loss = 0.0
                        num_batches = 1
                        print("LLEGO A EL DATASET DISTRIBUIDO")

                        # train_iter = iter(dist_dataset)
                        #
                        # for _ in range(5):
                        #     print(_)
                        #     total_loss += distributed_train_step(next(train_iter))
                        # distributed_train_step((state_batch,q_values))
                        for i in range(8):
                            x = state_batch[indexes[i],:,:,:]
                            y = q_values[indexes[i],:]
                            total_loss += distributed_train_step((x,y))
                            num_batches += 1
                        train_loss = total_loss / num_batches

                        template = ("Epoch {}, Loss: {}, A")
                        print(template.format(epoch + 1, train_loss))




                    # self.model.fit(state_batch, q_values,batch_size=len(state_batch),epochs=20,verbose=0)
        else:
            if len(self.priority_memory) < BATCH_SIZE:
                return
            obs_batch, act_batch, rew_batch, next_obs_batch, not_done_mask, weights, indxes = self.priority_memory.sample(BATCH_SIZE,0.5)
            self.pesos = np.array(weights,dtype=np.float32)
            non_final_mask = np.where(not_done_mask==0)[0]
            act_batch = np.array([list(range(len(act_batch))), act_batch]).transpose()
            next_state_values = np.zeros([BATCH_SIZE], dtype=float)
            next_state_values[non_final_mask] = np.max(self.model.predict(next_obs_batch[non_final_mask]), axis=1)

            rew_batch = (rew_batch - np.mean(rew_batch)) / (np.std(rew_batch) + 0.001)
            # rew_batch = rew_batch/max(np.abs(rew_batch))

            q_update = (rew_batch + self.gamma * next_state_values)
            q_values = self.model.predict([obs_batch])
            q_values[act_batch[:, 0], act_batch[:, 1]] = q_update

            with tf.GradientTape() as tape:
                # tape.watch(self.model.trainable_variables)
                y_pred = self.model([obs_batch],training=True)


                errors = tf.pow(tf.reduce_sum(q_values-y_pred,axis=1),2)

                # loss = tf.reduce_mean(tf.multiply(weights,errors))
                loss = tf.reduce_mean(errors)

            grads = tape.gradient(loss, self.model.trainable_variables)
            del tape
            # grads = self.optimizer.compute_gradients(f,self.model.trainable_variables)
            # for i,elem in enumerate(grads):
            #     grads[i] =elem[1].numpy()
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            # salidas = self.model.fit(obs_batch, q_values, batch_size=len(q_values), epochs=20, verbose=0)
            # print(salidas.history["loss"])
            td_error = self.model.predict([obs_batch])[act_batch[:, 0], act_batch[:, 1]]-q_update
            self.priority_memory.update_priorities(indxes, abs(td_error))






class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        """You can initialize Q-values here..."""
        ReinforcementAgent.__init__(self, **args)
        self.actions = [NORTH, WEST, SOUTH,EAST, STOP]
        # pdb.set_trace()
        self.num_episodes = 1
        self.num_datos=0
        self.task=args["task"]
        "*** YOUR CODE HERE ***"
        self.memory = []
        self.memory_length = 4
        self.prueba =False


        layout = args["layout"]
        width = layout.width
        height = layout.height
        self.BREAK=False
        self.phi=0.8

        self.num_trans = 0
        self.lastReward= 0
        self.n2 = 0
        self.eps_start = EPS_START
        self.eps_end = EPS_END
        self.similarity_function = None
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        print('caller name QLEARNING AGENT:', calframe[1][3])
        if "transfer" in args.keys():
            print("ENTRE AL IF DEL TRASNFER")
            self.policy_first = Policy(width, height, 5,use_image=True,use_prior=False)
            self.policy_second = Policy(width, height, 5,use_image=True,use_prior=False)
            num_first = args["transfer"][0]
            num_second = args ["transfer"][1]

            self.n1 = num_first
            self.n2 = num_second
            if num_first == 0:
                name = "modelo_imagen_20000_04_01_dif0_1575607728_gamma_0.9_attemp_8"
            if num_first == 1:
                name = "modelo_imagen_25000_04_01_dif1_1576737275_attemp_3_gamma0.9"
            # if difficulty == 2:
            #     name = "modelo_imagen_25000_04_01_dif2_1577007228_attemp_2_gamma0.9"
            self.policy_first.load_Model("models/" + name + ".h5")

            f = open(f"datos/"+args["sim_function"],"r+b")
            # self.similarity_function = pickle.load(f)
            self.similarity_function = keras.models.load_model("datos/"+args["sim_function"])
            f.close()



        else:
            self.policy_second = Policy(width, height, 5, use_image=True, use_prior=False)


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"

        util.raiseNotDefined()

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def set_start_time(self):
        self.episodeStartTime =time.time()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        features = dar_features(self.policy_second,state)
        shape = [1,-1]
        if self.policy_second.use_image:
            shape = [1]
            shape.extend(self.policy_second.state_space)
            Q_actual =np.array(self.policy_second.model(features.reshape(shape)))


        else:

            Q_actual = self.policy_second.model.predict(np.array(features).reshape(1,-1))


        accion = None

        assert len(self.memory) <= self.memory_length, f"La memoria tiene más de {self.memory_length:d} elementos"
        pedazo = dar_pedazo_de_imagenstate(state, self.policy_second)
        self.memory.append(pedazo)
        if len(self.memory) == self.memory_length:  # and self.num_datos < MAX_GUARDAR:
            if self.num_datos < MAX_GUARDAR and self.prueba :
                guardar = []
                for elem in self.memory:
                    if len(guardar) == 0:
                        guardar = elem
                    else:
                        guardar = np.append(guardar, elem)
                filename = f"datos/piezas_task_{self.task:d}"
                with open(filename, 'a+b') as fp:
                    pickle.dump(guardar, fp)
                self.num_datos += 1


            if self.num_datos >= MAX_GUARDAR:
                self.BREAK = True


            situacion = np.array(self.memory).reshape(1,4,27)
            # situacion = situacion.reshape(1,-1)
            if self.similarity_function is not None:
                pred =  self.similarity_function.predict(situacion)
                mse = np.mean(np.power(flatten(situacion) - flatten(pred), 2))

                if mse <= 0.02:
                    Q_pasado = self.policy_first.model.predict(features.reshape(shape))
                    Q_combinado = (self.phi*(Q_pasado-np.mean(Q_pasado))/np.std(Q_pasado)+(1-self.phi)*(Q_actual-np.mean(Q_actual))/np.std(Q_actual))
                    accion = np.argmax(Q_combinado) if np.random.rand() > self.epsilon else np.random.choice(
                        range(len(self.actions)))
            self.memory.pop(0)
        else:
                accion = np.argmax(Q_actual) if np.random.rand() > self.epsilon else np.random.choice(range(len(self.actions)))


        if accion is None:
            accion = np.argmax(Q_actual) if np.random.rand() > self.epsilon else np.random.choice(
                range(len(self.actions)))
        action = self.actions[accion]



        if not self.prueba:
                a =(EPS_END-EPS_START)/self.num_episodes
                eps_threshold = EPS_START + a * (self.episodesSoFar)
                self.epsilon = eps_threshold
                self.phi = self.phi*EPS_DECAY
                self.n +=1



        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        if self.policy_second.priority:
            self.policy_second.priority_memory.add(dar_features(self.policy_second, state), self.actions.index(action), reward,
                                            dar_features(self.policy_second, nextState),
                                            nextState.data._win or nextState.data._lose)
        else:
            self.policy_second.memory.push(dar_features( self.policy_second,state), self.actions.index(action), nextState, reward)




        self.lastReward = reward




        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        print('caller name PACMAN AGENT:', calframe[1][3])
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?

        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            print("ACABA AL FIN")
            pass
