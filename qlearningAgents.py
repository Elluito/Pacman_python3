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
import pdb;
import tensorflow as tf
from tensorflow import keras
import tensorflow.contrib.eager as tfe
# from tensorflow.python.client import device_lib

import graphicsUtils as graphix

import random,util,math

from collections import namedtuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
BATCH_SIZE=6

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




class Policy:

    def __init__( self, width,height, dim_action, gamma=0.9, load_name=None):
        tf.enable_eager_execution()
        self.width = width
        self.height = height
        tf.logging.set_verbosity(tf.logging.ERROR)

        # self.state_space = dim_state
        self.action_space = dim_action

        self.gamma = gamma
        self.memory = ReplayMemory(10000)



        self.global_step = tfe.Variable(0)
        self.loss_avg = tfe.metrics.Mean()
        self.mapeo = {"%": 200, "<": 30, ">": 35, "v": 40, "^": 45, ".": 90, "G": 150, " ": 10}



        self.model = keras.Sequential([
            keras.layers.Dense( 64, activation=tf.nn.relu, use_bias=False,input_shape=(self.height*self.width,)),
            keras.layers.Dropout( rate=0.6 ),
            keras.layers.Dense( self.action_space, activation=tf.nn.softmax )])
        self.model.summary()

        if load_name is not None : self.model = keras.models.load_model( load_name )

        self.optimizer = tf.train.AdamOptimizer()

        self.device = "CPU:0"
        # gpus = getGPUs( )
        # if gpus : self.device = gpus[0]

        # Episode policy and reward history
        self.state_history = []
        self.action_history = []
        self.reward_episode = []

        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
    def load_Model(self,load_name=None):
        self.model = keras.models.load_model( load_name )


    def update_policy_supervised( self, states, actions ) :
        states = np.array(states)
        epochs = 100
        f=open("loss.txt","a")
        for e in range(epochs) :
            with tf.device( self.device ) :
                with tf.GradientTape() as tape:
                    actions_ = self.model( [states[:,:-14],states[:,-14:] ])
                    # actions = tf.multiply(actions,1/np.sum(actions,axis=1,dtype=np.float).reshape(-1,1))
                    loss = tf.losses.softmax_cross_entropy( onehot_labels=tf.multiply(actions,1/np.sum(actions,axis=1,dtype=np.float).reshape(-1,1)), logits=actions_ )
                grads = tape.gradient( loss, self.model.trainable_variables )
                # del tape
                del tape
                self.optimizer.apply_gradients( zip( grads, self.model.trainable_variables ), self.global_step )

            f.write(str(loss.numpy())+"\n")
            # self.accuracy(coso,actions)
            print( f'\tEpoch {e+1:d}/{epochs}... | Loss: {loss:.3f}' )
        f.close()

    def saveModel( self, name ) :
        self.model.save('models/' + name + '.h5')

    def mapeo(self, state):
        filas = state.split("\n")

        imagen = np.zeros((self.height, self.policy.width))
        for i in range(self.height):
            for j in range(self.width):
                imagen[i, j] = self.mapeo[filas[i][j]] / 255

        return imagen.reshape((-1,1))



    def update_policy(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        state_batch = np.array(batch.state)
        action_batch = np.array(batch.action)
        reward_batch = np.array(batch.reward)


        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = np.array((tuple(map(lambda s: not s.data._lose and s.data._win,batch.next_state))),dtype=np.int)
        non_final_next_states =[s for s in batch.next_state
                                           if not s.data._lose and not s.data._win]
        next_state_values = np.zeros(BATCH_SIZE)
        non_final_next_states =list(map(str,non_final_next_states))
        non_final_next_states = list(map(self.mapeo,non_final_next_states))


        # TODO aquí tengo que escoger epsilon greedy la acion en el proximo estado
        next_state_values[non_final_mask] =self.model(non_final_next_states)




        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch



        # Calculate loss
        #
        with tf.GradientTape() as tape:
            logits =self.model([np.array(self.state_history).reshape(-1,56)])
            # index =  np.array([range(int(logits.shape[0])), np.array(self.action_history).reshape(-1, )]).reshape(-1, 2)
            real_logits = tf.log(tf.gather_nd(logits,self.action_history) )

            loss =tf.losses.huber_loss()

            # actions = tf.multiply(actions,1/np.sum(actions,axis=1,dtype=np.float).reshape(-1,1))


        grads = tape.gradient( loss, self.model.trainable_variables )
        # del tape
        del tape
        self.optimizer.apply_gradients( zip( grads, self.model.trainable_variables ), self.global_step )





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
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.actions = ['South', 'North', 'East', 'West', 'Stop']
        # pdb.set_trace()

        "*** YOUR CODE HERE ***"
        self.mapeo = {"%":200,"<":30,">":35,"v":40,"^":45,".":90,"G":150," ":10}
        self.escala = 255


        layout = args["layout"]
        width = layout.width
        height = layout.height



        self.policy = Policy(width,height,5)




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
        filas = str(state).split("\n")



        imagen = np.zeros((self.policy.height,self.policy.width))
        for i in range(self.policy.height):
            for j in range(self.policy.width):
                imagen[i,j]=self.mapeo[filas[i][j]]/self.escala


        # legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        new_state = np.transpose(imagen.reshape((-1,1)))
        logits = self.policy.model([new_state]).numpy()[0]
        logits = logits/(np.sum(logits)+0.01)
        accion = np.argmax(logits) if np.random.rand()> self.epsilon else np.random.choice(range(len(self.actions)))



        action = self.actions[accion]






        # util.raiseNotDefined()

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




        self.policy.memory.push(state,self.actions.index(action),nextState,reward)



        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
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
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
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
            pass
