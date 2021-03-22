import gym
import numpy as np
import math
import random
from copy import deepcopy, copy
import time
import ray
import json

datafile_name = "trained_weights_pso.json"

class NeuralNet:

    def __init__(self, numLayers, numNodes):
        super().__init__()

        self.numLayers = numLayers
        self.numNodes = numNodes

        # print("##########")
        # print(self.numNodes)

        self.weights = []
        self.baises = []

        self.fitness = 0.0

        self.initWeights()


    def initWeights(self):

        for i in range(0, self.numLayers - 1):
            self.weights.append(np.random.uniform(low = -1.0, high = 1.0, size = (self.numNodes[i], self.numNodes[i+1])))
            self.baises.append(np.random.uniform(low = -1.0, high = 1.0, size = (1, self.numNodes[i+1])))
    

    def getAction(self, observation):
        
        action = observation
        for i in range(0, self.numLayers - 1):
            action = np.matmul(np.asarray(action), self.weights[i]) + self.baises[i]
            action = np.reshape(action, (1, self.numNodes[i+1]))
            if(i == self.numLayers - 2):
                action = np.tanh(action)
            else:
                action = self.relu(action)
            action = np.reshape(action, (self.numNodes[i+1]))

        return action

    def relu(self, x):
        return np.maximum(0, x)


ENVIRONMENT = 'BipedalWalker-v3'
MAX_STEPS = 1000
MAX_GENERATIONS = 1000
POPULATION_SIZE = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.9

env = gym.make(ENVIRONMENT)
observation = env.reset()

env.render()

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

obs_range = (env.observation_space.low, env.observation_space.high)
action_range = (env.action_space.low, env.action_space.high)

print("OBSERVATION --> \nSHAPE:" + str(obs_dim) + "x1, \nRANGE: (" + str(obs_range[0]) + ", " + str(obs_range[1]) + ")")
print("\n")
print("Action --> \nSHAPE:" + str(action_dim) + "x1, \nRANGE: (" + str(action_range[0]) + ", " + str(action_range[1]) + ")")

def play_individual(individual, steps):
    observation = env.reset()
    for step in range(steps):
        env.render()
        action = individual.getAction(observation)
        observation, reward, done, info = env.step(action)
        if done:
            break


trained_weights = None

#NN that we need to play
NNLayers = (obs_dim, 64, 128, 32, action_dim)

try:
    with open(datafile_name, 'r') as infile:
        trained_weights = json.load(infile)

except Exception as e:
    print("JSON file not found!")
    print(e)

if not(trained_weights is None):
    parent = NeuralNet(len(NNLayers), NNLayers)
    for i in range(1, len(trained_weights[str(NNLayers)])):
        print("\t\tGENERATION " + str(i))

        # if i==0:
        #     continue

        for j in range(len(NNLayers) - 1):
            parent.weights[j] = np.asarray(trained_weights[str(NNLayers)][str(i)]['weights'][j])
            parent.baises[j] = np.asarray(trained_weights[str(NNLayers)][str(i)]['baises'][j])

        observation = env.reset()
        parent_reward = 0
        for step in range(MAX_STEPS):
            env.render()
            action = parent.getAction(observation)
            observation, reward, done, info = env.step(action)
            parent_reward += reward
            if done:
                break

        print("\t\t--> FITNESS: " + str(parent_reward) + "\n")


    pass