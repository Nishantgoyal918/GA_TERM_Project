import gym
import numpy as np
import math
import random
from copy import deepcopy, copy
import time
import ray
import json
import wandb
from matplotlib import animation
import matplotlib.pyplot as plt
from gym import wrappers

ray.init(num_cpus=8)
wandb.init(project="genetic")


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
            self.baises.append(np.random.uniform(low = -1.0, high = 1.0, size = (self.numNodes[i+1])))

        self.weights = np.array([np.array(xi) for xi in self.weights], dtype=object)
        self.baises = np.array([np.array(xi) for xi in self.baises], dtype=object)
    

    def getAction(self, observation):

        action = observation
        for i in range(0, self.numLayers - 1):
            action = np.matmul(np.asarray(action),
                               self.weights[i]) + self.baises[i]
            action = np.reshape(action, (1, self.numNodes[i+1]))
            # if(i == self.numLayers - 2):
            #     action = np.tanh(action)
            # else:
            #     action = self.relu(action)
            action = np.tanh(action)
            action = np.reshape(action, (self.numNodes[i+1]))

        return action

    def relu(self, x):
        return np.maximum(0, x)


class Population:

    def __init__(self, populationSize, mutationRate, learningRate, NNLayers):
        super().__init__()
        
        self.populationSize = populationSize
        self.mutationRate = mutationRate
        self.learningRate = learningRate
        self.NNLayers = NNLayers
        self.population = []
        self.parent = None
        self.currentGeneration = 0

        self.weights_noise = []
        self.baises_noise = []

        self.data = {}
        self.datafile_name = "trained_weights.json"


    def dump_data(self):

        temp_json = None

        try:
            with open(self.datafile_name, "r") as infile:
                temp_json = json.load(infile)
        except:
            print("File not found will create one!")

        if(temp_json is None):
            temp_json = {}

        with open(self.datafile_name, "w") as outfile:
            temp_json[str(self.NNLayers)] = self.data
            json.dump(temp_json, outfile)


    def initPopulation(self):
        self.parent = NeuralNet(len(self.NNLayers), self.NNLayers)
        # self.dump_data()


    def mutation(self):
        # new_individual = NeuralNet(len(self.NNLayers), self.NNLayers)
        
        self.weights_noise.clear()
        self.baises_noise.clear()
        for i in range(len(self.NNLayers) - 1):
            self.weights_noise.append(np.random.randn(self.populationSize, self.NNLayers[i], self.NNLayers[i+1]))
            self.baises_noise.append(np.random.randn(self.populationSize, self.NNLayers[i+1]))
            pass

        
        for i in range(self.populationSize):
            self.population.append(NeuralNet(len(self.NNLayers), self.NNLayers))
            for j in range(len(self.NNLayers) - 1):
                self.population[i].weights[j] = self.parent.weights[j] + self.mutationRate * self.weights_noise[j][i]
                self.population[i].baises[j] = self.parent.baises[j] + self.mutationRate * self.baises_noise[j][i]

    def incrementGeneration(self, population_fitness):
        temp_data = {}
        temp_data["weights"] = []
        temp_data["baises"] = []

        for i in range(len(self.NNLayers) - 1):
            temp_data["weights"].append(self.parent.weights[i].tolist())
            temp_data["baises"].append(self.parent.baises[i].tolist())

        self.data[self.currentGeneration] = temp_data

        self.dump_data()

        self.currentGeneration = self.currentGeneration + 1

        # min_fitness = min(population_fitness)
        # total_fitness = 0

        # for i in range(self.populationSize):
        #     total_fitness += population_fitness[i] - min_fitness

        # for i in range(len(self.NNLayers) - 1):
        #     self.parent.weights[i] = np.zeros(shape=(self.NNLayers[i], self.NNLayers[i+1]))
        #     self.parent.baises[i] = np.zeros(shape=(1, self.NNLayers[i+1]))
        #     for j in range(self.populationSize):
        #         self.parent.weights[i] += ((population_fitness[j] - min_fitness) / total_fitness) * self.population[j].weights[i]
        #         self.parent.baises[i] += ((population_fitness[j] - min_fitness) / total_fitness) * self.population[j].baises[i]

        # self.population.clear()

        population_fitness = np.asarray(population_fitness)
        A = (population_fitness - np.mean(population_fitness)) / (np.std(population_fitness))
        # print(A.shape)
        for i in range(len(self.NNLayers) - 1):
            # print(self.weights_noise[i].transpose(1, 2, 0).shape)
            self.parent.weights[i] = self.parent.weights[i] + (self.learningRate) * np.dot(np.asarray(self.weights_noise[i]).transpose(1, 2, 0), A)
            self.parent.baises[i] = self.parent.baises[i] + (self.learningRate) * np.dot(np.asarray(self.baises_noise[i]).transpose(1, 0), A)

        self.population.clear()

        # population_fitness = np.asarray(population_fitness)
        # A = (population_fitness - np.min(population_fitness))
        # A = A / np.sum(A)

        # for i in range(len(self.NNLayers) - 1):
        #     self.parent.weights[i] = self.parent.weights[i] + self.learningRate * np.dot(np.asarray(self.weights_noise[i]).transpose(1, 2, 0), A)
        #     self.parent.baises[i] = self.parent.baises[i] + self.learningRate * np.dot(np.asarray(self.baises_noise[i]).transpose(1, 0), A)
        # self.population.clear()


ENVIRONMENT = 'BipedalWalker-v3'
MAX_STEPS = 1000
MAX_GENERATIONS = 300
POPULATION_SIZE = 200
MUTATION_RATE = 0.1
LEARNING_RATE = 0.01

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
    frames = []
    total_reward = 0
    for step in range(steps):
        frames.append(env.render(mode="rgb_array"))
        action = individual.getAction(observation)
        observation, reward, done, info = env.step(action)
        total_reward = total_reward + reward
        if done:
            break
    return np.array(frames), total_reward


@ray.remote
def run_individual(individual, steps):
    totalReward = 0
    env_local = gym.make(ENVIRONMENT)
    obs = env_local.reset()

    for step in range(steps):
        action = individual.getAction(obs)
        obs, reward, done, info = env_local.step(action)
        totalReward += reward
        if done:
            break
    
    return totalReward
    pass

population = Population(POPULATION_SIZE, MUTATION_RATE, LEARNING_RATE, (obs_dim, 64, 64, 32, action_dim))

population.initPopulation()

for i in range(MAX_GENERATIONS):
    print("\n\n\t GENERATION " + str(population.currentGeneration) + "\n")

    env_list = []
    for i in range(POPULATION_SIZE):
        env_list.append(gym.make(ENVIRONMENT))

    population.mutation()

    start_time = time.time()

    future_rewards = [run_individual.remote(individual, MAX_STEPS) for individual in population.population]
    totalRewards = ray.get(future_rewards)

    end_time = time.time()

    print("\t\tElapsed time to simulate generation: " + str(end_time - start_time) + "\n")
    # play_individual(population.parent, MAX_STEPS)
    observation = env.reset()
    parent_reward = 0

    for step in range(MAX_STEPS):
        # env.render()
        action = population.parent.getAction(observation)
        observation, reward, done, info = env.step(action)
        parent_reward += reward
        if done:
            break

    population.parent.fitness = parent_reward


    average_fitness = np.mean(np.asarray(totalRewards))
    population.incrementGeneration(totalRewards)
    # gif, parent_reward = play_individual(population.parent, MAX_STEPS)

    # population.parent.fitness = parent_reward
    print("\t\tGENERATION " + str(population.currentGeneration) + ", PARENT FITNESS: " + str(parent_reward))

    # gif = np.swapaxes(gif, 1, -1)
    # gif = np.swapaxes(gif, 2, -1)

    wandb.log(
        {"Parent Fitness": population.parent.fitness,
         "Average Population Fitness": average_fitness
         }, step=population.currentGeneration)

tmp = input("Press enter to continue...")