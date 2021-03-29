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

ENVIRONMENT = 'BipedalWalker-v3'
MAX_STEPS = 1000
MAX_GENERATIONS = 1000
POPULATION_SIZE = 200
INERTIA = 0.9
INERTIA_DECAY_RATE = 0.001
GBEST_MAX_CONTRIB = 0.5
IBEST_MAX_CONTRIB = 0.3

env = gym.make(ENVIRONMENT)
observation = env.reset()

env.render()

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

NNLayers = (obs_dim, 64, 128, 32, action_dim)


def play_individual(individual, steps):
    observation = env.reset()
    frames = []
    for step in range(steps):
        frames.append(env.render(mode="rgb_array"))
        action = individual.getAction(observation)
        observation, reward, done, info = env.step(action)
        if done:
            break
    return np.array(frames)


class NeuralNet:

    def __init__(self, numLayers, numNodes):
        super().__init__()

        self.numLayers = numLayers
        self.numNodes = numNodes

        self.weights = []
        self.baises = []

        self.velocity_weights = []
        self.velocity_baises = []

        self.IBEST_weights = []
        self.IBEST_baises = []
        self.IBEST_fitness = float('-inf')

        self.fitness = 0.0

        self.initWeights()

    def initWeights(self):

        for i in range(0, self.numLayers - 1):
            self.weights.append(np.random.uniform(
                low=-1.0, high=1.0, size=(self.numNodes[i], self.numNodes[i+1])))
            self.baises.append(np.random.uniform(
                low=-1.0, high=1.0, size=(self.numNodes[i+1])))

            self.velocity_weights.append(np.random.uniform(
                low=-1.0, high=1.0, size=(self.numNodes[i], self.numNodes[i+1])))
            self.velocity_baises.append(np.random.uniform(
                low=-1.0, high=1.0, size=(self.numNodes[i+1])))

        self.IBEST_weights = self.weights.copy()
        self.IBEST_baises = self.baises.copy()

        self.weights = np.array([np.array(xi)
                                 for xi in self.weights], dtype=object)
        self.baises = np.array([np.array(xi)
                                for xi in self.baises], dtype=object)

        self.velocity_weights = np.array(
            [np.array(xi) for xi in self.velocity_weights], dtype=object)
        self.velocity_baises = np.array(
            [np.array(xi) for xi in self.velocity_baises], dtype=object)

        self.IBEST_weights = np.array(
            [np.array(xi) for xi in self.IBEST_weights], dtype=object)
        self.IBEST_baises = np.array([np.array(xi)
                                      for xi in self.IBEST_baises], dtype=object)

    def getAction(self, observation):

        action = observation
        for i in range(0, self.numLayers - 1):
            action = np.matmul(np.asarray(action),
                               self.weights[i]) + self.baises[i]
            action = np.reshape(action, (1, self.numNodes[i+1]))
            if(i == self.numLayers - 2):
                action = np.tanh(action)
            else:
                action = self.relu(action)
            action = np.reshape(action, (self.numNodes[i+1]))

        return action

    def relu(self, x):
        return np.maximum(0, x)


class Population:

    def __init__(self, populationSize, inertia, GBEST_max_contrib, IBEST_max_contrib, NNLayers):
        super().__init__()

        self.populationSize = populationSize
        self.inertia = inertia
        self.GBEST_max_contrib = GBEST_max_contrib
        self.IBEST_max_contrib = IBEST_max_contrib
        self.NNLayers = NNLayers
        self.population = []
        self.currentGeneration = 0

        self.data = {}
        self.datafile_name = "trained_weights_pso.json"

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

        for i in range(self.populationSize):
            self.population.append(
                NeuralNet(len(self.NNLayers), self.NNLayers))


@ ray.remote
def updatePositions(individual, GBEST):
    R1 = random.random()
    R2 = random.random()

    individual.velocity_weights = INERTIA * individual.velocity_weights + GBEST_MAX_CONTRIB * R1 * \
        (GBEST.weights - individual.weights) + IBEST_MAX_CONTRIB * \
        R2 * (individual.IBEST_weights - individual.weights)
    individual.velocity_baises = INERTIA * individual.velocity_baises + GBEST_MAX_CONTRIB * R1 * \
        (GBEST.baises - individual.baises) + IBEST_MAX_CONTRIB * \
        R2 * (individual.IBEST_baises - individual.baises)

    individual.weights = individual.weights + individual.velocity_weights
    individual.baises = individual.baises + individual.velocity_baises

    return [individual.velocity_weights, individual.velocity_baises, individual.weights, individual.baises]


@ ray.remote
def run_individual(individual, steps, weights, baises):
    totalReward = 0

    individual.weights = np.copy(weights)
    individual.baises = np.copy(baises)

    env_local = gym.make(ENVIRONMENT)
    obs = env_local.reset()

    for step in range(steps):
        action = individual.getAction(obs)
        obs, reward, done, info = env_local.step(action)
        totalReward += reward
        if done:
            break

    return totalReward


@ ray.remote
def run_individual2(individual, steps):
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


obs_range = (env.observation_space.low, env.observation_space.high)
action_range = (env.action_space.low, env.action_space.high)

print("OBSERVATION --> \nSHAPE:" + str(obs_dim) +
      "x1, \nRANGE: (" + str(obs_range[0]) + ", " + str(obs_range[1]) + ")")
print("\n")
print("Action --> \nSHAPE:" + str(action_dim) + "x1, \nRANGE: (" +
      str(action_range[0]) + ", " + str(action_range[1]) + ")")


population = Population(POPULATION_SIZE, INERTIA,
                        GBEST_MAX_CONTRIB, IBEST_MAX_CONTRIB, NNLayers)
population.initPopulation()

GBEST = NeuralNet(len(NNLayers), NNLayers)
GBEST.fitness = float('-inf')

init_fitness_futures = [run_individual2.remote(
    population.population[i], MAX_STEPS) for i in range(POPULATION_SIZE)]
init_fitness = ray.get(init_fitness_futures)

for i in range(POPULATION_SIZE):
    population.population[i].fitness = init_fitness[i]

    if(init_fitness[i] > GBEST.fitness):
        GBEST = deepcopy(population.population[i])

init_fitness_futures.clear()
init_fitness.clear()


for i in range(MAX_GENERATIONS):

    population.currentGeneration = population.currentGeneration + 1
    print("\n\n\t GENERATION " + str(population.currentGeneration) + "\n")

    start_time = time.time()

    future_updates = [updatePositions.remote(
        population.population[i], GBEST) for i in range(POPULATION_SIZE)]
    after_update = ray.get(future_updates)

    future_rewards = [run_individual.remote(
        population.population[i], MAX_STEPS, after_update[i][2], after_update[i][3]) for i in range(POPULATION_SIZE)]
    rewards = ray.get(future_rewards)

    end_time = time.time()

    print("\t\tElapsed time to simulate generation: " +
          str(end_time - start_time) + "\n")

    max_fitness = float('-inf')
    min_fitness = float('inf')
    average_fitness = 0
    best_individual = None

    counter = 0
    for individual in population.population:
        individual.fitness = rewards[counter]
        individual.weights = np.copy(after_update[counter][2])
        individual.baises = np.copy(after_update[counter][3])
        individual.velocity_weights = np.copy(after_update[counter][0])
        individual.velocity_baises = np.copy(after_update[counter][1])

        if(individual.IBEST_fitness <= individual.fitness):
            individual.IBEST_weights = np.copy(individual.weights)
            individual.IBEST_baises = np.copy(individual.baises)
            individual.IBEST_fitness = individual.fitness

        if(individual.fitness > GBEST.fitness):
            GBEST = deepcopy(individual)

        if(individual.fitness > max_fitness):
            max_fitness = individual.fitness
            best_individual = deepcopy(individual)

        if(individual.fitness < min_fitness):
            min_fitness = individual.fitness

        average_fitness += individual.fitness

        counter = counter + 1

    average_fitness = average_fitness / counter

    print("\t\tGENERATION " + str(population.currentGeneration) + ", BEST FITNESS: " +
          str(best_individual.fitness) + ", MIN FITNESS: " + str(min_fitness) + ", AVG. FITNESS: " + str(average_fitness))
    print("\t\t-->GBEST fitness: " + str(GBEST.fitness))
    gif = play_individual(best_individual, MAX_STEPS)

    gif = np.swapaxes(gif, 1, -1)
    gif = np.swapaxes(gif, 2, -1)

    wandb.log(
        {"Best Fitness": best_individual.fitness,
         "Minimum Fitness": min_fitness,
         "Average Fitness": average_fitness,
         "Global Best Fitness": GBEST.fitness,
         "Agent": wandb.Video(gif, fps=30, format="mp4")
         }, step=population.currentGeneration)

    # Updating inertia
    INERTIA = INERTIA * (1 / (1 + INERTIA_DECAY_RATE *
                              population.currentGeneration))

    temp_data = {}
    temp_data["weights"] = []
    temp_data["baises"] = []

    for i in range(len(population.NNLayers) - 1):
        temp_data["weights"].append(best_individual.weights[i].tolist())
        temp_data["baises"].append(best_individual.baises[i].tolist())

    population.data[population.currentGeneration] = temp_data
    population.dump_data()


tmp = input("Press enter to continue...")
