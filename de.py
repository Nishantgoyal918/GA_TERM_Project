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

wandb.init(project="genetic")

ENVIRONMENT = 'BipedalWalker-v3'
MAX_STEPS = 1000
MAX_GENERATIONS = 200
POPULATION_SIZE = 200
MUTATION_RATE = 0.3
CROSSOVER_RATE = 0.1

env = gym.make(ENVIRONMENT)
observation = env.reset()

env.render()

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

NNLayers = (obs_dim, 64, 64, 32, action_dim)

ray.init(num_cpus=8)

def play_individual_wandb(individual, steps):
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


class Population:

    def __init__(self, populationSize, mutationRate, crossoverRate, NNLayers):
        super().__init__()
        
        self.populationSize = populationSize
        self.mutationRate = mutationRate
        self.crossoverRate = crossoverRate
        self.NNLayers = NNLayers
        self.population = []
        self.currentGeneration = 0

        self.data = {}
        self.datafile_name = "trained_weights_de_2.json"

    
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

        if(self.populationSize == 0):
            numVariables = 0
            for i in range(len(self.NNLayers) - 1):
                numVariables += self.NNLayers[i] * self.NNLayers[i+1] + self.NNLayers[i+1]
            
            self.populationSize = numVariables * 10


        if(self.populationSize == -1):
            numVariables = 0
            for i in range(len(self.NNLayers) - 1):
                numVariables += self.NNLayers[i] * self.NNLayers[i+1] + self.NNLayers[i+1]
            
            if(numVariables % 2 == 0):
                self.populationSize = numVariables
            else:
                self.populationSize = numVariables + 1


        for i in range(self.populationSize):
            self.population.append(NeuralNet(len(self.NNLayers), self.NNLayers))

    
    def crossover(self, parent1_index, parent2_index):
        # parent1 = random.choice(self.population)
        # parent2 = random.choice(self.population)
        parent1 = self.population[parent1_index]
        parent2 = self.population[parent2_index]

        child = NeuralNet(len(self.NNLayers), self.NNLayers)
        child.weights = parent1.weights.copy()
        child.baises = parent1.baises.copy()

        # masks_weights = []
        # masks_biases = []

        # for i in range(0, len(self.NNLayers) - 1):
            # masks_weights.append(np.random.randint(low=0, high=2, size=(self.NNLayers[i], self.NNLayers[i+1])))
            # masks_biases.append(np.random.randint(low=0, high=2, size=(1, self.NNLayers[i+1])))
            # masks_weights.append(np.random.choice(2, (self.NNLayers[i], self.NNLayers[i+1]), p=[1-self.crossoverRate, self.crossoverRate]))
            # masks_biases.append(np.random.choice(2, (1, self.NNLayers[i+1]), p=[1-self.crossoverRate, self.crossoverRate]))
        
        for i in range(0, len(self.NNLayers) - 1):
            mask_weight = np.random.choice(2, (self.NNLayers[i], self.NNLayers[i+1]), p=[1-self.crossoverRate, self.crossoverRate])
            child.weights[i] = (1 - mask_weight) * child.weights[i] + mask_weight * parent2.weights[i]
            mask_biases = np.random.choice(2, (1, self.NNLayers[i+1]), p=[1-self.crossoverRate, self.crossoverRate])
            child.baises[i] = (1 - mask_biases) * child.baises[i] + mask_biases * parent2.baises[i]

        return child

    def mutation(self, parent, parent1_index, parent2_index):
        # parent1 = random.choice(self.population)
        # parent2 = random.choice(self.population)

        parent1 = self.population[parent1_index]
        parent2 = self.population[parent2_index]

        child = NeuralNet(len(self.NNLayers), self.NNLayers)
        child.weights = parent.weights.copy()
        child.baises = parent.baises.copy()

        for i in range(0, len(self.NNLayers) - 1):
            child.weights[i] = child.weights[i] + self.mutationRate * (parent1.weights[i] - parent2.weights[i])
            child.baises[i] = child.baises[i] + self.mutationRate * (parent1.baises[i] - parent2.baises[i])

        
        return child

    def run_individual(self, individual):
        totalReward = 0
        env_local = gym.make(ENVIRONMENT)
        obs = env_local.reset()

        for step in range(MAX_STEPS):
            action = individual.getAction(obs)
            obs, reward, done, info = env_local.step(action)
            totalReward += reward
            if done:
                break
    
        return totalReward

    def selection(self, individual1, individual2):
        individual1.fitness = self.run_individual(individual1)
        individual2.fitness = self.run_individual(individual2)

        if(individual2.fitness >= individual1.fitness):
            return individual2
        else:
            return individual1


@ray.remote
def crossover_parallel(parent1, parent2):
    # parent1 = self.population[parent1_index]
    # parent2 = self.population[parent2_index]

    child = NeuralNet(len(NNLayers), NNLayers)
    child.weights = parent1.weights.copy()
    child.baises = parent1.baises.copy()

        # masks_weights = []
        # masks_biases = []

        # for i in range(0, len(self.NNLayers) - 1):
            # masks_weights.append(np.random.randint(low=0, high=2, size=(self.NNLayers[i], self.NNLayers[i+1])))
            # masks_biases.append(np.random.randint(low=0, high=2, size=(1, self.NNLayers[i+1])))
            # masks_weights.append(np.random.choice(2, (self.NNLayers[i], self.NNLayers[i+1]), p=[1-self.crossoverRate, self.crossoverRate]))
            # masks_biases.append(np.random.choice(2, (1, self.NNLayers[i+1]), p=[1-self.crossoverRate, self.crossoverRate]))
        
    for i in range(0, len(NNLayers) - 1):
            mask_weight = np.random.choice(2, (NNLayers[i], NNLayers[i+1]), p=[1-CROSSOVER_RATE, CROSSOVER_RATE])
            child.weights[i] = (1 - mask_weight) * child.weights[i] + mask_weight * parent2.weights[i]
            mask_biases = np.random.choice(2, (1, NNLayers[i+1]), p=[1-CROSSOVER_RATE, CROSSOVER_RATE])
            child.baises[i] = (1 - mask_biases) * child.baises[i] + mask_biases * parent2.baises[i]

    return child

@ray.remote
def mutation_parallel(parent, parent1, parent2):
        # parent1 = random.choice(self.population)
        # parent2 = random.choice(self.population)

        # parent1 = self.population[parent1_index]
        # parent2 = self.population[parent2_index]

        child = NeuralNet(len(NNLayers), NNLayers)
        child.weights = parent.weights.copy()
        child.baises = parent.baises.copy()

        for i in range(0, len(NNLayers) - 1):
            child.weights[i] = child.weights[i] + MUTATION_RATE * (parent1.weights[i] - parent2.weights[i])
            child.baises[i] = child.baises[i] + MUTATION_RATE * (parent1.baises[i] - parent2.baises[i])

        
        return child

def run_individual_parallel(individual):
        totalReward = 0
        env_local = gym.make(ENVIRONMENT)
        obs = env_local.reset()

        for step in range(MAX_STEPS):
            action = individual.getAction(obs)
            obs, reward, done, info = env_local.step(action)
            totalReward += reward
            if done:
                break
    
        return totalReward

def selection_parallel(individual1, individual2):
        individual1.fitness = run_individual_parallel(individual1)
        individual2.fitness = run_individual_parallel(individual2)

        if(individual2.fitness >= individual1.fitness):
            return individual2
        else:
            return individual1

@ray.remote
def incrementGeneration(i, population):
    candidates = list(range(0, POPULATION_SIZE))
    candidates.remove(i)
    random_members = random.sample(candidates, 3)

    child1 = population.crossover(i, random_members[0])
    child1_mutated = population.mutation(child1, random_members[1], random_members[2])

    return population.selection(child1, child1_mutated)

@ray.remote
def incrementGeneration___1(child1, child1_mutated):
    return selection_parallel(child1, child1_mutated)


def play_individual(individual, steps):
    observation = env.reset()
    for step in range(steps):
        env.render()
        action = individual.getAction(observation)
        observation, reward, done, info = env.step(action)
        if done:
            break

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


obs_range = (env.observation_space.low, env.observation_space.high)
action_range = (env.action_space.low, env.action_space.high)

print("OBSERVATION --> \nSHAPE:" + str(obs_dim) + "x1, \nRANGE: (" + str(obs_range[0]) + ", " + str(obs_range[1]) + ")")
print("\n")
print("Action --> \nSHAPE:" + str(action_dim) + "x1, \nRANGE: (" + str(action_range[0]) + ", " + str(action_range[1]) + ")")

population = Population(POPULATION_SIZE, MUTATION_RATE, CROSSOVER_RATE, (obs_dim, 64, 64, 32, action_dim))

population.initPopulation()

for i in range(MAX_GENERATIONS):
    population.currentGeneration = population.currentGeneration + 1
    print("\n\n\t GENERATION " + str(population.currentGeneration) + "\n")

    start_time = time.time()
    # run_individual(env_list, population.population, MAX_STEPS)

    # future_rewards = [incrementGeneration.remote(i, population) for i in range(POPULATION_SIZE)]
    # next_population = ray.get(future_rewards)

    candidates=[]
    for j in range(POPULATION_SIZE):
        temp_1 = list(range(0, POPULATION_SIZE))
        temp_1.remove(j)
        random_members = random.sample(temp_1, 3)
        candidates.append(temp_1)

    mutation_future_childs = [mutation_parallel.remote(population.population[candidates[i][0]], population.population[candidates[i][1]], population.population[candidates[i][2]]) for i in range(POPULATION_SIZE)]
    mutation_childs = ray.get(mutation_future_childs)

    crossover_future_childs = [crossover_parallel.remote(population.population[i], mutation_childs[i]) for i in range(POPULATION_SIZE)]
    crossover_childs = ray.get(crossover_future_childs)

    future_childs = [incrementGeneration___1.remote(population.population[i], crossover_childs[i]) for i in range(POPULATION_SIZE)]
    next_population = ray.get(future_childs)

    end_time = time.time()

    print("\t\tElapsed time to simulate generation: " + str(end_time - start_time) + "\n")

    best_individual = None
    max_fitness = float('-inf')
    min_fitness = float('inf')
    average_fitness = 0

    population.population = next_population.copy()

    counter = 0
    for individual in population.population:
        if(individual.fitness > max_fitness):
            max_fitness = individual.fitness
            best_individual = deepcopy(individual)
        if(individual.fitness < min_fitness):
            min_fitness = individual.fitness
        
        average_fitness += individual.fitness
        
        counter = counter + 1
    
    average_fitness = average_fitness / counter

    print("\t\tGENERATION " + str(population.currentGeneration) + ", BEST FITNESS: " + str(best_individual.fitness) + ", MIN FITNESS: " + str(min_fitness) + ", AVG. FITNESS: " + str(average_fitness))    
    #play_individual(best_individual, MAX_STEPS)

    gif = play_individual_wandb(best_individual, MAX_STEPS)

    gif = np.swapaxes(gif, 1, -1)
    gif = np.swapaxes(gif, 2, -1)

    wandb.log(
        {"Best Fitness": best_individual.fitness,
         "Minimum Fitness": min_fitness,
         "Average Fitness": average_fitness,
         "Agent": wandb.Video(gif, fps=30, format="mp4")
         }, step=population.currentGeneration)

    temp_data = {}
    temp_data["weights"] = []
    temp_data["baises"] = []

    for i in range(len(population.NNLayers) - 1):
            temp_data["weights"].append(best_individual.weights[i].tolist())
            temp_data["baises"].append(best_individual.baises[i].tolist())

    population.data[population.currentGeneration] = temp_data

    population.dump_data()


tmp = input("Press enter to continue...")