import math
import random
import numpy as np
import copy
import matplotlib.pyplot as plt

num_candidates = 40

input_neurons = 2
hidden_neurons = 2
output_neurons = 1
hidden_weights = input_neurons*hidden_neurons
output_weights = hidden_neurons*output_neurons
total_num_weights = hidden_weights + output_weights
total_num_biases = hidden_neurons + output_neurons
genome_length = total_num_weights + total_num_biases

candidates = []
for _ in range(num_candidates):
    weights = [random.uniform(-5.0, 5.0) for _ in range(total_num_weights)]
    biases = [random.uniform(-1.0, 1.0) for _ in range(total_num_biases)]
    genome = weights[:hidden_weights]+biases[:hidden_neurons]+weights[hidden_weights:]+biases[hidden_neurons:]
    candidates.append(genome)

states = [(0,0),(0,1),(1,0),(1,1)]
answers = [0,1,1,0]

sigmoid = lambda x: 1/(1+np.exp(-x))

def fitness(candidates, states, answers):
    fitness_list = []
    for genomes in candidates:
        wrong = 0
        total_error = 0
        for j, i in enumerate(states):
            hidden_n1 = sigmoid(i[0]*genomes[0] + i[1]*genomes[2] + genomes[4])
            hidden_n2 = sigmoid(i[0]*genomes[1] + i[1]*genomes[3] + genomes[5])
            output_n0 = sigmoid(hidden_n1*genomes[6] + hidden_n2*genomes[7] + genomes[8])
            # result = 1 if output_n0>=0.5 else 0
            # if result != answers[j]:
                # wrong+=1
                # total_error += abs(output_n0 - answers[j])
            total_error += abs(output_n0 - answers[j])**15
        # fit = (4-wrong)**2 - 0.5*total_error    #changed from 1/(1+wrong)
        fit = (1/total_error)**5
        fitness_list.append(fit)
    return fitness_list

def selection(candidates, states, answers):
    random.shuffle(candidates)
    fitness_list = fitness(candidates, states, answers)
    elite_index = fitness_list.index(max(fitness_list))
    survived = []
    best = candidates[elite_index][:]
    survived.append(best)
    survived.append(best)
    
    players = 3
    for x in range(0, num_candidates, players):
        slice_start = x
        slice_end = min(num_candidates, x+players)
        local_slice = fitness_list[slice_start:slice_end]
        winner = max(local_slice)
        winner_idx = x+local_slice.index(winner)
        survived.append(candidates[winner_idx])
    survivors = survived[:]
    return survivors, fitness_list, best

def crossover(survivors,best):
    next_gen = []
    next_gen.extend(random.sample(survivors, math.ceil(num_candidates*0.1)))
    while len(next_gen)<num_candidates:
        parent_1, parent_2 = random.sample(survivors, 2)
        child = []
        for x in range(genome_length):
            flag = random.randint(1,10)
            if flag>5:
                child.append(parent_1[x])
            else:
                child.append(parent_2[x])
        next_gen.append(child)
    next_gen[num_candidates-1] = best
    next_gen[num_candidates-2] = best
    return next_gen, best

def mutation(next_gen, best):
    mutation_rate = 15
    noise_list = [0.1,-0.1]
    for x in next_gen:
        for i, j in enumerate(x):
            flag = random.randint(1,100)
            noise = random.choice(noise_list)
            if mutation_rate>flag:
                x[i] += noise
    next_gen[num_candidates-1] = best
    next_gen[num_candidates-2] = best
    return next_gen, best

def new_mutation(survivors, best):
    next_gen = survivors[:]
    mutation_rate = 20
    while len(next_gen)<num_candidates:
        to_mutate = copy.deepcopy(random.choice(survivors))
        noise_list = [0.2, -0.2]
        for i, x in enumerate(to_mutate):
            flag = random.randint(1,100)
            noise = random.choice(noise_list)
            if mutation_rate>flag:
                to_mutate[i] += noise
        next_gen.append(to_mutate)
    next_gen[num_candidates-1] = best
    next_gen[num_candidates-2] = best
    return next_gen, best

num_gen = 300
best_each_gen = []
avg_each_gen = []
for _ in range(num_gen):
    survivors, fitness_list, best = selection(candidates, states, answers)
    best_each_gen.append(max(fitness_list))
    avg_each_gen.append(sum(fitness_list)/num_candidates)
    candidates, best = new_mutation(survivors, best)
    # next_gen, best = crossover(survivors, best)
    # candidates, best = mutation(next_gen, best)

final_fitness = fitness(candidates, states, answers)
best_last_gen = candidates[final_fitness.index(max(fitness_list))]
output = []
for i in states:
    h1 = sigmoid(i[0]*best_last_gen[0] + i[1]*best_last_gen[2] + best_last_gen[4])
    h2 = sigmoid(i[0]*best_last_gen[1] + i[1]*best_last_gen[3] + best_last_gen[5])
    output.append(sigmoid(h1*best_last_gen[6] + h2*best_last_gen[7] + best_last_gen[8]))

print(output)
