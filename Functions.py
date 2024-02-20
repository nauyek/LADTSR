import random
import math
import numpy as np
import matplotlib.pyplot as plt

def create_hard_instance(instance, K, C_s, C_c):
    demand_vec = np.zeros(K).astype(int)
    heuristic = np.zeros(K)
    new_instance = []

    for task in instance:
        demand_vec[task] += 1

    k = K - (C_c // C_s + 1)
    count = 0
    demand_idx = np.argsort(-demand_vec)
    for idx in demand_idx:
        if count < k:
            heuristic[idx] = 1
            count += 1
        else:
            heuristic[idx] = C_c
        new_instance.extend(demand_vec[idx] * [idx])
    return new_instance, heuristic

def DTSR_generator_total(K, S, key="Uniform"):
    instance = []

    # total_demand = np.random.randint(1, 8*K*S)
    T = int(random.gauss(K*S, 100))
    T = max(5, T)
    for t in range(T):
        demand = max(1, np.random.poisson(1))
        if demand != 0:
            if key == "Uniform":
                instance.append((demand, np.random.randint(0, K)))
            if key == "Long":
                if np.random.random() < 0.8:
                    instance.append((demand, np.random.randint(0, int(K*0.2))))
                else:
                    instance.append((demand, np.random.randint(int(K*0.2), K)))
    return instance

def DTSR_generator(K, S, key="Uniform"):
    instance = []
    if key == "Uniform":
        for k in range(K):
            total_demand_k = np.random.normal(S, 100)
            total_demand_k = max(1, total_demand_k)
            # total_demand_k = np.random.randint(1, 10*S)
            while total_demand_k > 0:
                demand = np.random.poisson(1)
                if demand != 0:
                    instance.append((demand, k))
                total_demand_k -= demand

        # generate the sequence for each item

    if key == "Heavy":
        for k in range(K):
            if k <= int(0.2 * K):
                # total_demand_k = np.random.normal(4 * S, 100)
                total_demand_k = np.random.randint(1, 4 * S)
            else:
                total_demand_k = np.random.randint(1, 0.2 * S)
                # total_demand_k = np.random.normal(0.2 * S, 100)
            total_demand_k = max(1, total_demand_k)

            while total_demand_k > 0:
                demand = np.random.poisson(1)
                if demand != 0:
                    instance.append((demand, k))
                total_demand_k -= demand
    random.shuffle(instance)
    return instance

def Heavy_random_generator(K, S):
    demands = K*S
    instance = []
    heavy_tail = 0.2 * np.random.random()

    for i in range(demands):
        if random.random() >= heavy_tail:
            instance.append(random.randint(0, int(K*0.1)))
        else:
            instance.append(random.randint(int(K*0.1)+1, K-1))
    instance.sort()
    return instance

def DTSR_heuristic(instance, K, noise, type):
    demand_vec = np.zeros(K)
    heuristic = np.zeros(K)

    for demand, idx in instance:
        demand_vec[idx] += demand

    for idx in range(K):
        if type == 'sigma':
            heuristic[idx] = demand_vec[idx] + np.random.normal(0, noise)
            heuristic[idx] = max(0, heuristic[idx])
        if type == 'mu':
            heuristic[idx] = demand_vec[idx] + noise
            heuristic[idx] = max(0, heuristic[idx])
    return heuristic

def sin_demand_generator(periods, points_per_period, price):
    phase = 0.5 * np.pi * np.random.random()
    amplitude = 10 * price * np.random.random()
    X = np.linspace(phase, periods * 2 * np.pi - 2.0 * np.pi / points_per_period + phase, periods * points_per_period)
    Y = price + amplitude * np.sin(X)
    Y = np.clip(Y, a_min=1, a_max=None)

    return X, Y

def Azure_instance_generator(X):
    items = X.shape[0]
    instance = []

    for k in range(items):
        total_demand_k = X[k]
        total_demand_k = max(0, total_demand_k)
        # total_demand_k = np.random.randint(1, 10*S)
        while total_demand_k > 0:
            demand = np.random.poisson(1)
            if demand != 0:
                instance.append((demand, k))
            total_demand_k -= demand

    # generate the sequence for each item

    random.shuffle(instance)
    return instance














#generates one random entry
def entry_generator(mean, n, shape, key):
    
    if(key == "Pareto"):
        return round(np.random.pareto(shape))
    
    X = mean
    for a in range(0,n):
        X = np.random.poisson(X)
    
    return X

#generates a full instance of some length using the function entry_generator for each entry
#mean value, length of instance, number of iterations n as inputs
def instance_generator(mean, length, n, shape, key):
    
    instance = []
    for i in range(0, length):
        x = entry_generator(mean, n, shape, key)
        for j in range(0,x):
            instance.append(i)
        
    return instance

#from an instance described as a list of requests, generates an array T with T[i]=# of requests at time i
def agreggate_instance(instance):
    
    instance=sorted(instance)
    
    if(len(instance)==0): return []
    
    min_time = instance[0]
    max_time = instance[len(instance)-1]
    
    instance_agreggated = [0]*min_time
    
    current = 0
    for time in range(min_time, max_time+1):
        instance_agreggated.append(0)
        while(current<len(instance) and instance[current] == time):
            instance_agreggated[time]+=1
            current+=1
            
    return instance_agreggated

#reverse operation of agreggate_instance, from an array T with T[i]=# of requests at time i, outputs an instance described as a list of requests
def deagreggate_instance(instance):
    
    result = []
    
    for i in range(0, len(instance)):
        for x in range(0, instance[i]): result.append(i)
            
    return result

#generates a noisy instance from an instance
#replacement_rate is the replacement rate described in the main paper
def noisy_instance(instance, mean, n, shape, key, replacement_rate):
        
    instance_agreggated = agreggate_instance(instance)
    
    instance_agreggated_noisy = [0]*len(instance_agreggated)
    
    for i in range(0,len(instance_agreggated)):
        drop = (random.random()<replacement_rate)
        insert = (random.random()<replacement_rate)
        instance_agreggated_noisy[i]=instance_agreggated[i]
        if(drop): instance_agreggated_noisy[i]=0
        if(insert): instance_agreggated_noisy[i]+=entry_generator(mean, n, shape, key)
                
            
    final_instance = deagreggate_instance(instance_agreggated_noisy)
    
    return final_instance

#plots an instance as an histogram
def plot_instance(instance):
    
    instance_agreggated=agreggate_instance(instance)
    
    plt.plot(instance_agreggated)
    plt.ylabel('number of requests')
    plt.xlabel('time')
    plt.show()
    
    return 0

