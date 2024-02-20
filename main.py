import matplotlib.pyplot as plt
import numpy as np

from Functions import *
from Algorithms import DTSR_LA, DTSR_OFFLINE, TLLA_DTSR
from matplotlib.ticker import MaxNLocator

K = 6
C_s = 9
S = C_s
C_c = 36

lambdas = [1, 0.75, 0.5, 0.25, 0]
# lambdas = [0.8, 0.6, 0.4, 0.3, 0.25, 0.2]
# sigmas = np.linspace(0, 500, 50)
# sigmas = np.linspace(-100, 0, 5)
# sigmas = list(sigmas)
# sigmas.extend(list(np.linspace(1, 10, 5)))
# sigmas = np.sort(sigmas)
sigmas = np.linspace(-60, 20, 100)



random.seed(12)
np.random.seed(seed=12)
colors = {}
colors[lambdas[0]]='#808A87'
colors[lambdas[1]]='#f5d389'
colors[lambdas[2]]='#d98175'
colors[lambdas[3]]='#af6300'
colors[lambdas[4]]='#000000'
# colors[lambdas[5]]='#556B2F'

shape = {}
shape[lambdas[0]]='-^'
shape[lambdas[1]]='-'
shape[lambdas[2]]='-.'
shape[lambdas[3]]='--'
shape[lambdas[4]]=':'
# shape[lambdas[5]]='-'

def sequence_generation(K, key="Uniform", multiple=False):
    instance = []

    # total_demand = np.random.randint(1, 8*K*S)
    S = random.randint(1, 60)
    for t in range(S):
        if multiple:
            demand = max(1, np.random.poisson(1))
        else:
            demand = 1
        if key == "Uniform":
            instance.append((demand, np.random.randint(0, K-1)))
        if key == "Long":
            if np.random.random() < 0.8:
                instance.append((demand, np.random.randint(0, 1)))
            else:
                instance.append((demand, np.random.randint(2, K-1)))
    return instance


iterations_poisson = 1
nb_experiments = 200

results = {}

for s in sigmas:
    for l in lambdas:
        results[(s, l)] = 0

for i in range(0, nb_experiments):
    print(i)
    # instance = DTSR_generator(K, S, key='Uniform')
    if i < nb_experiments * 0.4:
        instance = sequence_generation(K,  key='Uniform', multiple=True)
    else:
        instance = sequence_generation(K, key='Long', multiple=True)
    # print(instance)
    Offline, offline_single, offline_combo, demand = DTSR_OFFLINE(instance, K=K, C_s=C_s, C_c=C_c)
    for s in sigmas:
        heuristic = DTSR_heuristic(instance, K=K, noise=s, type="mu")
        # print("heuristic", heuristic)

        for l in lambdas:
            Online, online_single, online_combo = TLLA_DTSR(instance=instance, K=K, C_s=C_s, C_c=C_c, Lambda=l,
                                                          heuristic=heuristic)
            # print(" Lambda = ", l, " CR = ", Online / Offline)
            results[(s, l)] += (Online / Offline)
            # results[(s, l)] += (Online / Offline) / nb_experiments
            # if results[(s, l)] < (Online / Offline):
            #     results[s, l] = Online / Offline

for s in sigmas:
    for l in lambdas:
        results[(s, l)] /= nb_experiments



# text_file = open("poisson.txt", "w")
#
# for p in replacement_rate:
#     for l in lambdas:
#         n = text_file.write("====================\n")
#         n = text_file.write("Lambda = {} replacement rate = {}\n".format(l, p))
#         n = text_file.write("{}\n".format(results_1[(p, l)]))
#         print("====================")
#         print("Lambda = ", l, " replacement rate = ", p)
#         print(results_1[(p, l)])



#plotting data
fig = plt.figure(figsize=(8, 9), dpi=300)
index = np.arange(len(sigmas))
# index_label = [str(int(x)) for x in sigmas]
# bax = brokenaxes(xlims=((-100, -95), (-5, 10)),ylims=((1,3),), hspace=0.05, despine=False)
for l in lambdas:
    plot = []
    for s in sigmas:
        plot.append(results[s,l])
    if l == 0:
        plt.plot(sigmas, plot, shape[l], markevery=10, color=colors[l], label=r"$\theta$ = {} (FTP)".format(l), linewidth=5, markersize=10)
    elif l == 1:
        plt.plot(sigmas, plot, shape[l], markevery=10, color=colors[l], label=r"$\theta$ = {} (RDTSR)".format(l), linewidth=5, markersize=10)
    else:
        plt.plot(sigmas, plot, shape[l], markevery=10, color=colors[l], label=r"$\theta$ = {}".format(l), linewidth=5, markersize=10)
# plt.xticks(index, index_label)
plt.ylim(bottom=0.95)
plt.grid()
label_fontsize = 47
legned_fontsize = 35
plt.xticks(fontsize=label_fontsize)
plt.yticks(fontsize=label_fontsize)
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(7))
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
y_ticks = np.linspace(1, 3.5, 6)
plt.yticks(y_ticks)
plt.xlabel("Bias $\mu$", fontsize=label_fontsize)
plt.ylabel("Average Cost Ratio", fontsize=label_fontsize)
plt.legend(loc='upper left', fontsize=legned_fontsize, frameon=False)
plt.tight_layout()
plt.savefig('LA_synthetic.pdf', dpi=300)
plt.show()




# print(offline_single)
# print(offline_combo)
# print(Online)
# print(Offline)
# print("offline:-----------------------")
# print("combo:", offline_combo)
# print("single", offline_single)
# print("demand", demand)
# print("=====================")
# print("combo:", online_combo)
# print("single", online_single)
# print("=====================")
# print(Online / Offline)





