import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import seaborn as sns
sns.set()


# sns.set(style="whitegrid")


# Function: Change of Population
def changePop(b, d):
    r = np.random.random_sample()
    if r < b:
        return 1
    elif r > (b+d):
        return 0
    else:
        return -1


# Function: Find Probability of Birth/Dearth/None
def findProp(i):
    b = r*(i - np.power(i, 2)/N)
    d = r*np.power(i, 2)/N
    return b, d


# Function: Generate One Population Trajectory
def popTrajectory(t, P0):
    pop = np.zeros(t)
    pop[0] = P0
    for k in range(t-1):
        if pop[k] < 1:
            break
        else:
            b, d = findProp(pop[k])
            pop[k+1] = pop[k] + changePop(b, d)
    return pop


# Generate and plot some population trajectories
# --- Intial Parameters
r = 0.004
K = 50
t = 3000
aP0 = [8, 50, 100]  # Initial population

N = 2*K


#--- Plot
current_palette = sns.color_palette()
for P0 in aP0:
    pop = popTrajectory(t, P0)
    time = np.linspace(0, t-1, t)
    time_pop = pd.DataFrame(pop, time, [P0])
    time_pop = time_pop.rolling(7).mean()
    ax = sns.lineplot(data=time_pop, palette=[
                      current_palette[np.random.randint(0, len(current_palette))]], linewidth=1)

ax.set(xlabel='Time', ylabel='Population')
plt.show()


#--- Histogram
N_trial = 500
P0 = 100
t = 3000

s = []
for i in range(N_trial):
    pop = popTrajectory(t, P0)
    s.append(pop[t-1])

sns.distplot(s, label='Histogram of Population')
plt.legend(loc="upper left")
plt.show()


# # Find the extinction time
# # --- Intial Parameters

# r = 0.015
# K = 8
# t = 8000
# P0 = 100  #Initial population

# N = 2*K


# # Function: Extinction time
# def extinctionTime(pop):
#     extime = 0
#     while pop > 0:
#         extime += 1
#         b, d = findProp(pop)
#         pop += changePop(b, d)
#     return extime


# print("Extinction when ", extinctionTime(P0))

# start = time.time()
# p0 = np.random.randint(1, 100, 15)
# nloop = len(p0)
# aextime = np.zeros(nloop)
# for i in range(nloop):
#     aextime[i] = extinctionTime(p0[i])

# print(aextime)
# end = time.time()
# print(end-start)
