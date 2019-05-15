#!/usr/bin/python 
import numpy as np
import networkx as nx
#import matplotlib.pyplot as plt
import math
import heapq
import pickle as pkl
import time

F=0.001

to_save={}
save_loc="/home/kabir/SOC_model/data.pkl"
save_loc="/storage/subhadra/kabir/output/SOC_model/data.pkl"

time_loc="/home/kabir/SOC_model/time"+str(F)+".txt"

def WRITE(text):
    with open(time_loc,'a') as time_file:
        time_file.write(text+"\n")

def avalanche(x,A,f,N):

    count = 0
    degree = np.array(np.sum(A,0))[0]
    degree = np.reshape(degree, (N,1))
    xc = np.reshape(degree + (degree==0), (N,1))
    spikes = np.multiply(x>=xc, 1)
    ava = spikes.copy()
    while np.sum(np.multiply(degree,spikes))> 0:
        ava = np.multiply((ava+spikes)>0,1)
        spikes = np.multiply(x>=degree, 1)
        spikes = np.reshape(spikes,(N,1))
        x = x + A*spikes
        x = x - np.multiply(spikes,degree)
        #x = x - (np.random.rand(N,1)>f)
        x = np.multiply(x,(x>0))
        count = count + 1

    return [x,ava]
    
    
def my_kmax(R,k):
    N = len(R)
    f = np.ravel(R)
    #print(f)
    indices = np.array(heapq.nlargest(k, range(len(f)), f.__getitem__))
    j = np.mod(indices,N)
    i = np.floor(indices/N)
    #print(indices)
    return [i,j]
    


WRITE("Starting")
N = 1000
f = F#0.1 #f leak (fraction of leak)
f = 1-f
mean_degree = 4

K = 4
steps = 500
#tic
np.random.seed(1)
WRITE("Initializing graph")
G = nx.erdos_renyi_graph(N,mean_degree/N)
WRITE("Done initializing graph")

A = nx.to_numpy_matrix(G, dtype=np.int)
to_save['initA']=A
#fig = plt.figure(figsize=(5, 5))
#plt.title("Adjacency Matrix")
#plt.imshow(A,cmap="Greys",interpolation="none")   
#%matplotlib inline
#sns.heatmap(A, cmap="Greys")
#plt.show()

#storex = np.zeros((N,steps))
#storex_noava = np.zeros((N,steps))


temp = np.arange(0,N**2)
R = temp.reshape(N,N)
R = np.multiply(R,A)
A_ini = A.copy()
degree = np.array(np.sum(A,0))[0]



#Initial State
x = np.zeros((N,1));
for i in range(N):
    if degree[i]>0:
        np.random.seed(1)
        x[i]=np.random.randint(0,degree[i])
    else:
        x[i]=0
        
xb = x

count = 0
#xsave = [0]*steps

save_distribution = []
WRITE("Starting simulation")
start = time.time()
for i in range(steps):
    
    if np.mod(i,float(steps)/50) == 0:
        dist = np.bincount(list(degree.flat))
        #x_axis = np.arange(0,max(degree)+1)
        save_distribution.append((dist,degree))
    if np.mod(i,float(steps)/500) == 0:
        WRITE("Step:"+str(i))
    #Particle addition
    
    add_site = np.random.randint(0,N-1)

    x[add_site] = x[add_site] + 1
    WRITE("Starting avalanche processing")
    #Avalanche processing
    degree = np.array(np.sum(A,0))[0]
    degree = np.reshape(degree, (N,1))
    xc = np.reshape(degree + (degree==0), (N,1))
    spikes = np.multiply(x>=xc, 1)
    ava = spikes.copy()
    temp1 = np.zeros((N,1))
    if np.sum(np.multiply(degree,spikes))>0: 
        [x,temp1] = avalanche(x,A,f,N)
        x = x - (np.random.rand(N,1)>f)
        count = count + 1
    ava = np.multiply((ava+temp1)>0,1)
    #xsave[i] = x
    a = np.sum(ava)
    WRITE("Avalanche processing over")

    WRITE("Starting Rewiring")    
    ##Rewiring
    
    R = np.multiply(R,A)
    Ruse = R.copy()
    """
    for c1 in range(N):
        for c2 in range(N):
            if Ruse[c1,c2] <= Ruse[c2,c1]:
                Ruse[c1,c2] = -1
            else:
                Ruse[c2,c1] = -1
    #"""
    
    max_endpoints=[]
    max_endpoints_r=[]
    for edge in G.edges():
        c1,c2=edge
        if R[c1,c2] <= R[c2,c1]:
            max_endpoints.append((c1,c2))
            max_endpoints_r.append(R[c1,c2])
        else:
            max_endpoints.append((c2,c1))
            max_endpoints_r.append(R[c2,c1])
    
    indices = np.array(heapq.nlargest(a, range(len(max_endpoints_r)), max_endpoints_r.__getitem__))
    i,j=tuple(zip(*[max_endpoints[i] for i in indices]))
    
    WRITE("Finding kmax")
    ie,je = my_kmax(Ruse,a)
    
    WRITE("Done kmax")    
    
    

    for j in range(a):
        if (je[j] != add_site) & (A[int(ie[j]),int(je[j])] != 0) & (A[add_site,je[j]] == 0):
            #print('hi')
            A[int(ie[j]),int(je[j])] = 0
            A[int(je[j]),int(ie[j])] = 0

            A[add_site,je[j]] = 1
            A[je[j],add_site] = 1

            R = np.multiply((R+1),A)
    WRITE("Rewiring over")    
    
end = time.time()
WRITE("Iteration Time: "+str(end - start)+"\nStep:"+str(i))    
#to_save['save_distribution']=save_distribution

#saving pickle

#with open("/home/kabir/SOC_model/data.pkl","wb") as pickle_out:
#    pkl.dump(to_save, pickle_out)


