#!/usr/bin/python 
import numpy as np
import networkx as nx
#import matplotlib.pyplot as plt
import math
import heapq
import pickle as pkl
import time
import os
import sys



F=0.1

with open(os.getcwd()+"/outputloc.txt",'r') as f:
    outputloc=f.read().replace('\n','')

to_save={}
save_loc=outputloc+"data.pkl"
#save_loc="/storage/subhadra/kabir/output/SOC_model/data.pkl"

time_loc=outputloc+"time_2_"+str(F)+".txt"

#time_loc="/storage/subhadra/kabir/output/SOC_model/time_2_"+str(F)+".txt"
if os.path.isfile(time_loc):
    os.remove(time_loc)
########Functions###################################################### 
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
        spikes = np.multiply(x>=xc, 1)
        spikes = np.reshape(spikes,(N,1))
        x = x + A*spikes
        x = x - np.multiply(spikes,degree)
        x = x - (np.random.rand(N,1)>f)
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
    

WRITE("Networkx version:"+str(nx.__version__))

N = 5000
f = F #f leak (fraction of leak)
f = 1-f
mean_degree = 4.0

K = 4
steps = 1000

WRITE("N="+str(N)+"\nf="+str(f)+"\nmean_degree="+str(mean_degree)+"\nK="+str(K)+"\nsteps="+str(steps))


#np.random.seed(1)
G_undir = nx.erdos_renyi_graph(N,mean_degree/N)

G_dir = nx.DiGraph()
G_dir.add_nodes_from(range(N))

G_dir.add_edges_from(G_undir.edges())


A = nx.to_numpy_matrix(G_dir.to_undirected(), dtype=np.int)
#fig = plt.figure(figsize=(5, 5))
#plt.title("Adjacency Matrix")
#plt.imshow(A,cmap="Greys",interpolation="none")   
#%matplotlib inline
#sns.heatmap(A, cmap="Greys")
#plt.show()

#storex = np.zeros((N,steps))
#storex_noava = np.zeros((N,steps))

temp = np.arange(0,2*G_undir.number_of_edges())
R = [(temp[2*i],temp[2*i+1]) for i in range(G_undir.number_of_edges())]
recency={}
for edge,rec in zip(G_dir.edges(),R):
    recency[edge]=rec

nx.set_edge_attributes(G_dir,"recency",recency)
#print()
#print(G_dir.edges(data=True))

degree = np.array(np.sum(A,0))[0]
G_undir.clear()
del(R)
del(temp)
del(recency)



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
save_As=[]
save_distribution = []
WRITE("Starting simulation")

avalanche_size=[]


start = time.time()
for i in range(steps):
    WRITE("Step: "+str(i))
    if np.mod(i,float(steps)/10) == 0: #
        dist = np.bincount(list(degree.flat))
        x_axis = np.arange(0,max(degree)+1)
        save_distribution.append([x_axis,dist])
        save_As.append(A.copy())
    #if np.mod(i,float(steps)/500) == 0:
        #WRITE("Step:"+str(i))
    #Particle addition
    
    add_site = np.random.randint(0,N-1)
    
    x[add_site] = x[add_site] + 1
    #print("Add site:",add_site)
    #print('x',list(x.flat))
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
        count = count + 1
    ava = np.multiply((ava+temp1)>0,1)
    #xsave[i] = x
    a = np.sum(ava)
    avalanche_size.append(a)
    WRITE("Avalanche processing over. Avalanche size:"+str(a))

    
    ##Rewiring
    
    """
    for c1 in range(N):
        for c2 in range(N):
            if Ruse[c1,c2] <= Ruse[c2,c1]:
                Ruse[c1,c2] = -1
            else:
                Ruse[c2,c1] = -1
    #"""
    WRITE("Finding "+str(a)+" max indices")    
    if a!=0:
        edges=list(G_dir.edges())
        
        R=[max(G_dir[edge[0]][edge[1]]['recency']) for edge in edges] #takes maximum of the two endpoints for each edge
        indices = np.array(heapq.nlargest(a, range(len(R)), R.__getitem__))
        #print(indices)        
        ie,je=tuple(zip(*[edges[ind] for ind in indices]))
        """
        print('Steps:',i)
        print('a',a)
        print('R',R)
        print('indices',indices)
        print(ie[0],je[0])
        print(G_dir[ie[0]][je[0]]['recency'])#"""
    WRITE("Done kmax")    
    
    
    
    WRITE("Starting Rewiring")
    
    for j in range(a):
        if (je[j] != add_site) & (A[add_site,je[j]]==0):

            if (ie[j],je[j]) in G_dir.edges():
                r=G_dir[ie[j]][je[j]]['recency'][1]
                G_dir.remove_edge(ie[j],je[j])
                G_dir.add_edge(add_site,je[j],recency=(0,r))
                
            else:
                r=G_dir[je[j]][ie[j]]['recency'][0]
                G_dir.remove_edge(je[j],ie[j])
                G_dir.add_edge(je[j],add_site,recency=(r,0))
            
       
            A[int(ie[j]),int(je[j])] = 0
            A[int(je[j]),int(ie[j])] = 0
            A[add_site,je[j]] = 1
            A[je[j],add_site] = 1
            
            
            
            #this part looks ugly but it's just adding 1 to all the recencies
            for edge in G_dir.edges():
                r=G_dir[edge[0]][edge[1]]['recency']
                G_dir[edge[0]][edge[1]]['recency']=(r[0]+1,r[1]+1)
    #print(np.all(A==nx.to_numpy_matrix(G_dir.to_undirected(), dtype=np.int)))        
    WRITE("Rewiring over")
    #A = nx.to_numpy_matrix(G_dir.to_undirected(), dtype=np.int)
end = time.time()
WRITE("Iteration Time: "+str(end - start))    
to_save['save_distribution']=save_distribution
to_save['save_As']=save_As
to_save['avalanche_size']=avalanche_size
#saving pickle

with open(save_loc,"wb") as pickle_out:
    pkl.dump(to_save, pickle_out)



