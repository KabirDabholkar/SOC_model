#!/usr/bin/python 
import bohrium as np
import networkx as nx
#import matplotlib.pyplot as plt
import math
import heapq
import pickle as pkl
import time
import os
import sys
import numpy


#Initial Conditions
N = 40
F = 0.001 #f leak (fraction of leak)
f = 1-F
mean_degree = 4.0

K = 4
steps = 10
seed=1
#np.random.seed(seed)

with open(os.getcwd()+"/outputloc.txt",'r') as outputlocfile:
    outputloc=outputlocfile.read().replace('\n','')

#to_save={}
save_loc=outputloc+"data_f"+str(F)+"N"+str(N)+"/"

time_loc=save_loc+"time.txt"

#time_loc="/storage/subhadra/kabir/output/SOC_model/time_2_"+str(F)+".txt"
#if os.path.isfile(time_loc):
#    os.remove(time_loc)
if not os.path.exists(save_loc[:-1]):
    os.makedirs(save_loc[:-1])    
    
########Functions###################################################### 
def WRITE(text):
    with open(time_loc,'a') as time_file:
        time_file.write(text+"\n")
        
        
def save_frame(step,A,dist):
    step_data={'A':A.copy2numpy(),'dist':dist}
    with open(save_loc+"step_"+str(step)+"_data.pkl","wb") as pickle_out:
        pkl.dump(step_data, pickle_out)
    

def avalanche(x,A,f,N):

    count = 0
    degree = np.sum(A,0,dtype=np.int32)
    #print(degree.shape)
    xc = degree + (degree==0)
    spikes = x>=xc
    ava = spikes.copy()
    while np.dot(degree,spikes)> 0:
        ava = (ava+spikes)>0
        spikes = x>=xc
		
        add_particles=np.dot(A,spikes)
        #subtract=np.multiply(np.random.rand(N),spikes)>f
        nonzero=add_particles.nonzero()[0]
        add_particles[nonzero]=np.random.rand(len(nonzero))>f #leak of avalanching particles

        x = x + add_particles
        x = x - np.multiply(spikes,degree)
        #x = x - (np.random.rand(N)>f)
        #x = np.multiply(x,(x>0))
        count = count + 1

    return [x,ava]
"""
def avalanche(x,A,f,N):

    count = 0
    degree = np.array(np.sum(A,0))
    degree = np.reshape(degree, (N,1))
    xc = np.reshape(degree + (degree==0), (N,1))
    spikes = np.multiply(x>=xc, 1)
    spikes = np.reshape(spikes,(N,1)) 
    ava = spikes.copy()
    while np.sum(np.multiply(degree,spikes))> 0:
        ava = np.multiply((ava+spikes)>0,1)

        spikes = x>=xc

        x = x + np.matmul(A,spikes)
        x = x - np.multiply(spikes,degree)
        x = x - (np.random.rand(N,1)>f)
        x = np.multiply(x,(x>0))
        count = count + 1
    return [x,ava]    
"""    
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
print("Networkx version:"+str(nx.__version__))


init_cond={'N':N,'f':f,'mean_degree':mean_degree,'K':K,'steps':steps}

WRITE("N="+str(N)+"\nf="+str(f)+"\nmean_degree="+str(mean_degree)+"\nK="+str(K)+"\nsteps="+str(steps))


#np.random.seed(1)
G_undir = nx.erdos_renyi_graph(N,mean_degree/N,seed=seed)

G_dir = nx.DiGraph()
G_dir.add_nodes_from(range(N))

G_dir.add_edges_from(G_undir.edges())


A = nx.to_numpy_matrix(G_dir.to_undirected(), dtype=np.bool)
A=np.asarray(A,dtype=np.bool)

#fig = plt.figure(figsize=(5, 5))
#plt.title("Adjacency Matrix")
#plt.imshow(A,cmap="Greys",interpolation="none")   
#%matplotlib inline
#sns.heatmap(A, cmap="Greys")
#plt.show()

#storex = np.zeros((N,steps))
#storex_noava = np.zeros((N,steps))

temp = np.arange(0,int(2*G_undir.number_of_edges()))
R = [(temp[2*i],temp[2*i+1]) for i in range(G_undir.number_of_edges())]
recency={}
for edge,rec in zip(G_dir.edges(),R):
    recency[edge]={'recency':rec}

nx.set_edge_attributes(G_dir,recency)
#print()
#print(G_dir.edges(data=True))

degree = np.array(np.sum(A,0),dtype=np.int32)

G_undir.clear()
del(R)
del(temp)
del(recency)



#Initial State
x = np.zeros(N,dtype=np.int32)
for i in range(N):
    if degree[i]>0:
        #np.random.seed(1)
        x[i]=np.random.randint(0,degree[i])
    else:
        x[i]=0
    
xb = x

count = 0
#xsave = [0]*steps
#save_As=[]
#save_distribution = []
WRITE("Starting simulation")

avalanche_sizes=[]


start = time.time()
for i in range(steps):
    print('Step:',i)
    WRITE("Step: "+str(i))
    if np.mod(i,float(steps)/100) == 0: #
        dist = list(np.bincount(list(degree.flat)))
        x_axis = list(np.arange(0,int(np.max(degree)+1)))   
        save_frame(i,A,[x_axis,dist])
        
                
        
    #Particle addition
    
    add_site = np.random.randint(0,N-1)
    
    x[add_site] = x[add_site] + 1



    WRITE("Starting avalanche processing")

    #Avalanche processing
    """
    degree = np.array(np.sum(A,0))
    xc = degree + (degree==0)
    
    spikes = x>=xc

    temp1 = np.zeros(N)
    ava = spikes.copy()
    if np.sum(np.multiply(degree,spikes))>0: 
        [x,temp1] = avalanche(x,A,f,N)
        count = count + 1
    ava = (ava+temp1)>0
    """
    degree = np.sum(A,0,dtype=np.int32)
    print("Particles",np.sum(x))
    #degree = np.reshape(degree, (N,1))
    #print(degree.shape)
    xc = degree + (degree==0)
    spikes = x>=xc
    ava = spikes.copy()
    temp1 = np.zeros(N,dtype=np.bool)
    if np.sum(np.multiply(degree,spikes))>0: 
        [x,temp1] = avalanche(x,A,f,N)
        count = count + 1
        
    #xsave[i] = x
    ava = (ava+temp1)>0
    a = int(np.sum(ava))
    avalanche_sizes.append(a)
    print('Avalanche size',a)
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
    max_edges=[]
    

    if a!=0:
        edges=list(G_dir.edges())
        R=[max(G_dir[edge[0]][edge[1]]['recency']) for edge in edges] #takes maximum of the two endpoints for each edge
        indices = np.array(heapq.nlargest(a, range(len(R)), R.__getitem__))
        #print(indices)        
        max_edges=[edges[ind] for ind in indices]

   
    for (ie,je) in max_edges:
        rec=G_dir[ie][je]['recency']
        left_max=rec[0]>rec[1]
        
        fixed_endpoint=(ie,je)[not left_max]
        ne=(add_site,fixed_endpoint)
            
        if (fixed_endpoint != add_site) & (A[add_site,fixed_endpoint]==0): #A is symmetric so it doesn't matter
            dir_cor=(ie,je) in G_dir.edges()
            oe=(ie,je) #oe is old_edge
            
            r=G_dir[oe[0]][oe[1]]['recency'][left_max]
            G_dir.remove_edge(*oe)
            rec=(0,r)

            G_dir.add_edge(*ne,recency=rec)


            A[ie,je] = 0
            A[je,ie] = 0
            A[add_site,fixed_endpoint] = 1
            A[fixed_endpoint,add_site] = 1
    
    #print(np.all(A == nx.to_numpy_matrix(G_dir.to_undirected(), dtype=np.int)))
    #print(np.all(A==nx.to_numpy_matrix(G_dir.to_undirected(), dtype=np.int)))        
    WRITE("Rewiring over")
    #A = nx.to_numpy_matrix(G_dir.to_undirected(), dtype=np.int)
end = time.time()
WRITE("Iteration Time: "+str(end - start))    
#to_save['save_distribution']=save_distribution
#to_save['save_As']=save_As
#to_save['avalanche_size']=avalanche_size
#to_save['init_cond']=init_cond
#saving pickle

with open(save_loc+"avalanche_sizes.pkl","wb") as pickle_out:
    pkl.dump(avalanche_sizes, pickle_out)



